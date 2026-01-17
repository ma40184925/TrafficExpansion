"""
PIM-Net 推理脚本
================

将训练好的模型应用于新数据，进行流量映射推理

用法:
    python inference.py --input new_data.csv --output predictions.csv
    python inference.py --checkpoint checkpoints/pim_net_best.pt
"""

import sys
import argparse
import pickle
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# 路径设置
current_file = Path(__file__).resolve()
src_dir = current_file.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, CATEGORY_CONFIG
from models import build_model, get_feature_dims


class PIMNetInference:
    """PIM-Net推理器"""
    
    def __init__(self, checkpoint_path, processor_path=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = self._load_model(checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 加载数据处理器
        if processor_path and Path(processor_path).exists():
            with open(processor_path, 'rb') as f:
                self.processor = pickle.load(f)
            print(f"加载数据处理器: {processor_path}")
        else:
            from models.data_loader import DataProcessor
            self.processor = DataProcessor()
            print("使用新的数据处理器")
    
    def _load_model(self, checkpoint_path):
        """加载模型"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        feature_dims = get_feature_dims()
        config = checkpoint.get('config', {}).get('model', MODEL_CONFIG)
        
        model = build_model(
            feature_dims['fcd_dim'],
            feature_dims['physics_dim'],
            feature_dims['time_dim'],
            config
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"加载模型: {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        
        return model
    
    def preprocess_data(self, df):
        """预处理输入数据"""
        df = df.copy()
        
        # 道路类型处理
        df['kind_x'] = df['kind_x'].astype(str).str.zfill(2)
        df['road_type_idx'] = df['kind_x'].map(
            CATEGORY_CONFIG['road_type_mapping']
        ).fillna(4).astype(int)
        
        # 车道数处理
        df['lane_idx'] = df['width'].map(
            CATEGORY_CONFIG['lane_mapping']
        ).fillna(1).astype(int)
        
        # 时段处理
        df['time_period_idx'] = df['time_period'].map(
            CATEGORY_CONFIG['time_period_mapping']
        ).fillna(0).astype(int)
        
        # 数值保护
        df['theoretical_flow'] = df['theoretical_flow'].clip(lower=0, upper=10000)
        df['density_proxy'] = df['density_proxy'].clip(lower=0, upper=1000)
        
        return df
    
    def create_batch(self, df, idx_start, idx_end):
        """创建batch数据"""
        subset = df.iloc[idx_start:idx_end]
        
        # 浮动车特征
        fcd_features = subset[DATA_CONFIG['fcd_features']].values
        
        # 物理特征
        physics_features = subset[DATA_CONFIG['physics_features']].values
        
        # 时间特征
        time_features = subset[DATA_CONFIG['time_features']].values
        
        # 标准化 (如果处理器已拟合)
        if self.processor.is_fitted:
            fcd_features = self.processor.fcd_scaler.transform(fcd_features)
            physics_features = self.processor.physics_scaler.transform(physics_features)
            time_features = self.processor.time_scaler.transform(time_features)
        
        batch = {
            'fcd_features': torch.FloatTensor(fcd_features).to(self.device),
            'physics_features': torch.FloatTensor(physics_features).to(self.device),
            'time_features': torch.FloatTensor(time_features).to(self.device),
            'road_type': torch.LongTensor(subset['road_type_idx'].values).to(self.device),
            'lane': torch.LongTensor(subset['lane_idx'].values).to(self.device),
            'time_period': torch.LongTensor(subset['time_period_idx'].values).to(self.device),
            'length': torch.FloatTensor(subset['length'].values).to(self.device),
            'theoretical_flow': torch.FloatTensor(subset['theoretical_flow'].values).to(self.device),
        }
        
        return batch
    
    @torch.no_grad()
    def predict(self, df, batch_size=512):
        """
        对DataFrame进行预测
        
        Args:
            df: 输入数据DataFrame
            batch_size: 批大小
        
        Returns:
            predictions: 预测结果数组
            alphas: Alpha系数数组 (如果有)
        """
        # 预处理
        df = self.preprocess_data(df)
        
        all_preds = []
        all_alphas = []
        
        n_samples = len(df)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for i in tqdm(range(n_batches), desc="Inference"):
            idx_start = i * batch_size
            idx_end = min((i + 1) * batch_size, n_samples)
            
            batch = self.create_batch(df, idx_start, idx_end)
            output = self.model(batch)
            
            all_preds.append(output['prediction'].cpu().numpy())
            if 'alpha' in output:
                all_alphas.append(output['alpha'].cpu().numpy())
        
        predictions = np.concatenate(all_preds)
        alphas = np.concatenate(all_alphas) if all_alphas else None
        
        return predictions, alphas
    
    def predict_file(self, input_path, output_path=None, batch_size=512):
        """
        对文件进行预测
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径 (可选)
            batch_size: 批大小
        
        Returns:
            结果DataFrame
        """
        print(f"读取输入文件: {input_path}")
        df = pd.read_csv(input_path, low_memory=False)
        print(f"样本数: {len(df)}")
        
        # 预测
        predictions, alphas = self.predict(df, batch_size)
        
        # 添加预测结果
        df['flow_pred'] = predictions
        if alphas is not None:
            df['alpha'] = alphas
        
        # 保存结果
        if output_path:
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"结果已保存: {output_path}")
        
        return df


def main():
    parser = argparse.ArgumentParser(description="PIM-Net Inference")
    parser.add_argument('--input', type=str, required=True, help='输入数据文件')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径')
    parser.add_argument('--processor', type=str, default=None, help='数据处理器路径')
    parser.add_argument('--batch_size', type=int, default=512, help='批大小')
    args = parser.parse_args()
    
    # 默认路径
    checkpoint_dir = Path(src_dir) / TRAIN_CONFIG['save_dir']
    
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = checkpoint_dir / "pim_net_best.pt"
    
    if args.processor:
        processor_path = Path(args.processor)
    else:
        processor_path = checkpoint_dir / "data_processor.pkl"
    
    # 输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        input_stem = Path(args.input).stem
        output_path = Path(args.input).parent / f"{input_stem}_predictions.csv"
    
    # 推理
    inferencer = PIMNetInference(checkpoint_path, processor_path)
    inferencer.predict_file(args.input, output_path, args.batch_size)


if __name__ == "__main__":
    main()
