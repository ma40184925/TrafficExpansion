"""
数据加载与预处理模块
====================

负责加载特征工程后的数据，进行标准化和数据集划分

主要改进：
1. 分层随机采样（按道路类型、时段分层）
2. 支持目标值Log变换
3. 更稳健的数据预处理
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

# 路径设置
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from config import DATA_CONFIG, CATEGORY_CONFIG, TRAIN_CONFIG


class TrafficFlowDataset(Dataset):
    """交通流量数据集"""
    
    def __init__(self, data_dict):
        """
        Args:
            data_dict: 包含各类特征张量的字典
        """
        self.fcd_features = torch.FloatTensor(data_dict['fcd_features'])
        self.physics_features = torch.FloatTensor(data_dict['physics_features'])
        self.time_features = torch.FloatTensor(data_dict['time_features'])
        self.road_type = torch.LongTensor(data_dict['road_type'])
        self.lane = torch.LongTensor(data_dict['lane'])
        self.time_period = torch.LongTensor(data_dict['time_period'])
        self.length = torch.FloatTensor(data_dict['length'])
        self.target = torch.FloatTensor(data_dict['target'])
        
        # 理论流量 (用于物理分支)
        self.theoretical_flow = torch.FloatTensor(data_dict['theoretical_flow'])
        
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        return {
            'fcd_features': self.fcd_features[idx],
            'physics_features': self.physics_features[idx],
            'time_features': self.time_features[idx],
            'road_type': self.road_type[idx],
            'lane': self.lane[idx],
            'time_period': self.time_period[idx],
            'length': self.length[idx],
            'theoretical_flow': self.theoretical_flow[idx],
            'target': self.target[idx],
        }


class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config=None, train_config=None):
        self.config = config or DATA_CONFIG
        self.train_config = train_config or TRAIN_CONFIG
        self.category_config = CATEGORY_CONFIG
        
        # 标准化器
        self.fcd_scaler = StandardScaler()
        self.physics_scaler = StandardScaler()
        self.time_scaler = StandardScaler()
        self.length_scaler = StandardScaler()
        self.theo_scaler = StandardScaler()  # 理论流量标准化
        
        self.is_fitted = False
        
        # Log变换设置
        self.use_log_transform = self.train_config.get('use_log_transform', True)
        
    def load_data(self, file_path):
        """加载数据文件"""
        print(f"加载数据: {file_path}")
        df = pd.read_csv(file_path, low_memory=False)
        print(f"数据量: {len(df)} 条, {df['卡口编号'].nunique()} 个卡口")
        return df
    
    def preprocess(self, df):
        """预处理数据"""
        df = df.copy()
        
        # 1. 处理道路类型
        df['kind_x'] = df['kind_x'].astype(str).str.zfill(2)
        df['road_type_idx'] = df['kind_x'].map(
            self.category_config['road_type_mapping']
        ).fillna(4).astype(int)  # 未知类型映射到市镇村道
        
        # 2. 处理车道数
        df['lane_idx'] = df[self.config['lane_col']].map(
            self.category_config['lane_mapping']
        ).fillna(1).astype(int)  # 未知映射到2-3车道
        
        # 3. 处理时段
        df['time_period_idx'] = df[self.config['time_period_col']].map(
            self.category_config['time_period_mapping']
        ).fillna(0).astype(int)
        
        # 4. 过滤无效数据
        # 移除目标值为0或负的记录
        df = df[df[self.config['target_col']] > 0].copy()
        
        # 移除浮动车流量为0的记录
        df = df[df['fcd_flow'] > 0].copy()
        
        # 5. 处理异常值 - 更严格的数值保护
        # theoretical_flow 数值保护：基于物理边界条件
        df['theoretical_flow'] = df['theoretical_flow'].clip(lower=0.01, upper=5000)
        df['density_proxy'] = df['density_proxy'].clip(lower=0, upper=500)
        df['fcd_flow_per_length'] = df['fcd_flow_per_length'].clip(lower=0, upper=1000)
        
        # 6. 创建分层标签（用于分层采样）
        df['stratify_label'] = (
            df['road_type_idx'].astype(str) + '_' + 
            df['time_period_idx'].astype(str)
        )
        
        print(f"预处理后数据量: {len(df)} 条")
        
        return df
    
    def extract_features(self, df):
        """提取特征"""
        # 浮动车特征
        fcd_features = df[self.config['fcd_features']].values
        
        # 物理特征
        physics_features = df[self.config['physics_features']].values
        
        # 时间特征
        time_features = df[self.config['time_features']].values
        
        # 类别特征
        road_type = df['road_type_idx'].values
        lane = df['lane_idx'].values
        time_period = df['time_period_idx'].values
        
        # 路段长度
        length = df[self.config['length_col']].values.reshape(-1, 1)
        
        # 理论流量 (单独提取，用于物理分支)
        theoretical_flow = df['theoretical_flow'].values.reshape(-1, 1)
        
        # 目标值
        target = df[self.config['target_col']].values.reshape(-1, 1)
        
        return {
            'fcd_features': fcd_features,
            'physics_features': physics_features,
            'time_features': time_features,
            'road_type': road_type,
            'lane': lane,
            'time_period': time_period,
            'length': length,
            'theoretical_flow': theoretical_flow,
            'target': target,
        }
    
    def fit_scalers(self, features_dict):
        """拟合标准化器"""
        self.fcd_scaler.fit(features_dict['fcd_features'])
        self.physics_scaler.fit(features_dict['physics_features'])
        self.time_scaler.fit(features_dict['time_features'])
        self.length_scaler.fit(features_dict['length'])
        self.theo_scaler.fit(features_dict['theoretical_flow'])
        self.is_fitted = True
        
    def transform_features(self, features_dict, transform_target=True):
        """标准化特征"""
        if not self.is_fitted:
            raise RuntimeError("Scalers not fitted. Call fit_scalers first.")
        
        # 目标值处理
        target = features_dict['target'].flatten()
        if self.use_log_transform and transform_target:
            target = np.log1p(target)  # log(1 + x)
        
        return {
            'fcd_features': self.fcd_scaler.transform(features_dict['fcd_features']),
            'physics_features': self.physics_scaler.transform(features_dict['physics_features']),
            'time_features': self.time_scaler.transform(features_dict['time_features']),
            'road_type': features_dict['road_type'],
            'lane': features_dict['lane'],
            'time_period': features_dict['time_period'],
            'length': self.length_scaler.transform(features_dict['length']).flatten(),
            'theoretical_flow': self.theo_scaler.transform(features_dict['theoretical_flow']).flatten(),
            'target': target,
        }
    
    def inverse_transform_target(self, target):
        """反变换目标值"""
        if self.use_log_transform:
            return np.expm1(target)  # exp(x) - 1
        return target
    
    def split_data_stratified(self, df, random_seed=None):
        """
        分层随机划分数据集
        
        不按卡口划分，而是将所有(卡口, 时间)样本混合，
        按道路类型和时段进行分层采样，确保各集合中样本分布一致。
        
        学术依据：映射关系主要受道路属性和流体力学规律支配，
        而非受特定地理位置的ID支配。
        """
        if random_seed is None:
            random_seed = self.config['random_seed']
        
        train_ratio = self.config['train_ratio']
        val_ratio = self.config['val_ratio']
        test_ratio = self.config['test_ratio']
        
        # 使用分层标签
        stratify_col = df['stratify_label']
        
        # 处理稀有类别：如果某个分层组样本太少，合并到相邻组
        label_counts = stratify_col.value_counts()
        min_samples = 10  # 每个分层组最少需要10个样本
        
        # 对于样本过少的组，使用简化的分层（只按道路类型）
        if (label_counts < min_samples).any():
            print("  注意: 部分分层组样本较少，使用道路类型进行分层")
            stratify_col = df['road_type_idx']
        
        # 第一次划分：分出测试集
        train_val_idx, test_idx = train_test_split(
            df.index,
            test_size=test_ratio,
            random_state=random_seed,
            stratify=stratify_col
        )
        
        # 第二次划分：从训练验证集中分出验证集
        train_val_stratify = stratify_col.loc[train_val_idx]
        val_size = val_ratio / (train_ratio + val_ratio)
        
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
            random_state=random_seed,
            stratify=train_val_stratify
        )
        
        train_df = df.loc[train_idx]
        val_df = df.loc[val_idx]
        test_df = df.loc[test_idx]
        
        print(f"数据划分 (分层随机采样):")
        print(f"  训练集: {len(train_df)} 条 ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  验证集: {len(val_df)} 条 ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  测试集: {len(test_df)} 条 ({len(test_df)/len(df)*100:.1f}%)")
        
        # 打印分布信息
        print(f"\n各集合道路类型分布:")
        for name, subset in [('训练', train_df), ('验证', val_df), ('测试', test_df)]:
            dist = subset['road_type_idx'].value_counts(normalize=True).sort_index()
            dist_str = ', '.join([f"{i}:{v:.1%}" for i, v in dist.items()])
            print(f"  {name}: {dist_str}")
        
        return train_df, val_df, test_df
    
    def prepare_dataloaders(self, file_path, batch_size=256):
        """
        完整的数据准备流程
        
        Returns:
            train_loader, val_loader, test_loader
        """
        # 加载数据
        df = self.load_data(file_path)
        
        # 预处理
        df = self.preprocess(df)
        
        # 分层随机划分数据集
        train_df, val_df, test_df = self.split_data_stratified(df)
        
        # 提取特征
        train_features = self.extract_features(train_df)
        val_features = self.extract_features(val_df)
        test_features = self.extract_features(test_df)
        
        # 拟合标准化器 (只用训练集)
        self.fit_scalers(train_features)
        
        # 标准化
        train_features = self.transform_features(train_features)
        val_features = self.transform_features(val_features)
        test_features = self.transform_features(test_features)
        
        # 创建数据集
        train_dataset = TrafficFlowDataset(train_features)
        val_dataset = TrafficFlowDataset(val_features)
        test_dataset = TrafficFlowDataset(test_features)
        
        # 创建DataLoader
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader


def get_feature_dims():
    """获取各特征维度"""
    return {
        'fcd_dim': len(DATA_CONFIG['fcd_features']),
        'physics_dim': len(DATA_CONFIG['physics_features']),
        'time_dim': len(DATA_CONFIG['time_features']),
    }


if __name__ == "__main__":
    # 测试数据加载
    from utils.path_manager import pm
    
    processor = DataProcessor()
    data_path = pm.get_processed_path(DATA_CONFIG['input_file'])
    
    train_loader, val_loader, test_loader = processor.prepare_dataloaders(
        data_path, 
        batch_size=256
    )
    
    # 打印一个batch的信息
    batch = next(iter(train_loader))
    print("\nBatch info:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")
    
    # 验证Log变换
    print(f"\n目标值范围 (Log变换后): [{batch['target'].min():.2f}, {batch['target'].max():.2f}]")
