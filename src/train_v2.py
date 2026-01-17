"""
PIM-Net v2 训练脚本
===================

用法:
    python train_v2.py
    python train_v2.py --epochs 200 --batch_size 512
    python train_v2.py --ablation all  # 运行所有消融实验
"""

import os
import sys
import argparse
import time
import pickle
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from tqdm import tqdm

# 路径设置
current_file = Path(__file__).resolve()
src_dir = current_file.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# 使用v2配置
from config.pim_net_v2_config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG
from models.pim_net_v2 import build_model_v2, PIMNetV2Ablation
from models.data_loader import DataProcessor, get_feature_dims
from utils.path_manager import pm


class TrainerV2:
    """PIM-Net v2 训练器"""
    
    def __init__(self, model, train_loader, val_loader, processor, config=None):
        self.config = config or TRAIN_CONFIG
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.processor = processor
        
        # 设备
        self.device = self._get_device()
        self.model = self.model.to(self.device)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # 学习率调度器
        if self.config['lr_scheduler'] == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config['lr_factor'],
                patience=self.config['lr_patience']
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs']
            )
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_metrics = {}
        self.patience_counter = 0
        
        # 保存路径
        self.save_dir = Path(src_dir) / self.config['save_dir']
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Log变换
        self.use_log_transform = self.config.get('use_log_transform', True)
        
    def _get_device(self):
        if self.config['device'] == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("使用CPU")
        return device
    
    def _to_device(self, batch):
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            batch = self._to_device(batch)
            
            self.optimizer.zero_grad()
            output = self.model(batch)
            
            loss = self.criterion(output['prediction'], batch['target'])
            loss.backward()
            
            if self.config['grad_clip'] > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_preds = []
        all_targets = []
        all_alphas = []
        all_gates = []
        
        for batch in self.val_loader:
            batch = self._to_device(batch)
            output = self.model(batch)
            
            loss = self.criterion(output['prediction'], batch['target'])
            total_loss += loss.item()
            num_batches += 1
            
            all_preds.append(output['prediction'].cpu().numpy())
            all_targets.append(batch['target'].cpu().numpy())
            
            if output.get('alpha') is not None:
                all_alphas.append(output['alpha'].cpu().numpy())
            if output.get('gate') is not None:
                all_gates.append(output['gate'].cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        
        # 反变换计算原始尺度指标
        if self.use_log_transform:
            preds_orig = np.expm1(preds)
            targets_orig = np.expm1(targets)
        else:
            preds_orig = preds
            targets_orig = targets
        
        preds_orig = np.maximum(preds_orig, 0)
        
        # 计算指标
        mae = np.mean(np.abs(preds_orig - targets_orig))
        rmse = np.sqrt(np.mean((preds_orig - targets_orig) ** 2))
        
        mask = targets_orig > 10
        mape = np.mean(np.abs(preds_orig[mask] - targets_orig[mask]) / targets_orig[mask]) * 100 if mask.sum() > 0 else np.nan
        
        ss_res = np.sum((targets_orig - preds_orig) ** 2)
        ss_tot = np.sum((targets_orig - np.mean(targets_orig)) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        metrics = {
            'loss': avg_loss,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
        }
        
        # Alpha和Gate统计
        if all_alphas:
            alphas = np.concatenate(all_alphas)
            metrics['alpha_mean'] = np.mean(alphas)
            metrics['alpha_std'] = np.std(alphas)
        
        if all_gates:
            gates = np.concatenate(all_gates)
            metrics['gate_mean'] = np.mean(gates)
            metrics['gate_std'] = np.std(gates)
        
        return metrics
    
    def train(self, epochs=None):
        """完整训练流程"""
        epochs = epochs or self.config['epochs']
        patience = self.config['patience']
        
        print(f"\n开始训练 (共 {epochs} 轮)")
        print(f"Log变换: {'启用' if self.use_log_transform else '禁用'}")
        print("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            val_metrics = self.validate()
            val_loss = val_metrics['loss']
            self.val_losses.append(val_loss)
            
            # 学习率调整
            old_lr = self.optimizer.param_groups[0]['lr']
            if self.config['lr_scheduler'] == 'plateau':
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]['lr']
            
            lr_info = f" | LR: {old_lr:.2e}→{new_lr:.2e}" if new_lr != old_lr else ""
            
            # 打印信息
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"MAE: {val_metrics['mae']:.1f} | "
                  f"RMSE: {val_metrics['rmse']:.1f} | "
                  f"R²: {val_metrics['r2']:.4f}{lr_info}")
            
            # 打印Alpha和Gate
            if 'alpha_mean' in val_metrics and 'gate_mean' in val_metrics:
                print(f"         Alpha: {val_metrics['alpha_mean']:.3f}±{val_metrics['alpha_std']:.3f} | "
                      f"Gate: {val_metrics['gate_mean']:.3f}±{val_metrics['gate_std']:.3f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_metrics = val_metrics.copy()
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"         ✓ 保存最佳模型 (R²: {val_metrics['r2']:.4f})")
            else:
                self.patience_counter += 1
            
            # 早停
            if self.patience_counter >= patience:
                print(f"\n早停触发 (耐心值: {patience})")
                break
        
        total_time = time.time() - start_time
        print("=" * 80)
        print(f"训练完成 | 总用时: {total_time/60:.1f} 分钟")
        print(f"最佳验证结果: R²={self.best_metrics['r2']:.4f}, "
              f"MAE={self.best_metrics['mae']:.1f}, "
              f"RMSE={self.best_metrics['rmse']:.1f}")
        
        return self.best_val_loss
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'config': {
                'model': MODEL_CONFIG,
                'train': self.config,
            }
        }
        
        path = self.save_dir / f"{self.config['model_name']}_latest.pt"
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.save_dir / f"{self.config['model_name']}_best.pt"
            torch.save(checkpoint, best_path)


def run_ablation_v2(ablation_type, train_loader, val_loader, processor,
                    fcd_dim, physics_dim, time_dim):
    """
    运行消融实验
    
    Args:
        ablation_type: 
            - 'full': 完整模型
            - 'no_se': 去掉SE-Block
            - 'no_dcn': 去掉DCN（只用Deep）
            - 'no_gate': 去掉门控融合（用加法）
            - 'no_physics': 去掉物理分支
    """
    print(f"\n{'='*70}")
    print(f"消融实验: {ablation_type}")
    print(f"{'='*70}")
    
    # 根据消融类型构建模型
    if ablation_type == 'full':
        model = build_model_v2(fcd_dim, physics_dim, time_dim)
    elif ablation_type == 'no_se':
        model = PIMNetV2Ablation(fcd_dim, physics_dim, time_dim,
                                 use_se=False, use_dcn=True, use_gate=True, use_physics=True)
    elif ablation_type == 'no_dcn':
        model = PIMNetV2Ablation(fcd_dim, physics_dim, time_dim,
                                 use_se=True, use_dcn=False, use_gate=True, use_physics=True)
    elif ablation_type == 'no_gate':
        model = PIMNetV2Ablation(fcd_dim, physics_dim, time_dim,
                                 use_se=True, use_dcn=True, use_gate=False, use_physics=True)
    elif ablation_type == 'no_physics':
        model = PIMNetV2Ablation(fcd_dim, physics_dim, time_dim,
                                 use_se=True, use_dcn=True, use_gate=False, use_physics=False)
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")
    
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,}")
    
    # 修改保存名称
    config = TRAIN_CONFIG.copy()
    config['model_name'] = f"pim_net_v2_{ablation_type}"
    
    trainer = TrainerV2(model, train_loader, val_loader, processor, config)
    best_loss = trainer.train()
    
    return best_loss, trainer.best_metrics


def main():
    parser = argparse.ArgumentParser(description="PIM-Net v2 Training")
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--ablation', type=str, default=None,
                        choices=['full', 'no_se', 'no_dcn', 'no_gate', 'no_physics', 'all'])
    parser.add_argument('--data_path', type=str, default=None)
    args = parser.parse_args()
    
    # 更新配置
    train_config = TRAIN_CONFIG.copy()
    if args.epochs:
        train_config['epochs'] = args.epochs
    if args.lr:
        train_config['learning_rate'] = args.lr
    
    batch_size = args.batch_size or train_config['batch_size']
    
    # 数据路径
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = pm.get_processed_path(DATA_CONFIG['input_file'])
    
    print("=" * 70)
    print("PIM-Net v2: Physics-Informed Mapping Network")
    print("           with Deep Feature Interaction")
    print("=" * 70)
    print(f"数据路径: {data_path}")
    
    # 加载数据
    processor = DataProcessor(train_config=train_config)
    train_loader, val_loader, test_loader = processor.prepare_dataloaders(
        data_path,
        batch_size=batch_size
    )
    
    # 获取特征维度
    feature_dims = get_feature_dims()
    fcd_dim = feature_dims['fcd_dim']
    physics_dim = feature_dims['physics_dim']
    time_dim = feature_dims['time_dim']
    
    # 运行训练
    if args.ablation == 'all':
        # 运行所有消融实验
        results = {}
        ablation_types = ['full', 'no_se', 'no_dcn', 'no_gate', 'no_physics']
        
        for ablation_type in ablation_types:
            best_loss, best_metrics = run_ablation_v2(
                ablation_type, train_loader, val_loader, processor,
                fcd_dim, physics_dim, time_dim
            )
            results[ablation_type] = best_metrics
        
        # 汇总结果
        print("\n" + "=" * 70)
        print("消融实验结果汇总")
        print("=" * 70)
        print(f"{'模型':<15} {'MAE':>10} {'RMSE':>10} {'MAPE':>10} {'R²':>10}")
        print("-" * 55)
        for name, metrics in results.items():
            mape_str = f"{metrics['mape']:.1f}%" if not np.isnan(metrics['mape']) else "N/A"
            print(f"{name:<15} {metrics['mae']:>10.1f} {metrics['rmse']:>10.1f} "
                  f"{mape_str:>10} {metrics['r2']:>10.4f}")
    
    elif args.ablation:
        # 单个消融实验
        run_ablation_v2(
            args.ablation, train_loader, val_loader, processor,
            fcd_dim, physics_dim, time_dim
        )
    
    else:
        # 正常训练完整模型
        model = build_model_v2(fcd_dim, physics_dim, time_dim)
        trainer = TrainerV2(model, train_loader, val_loader, processor, train_config)
        trainer.train()
        
        # 保存数据处理器
        processor_path = trainer.save_dir / "data_processor.pkl"
        with open(processor_path, 'wb') as f:
            pickle.dump(processor, f)
        print(f"数据处理器已保存: {processor_path}")


if __name__ == "__main__":
    main()
