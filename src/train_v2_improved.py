"""
PIM-Net v2 改进版训练脚本
========================

改进内容：
1. 物理一致性正则化 (Physical Consistency Regularization)
2. 渐进式λ调度 (Progressive Lambda Scheduling)
3. 门控平衡正则化 (Gate Balance Regularization)

用法:
    python train_v2_improved.py
    python train_v2_improved.py --lambda_phy 0.2
    python train_v2_improved.py --lambda_gate 0.1
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

from config.pim_net_v2_config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG
from models.pim_net_v2 import build_model_v2
from models.data_loader import DataProcessor, get_feature_dims
from utils.path_manager import pm


class PhysicsAwareLoss(nn.Module):
    """
    物理感知损失函数
    ================
    
    包含四个组成部分：
    1. 主损失 (Main Loss): 最终预测与真实值的加权MSE
    2. 物理一致性损失 (Physics Consistency Loss): 物理分支独立预测能力
    3. 门控平衡损失 (Gate Balance Loss): 防止Gate极端化
    4. 高流量加权 (High-Flow Weighting): 对高流量样本增加权重
    
    公式：
        L = L_main_weighted + λ_phy * L_phy + λ_gate * L_gate
    
    其中：
        L_main_weighted = mean(w * (Q_pred - Q_true)^2)
        w = 1 + α * (Q_true - Q_mean) / Q_std  (高流量权重更大)
        L_phy = MSE(Q_phy, Q_true)
        L_gate = (mean(gate) - target_gate)^2
    """
    
    def __init__(self, lambda_phy=0.1, lambda_gate=0.0, target_gate=0.5,
                 use_progressive=True, warmup_epochs=10,
                 use_flow_weighting=True, flow_weight_alpha=0.3):
        """
        Args:
            lambda_phy: 物理一致性正则化系数
            lambda_gate: 门控平衡正则化系数
            target_gate: 目标门控均值
            use_progressive: 是否使用渐进式λ
            warmup_epochs: 渐进预热轮数
            use_flow_weighting: 是否使用高流量加权
            flow_weight_alpha: 流量加权强度 (0=不加权, 1=强加权)
        """
        super().__init__()
        self.lambda_phy_base = lambda_phy
        self.lambda_gate = lambda_gate
        self.target_gate = target_gate
        self.use_progressive = use_progressive
        self.warmup_epochs = warmup_epochs
        self.use_flow_weighting = use_flow_weighting
        self.flow_weight_alpha = flow_weight_alpha
        
        self.mse = nn.MSELoss(reduction='none')  # 改为none以支持加权
        self.current_epoch = 0
        
    def set_epoch(self, epoch):
        """设置当前epoch（用于渐进式调度）"""
        self.current_epoch = epoch
        
    def get_lambda_phy(self):
        """获取当前的λ_phy值"""
        if not self.use_progressive:
            return self.lambda_phy_base
        
        # 渐进式策略：前期λ较大，后期衰减
        # 这样前期强迫物理分支学习，后期让模型自由融合
        if self.current_epoch < self.warmup_epochs:
            # Warmup阶段：λ从2倍逐渐降到1倍
            progress = self.current_epoch / self.warmup_epochs
            factor = 2.0 - progress  # 2.0 → 1.0
        else:
            # 衰减阶段：λ缓慢衰减
            decay_epoch = self.current_epoch - self.warmup_epochs
            factor = 1.0 / (1.0 + 0.05 * decay_epoch)  # 缓慢衰减
        
        return self.lambda_phy_base * factor
    
    def forward(self, output, target):
        """
        计算总损失
        
        Args:
            output: 模型输出字典
            target: 真实目标值 [B]
        
        Returns:
            total_loss: 总损失
            loss_dict: 各分量损失（用于监控）
        """
        prediction = output['prediction']
        q_phy = output.get('q_phy')
        gate = output.get('gate')
        
        # 1. 计算样本权重（高流量加权）
        if self.use_flow_weighting:
            # target是log空间的值，exp后得到原始流量
            # 使用softplus确保权重为正且平滑
            # 权重公式: w = 1 + alpha * softplus(target - median)
            # 这样高于中位数的样本权重更大
            target_centered = target - target.mean()
            weights = 1.0 + self.flow_weight_alpha * torch.tanh(target_centered)
            weights = weights / weights.mean()  # 归一化使均值为1
        else:
            weights = torch.ones_like(target)
        
        # 2. 主损失（加权MSE）
        loss_main_elements = self.mse(prediction, target)  # [B]
        loss_main = (weights * loss_main_elements).mean()
        
        loss_dict = {
            'main': loss_main.item(),
        }
        
        total_loss = loss_main
        
        # 3. 物理一致性损失
        if q_phy is not None and self.lambda_phy_base > 0:
            loss_phy_elements = self.mse(q_phy, target)
            loss_phy = (weights * loss_phy_elements).mean()  # 同样加权
            lambda_phy = self.get_lambda_phy()
            total_loss = total_loss + lambda_phy * loss_phy
            loss_dict['phy'] = loss_phy.item()
            loss_dict['lambda_phy'] = lambda_phy
        
        # 4. 门控平衡损失
        if gate is not None and self.lambda_gate > 0:
            gate_mean = gate.mean()
            loss_gate = (gate_mean - self.target_gate) ** 2
            total_loss = total_loss + self.lambda_gate * loss_gate
            loss_dict['gate'] = loss_gate.item()
            loss_dict['gate_mean'] = gate_mean.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


class ImprovedTrainerV2:
    """改进版PIM-Net v2训练器"""
    
    def __init__(self, model, train_loader, val_loader, processor, config=None,
                 lambda_phy=0.1, lambda_gate=0.0, target_gate=0.5,
                 use_progressive=True, use_flow_weighting=True, 
                 flow_weight_alpha=0.3):
        self.config = config or TRAIN_CONFIG
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.processor = processor
        
        # 设备
        self.device = self._get_device()
        self.model = self.model.to(self.device)
        
        # 改进的损失函数
        self.criterion = PhysicsAwareLoss(
            lambda_phy=lambda_phy,
            lambda_gate=lambda_gate,
            target_gate=target_gate,
            use_progressive=use_progressive,
            warmup_epochs=10,
            use_flow_weighting=use_flow_weighting,
            flow_weight_alpha=flow_weight_alpha
        )
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config['lr_factor'],
            patience=self.config['lr_patience']
        )
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.loss_history = []  # 详细损失记录
        self.best_val_loss = float('inf')
        self.best_metrics = {}
        self.patience_counter = 0
        
        # 保存路径
        self.save_dir = Path(src_dir) / self.config['save_dir']
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Log变换
        self.use_log_transform = self.config.get('use_log_transform', True)
        
        # 正则化参数（用于保存）
        self.lambda_phy = lambda_phy
        self.lambda_gate = lambda_gate
        self.use_flow_weighting = use_flow_weighting
        self.flow_weight_alpha = flow_weight_alpha
        
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
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        self.criterion.set_epoch(epoch)
        
        total_loss = 0
        total_loss_main = 0
        total_loss_phy = 0
        total_gate_mean = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            batch = self._to_device(batch)
            
            self.optimizer.zero_grad()
            output = self.model(batch)
            
            loss, loss_dict = self.criterion(output, batch['target'])
            loss.backward()
            
            if self.config['grad_clip'] > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            total_loss += loss_dict['total']
            total_loss_main += loss_dict['main']
            if 'phy' in loss_dict:
                total_loss_phy += loss_dict['phy']
            if 'gate_mean' in loss_dict:
                total_gate_mean += loss_dict['gate_mean']
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'main': f"{loss_dict['main']:.4f}",
                'gate': f"{loss_dict.get('gate_mean', 0):.3f}"
            })
        
        epoch_loss = {
            'total': total_loss / num_batches,
            'main': total_loss_main / num_batches,
            'phy': total_loss_phy / num_batches if total_loss_phy > 0 else 0,
            'gate_mean': total_gate_mean / num_batches if total_gate_mean > 0 else 0,
            'lambda_phy': self.criterion.get_lambda_phy(),
        }
        
        return epoch_loss
    
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
        all_q_phy = []
        all_q_res = []
        
        for batch in self.val_loader:
            batch = self._to_device(batch)
            output = self.model(batch)
            
            # 只用主损失评估
            loss = nn.MSELoss()(output['prediction'], batch['target'])
            total_loss += loss.item()
            num_batches += 1
            
            all_preds.append(output['prediction'].cpu().numpy())
            all_targets.append(batch['target'].cpu().numpy())
            
            if output.get('alpha') is not None:
                all_alphas.append(output['alpha'].cpu().numpy())
            if output.get('gate') is not None:
                all_gates.append(output['gate'].cpu().numpy())
            if output.get('q_phy') is not None:
                all_q_phy.append(output['q_phy'].cpu().numpy())
            if output.get('q_res') is not None:
                all_q_res.append(output['q_res'].cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        
        # 反变换
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
        
        # 计算物理分支贡献比
        if all_q_phy and all_gates:
            q_phy = np.concatenate(all_q_phy)
            q_res = np.concatenate(all_q_res)
            gates = np.concatenate(all_gates)
            
            phy_contrib = np.mean(np.abs((1 - gates) * q_phy))
            res_contrib = np.mean(np.abs(gates * q_res))
            total_contrib = phy_contrib + res_contrib + 1e-8
            
            metrics['phy_ratio'] = phy_contrib / total_contrib
            metrics['res_ratio'] = res_contrib / total_contrib
        
        return metrics
    
    def train(self, epochs=None):
        """完整训练流程"""
        epochs = epochs or self.config['epochs']
        patience = self.config['patience']
        
        print(f"\n{'='*70}")
        print("PIM-Net v2 改进版训练")
        print(f"{'='*70}")
        print(f"物理一致性正则化: λ_phy = {self.lambda_phy}")
        print(f"门控平衡正则化: λ_gate = {self.lambda_gate}")
        print(f"高流量加权: {'启用' if self.use_flow_weighting else '禁用'} (α={self.flow_weight_alpha})")
        print(f"渐进式调度: 启用")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss['total'])
            self.loss_history.append(train_loss)
            
            val_metrics = self.validate()
            val_loss = val_metrics['loss']
            self.val_losses.append(val_loss)
            
            # 学习率调整
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            lr_info = f" | LR: {old_lr:.2e}→{new_lr:.2e}" if new_lr != old_lr else ""
            
            # 打印信息
            phy_ratio = val_metrics.get('phy_ratio', 0) * 100
            gate_mean = val_metrics.get('gate_mean', 0)
            
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {train_loss['total']:.4f}/{val_loss:.4f} | "
                  f"MAE: {val_metrics['mae']:.1f} | "
                  f"R²: {val_metrics['r2']:.4f} | "
                  f"λ_phy: {train_loss['lambda_phy']:.3f}{lr_info}")
            
            print(f"         Alpha: {val_metrics.get('alpha_mean', 0):.2f}±{val_metrics.get('alpha_std', 0):.2f} | "
                  f"Gate: {gate_mean:.3f} | "
                  f"PhyRatio: {phy_ratio:.1f}%")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_metrics = val_metrics.copy()
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"         ✓ 保存最佳模型 (R²: {val_metrics['r2']:.4f}, PhyRatio: {phy_ratio:.1f}%)")
            else:
                self.patience_counter += 1
            
            # 早停
            if self.patience_counter >= patience:
                print(f"\n早停触发 (耐心值: {patience})")
                break
        
        total_time = time.time() - start_time
        print("=" * 70)
        print(f"训练完成 | 总用时: {total_time/60:.1f} 分钟")
        print(f"最佳结果: R²={self.best_metrics['r2']:.4f}, "
              f"MAE={self.best_metrics['mae']:.1f}, "
              f"PhyRatio={self.best_metrics.get('phy_ratio', 0)*100:.1f}%")
        
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
            'loss_history': self.loss_history,
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'config': {
                'model': MODEL_CONFIG,
                'train': self.config,
                'lambda_phy': self.lambda_phy,
                'lambda_gate': self.lambda_gate,
            }
        }
        
        path = self.save_dir / f"{self.config['model_name']}_improved_latest.pt"
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.save_dir / f"{self.config['model_name']}_improved_best.pt"
            torch.save(checkpoint, best_path)


def main():
    parser = argparse.ArgumentParser(description="PIM-Net v2 Improved Training")
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--lambda_phy', type=float, default=0.1,
                        help='物理一致性正则化系数')
    parser.add_argument('--lambda_gate', type=float, default=0.0,
                        help='门控平衡正则化系数')
    parser.add_argument('--target_gate', type=float, default=0.5,
                        help='目标门控均值')
    parser.add_argument('--no_progressive', action='store_true',
                        help='禁用渐进式λ调度')
    parser.add_argument('--no_flow_weight', action='store_true',
                        help='禁用高流量加权')
    parser.add_argument('--flow_weight_alpha', type=float, default=0.3,
                        help='高流量加权强度')
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
    print("PIM-Net v2 Improved: 物理一致性正则化训练")
    print("=" * 70)
    print(f"数据路径: {data_path}")
    print(f"λ_phy: {args.lambda_phy}")
    print(f"λ_gate: {args.lambda_gate}")
    print(f"高流量加权: {'禁用' if args.no_flow_weight else '启用'} (α={args.flow_weight_alpha})")
    print(f"渐进式调度: {'禁用' if args.no_progressive else '启用'}")
    
    # 加载数据
    processor = DataProcessor(train_config=train_config)
    train_loader, val_loader, test_loader = processor.prepare_dataloaders(
        data_path,
        batch_size=batch_size
    )
    
    # 获取特征维度
    feature_dims = get_feature_dims()
    
    # 构建模型
    model = build_model_v2(
        feature_dims['fcd_dim'],
        feature_dims['physics_dim'],
        feature_dims['time_dim']
    )
    
    # 创建训练器
    trainer = ImprovedTrainerV2(
        model, train_loader, val_loader, processor, train_config,
        lambda_phy=args.lambda_phy,
        lambda_gate=args.lambda_gate,
        target_gate=args.target_gate,
        use_progressive=not args.no_progressive,
        use_flow_weighting=not args.no_flow_weight,
        flow_weight_alpha=args.flow_weight_alpha
    )
    
    # 训练
    trainer.train()
    
    # 保存处理器
    processor_path = trainer.save_dir / "data_processor_improved.pkl"
    with open(processor_path, 'wb') as f:
        pickle.dump(processor, f)
    print(f"数据处理器已保存: {processor_path}")


if __name__ == "__main__":
    main()
