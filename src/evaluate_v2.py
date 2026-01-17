"""
PIM-Net v2 评估脚本
===================

对训练好的v2模型进行全面评估，包括：
1. 整体性能指标
2. 分道路类型/时段评估
3. Alpha系数分析
4. Gate门控系数分析（新增）
5. SE注意力权重分析（新增）
6. 可视化分析

用法:
    python evaluate_v2.py
    python evaluate_v2.py --checkpoint checkpoints/pim_net_v2_best.pt
"""

import os
import sys
import argparse
import pickle
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# 路径设置
current_file = Path(__file__).resolve()
src_dir = current_file.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from config.pim_net_v2_config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, EVAL_CONFIG, CATEGORY_CONFIG
from models.pim_net_v2 import build_model_v2
from models.data_loader import DataProcessor, get_feature_dims
from utils.path_manager import pm

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class EvaluatorV2:
    """PIM-Net v2 评估器"""
    
    def __init__(self, model, test_loader, processor, device='cuda'):
        self.model = model
        self.test_loader = test_loader
        self.processor = processor
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.use_log_transform = TRAIN_CONFIG.get('use_log_transform', True)
        
        self.road_type_names = {
            0: '城市高速', 1: '国道', 2: '省道', 3: '县道', 4: '市镇村道'
        }
        self.time_period_names = {v: k for k, v in CATEGORY_CONFIG['time_period_mapping'].items()}
        
        # 特征名称（用于SE权重可视化）
        self.feature_names = [
            'fcd_flow', 'fcd_speed', 'fcd_status',  # FCD特征
            'Q_theo', 'K_proxy',                      # 物理特征
            'h_sin', 'h_cos', 'w_sin', 'w_cos', 'weekend',  # 时间特征
        ] + [f'road_emb_{i}' for i in range(16)] \
          + [f'lane_emb_{i}' for i in range(8)] \
          + [f'time_emb_{i}' for i in range(16)] \
          + ['length']
        
    def _to_device(self, batch):
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}
    
    @torch.no_grad()
    def predict_all(self):
        """完整预测"""
        results = {
            'predictions_log': [], 'targets_log': [],
            'predictions_orig': [], 'targets_orig': [],
            'alphas': [], 'gates': [], 'se_weights': [],
            'q_phy': [], 'q_res': [],
            'road_types': [], 'time_periods': [],
            'fcd_speeds': [], 'theoretical_flows': [],
        }
        
        for batch in tqdm(self.test_loader, desc="Predicting"):
            batch = self._to_device(batch)
            output = self.model(batch)
            
            pred_log = output['prediction'].cpu().numpy()
            target_log = batch['target'].cpu().numpy()
            
            results['predictions_log'].append(pred_log)
            results['targets_log'].append(target_log)
            
            if self.use_log_transform:
                pred_orig = np.expm1(pred_log)
                target_orig = np.expm1(target_log)
            else:
                pred_orig = pred_log
                target_orig = target_log
            
            pred_orig = np.maximum(pred_orig, 0)
            
            results['predictions_orig'].append(pred_orig)
            results['targets_orig'].append(target_orig)
            
            results['road_types'].append(batch['road_type'].cpu().numpy())
            results['time_periods'].append(batch['time_period'].cpu().numpy())
            results['fcd_speeds'].append(batch['fcd_features'][:, 1].cpu().numpy())
            results['theoretical_flows'].append(batch['theoretical_flow'].cpu().numpy())
            
            if output.get('alpha') is not None:
                results['alphas'].append(output['alpha'].cpu().numpy())
            if output.get('gate') is not None:
                results['gates'].append(output['gate'].cpu().numpy())
            if output.get('se_weights') is not None:
                results['se_weights'].append(output['se_weights'].cpu().numpy())
            if output.get('q_phy') is not None:
                results['q_phy'].append(output['q_phy'].cpu().numpy())
            if output.get('q_res') is not None:
                results['q_res'].append(output['q_res'].cpu().numpy())
        
        # 拼接
        for key in results:
            if results[key]:
                results[key] = np.concatenate(results[key])
        
        return results
    
    def compute_metrics(self, predictions, targets):
        """计算指标"""
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        
        mask = targets > 10
        mape = np.mean(np.abs(predictions[mask] - targets[mask]) / targets[mask]) * 100 if mask.sum() > 0 else np.nan
        
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2, 'Count': len(predictions)}
    
    def evaluate(self, save_dir=None):
        """完整评估"""
        print("\n" + "=" * 70)
        print("PIM-Net v2 模型评估")
        print("=" * 70)
        
        results = self.predict_all()
        
        pred_orig = results['predictions_orig']
        target_orig = results['targets_orig']
        
        # === 1. 整体指标 ===
        print("\n" + "=" * 70)
        print("[1] 整体性能指标 (原始尺度)")
        print("=" * 70)
        
        metrics = self.compute_metrics(pred_orig, target_orig)
        print(f"  MAE:  {metrics['MAE']:.1f} pcu/h")
        print(f"  RMSE: {metrics['RMSE']:.1f} pcu/h")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  R²:   {metrics['R2']:.4f}")
        print(f"  样本数: {metrics['Count']}")
        
        # === 2. 分道路类型 ===
        print("\n" + "=" * 70)
        print("[2] 分道路类型评估")
        print("=" * 70)
        print(f"{'道路类型':<10} {'MAE':>10} {'RMSE':>10} {'MAPE':>10} {'R²':>10} {'N':>8}")
        print("-" * 60)
        
        for road_idx in sorted(self.road_type_names.keys()):
            mask = results['road_types'] == road_idx
            if mask.sum() > 0:
                m = self.compute_metrics(pred_orig[mask], target_orig[mask])
                print(f"{self.road_type_names[road_idx]:<10} {m['MAE']:>10.1f} {m['RMSE']:>10.1f} "
                      f"{m['MAPE']:>9.1f}% {m['R2']:>10.4f} {m['Count']:>8}")
        
        # === 3. 分时段 ===
        print("\n" + "=" * 70)
        print("[3] 分时段评估")
        print("=" * 70)
        print(f"{'时段':<10} {'MAE':>10} {'RMSE':>10} {'MAPE':>10} {'R²':>10} {'N':>8}")
        print("-" * 60)
        
        for time_idx in sorted(self.time_period_names.keys()):
            mask = results['time_periods'] == time_idx
            if mask.sum() > 0:
                m = self.compute_metrics(pred_orig[mask], target_orig[mask])
                print(f"{self.time_period_names[time_idx]:<10} {m['MAE']:>10.1f} {m['RMSE']:>10.1f} "
                      f"{m['MAPE']:>9.1f}% {m['R2']:>10.4f} {m['Count']:>8}")
        
        # === 4. Alpha分析 ===
        if len(results['alphas']) > 0:
            print("\n" + "=" * 70)
            print("[4] Alpha系数分析")
            print("=" * 70)
            alphas = results['alphas']
            print(f"  mean = {np.mean(alphas):.4f}")
            print(f"  std  = {np.std(alphas):.4f}")
            print(f"  range = [{np.min(alphas):.4f}, {np.max(alphas):.4f}]")
        
        # === 5. Gate分析（新增）===
        if len(results['gates']) > 0:
            print("\n" + "=" * 70)
            print("[5] Gate门控系数分析 (物理-数据融合权重)")
            print("=" * 70)
            gates = results['gates']
            print(f"  mean = {np.mean(gates):.4f} (0=信物理, 1=信数据)")
            print(f"  std  = {np.std(gates):.4f}")
            print(f"  range = [{np.min(gates):.4f}, {np.max(gates):.4f}]")
            
            # 按道路类型分析Gate
            print("\n  按道路类型:")
            for road_idx in sorted(self.road_type_names.keys()):
                mask = results['road_types'] == road_idx
                if mask.sum() > 0:
                    print(f"    {self.road_type_names[road_idx]}: "
                          f"gate={np.mean(gates[mask]):.4f}±{np.std(gates[mask]):.4f}")
            
            # 按时段分析Gate
            print("\n  按时段:")
            for time_idx in sorted(self.time_period_names.keys()):
                mask = results['time_periods'] == time_idx
                if mask.sum() > 0:
                    print(f"    {self.time_period_names[time_idx]}: "
                          f"gate={np.mean(gates[mask]):.4f}±{np.std(gates[mask]):.4f}")
        
        # === 6. 物理分支vs残差分支贡献分析 ===
        if len(results['q_phy']) > 0 and len(results['q_res']) > 0:
            print("\n" + "=" * 70)
            print("[6] 双流贡献分析")
            print("=" * 70)
            q_phy = results['q_phy']
            q_res = results['q_res']
            gates = results['gates'] if len(results['gates']) > 0 else np.zeros_like(q_phy)
            
            # 计算各分支的绝对贡献
            phy_contrib = np.abs((1 - gates) * q_phy)
            res_contrib = np.abs(gates * q_res)
            total_contrib = phy_contrib + res_contrib + 1e-8
            
            phy_ratio = phy_contrib / total_contrib
            res_ratio = res_contrib / total_contrib
            
            print(f"  物理分支贡献比: {np.mean(phy_ratio)*100:.1f}%")
            print(f"  残差分支贡献比: {np.mean(res_ratio)*100:.1f}%")
        
        # === 7. 可视化 ===
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self.plot_results(results, save_dir)
        
        return {'metrics': metrics, 'results': results}
    
    def plot_results(self, results, save_dir):
        """生成可视化图表"""
        print("\n" + "=" * 70)
        print("[7] 生成可视化图表")
        print("=" * 70)
        
        pred_orig = results['predictions_orig']
        target_orig = results['targets_orig']
        
        # === 图1: 主评估结果 (2x3) ===
        fig, axes = plt.subplots(2, 3, figsize=(16, 11))
        fig.suptitle('PIM-Net v2 评估结果', fontsize=16, fontweight='bold')
        
        # 1.1 预测vs真实
        ax = axes[0, 0]
        ax.scatter(target_orig, pred_orig, alpha=0.3, s=8, c='steelblue')
        max_val = max(target_orig.max(), pred_orig.max())
        ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='理想线')
        ax.set_xlabel('真实流量 (pcu/h)')
        ax.set_ylabel('预测流量 (pcu/h)')
        ax.set_title('预测值 vs 真实值')
        
        ss_res = np.sum((target_orig - pred_orig) ** 2)
        ss_tot = np.sum((target_orig - np.mean(target_orig)) ** 2)
        r2 = 1 - ss_res / ss_tot
        ax.text(0.05, 0.95, f'$R^2$ = {r2:.4f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.legend()
        
        # 1.2 残差分布
        ax = axes[0, 1]
        residuals = pred_orig - target_orig
        ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(0, color='red', linestyle='--', lw=2)
        ax.axvline(np.mean(residuals), color='orange', lw=2, label=f'mean={np.mean(residuals):.1f}')
        ax.set_xlabel('预测残差 (pcu/h)')
        ax.set_ylabel('频次')
        ax.set_title('残差分布')
        ax.legend()
        
        # 1.3 分道路类型MAE
        ax = axes[0, 2]
        road_names, road_maes, road_r2s = [], [], []
        for road_idx in sorted(self.road_type_names.keys()):
            mask = results['road_types'] == road_idx
            if mask.sum() > 0:
                road_names.append(self.road_type_names[road_idx])
                road_maes.append(np.mean(np.abs(pred_orig[mask] - target_orig[mask])))
                ss_res = np.sum((target_orig[mask] - pred_orig[mask]) ** 2)
                ss_tot = np.sum((target_orig[mask] - np.mean(target_orig[mask])) ** 2)
                road_r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)
        
        bars = ax.bar(road_names, road_maes, color='steelblue', edgecolor='black', alpha=0.8)
        ax.set_ylabel('MAE (pcu/h)')
        ax.set_title('分道路类型MAE')
        for bar, r2_val in zip(bars, road_r2s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'$R^2$={r2_val:.2f}', ha='center', va='bottom', fontsize=9)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        
        # 1.4 Gate分布
        ax = axes[1, 0]
        if len(results['gates']) > 0:
            gates = results['gates']
            ax.hist(gates, bins=50, edgecolor='black', alpha=0.7, color='green')
            ax.axvline(np.mean(gates), color='red', linestyle='--', lw=2,
                      label=f'mean={np.mean(gates):.3f}')
            ax.set_xlabel('Gate系数')
            ax.set_ylabel('频次')
            ax.set_title('门控系数分布 (0=物理, 1=数据)')
            ax.legend()
        else:
            ax.text(0.5, 0.5, '无Gate数据', ha='center', va='center', transform=ax.transAxes)
        
        # 1.5 Alpha分布
        ax = axes[1, 1]
        if len(results['alphas']) > 0:
            alphas = results['alphas']
            ax.hist(alphas, bins=50, edgecolor='black', alpha=0.7, color='purple')
            ax.axvline(np.mean(alphas), color='red', linestyle='--', lw=2,
                      label=f'mean={np.mean(alphas):.3f}')
            ax.set_xlabel('Alpha系数')
            ax.set_ylabel('频次')
            ax.set_title('物理校正系数分布')
            ax.legend()
        
        # 1.6 分时段MAE
        ax = axes[1, 2]
        time_names, time_maes = [], []
        for time_idx in sorted(self.time_period_names.keys()):
            mask = results['time_periods'] == time_idx
            if mask.sum() > 0:
                time_names.append(self.time_period_names[time_idx])
                time_maes.append(np.mean(np.abs(pred_orig[mask] - target_orig[mask])))
        
        ax.bar(time_names, time_maes, color='coral', edgecolor='black', alpha=0.8)
        ax.set_ylabel('MAE (pcu/h)')
        ax.set_title('分时段MAE')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        
        plt.tight_layout()
        fig_path = save_dir / 'evaluation_results_v2.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ 保存: {fig_path}")
        plt.close()
        
        # === 图2: Gate深度分析 ===
        if len(results['gates']) > 0:
            fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
            
            gates = results['gates']
            
            # 2.1 Gate vs 速度
            ax = axes2[0]
            ax.scatter(results['fcd_speeds'], gates, alpha=0.3, s=10,
                      c=results['road_types'], cmap='tab10')
            ax.set_xlabel('浮动车速度 (标准化)')
            ax.set_ylabel('Gate系数')
            ax.set_title('Gate与速度的关系')
            ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
            
            # 2.2 Gate vs 预测误差
            ax = axes2[1]
            abs_errors = np.abs(pred_orig - target_orig)
            ax.scatter(gates, abs_errors, alpha=0.3, s=10)
            ax.set_xlabel('Gate系数')
            ax.set_ylabel('绝对误差 (pcu/h)')
            ax.set_title('Gate与预测误差的关系')
            
            # 2.3 Gate箱线图（按道路类型）
            ax = axes2[2]
            gate_by_road = []
            labels = []
            for road_idx in sorted(self.road_type_names.keys()):
                mask = results['road_types'] == road_idx
                if mask.sum() > 0:
                    gate_by_road.append(gates[mask])
                    labels.append(self.road_type_names[road_idx])
            ax.boxplot(gate_by_road, tick_labels=labels)
            ax.set_ylabel('Gate系数')
            ax.set_title('分道路类型Gate分布')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
            
            plt.tight_layout()
            fig2_path = save_dir / 'gate_analysis.png'
            fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ 保存: {fig2_path}")
            plt.close()
        
        # === 图3: 双流贡献分析 ===
        if len(results['q_phy']) > 0 and len(results['gates']) > 0:
            fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
            
            q_phy = results['q_phy']
            q_res = results['q_res']
            gates = results['gates']
            
            # 3.1 物理vs残差输出
            ax = axes3[0]
            ax.scatter(q_phy, q_res, alpha=0.3, s=10, c=gates, cmap='RdYlGn')
            ax.set_xlabel('物理分支输出 Q_phy')
            ax.set_ylabel('残差分支输出 Q_res')
            ax.set_title('双流输出分布')
            cbar = plt.colorbar(ax.collections[0], ax=ax)
            cbar.set_label('Gate')
            
            # 3.2 贡献比例饼图
            ax = axes3[1]
            phy_contrib = np.mean(np.abs((1 - gates) * q_phy))
            res_contrib = np.mean(np.abs(gates * q_res))
            
            ax.pie([phy_contrib, res_contrib],
                   labels=['物理分支', '残差分支'],
                   autopct='%1.1f%%',
                   colors=['steelblue', 'coral'],
                   explode=[0.05, 0.05])
            ax.set_title('双流平均贡献比例')
            
            plt.tight_layout()
            fig3_path = save_dir / 'dual_stream_analysis.png'
            fig3.savefig(fig3_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ 保存: {fig3_path}")
            plt.close()


def load_model_v2(checkpoint_path, device='cuda'):
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    feature_dims = get_feature_dims()
    
    config = checkpoint.get('config', {}).get('model', MODEL_CONFIG)
    model = build_model_v2(
        feature_dims['fcd_dim'],
        feature_dims['physics_dim'],
        feature_dims['time_dim'],
        config
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"加载模型: {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best Val Loss: {checkpoint['best_val_loss']:.4f}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="PIM-Net v2 Evaluation")
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='evaluation_results_v2')
    args = parser.parse_args()
    
    # 数据路径
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = pm.get_processed_path(DATA_CONFIG['input_file'])
    
    # 加载数据
    processor = DataProcessor()
    train_loader, val_loader, test_loader = processor.prepare_dataloaders(
        data_path,
        batch_size=TRAIN_CONFIG['batch_size']
    )
    
    # 输出目录
    output_dir = Path(src_dir) / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        # checkpoint_path = Path(src_dir) / TRAIN_CONFIG['save_dir'] / "pim_net_v2_best.pt"
        checkpoint_path = Path(src_dir) / TRAIN_CONFIG['save_dir'] / "pim_net_v2_improved_best.pt"

    if not checkpoint_path.exists():
        print(f"错误: 检查点不存在 {checkpoint_path}")
        return
    
    model = load_model_v2(checkpoint_path)
    
    # 评估
    evaluator = EvaluatorV2(model, test_loader, processor)
    evaluator.evaluate(save_dir=output_dir)


if __name__ == "__main__":
    main()
