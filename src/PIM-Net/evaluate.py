"""
PIM-Net 评估脚本
================

对训练好的模型进行全面评估，包括：
1. 整体性能指标 (MAE, RMSE, MAPE, R²)
2. 分道路类型评估
3. 分时段评估
4. Alpha系数分析 (物理分支可解释性)
5. 可视化分析

用法:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/pim_net_best.pt
    python evaluate.py --ablation all  # 对比消融实验
"""

import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 路径设置
current_file = Path(__file__).resolve()
src_dir = current_file.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, CATEGORY_CONFIG
from models import build_model, DataProcessor, get_feature_dims
from utils.path_manager import pm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Evaluator:
    """PIM-Net评估器"""
    
    def __init__(self, model, test_loader, device='cuda'):
        self.model = model
        self.test_loader = test_loader
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 类别映射 (反向)
        self.road_type_names = {v: k for k, v in CATEGORY_CONFIG['road_type_mapping'].items()}
        self.road_type_names = {
            0: '城市高速', 1: '国道', 2: '省道', 3: '县道', 4: '市镇村道'
        }
        self.time_period_names = {v: k for k, v in CATEGORY_CONFIG['time_period_mapping'].items()}
        
    def _to_device(self, batch):
        """将batch数据移到设备上"""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
    
    @torch.no_grad()
    def predict_all(self):
        """对测试集进行完整预测"""
        all_results = {
            'predictions': [],
            'targets': [],
            'alphas': [],
            'bases': [],
            'residuals': [],
            'road_types': [],
            'time_periods': [],
            'fcd_flows': [],
            'fcd_speeds': [],
            'theoretical_flows': [],
        }
        
        for batch in tqdm(self.test_loader, desc="Predicting"):
            batch = self._to_device(batch)
            output = self.model(batch)
            
            all_results['predictions'].append(output['prediction'].cpu().numpy())
            all_results['targets'].append(batch['target'].cpu().numpy())
            all_results['road_types'].append(batch['road_type'].cpu().numpy())
            all_results['time_periods'].append(batch['time_period'].cpu().numpy())
            all_results['fcd_flows'].append(batch['fcd_features'][:, 0].cpu().numpy())
            all_results['fcd_speeds'].append(batch['fcd_features'][:, 1].cpu().numpy())
            all_results['theoretical_flows'].append(batch['theoretical_flow'].cpu().numpy())
            
            if 'alpha' in output:
                all_results['alphas'].append(output['alpha'].cpu().numpy())
                all_results['bases'].append(output['base'].cpu().numpy())
                all_results['residuals'].append(output['residual'].cpu().numpy())
        
        # 拼接所有结果
        for key in all_results:
            if all_results[key]:
                all_results[key] = np.concatenate(all_results[key])
        
        return all_results
    
    def compute_metrics(self, predictions, targets, prefix=''):
        """计算评估指标"""
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        
        # MAPE (避免除零)
        mask = targets > 1
        if mask.sum() > 0:
            mape = np.mean(np.abs(predictions[mask] - targets[mask]) / targets[mask]) * 100
        else:
            mape = np.nan
        
        # R²
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        
        metrics = {
            f'{prefix}MAE': mae,
            f'{prefix}RMSE': rmse,
            f'{prefix}MAPE': mape,
            f'{prefix}R2': r2,
            f'{prefix}Count': len(predictions),
        }
        
        return metrics
    
    def evaluate(self, save_dir=None):
        """完整评估流程"""
        print("\n" + "=" * 60)
        print("模型评估")
        print("=" * 60)
        
        # 预测
        results = self.predict_all()
        predictions = results['predictions']
        targets = results['targets']
        
        # === 1. 整体指标 ===
        print("\n[1] 整体性能指标")
        print("-" * 40)
        overall_metrics = self.compute_metrics(predictions, targets)
        for key, value in overall_metrics.items():
            if key.endswith('Count'):
                print(f"  样本数: {value}")
            elif key.endswith('MAPE'):
                print(f"  {key}: {value:.2f}%")
            elif key.endswith('R2'):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value:.2f}")
        
        # === 2. 分道路类型评估 ===
        print("\n[2] 分道路类型评估")
        print("-" * 40)
        road_metrics = {}
        for road_idx, road_name in self.road_type_names.items():
            mask = results['road_types'] == road_idx
            if mask.sum() > 0:
                metrics = self.compute_metrics(predictions[mask], targets[mask])
                road_metrics[road_name] = metrics
                print(f"  {road_name}:")
                print(f"    MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, "
                      f"R²={metrics['R2']:.4f}, N={metrics['Count']}")
        
        # === 3. 分时段评估 ===
        print("\n[3] 分时段评估")
        print("-" * 40)
        time_metrics = {}
        for time_idx, time_name in self.time_period_names.items():
            mask = results['time_periods'] == time_idx
            if mask.sum() > 0:
                metrics = self.compute_metrics(predictions[mask], targets[mask])
                time_metrics[time_name] = metrics
                print(f"  {time_name}:")
                print(f"    MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, "
                      f"R²={metrics['R2']:.4f}, N={metrics['Count']}")
        
        # === 4. Alpha系数分析 ===
        if len(results['alphas']) > 0:
            print("\n[4] Alpha系数分析 (物理分支可解释性)")
            print("-" * 40)
            alphas = results['alphas']
            print(f"  整体: mean={np.mean(alphas):.4f}, std={np.std(alphas):.4f}")
            print(f"  范围: [{np.min(alphas):.4f}, {np.max(alphas):.4f}]")
            
            # 按路况分析
            speeds = results['fcd_speeds']
            low_speed_mask = speeds < 20  # 低速/拥堵
            high_speed_mask = speeds > 40  # 高速/畅通
            
            if low_speed_mask.sum() > 0:
                print(f"  低速状态 (<20km/h): alpha={np.mean(alphas[low_speed_mask]):.4f}")
            if high_speed_mask.sum() > 0:
                print(f"  高速状态 (>40km/h): alpha={np.mean(alphas[high_speed_mask]):.4f}")
            
            # 按道路类型分析Alpha
            print("\n  按道路类型:")
            for road_idx, road_name in self.road_type_names.items():
                mask = results['road_types'] == road_idx
                if mask.sum() > 0:
                    print(f"    {road_name}: alpha={np.mean(alphas[mask]):.4f} ± {np.std(alphas[mask]):.4f}")
        
        # === 5. 可视化 ===
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self.plot_results(results, save_dir)
        
        return {
            'overall': overall_metrics,
            'by_road_type': road_metrics,
            'by_time_period': time_metrics,
            'results': results,
        }
    
    def plot_results(self, results, save_dir):
        """生成可视化图表"""
        print("\n[5] 生成可视化图表")
        print("-" * 40)
        
        predictions = results['predictions']
        targets = results['targets']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('PIM-Net 评估结果', fontsize=14, fontweight='bold')
        
        # 1. 预测值 vs 真实值散点图
        ax1 = axes[0, 0]
        ax1.scatter(targets, predictions, alpha=0.3, s=5)
        max_val = max(targets.max(), predictions.max())
        ax1.plot([0, max_val], [0, max_val], 'r--', label='理想线')
        ax1.set_xlabel('真实流量')
        ax1.set_ylabel('预测流量')
        ax1.set_title('预测值 vs 真实值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 残差分布
        ax2 = axes[0, 1]
        residuals = predictions - targets
        ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--')
        ax2.set_xlabel('预测残差')
        ax2.set_ylabel('频次')
        ax2.set_title(f'残差分布 (mean={np.mean(residuals):.2f})')
        
        # 3. 分道路类型的MAE
        ax3 = axes[0, 2]
        road_names = []
        road_maes = []
        for road_idx, road_name in self.road_type_names.items():
            mask = results['road_types'] == road_idx
            if mask.sum() > 0:
                road_names.append(road_name)
                road_maes.append(np.mean(np.abs(predictions[mask] - targets[mask])))
        ax3.bar(road_names, road_maes, color='steelblue', edgecolor='black')
        ax3.set_ylabel('MAE')
        ax3.set_title('分道路类型MAE')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 分时段的MAE
        ax4 = axes[1, 0]
        time_names = []
        time_maes = []
        for time_idx in sorted(self.time_period_names.keys()):
            time_name = self.time_period_names[time_idx]
            mask = results['time_periods'] == time_idx
            if mask.sum() > 0:
                time_names.append(time_name)
                time_maes.append(np.mean(np.abs(predictions[mask] - targets[mask])))
        ax4.bar(time_names, time_maes, color='coral', edgecolor='black')
        ax4.set_ylabel('MAE')
        ax4.set_title('分时段MAE')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Alpha分布 (如果有)
        ax5 = axes[1, 1]
        if len(results['alphas']) > 0:
            ax5.hist(results['alphas'], bins=50, edgecolor='black', alpha=0.7, color='green')
            ax5.axvline(np.mean(results['alphas']), color='red', linestyle='--', 
                       label=f'mean={np.mean(results["alphas"]):.3f}')
            ax5.set_xlabel('Alpha系数')
            ax5.set_ylabel('频次')
            ax5.set_title('物理系数Alpha分布')
            ax5.legend()
        else:
            ax5.text(0.5, 0.5, '无Alpha系数\n(物理分支未启用)', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('物理系数Alpha分布')
        
        # 6. 理论流量 vs 真实流量
        ax6 = axes[1, 2]
        if len(results['theoretical_flows']) > 0:
            ax6.scatter(results['theoretical_flows'], targets, alpha=0.3, s=5, label='理论流量')
            ax6.scatter(results['theoretical_flows'], predictions, alpha=0.3, s=5, label='预测流量')
            ax6.set_xlabel('理论流量')
            ax6.set_ylabel('流量')
            ax6.set_title('理论流量 vs 真实/预测流量')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        fig_path = save_dir / 'evaluation_results.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ 保存图表: {fig_path}")
        
        plt.close()
        
        # === 额外图表: Alpha与速度的关系 ===
        if len(results['alphas']) > 0:
            fig2, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(results['fcd_speeds'], results['alphas'], 
                               c=results['road_types'], cmap='tab10', alpha=0.5, s=10)
            ax.set_xlabel('浮动车速度 (km/h)')
            ax.set_ylabel('Alpha系数')
            ax.set_title('Alpha系数与速度的关系')
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('道路类型')
            
            fig2_path = save_dir / 'alpha_vs_speed.png'
            fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ 保存图表: {fig2_path}")
            plt.close()


def load_model(checkpoint_path, device='cuda'):
    """加载训练好的模型"""
    # PyTorch 2.6+ 需要设置 weights_only=False 来加载包含numpy数组的checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 获取特征维度
    feature_dims = get_feature_dims()

    # 构建模型
    config = checkpoint.get('config', {}).get('model', MODEL_CONFIG)
    model = build_model(
        feature_dims['fcd_dim'],
        feature_dims['physics_dim'],
        feature_dims['time_dim'],
        config
    )

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"加载模型: {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best Val Loss: {checkpoint['best_val_loss']:.4f}")

    return model


def compare_ablations(checkpoint_dir, test_loader, device='cuda'):
    """对比消融实验结果"""
    print("\n" + "=" * 60)
    print("消融实验对比")
    print("=" * 60)

    checkpoint_dir = Path(checkpoint_dir)
    ablation_types = ['full', 'no_physics', 'no_embedding', 'no_residual']

    results = {}

    for ablation in ablation_types:
        ckpt_path = checkpoint_dir / f"pim_net_{ablation}_best.pt"
        if not ckpt_path.exists():
            print(f"  跳过 {ablation}: 检查点不存在")
            continue

        print(f"\n评估: {ablation}")
        model = load_model(ckpt_path, device)
        evaluator = Evaluator(model, test_loader, device)
        eval_results = evaluator.evaluate()
        results[ablation] = eval_results['overall']

    # 汇总对比
    if results:
        print("\n" + "=" * 60)
        print("消融实验结果汇总")
        print("=" * 60)
        print(f"{'模型':<15} {'MAE':>10} {'RMSE':>10} {'MAPE':>10} {'R²':>10}")
        print("-" * 55)
        for name, metrics in results.items():
            print(f"{name:<15} {metrics['MAE']:>10.2f} {metrics['RMSE']:>10.2f} "
                  f"{metrics['MAPE']:>10.2f}% {metrics['R2']:>10.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="PIM-Net Evaluation")
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径')
    parser.add_argument('--data_path', type=str, default=None, help='数据路径')
    parser.add_argument('--ablation', type=str, default=None,
                        choices=['all'], help='对比消融实验')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='输出目录')
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

    if args.ablation == 'all':
        # 对比消融实验
        checkpoint_dir = Path(src_dir) / TRAIN_CONFIG['save_dir']
        compare_ablations(checkpoint_dir, test_loader)
    else:
        # 评估单个模型
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
        else:
            checkpoint_path = Path(src_dir) / TRAIN_CONFIG['save_dir'] / "pim_net_best.pt"

        if not checkpoint_path.exists():
            print(f"错误: 检查点不存在 {checkpoint_path}")
            return

        model = load_model(checkpoint_path)
        evaluator = Evaluator(model, test_loader)
        evaluator.evaluate(save_dir=output_dir)


if __name__ == "__main__":
    main()