"""
PIM-Net 评估脚本
================

对训练好的模型进行全面评估，包括：
1. 整体性能指标 (MAE, RMSE, MAPE, R²) - 同时展示Log空间和原始尺度
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

    def __init__(self, model, test_loader, processor, device='cuda'):
        self.model = model
        self.test_loader = test_loader
        self.processor = processor  # 用于反变换
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

        # Log变换设置
        self.use_log_transform = TRAIN_CONFIG.get('use_log_transform', True)

        # 类别映射 (反向)
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
            'predictions_log': [],  # Log空间预测
            'targets_log': [],  # Log空间目标
            'predictions_orig': [],  # 原始尺度预测
            'targets_orig': [],  # 原始尺度目标
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

            pred_log = output['prediction'].cpu().numpy()
            target_log = batch['target'].cpu().numpy()

            all_results['predictions_log'].append(pred_log)
            all_results['targets_log'].append(target_log)

            # 反变换到原始尺度
            if self.use_log_transform:
                pred_orig = np.expm1(pred_log)  # exp(x) - 1
                target_orig = np.expm1(target_log)
            else:
                pred_orig = pred_log
                target_orig = target_log

            # 确保非负
            pred_orig = np.maximum(pred_orig, 0)

            all_results['predictions_orig'].append(pred_orig)
            all_results['targets_orig'].append(target_orig)

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

        # MAPE (避免除零，使用更宽松的阈值)
        mask = targets > 10  # 原始尺度用10，避免小流量的相对误差过大
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
        print("\n" + "=" * 70)
        print("PIM-Net 模型评估")
        print("=" * 70)

        # 预测
        results = self.predict_all()

        # Log空间和原始尺度的预测/目标
        pred_log = results['predictions_log']
        target_log = results['targets_log']
        pred_orig = results['predictions_orig']
        target_orig = results['targets_orig']

        # === 1. 整体指标 ===
        print("\n" + "=" * 70)
        print("[1] 整体性能指标")
        print("=" * 70)

        # Log空间指标
        log_metrics = self.compute_metrics(pred_log, target_log)
        print("\n【Log空间】(训练时使用的空间)")
        print("-" * 40)
        print(f"  MAE:  {log_metrics['MAE']:.4f}")
        print(f"  RMSE: {log_metrics['RMSE']:.4f}")
        print(f"  R²:   {log_metrics['R2']:.4f}")

        # 原始尺度指标
        orig_metrics = self.compute_metrics(pred_orig, target_orig)
        print("\n【原始尺度】(实际流量，单位: pcu/h)")
        print("-" * 40)
        print(f"  MAE:  {orig_metrics['MAE']:.1f} pcu/h")
        print(f"  RMSE: {orig_metrics['RMSE']:.1f} pcu/h")
        print(f"  MAPE: {orig_metrics['MAPE']:.2f}%")
        print(f"  R²:   {orig_metrics['R2']:.4f}")
        print(f"  样本数: {orig_metrics['Count']}")

        # 流量统计
        print("\n【流量统计】")
        print("-" * 40)
        print(f"  真实流量: mean={np.mean(target_orig):.1f}, "
              f"std={np.std(target_orig):.1f}, "
              f"range=[{np.min(target_orig):.1f}, {np.max(target_orig):.1f}]")
        print(f"  预测流量: mean={np.mean(pred_orig):.1f}, "
              f"std={np.std(pred_orig):.1f}, "
              f"range=[{np.min(pred_orig):.1f}, {np.max(pred_orig):.1f}]")

        # === 2. 分道路类型评估 ===
        print("\n" + "=" * 70)
        print("[2] 分道路类型评估 (原始尺度)")
        print("=" * 70)
        road_metrics = {}
        print(f"\n{'道路类型':<10} {'MAE':>10} {'RMSE':>10} {'MAPE':>10} {'R²':>10} {'样本数':>10}")
        print("-" * 62)
        for road_idx in sorted(self.road_type_names.keys()):
            road_name = self.road_type_names[road_idx]
            mask = results['road_types'] == road_idx
            if mask.sum() > 0:
                metrics = self.compute_metrics(pred_orig[mask], target_orig[mask])
                road_metrics[road_name] = metrics
                print(f"{road_name:<10} {metrics['MAE']:>10.1f} {metrics['RMSE']:>10.1f} "
                      f"{metrics['MAPE']:>9.1f}% {metrics['R2']:>10.4f} {metrics['Count']:>10}")

        # === 3. 分时段评估 ===
        print("\n" + "=" * 70)
        print("[3] 分时段评估 (原始尺度)")
        print("=" * 70)
        time_metrics = {}
        print(f"\n{'时段':<10} {'MAE':>10} {'RMSE':>10} {'MAPE':>10} {'R²':>10} {'样本数':>10}")
        print("-" * 62)
        for time_idx in sorted(self.time_period_names.keys()):
            time_name = self.time_period_names[time_idx]
            mask = results['time_periods'] == time_idx
            if mask.sum() > 0:
                metrics = self.compute_metrics(pred_orig[mask], target_orig[mask])
                time_metrics[time_name] = metrics
                print(f"{time_name:<10} {metrics['MAE']:>10.1f} {metrics['RMSE']:>10.1f} "
                      f"{metrics['MAPE']:>9.1f}% {metrics['R2']:>10.4f} {metrics['Count']:>10}")

        # === 4. Alpha系数分析 ===
        if len(results['alphas']) > 0:
            print("\n" + "=" * 70)
            print("[4] Alpha系数分析 (物理分支可解释性)")
            print("=" * 70)
            alphas = results['alphas']
            print(f"\n  整体统计:")
            print(f"    mean = {np.mean(alphas):.6f}")
            print(f"    std  = {np.std(alphas):.6f}")
            print(f"    range = [{np.min(alphas):.6f}, {np.max(alphas):.6f}]")

            # 按速度分析
            speeds = results['fcd_speeds']
            print(f"\n  按速度状态 (标准化后的速度值):")
            low_speed_mask = speeds < -0.5
            mid_speed_mask = (speeds >= -0.5) & (speeds <= 0.5)
            high_speed_mask = speeds > 0.5

            if low_speed_mask.sum() > 0:
                print(f"    低速 (z<-0.5): alpha={np.mean(alphas[low_speed_mask]):.6f}, N={low_speed_mask.sum()}")
            if mid_speed_mask.sum() > 0:
                print(f"    中速 (-0.5≤z≤0.5): alpha={np.mean(alphas[mid_speed_mask]):.6f}, N={mid_speed_mask.sum()}")
            if high_speed_mask.sum() > 0:
                print(f"    高速 (z>0.5): alpha={np.mean(alphas[high_speed_mask]):.6f}, N={high_speed_mask.sum()}")

            # 按道路类型分析Alpha
            print(f"\n  按道路类型:")
            for road_idx in sorted(self.road_type_names.keys()):
                road_name = self.road_type_names[road_idx]
                mask = results['road_types'] == road_idx
                if mask.sum() > 0:
                    print(f"    {road_name}: alpha={np.mean(alphas[mask]):.6f} ± {np.std(alphas[mask]):.6f}")

        # === 5. 误差分析 ===
        print("\n" + "=" * 70)
        print("[5] 误差分析")
        print("=" * 70)

        errors = pred_orig - target_orig
        abs_errors = np.abs(errors)
        rel_errors = np.abs(errors) / (target_orig + 1) * 100

        print(f"\n  绝对误差分布:")
        print(f"    mean = {np.mean(errors):.1f} (偏差方向)")
        print(f"    |mean| = {np.mean(abs_errors):.1f}")
        print(f"    median = {np.median(abs_errors):.1f}")
        print(f"    P90 = {np.percentile(abs_errors, 90):.1f}")
        print(f"    P95 = {np.percentile(abs_errors, 95):.1f}")

        print(f"\n  相对误差分布 (%):")
        print(f"    mean = {np.mean(rel_errors):.1f}%")
        print(f"    median = {np.median(rel_errors):.1f}%")
        print(f"    P90 = {np.percentile(rel_errors, 90):.1f}%")

        # 分流量区间分析
        print(f"\n  分流量区间误差 (原始尺度):")
        flow_bins = [(0, 100), (100, 300), (300, 500), (500, 1000), (1000, np.inf)]
        for low, high in flow_bins:
            mask = (target_orig >= low) & (target_orig < high)
            if mask.sum() > 0:
                bin_mae = np.mean(abs_errors[mask])
                bin_mape = np.mean(np.abs(errors[mask]) / (target_orig[mask] + 1)) * 100
                high_str = f"{high:.0f}" if high != np.inf else "∞"
                print(f"    [{low:.0f}, {high_str}): MAE={bin_mae:.1f}, MAPE={bin_mape:.1f}%, N={mask.sum()}")

        # === 6. 可视化 ===
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self.plot_results(results, save_dir)

        return {
            'log_metrics': log_metrics,
            'orig_metrics': orig_metrics,
            'by_road_type': road_metrics,
            'by_time_period': time_metrics,
            'results': results,
        }

    def plot_results(self, results, save_dir):
        """生成可视化图表"""
        print("\n" + "=" * 70)
        print("[6] 生成可视化图表")
        print("=" * 70)

        pred_orig = results['predictions_orig']
        target_orig = results['targets_orig']
        pred_log = results['predictions_log']
        target_log = results['targets_log']

        fig, axes = plt.subplots(2, 3, figsize=(16, 11))
        fig.suptitle('PIM-Net 评估结果', fontsize=16, fontweight='bold')

        # 1. 预测值 vs 真实值散点图 (原始尺度)
        ax1 = axes[0, 0]
        ax1.scatter(target_orig, pred_orig, alpha=0.3, s=8, c='steelblue')
        max_val = max(target_orig.max(), pred_orig.max())
        ax1.plot([0, max_val], [0, max_val], 'r--', lw=2, label='理想线')
        ax1.set_xlabel('真实流量 (pcu/h)', fontsize=11)
        ax1.set_ylabel('预测流量 (pcu/h)', fontsize=11)
        ax1.set_title('预测值 vs 真实值 (原始尺度)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 添加R²标注
        ss_res = np.sum((target_orig - pred_orig) ** 2)
        ss_tot = np.sum((target_orig - np.mean(target_orig)) ** 2)
        r2 = 1 - ss_res / ss_tot
        ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes,
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 2. 残差分布 (原始尺度)
        ax2 = axes[0, 1]
        residuals = pred_orig - target_orig
        ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax2.axvline(0, color='red', linestyle='--', lw=2)
        ax2.axvline(np.mean(residuals), color='orange', linestyle='-', lw=2, label=f'mean={np.mean(residuals):.1f}')
        ax2.set_xlabel('预测残差 (pcu/h)', fontsize=11)
        ax2.set_ylabel('频次', fontsize=11)
        ax2.set_title('残差分布 (原始尺度)', fontsize=12)
        ax2.legend()

        # 3. 分道路类型的MAE (原始尺度)
        ax3 = axes[0, 2]
        road_names = []
        road_maes = []
        road_r2s = []
        for road_idx in sorted(self.road_type_names.keys()):
            road_name = self.road_type_names[road_idx]
            mask = results['road_types'] == road_idx
            if mask.sum() > 0:
                road_names.append(road_name)
                road_maes.append(np.mean(np.abs(pred_orig[mask] - target_orig[mask])))
                ss_res = np.sum((target_orig[mask] - pred_orig[mask]) ** 2)
                ss_tot = np.sum((target_orig[mask] - np.mean(target_orig[mask])) ** 2)
                road_r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)

        x = np.arange(len(road_names))
        bars = ax3.bar(x, road_maes, color='steelblue', edgecolor='black', alpha=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels(road_names, rotation=30, ha='right')
        ax3.set_ylabel('MAE (pcu/h)', fontsize=11)
        ax3.set_title('分道路类型MAE', fontsize=12)

        # 在柱子上标注R²
        for i, (bar, r2_val) in enumerate(zip(bars, road_r2s)):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f'R²={r2_val:.2f}', ha='center', va='bottom', fontsize=9)

        # 4. 分时段的MAE (原始尺度)
        ax4 = axes[1, 0]
        time_names = []
        time_maes = []
        for time_idx in sorted(self.time_period_names.keys()):
            time_name = self.time_period_names[time_idx]
            mask = results['time_periods'] == time_idx
            if mask.sum() > 0:
                time_names.append(time_name)
                time_maes.append(np.mean(np.abs(pred_orig[mask] - target_orig[mask])))

        ax4.bar(time_names, time_maes, color='coral', edgecolor='black', alpha=0.8)
        ax4.set_ylabel('MAE (pcu/h)', fontsize=11)
        ax4.set_title('分时段MAE', fontsize=12)
        ax4.tick_params(axis='x', rotation=30)

        # 5. Alpha分布 (如果有)
        ax5 = axes[1, 1]
        if len(results['alphas']) > 0:
            alphas = results['alphas']
            ax5.hist(alphas, bins=50, edgecolor='black', alpha=0.7, color='green')
            ax5.axvline(np.mean(alphas), color='red', linestyle='--', lw=2,
                        label=f'mean={np.mean(alphas):.4f}')
            ax5.set_xlabel('Alpha系数', fontsize=11)
            ax5.set_ylabel('频次', fontsize=11)
            ax5.set_title('物理系数Alpha分布', fontsize=12)
            ax5.legend()
        else:
            ax5.text(0.5, 0.5, '无Alpha系数\n(物理分支未启用)',
                     ha='center', va='center', transform=ax5.transAxes, fontsize=12)
            ax5.set_title('物理系数Alpha分布', fontsize=12)

        # 6. 分流量区间的MAPE
        ax6 = axes[1, 2]
        flow_bins = [(0, 100), (100, 300), (300, 500), (500, 1000), (1000, 3000)]
        bin_names = []
        bin_mapes = []
        bin_counts = []
        for low, high in flow_bins:
            mask = (target_orig >= low) & (target_orig < high)
            if mask.sum() > 0:
                bin_names.append(f'{low}-{high}')
                mape = np.mean(np.abs(pred_orig[mask] - target_orig[mask]) / (target_orig[mask] + 1)) * 100
                bin_mapes.append(mape)
                bin_counts.append(mask.sum())

        bars = ax6.bar(bin_names, bin_mapes, color='purple', edgecolor='black', alpha=0.7)
        ax6.set_xlabel('流量区间 (pcu/h)', fontsize=11)
        ax6.set_ylabel('MAPE (%)', fontsize=11)
        ax6.set_title('分流量区间MAPE', fontsize=12)
        ax6.tick_params(axis='x', rotation=30)

        # 在柱子上标注样本数
        for bar, count in zip(bars, bin_counts):
            ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'N={count}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        fig_path = save_dir / 'evaluation_results.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ 保存图表: {fig_path}")
        plt.close()

        # === 额外图表: Alpha与速度/道路类型的关系 ===
        if len(results['alphas']) > 0:
            fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

            # Alpha vs 速度
            ax_left = axes2[0]
            scatter = ax_left.scatter(results['fcd_speeds'], results['alphas'],
                                      c=results['road_types'], cmap='tab10',
                                      alpha=0.5, s=10)
            ax_left.set_xlabel('浮动车速度 (标准化)', fontsize=11)
            ax_left.set_ylabel('Alpha系数', fontsize=11)
            ax_left.set_title('Alpha系数与速度的关系', fontsize=12)
            cbar = plt.colorbar(scatter, ax=ax_left)
            cbar.set_label('道路类型')

            # Alpha箱线图（按道路类型）
            ax_right = axes2[1]
            alpha_by_road = []
            labels = []
            for road_idx in sorted(self.road_type_names.keys()):
                road_name = self.road_type_names[road_idx]
                mask = results['road_types'] == road_idx
                if mask.sum() > 0:
                    alpha_by_road.append(results['alphas'][mask])
                    labels.append(road_name)

            ax_right.boxplot(alpha_by_road, labels=labels)
            ax_right.set_ylabel('Alpha系数', fontsize=11)
            ax_right.set_title('分道路类型Alpha分布', fontsize=12)
            ax_right.tick_params(axis='x', rotation=30)

            plt.tight_layout()
            fig2_path = save_dir / 'alpha_analysis.png'
            fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ 保存图表: {fig2_path}")
            plt.close()

        # === 额外图表: 预测误差热力图 ===
        fig3, ax3 = plt.subplots(figsize=(10, 8))

        # 创建道路类型×时段的误差矩阵
        error_matrix = np.zeros((len(self.road_type_names), len(self.time_period_names)))
        count_matrix = np.zeros_like(error_matrix)

        for road_idx in self.road_type_names.keys():
            for time_idx in self.time_period_names.keys():
                mask = (results['road_types'] == road_idx) & (results['time_periods'] == time_idx)
                if mask.sum() > 0:
                    error_matrix[road_idx, time_idx] = np.mean(np.abs(pred_orig[mask] - target_orig[mask]))
                    count_matrix[road_idx, time_idx] = mask.sum()

        im = ax3.imshow(error_matrix, cmap='YlOrRd', aspect='auto')

        # 设置标签
        ax3.set_xticks(range(len(self.time_period_names)))
        ax3.set_xticklabels([self.time_period_names[i] for i in sorted(self.time_period_names.keys())],
                            rotation=45, ha='right')
        ax3.set_yticks(range(len(self.road_type_names)))
        ax3.set_yticklabels([self.road_type_names[i] for i in sorted(self.road_type_names.keys())])

        ax3.set_xlabel('时段', fontsize=11)
        ax3.set_ylabel('道路类型', fontsize=11)
        ax3.set_title('MAE热力图 (道路类型 × 时段)', fontsize=12)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('MAE (pcu/h)')

        # 在格子中显示数值
        for i in range(error_matrix.shape[0]):
            for j in range(error_matrix.shape[1]):
                if count_matrix[i, j] > 0:
                    text = ax3.text(j, i, f'{error_matrix[i, j]:.0f}\n({int(count_matrix[i, j])})',
                                    ha='center', va='center', fontsize=8)

        plt.tight_layout()
        fig3_path = save_dir / 'error_heatmap.png'
        fig3.savefig(fig3_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ 保存图表: {fig3_path}")
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


def compare_ablations(checkpoint_dir, test_loader, processor, device='cuda'):
    """对比消融实验结果"""
    print("\n" + "=" * 70)
    print("消融实验对比")
    print("=" * 70)

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
        evaluator = Evaluator(model, test_loader, processor, device)
        eval_results = evaluator.evaluate()
        results[ablation] = eval_results['orig_metrics']

    # 汇总对比
    if results:
        print("\n" + "=" * 70)
        print("消融实验结果汇总 (原始尺度)")
        print("=" * 70)
        print(f"{'模型':<15} {'MAE':>12} {'RMSE':>12} {'MAPE':>12} {'R²':>12}")
        print("-" * 65)
        for name, metrics in results.items():
            print(f"{name:<15} {metrics['MAE']:>12.1f} {metrics['RMSE']:>12.1f} "
                  f"{metrics['MAPE']:>11.1f}% {metrics['R2']:>12.4f}")

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
        compare_ablations(checkpoint_dir, test_loader, processor)
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
        evaluator = Evaluator(model, test_loader, processor)
        evaluator.evaluate(save_dir=output_dir)


if __name__ == "__main__":
    main()