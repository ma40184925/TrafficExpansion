"""
评估脚本：卡口流量与浮动车流量比值分析
============================================

分析内容:
1. 整体比值分布（flow_std / fcd_flow）
2. 按时段的比值变化（早高峰/平峰/晚高峰/夜间）
3. 按道路类型的比值差异
4. 各卡口的比值稳定性（均值、标准差、变异系数）
5. 异常值识别

用法:
    python evaluate_flow_ratio.py
    python evaluate_flow_ratio.py --input final_training_data.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

# === 路径设置 ===
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from utils.path_manager import pm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# === 配置 ===
CONFIG = {
    'input_file': 'final_training_data.csv',
    'output_report': 'report_flow_ratio_analysis.csv',
    'output_checkpoint_stats': 'report_checkpoint_ratio_stats.csv',
    'output_figure': 'figure_flow_ratio_analysis.png',
    # 道路类型映射
    'road_kind_mapping': {
        '00': '高速公路',
        '01': '城市高速',
        '02': '国道',
        '03': '省道',
        '04': '县道',
        '06': '市镇村道'
    },
    # 时段划分
    'time_periods': {
        '夜间(0-6)': (0, 6),
        '早高峰(7-9)': (7, 9),
        '上午平峰(10-11)': (10, 11),
        '午间(12-13)': (12, 13),
        '下午平峰(14-16)': (14, 16),
        '晚高峰(17-19)': (17, 19),
        '晚间(20-23)': (20, 23),
    }
}


def load_data(input_file=None):
    """加载数据"""
    if input_file is None:
        input_file = CONFIG['input_file']
    
    file_path = pm.get_processed_path(input_file)
    print(f"读取数据: {file_path.name}")
    df = pd.read_csv(file_path, low_memory=False)
    
    # 解析时间
    df['start_time'] = pd.to_datetime(df['开始时间'])
    df['hour'] = df['start_time'].dt.hour
    df['date'] = df['start_time'].dt.date
    
    # 确保 kind_x 是字符串
    if 'kind_x' in df.columns:
        df['kind_x'] = df['kind_x'].astype(str).str.zfill(2)
    
    print(f"数据量: {len(df)} 条, {df['卡口编号'].nunique()} 个卡口")
    
    return df


def calculate_ratio(df):
    """
    计算比值 (卡口流量 / 浮动车流量)
    只对 fcd_flow > 0 的记录计算
    """
    df = df.copy()
    
    # 过滤有效数据
    valid_mask = (df['fcd_flow'] > 0) & (df['flow_std'] > 0)
    df_valid = df[valid_mask].copy()
    
    # 计算比值
    df_valid['ratio'] = df_valid['flow_std'] / df_valid['fcd_flow']
    
    # 计算渗透率 (浮动车/卡口)
    df_valid['penetration'] = df_valid['fcd_flow'] / df_valid['flow_std']
    
    print(f"\n有效记录数: {len(df_valid)} / {len(df)} ({len(df_valid)/len(df)*100:.1f}%)")
    print(f"无效记录 (fcd_flow=0): {(df['fcd_flow'] == 0).sum()}")
    
    return df_valid


def analyze_overall_distribution(df):
    """分析整体比值分布"""
    print("\n" + "=" * 60)
    print("1. 整体比值分布 (卡口流量 / 浮动车流量)")
    print("=" * 60)
    
    ratio = df['ratio']
    penetration = df['penetration']
    
    stats = {
        '记录数': len(ratio),
        '比值均值': ratio.mean(),
        '比值中位数': ratio.median(),
        '比值标准差': ratio.std(),
        '比值最小值': ratio.min(),
        '比值25%分位': ratio.quantile(0.25),
        '比值75%分位': ratio.quantile(0.75),
        '比值最大值': ratio.max(),
        '渗透率均值': penetration.mean(),
        '渗透率中位数': penetration.median(),
    }
    
    print(f"\n比值 (卡口/浮动车):")
    print(f"  均值: {stats['比值均值']:.2f}")
    print(f"  中位数: {stats['比值中位数']:.2f}")
    print(f"  标准差: {stats['比值标准差']:.2f}")
    print(f"  范围: [{stats['比值最小值']:.2f}, {stats['比值最大值']:.2f}]")
    print(f"  IQR: [{stats['比值25%分位']:.2f}, {stats['比值75%分位']:.2f}]")
    
    print(f"\n渗透率 (浮动车/卡口):")
    print(f"  均值: {stats['渗透率均值']:.2%}")
    print(f"  中位数: {stats['渗透率中位数']:.2%}")
    
    # 比值分布区间统计
    print(f"\n比值分布区间:")
    bins = [0, 1, 2, 5, 10, 20, 50, 100, float('inf')]
    labels = ['<1', '1-2', '2-5', '5-10', '10-20', '20-50', '50-100', '>100']
    df['ratio_bin'] = pd.cut(df['ratio'], bins=bins, labels=labels)
    bin_counts = df['ratio_bin'].value_counts().sort_index()
    
    for label, count in bin_counts.items():
        pct = count / len(df) * 100
        print(f"  {label:>8}: {count:>6} ({pct:>5.1f}%)")
    
    return stats


def analyze_by_time_period(df):
    """按时段分析比值"""
    print("\n" + "=" * 60)
    print("2. 按时段的比值变化")
    print("=" * 60)
    
    time_periods = CONFIG['time_periods']
    
    results = []
    for period_name, (start_h, end_h) in time_periods.items():
        mask = (df['hour'] >= start_h) & (df['hour'] <= end_h)
        subset = df[mask]
        
        if len(subset) > 0:
            results.append({
                '时段': period_name,
                '记录数': len(subset),
                '比值均值': subset['ratio'].mean(),
                '比值中位数': subset['ratio'].median(),
                '比值标准差': subset['ratio'].std(),
                '渗透率均值': subset['penetration'].mean(),
            })
    
    result_df = pd.DataFrame(results)
    
    print(f"\n{'时段':<20} {'记录数':>8} {'比值均值':>10} {'比值中位数':>10} {'渗透率':>10}")
    print("-" * 65)
    for _, row in result_df.iterrows():
        print(f"{row['时段']:<20} {row['记录数']:>8} {row['比值均值']:>10.2f} {row['比值中位数']:>10.2f} {row['渗透率均值']:>10.2%}")
    
    return result_df


def analyze_by_road_type(df):
    """按道路类型分析比值"""
    print("\n" + "=" * 60)
    print("3. 按道路类型的比值差异")
    print("=" * 60)
    
    if 'kind_x' not in df.columns:
        print("无道路类型数据，跳过此分析")
        return None
    
    kind_mapping = CONFIG['road_kind_mapping']
    
    results = []
    for kind_code in df['kind_x'].unique():
        subset = df[df['kind_x'] == kind_code]
        kind_name = kind_mapping.get(kind_code, '未知')
        
        results.append({
            '类型代码': kind_code,
            '道路类型': kind_name,
            '卡口数': subset['卡口编号'].nunique(),
            '记录数': len(subset),
            '比值均值': subset['ratio'].mean(),
            '比值中位数': subset['ratio'].median(),
            '比值标准差': subset['ratio'].std(),
            '渗透率均值': subset['penetration'].mean(),
        })
    
    result_df = pd.DataFrame(results).sort_values('记录数', ascending=False)
    
    print(f"\n{'道路类型':<12} {'卡口数':>6} {'记录数':>8} {'比值均值':>10} {'比值中位数':>10} {'渗透率':>10}")
    print("-" * 65)
    for _, row in result_df.iterrows():
        print(f"{row['道路类型']:<12} {row['卡口数']:>6} {row['记录数']:>8} {row['比值均值']:>10.2f} {row['比值中位数']:>10.2f} {row['渗透率均值']:>10.2%}")
    
    return result_df


def analyze_by_checkpoint(df):
    """按卡口分析比值稳定性"""
    print("\n" + "=" * 60)
    print("4. 各卡口比值稳定性分析")
    print("=" * 60)
    
    # 按卡口聚合统计
    ckpt_stats = df.groupby(['卡口编号', '卡口名称']).agg(
        记录数=('ratio', 'count'),
        比值均值=('ratio', 'mean'),
        比值中位数=('ratio', 'median'),
        比值标准差=('ratio', 'std'),
        比值最小=('ratio', 'min'),
        比值最大=('ratio', 'max'),
        渗透率均值=('penetration', 'mean'),
        卡口流量均值=('flow_std', 'mean'),
        浮动车流量均值=('fcd_flow', 'mean'),
    ).reset_index()
    
    # 计算变异系数 (CV = std / mean)
    ckpt_stats['变异系数'] = ckpt_stats['比值标准差'] / ckpt_stats['比值均值']
    
    # 按比值均值排序
    ckpt_stats = ckpt_stats.sort_values('比值均值', ascending=True)
    
    print(f"\n卡口总数: {len(ckpt_stats)}")
    
    # 比值均值分布
    print(f"\n各卡口比值均值分布:")
    ratio_mean = ckpt_stats['比值均值']
    print(f"  最小: {ratio_mean.min():.2f}")
    print(f"  25%分位: {ratio_mean.quantile(0.25):.2f}")
    print(f"  中位数: {ratio_mean.median():.2f}")
    print(f"  75%分位: {ratio_mean.quantile(0.75):.2f}")
    print(f"  最大: {ratio_mean.max():.2f}")
    
    # 变异系数分布（衡量稳定性）
    print(f"\n各卡口变异系数分布 (越小越稳定):")
    cv = ckpt_stats['变异系数']
    print(f"  最小: {cv.min():.2f}")
    print(f"  25%分位: {cv.quantile(0.25):.2f}")
    print(f"  中位数: {cv.median():.2f}")
    print(f"  75%分位: {cv.quantile(0.75):.2f}")
    print(f"  最大: {cv.max():.2f}")
    
    # 稳定性分类
    print(f"\n稳定性分类 (基于变异系数):")
    stable = (cv < 0.3).sum()
    moderate = ((cv >= 0.3) & (cv < 0.5)).sum()
    unstable = (cv >= 0.5).sum()
    print(f"  稳定 (CV<0.3): {stable} 个卡口 ({stable/len(cv)*100:.1f}%)")
    print(f"  中等 (0.3≤CV<0.5): {moderate} 个卡口 ({moderate/len(cv)*100:.1f}%)")
    print(f"  不稳定 (CV≥0.5): {unstable} 个卡口 ({unstable/len(cv)*100:.1f}%)")
    
    # 显示极端卡口
    print(f"\n比值最低的5个卡口 (浮动车占比高):")
    print(ckpt_stats[['卡口名称', '比值均值', '渗透率均值', '变异系数']].head(5).to_string(index=False))
    
    print(f"\n比值最高的5个卡口 (浮动车占比低):")
    print(ckpt_stats[['卡口名称', '比值均值', '渗透率均值', '变异系数']].tail(5).to_string(index=False))
    
    return ckpt_stats


def analyze_correlation(df):
    """分析卡口流量与浮动车流量的相关性"""
    print("\n" + "=" * 60)
    print("5. 流量相关性分析")
    print("=" * 60)
    
    # 整体相关系数
    corr = df['flow_std'].corr(df['fcd_flow'])
    print(f"\n整体Pearson相关系数: {corr:.4f}")
    
    # 按卡口计算相关系数
    ckpt_corr = df.groupby('卡口编号').apply(
        lambda x: x['flow_std'].corr(x['fcd_flow']) if len(x) > 10 else np.nan
    ).dropna()
    
    print(f"\n各卡口相关系数分布 (样本量>10):")
    print(f"  卡口数: {len(ckpt_corr)}")
    print(f"  均值: {ckpt_corr.mean():.4f}")
    print(f"  中位数: {ckpt_corr.median():.4f}")
    print(f"  最小: {ckpt_corr.min():.4f}")
    print(f"  最大: {ckpt_corr.max():.4f}")
    
    # 相关性分类
    high_corr = (ckpt_corr >= 0.7).sum()
    mid_corr = ((ckpt_corr >= 0.4) & (ckpt_corr < 0.7)).sum()
    low_corr = (ckpt_corr < 0.4).sum()
    
    print(f"\n相关性分类:")
    print(f"  高相关 (r≥0.7): {high_corr} 个卡口 ({high_corr/len(ckpt_corr)*100:.1f}%)")
    print(f"  中相关 (0.4≤r<0.7): {mid_corr} 个卡口 ({mid_corr/len(ckpt_corr)*100:.1f}%)")
    print(f"  低相关 (r<0.4): {low_corr} 个卡口 ({low_corr/len(ckpt_corr)*100:.1f}%)")
    
    return ckpt_corr


def plot_analysis(df, ckpt_stats, output_path):
    """生成可视化图表"""
    print("\n生成可视化图表...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('卡口流量与浮动车流量比值分析', fontsize=14, fontweight='bold')
    
    # 1. 比值分布直方图
    ax1 = axes[0, 0]
    ratio_clipped = df['ratio'].clip(upper=50)  # 截断极端值便于展示
    ax1.hist(ratio_clipped, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(df['ratio'].median(), color='red', linestyle='--', label=f'中位数: {df["ratio"].median():.2f}')
    ax1.set_xlabel('比值 (卡口/浮动车)')
    ax1.set_ylabel('频次')
    ax1.set_title('比值分布 (截断至50)')
    ax1.legend()
    
    # 2. 渗透率分布
    ax2 = axes[0, 1]
    penetration_pct = df['penetration'] * 100
    ax2.hist(penetration_pct, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax2.axvline(penetration_pct.median(), color='red', linestyle='--', label=f'中位数: {penetration_pct.median():.1f}%')
    ax2.set_xlabel('渗透率 (%)')
    ax2.set_ylabel('频次')
    ax2.set_title('浮动车渗透率分布')
    ax2.legend()
    
    # 3. 按小时的比值变化
    ax3 = axes[0, 2]
    hourly_stats = df.groupby('hour')['ratio'].agg(['mean', 'median']).reset_index()
    ax3.plot(hourly_stats['hour'], hourly_stats['mean'], 'o-', label='均值', markersize=4)
    ax3.plot(hourly_stats['hour'], hourly_stats['median'], 's--', label='中位数', markersize=4)
    ax3.set_xlabel('小时')
    ax3.set_ylabel('比值')
    ax3.set_title('按小时的比值变化')
    ax3.set_xticks(range(0, 24, 2))
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 各卡口比值均值分布
    ax4 = axes[1, 0]
    ratio_mean_clipped = ckpt_stats['比值均值'].clip(upper=50)
    ax4.hist(ratio_mean_clipped, bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax4.axvline(ckpt_stats['比值均值'].median(), color='red', linestyle='--', 
                label=f'中位数: {ckpt_stats["比值均值"].median():.2f}')
    ax4.set_xlabel('卡口比值均值')
    ax4.set_ylabel('卡口数')
    ax4.set_title('各卡口比值均值分布')
    ax4.legend()
    
    # 5. 各卡口变异系数分布
    ax5 = axes[1, 1]
    cv_clipped = ckpt_stats['变异系数'].clip(upper=2)
    ax5.hist(cv_clipped, bins=30, edgecolor='black', alpha=0.7, color='purple')
    ax5.axvline(0.3, color='green', linestyle='--', label='稳定阈值(0.3)')
    ax5.axvline(0.5, color='red', linestyle='--', label='不稳定阈值(0.5)')
    ax5.set_xlabel('变异系数')
    ax5.set_ylabel('卡口数')
    ax5.set_title('各卡口比值变异系数分布')
    ax5.legend()
    
    # 6. 卡口流量 vs 浮动车流量散点图
    ax6 = axes[1, 2]
    sample = df.sample(min(5000, len(df)))  # 采样避免过密
    ax6.scatter(sample['fcd_flow'], sample['flow_std'], alpha=0.3, s=5)
    # 添加对角线参考
    max_val = max(sample['fcd_flow'].max(), sample['flow_std'].max())
    ax6.plot([0, max_val], [0, max_val], 'r--', label='1:1线')
    ax6.plot([0, max_val], [0, max_val*5], 'g--', alpha=0.5, label='5:1线')
    ax6.plot([0, max_val], [0, max_val*10], 'b--', alpha=0.5, label='10:1线')
    ax6.set_xlabel('浮动车流量')
    ax6.set_ylabel('卡口流量')
    ax6.set_title('卡口流量 vs 浮动车流量')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图表已保存: {output_path.name}")
    
    plt.close()


def run(input_file=None):
    """执行完整分析"""
    print("=" * 60)
    print("卡口流量与浮动车流量比值分析")
    print("=" * 60)
    
    # 加载数据
    df = load_data(input_file)
    
    # 计算比值
    df_valid = calculate_ratio(df)
    
    if len(df_valid) == 0:
        print("❌ 无有效数据进行分析")
        return
    
    # 1. 整体分布
    overall_stats = analyze_overall_distribution(df_valid)
    
    # 2. 按时段分析
    time_stats = analyze_by_time_period(df_valid)
    
    # 3. 按道路类型分析
    road_stats = analyze_by_road_type(df_valid)
    
    # 4. 按卡口分析
    ckpt_stats = analyze_by_checkpoint(df_valid)
    
    # 5. 相关性分析
    corr_stats = analyze_correlation(df_valid)
    
    # 保存报告
    print("\n" + "=" * 60)
    print("保存分析报告")
    print("=" * 60)
    
    # 保存卡口统计
    ckpt_output = pm.get_processed_path(CONFIG['output_checkpoint_stats'])
    ckpt_stats.to_csv(ckpt_output, index=False, encoding='utf-8-sig')
    print(f"✅ 卡口统计: {ckpt_output.name}")
    
    # 生成图表
    fig_output = pm.get_processed_path(CONFIG['output_figure'])
    plot_analysis(df_valid, ckpt_stats, fig_output)
    
    # 总结建议
    print("\n" + "=" * 60)
    print("分析总结与建议")
    print("=" * 60)
    
    median_ratio = df_valid['ratio'].median()
    median_penetration = df_valid['penetration'].median()
    
    print(f"\n📊 核心指标:")
    print(f"  - 比值中位数: {median_ratio:.2f} (即平均每1辆浮动车对应{median_ratio:.1f}辆实际车)")
    print(f"  - 渗透率中位数: {median_penetration:.2%}")
    
    print(f"\n💡 建议:")
    if median_ratio > 5:
        print(f"  - 比值较高(>{median_ratio:.0f})，浮动车样本稀疏，建议使用扩样模型")
        print(f"  - 可尝试: 线性回归、随机森林、或时空图神经网络进行流量扩样")
    else:
        print(f"  - 比值适中，可以探索卡口流量与浮动车流量的函数关系")
        print(f"  - 可尝试: 分时段/分路段建立回归模型")
    
    cv_median = ckpt_stats['变异系数'].median()
    if cv_median > 0.5:
        print(f"  - 比值变异系数较大({cv_median:.2f})，建议分卡口/分时段建模")
    else:
        print(f"  - 比值相对稳定({cv_median:.2f})，可考虑统一扩样系数")
    
    print("\n✅ 分析完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="卡口流量与浮动车流量比值分析")
    parser.add_argument('--input', type=str, default=None,
                        help='输入文件名 (默认: final_training_data.csv)')
    args = parser.parse_args()
    
    run(input_file=args.input)
