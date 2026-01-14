"""
Stage 1: 数据清洗与质量筛选
合并原 1_clean_df.py 和 2_analyze_coverage_active.py

功能:
1. 读取原始Excel，透视车型，计算PCU标准车流量
2. 基于活跃时段覆盖率筛选优质卡口

用法:
    python 1_clean_and_filter.py                    # 完整流程
    python 1_clean_and_filter.py --skip-clean       # 跳过清洗，只做筛选
"""

import pandas as pd
import argparse
import sys
from pathlib import Path

# === 路径设置 ===
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from utils.path_manager import pm


# === 配置参数 ===
CONFIG = {
    # 原始数据文件
    'raw_files': [
        '2023_11_20_0点至26_24点.xlsx',
        '2024_01_08_0点至14_24点.xlsx'
    ],
    # 活跃时段
    'active_start_hour': 7,
    'active_end_hour': 22,
    # 覆盖率阈值
    'coverage_threshold': 0.75,
    # 总天数
    'total_days': 14,
    # 输出文件名
    'output_cleaned': 'checkpoint_flow_std_cleaned.csv',
    'output_filtered': 'checkpoint_flow_std_high_quality.csv',
    'output_coverage_report': 'coverage_report.csv',
}


def step1_clean_and_pivot():
    """
    Step 1: 读取原始数据，透视车型，计算PCU流量
    """
    print("=" * 50)
    print("Step 1: 数据清洗与PCU计算")
    print("=" * 50)

    all_data = []
    print(f"数据目录: {pm.data_raw}")

    for fname in CONFIG['raw_files']:
        file_path = pm.get_raw_path(fname)
        if file_path.exists():
            print(f"  ✓ 读取: {fname}")
            df = pd.read_excel(file_path)
            all_data.append(df)
        else:
            print(f"  ✗ 文件不存在: {fname}")

    if not all_data:
        raise FileNotFoundError("未找到任何原始数据文件")

    raw_df = pd.concat(all_data, ignore_index=True)

    # 仅保留品牌种类 1(大车) 和 4(小车)
    raw_df = raw_df[raw_df['品牌种类'].isin([1, 4])]

    # 透视表
    print("正在聚合车型数据...")
    pivot_df = pd.pivot_table(
        raw_df,
        values='流量',
        index=['卡口编号', '卡口名称', '开始时间', '结束时间'],
        columns='品牌种类',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    pivot_df.rename(columns={1: 'flow_large', 4: 'flow_small'}, inplace=True)

    # 计算PCU: 小车 + 大车 * 1.5
    print("正在计算标准车当量(PCU)...")
    pivot_df['flow_std'] = pivot_df['flow_small'] + pivot_df['flow_large'] * 1.5

    # 剔除全0卡口
    print("正在剔除全0流量的无效卡口...")
    checkpoint_stats = pivot_df.groupby('卡口编号')['flow_std'].sum()
    valid_checkpoints = checkpoint_stats[checkpoint_stats > 0].index.tolist()
    clean_df = pivot_df[pivot_df['卡口编号'].isin(valid_checkpoints)].copy()

    # 统计
    original_count = pivot_df['卡口编号'].nunique()
    final_count = clean_df['卡口编号'].nunique()
    print("-" * 30)
    print(f"原始卡口数量: {original_count}")
    print(f"清洗后卡口数量: {final_count}")
    print(f"剔除无效卡口数: {original_count - final_count}")
    print("-" * 30)

    # 保存
    output_path = pm.get_processed_path(CONFIG['output_cleaned'])
    clean_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存: {output_path.name}")

    return clean_df


def step2_filter_by_coverage(df=None):
    """
    Step 2: 基于活跃时段覆盖率筛选优质卡口
    """
    print("\n" + "=" * 50)
    print("Step 2: 活跃时段覆盖率筛选")
    print("=" * 50)

    # 如果没传入df，从文件读取
    if df is None:
        input_path = pm.get_processed_path(CONFIG['output_cleaned'])
        print(f"读取数据: {input_path.name}")
        df = pd.read_csv(input_path)

    active_start = CONFIG['active_start_hour']
    active_end = CONFIG['active_end_hour']
    threshold = CONFIG['coverage_threshold']
    total_days = CONFIG['total_days']

    # 理论活跃时间片数
    daily_active_hours = active_end - active_start + 1
    total_expected_slots = total_days * daily_active_hours

    print(f"活跃时段: {active_start}:00 - {active_end}:00")
    print(f"理论满额时间片: {total_expected_slots} 小时")

    # 提取小时
    df['start_time'] = pd.to_datetime(df['开始时间'])
    df['hour'] = df['start_time'].dt.hour

    # 筛选活跃时段
    mask_active = (df['hour'] >= active_start) & (df['hour'] <= active_end)
    df_active = df[mask_active].copy()

    # 统计覆盖率
    coverage_stats = (df_active[df_active['flow_std'] > 0]
                      .groupby(['卡口编号', '卡口名称'])
                      .agg(valid_slots=('start_time', 'nunique'))
                      .reset_index())

    coverage_stats['coverage_rate'] = coverage_stats['valid_slots'] / total_expected_slots

    # 划分优质/劣质
    good_sensors = coverage_stats[coverage_stats['coverage_rate'] >= threshold]
    bad_sensors = coverage_stats[coverage_stats['coverage_rate'] < threshold]
    good_ids = good_sensors['卡口编号'].tolist()

    print("-" * 40)
    print(f"卡口总数: {len(coverage_stats)}")
    print(f"优质卡口 (覆盖率≥{threshold*100:.0f}%): {len(good_sensors)}")
    print(f"劣质卡口 (覆盖率<{threshold*100:.0f}%): {len(bad_sensors)}")
    print("-" * 40)

    if not bad_sensors.empty:
        print("\n[劣质卡口示例]")
        print(bad_sensors[['卡口名称', 'valid_slots', 'coverage_rate']].head(3).to_string(index=False))

    # 回捞优质卡口的全天数据
    final_df = df[df['卡口编号'].isin(good_ids)].copy()
    final_df.drop(columns=['start_time', 'hour'], inplace=True, errors='ignore')

    # 保存覆盖率报告
    report_path = pm.get_processed_path(CONFIG['output_coverage_report'])
    coverage_stats.sort_values('coverage_rate', ascending=False).to_csv(
        report_path, index=False, encoding='utf-8-sig'
    )

    # 保存高质量数据
    output_path = pm.get_processed_path(CONFIG['output_filtered'])
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n✅ 覆盖率报告: {report_path.name}")
    print(f"✅ 高质量数据: {output_path.name}")
    print("   (包含优质卡口的全天24小时数据)")

    return final_df


def run(skip_clean=False):
    """执行完整流程"""
    if skip_clean:
        print("跳过清洗步骤，直接读取已清洗数据...")
        df_cleaned = None
    else:
        df_cleaned = step1_clean_and_pivot()

    df_final = step2_filter_by_coverage(df_cleaned)
    return df_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: 数据清洗与质量筛选")
    parser.add_argument('--skip-clean', action='store_true',
                        help='跳过清洗步骤，直接读取已清洗数据')
    args = parser.parse_args()

    run(skip_clean=args.skip_clean)
