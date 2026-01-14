"""
Stage 6: 融合浮动车数据
将浮动车数据（流量、速度、状态）融合到卡口数据

特殊处理:
- 多Link (如 "100986499+60322527") 需要聚合所有Link的浮动车数据
- 剔除全时段无浮动车数据的卡口
- 统计每个卡口的浮动车覆盖率

用法:
    python 6_merge_fcd.py
    python 6_merge_fcd.py --fcd-root "F:/jinan_temp"  # 指定浮动车数据目录
"""

import pandas as pd
import numpy as np
import os
import glob
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# === 路径设置 ===
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from utils.path_manager import pm


# === 配置参数 ===
CONFIG = {
    # 输入
    'input_checkpoint': 'checkpoint_with_road_attrs.csv',  # Stage 5 输出
    'fcd_root': r'F:\jinan_temp',  # 浮动车数据根目录
    # 黑名单日期
    'skip_dates': ['20231122'],
    # 浮动车数据配置
    'fcd_cols': ['linkId', 'roadLen', 'dataTime', 'travelTime', 'status', 'carNum'],
    'invalid_roadlen': {0, 65535},
    'min_speed_kmh': 1,
    'max_speed_kmh': 120,
    # 覆盖率阈值 (低于此值的卡口将被剔除)
    'min_coverage_rate': 0.75,
    # 输出
    'output_final': 'final_training_data.csv',
    'output_dropped': 'report_dropped_no_fcd.csv',
    'output_coverage': 'report_fcd_coverage.csv',
}


def normalize_id(series):
    """标准化ID字段"""
    return (series.astype(str)
            .str.replace(r'\.0$', '', regex=True)
            .str.strip())


def parse_link_ids(link_id_str):
    """
    解析 Link_ID 字符串，可能是单个或多个（用+分隔）
    返回: list of link_ids
    """
    link_id_str = str(link_id_str).strip()
    if '+' in link_id_str:
        return [lid.strip() for lid in link_id_str.split('+')]
    else:
        return [link_id_str]


def load_fcd_for_dates(fcd_root, target_dates, target_links, skip_dates):
    """
    加载指定日期的浮动车数据
    
    Args:
        fcd_root: 浮动车数据根目录
        target_dates: 目标日期列表 (YYYYMMDD)
        target_links: 目标Link ID集合
        skip_dates: 跳过的日期列表
    
    Returns:
        聚合后的浮动车数据 DataFrame
    """
    fcd_cols = CONFIG['fcd_cols']
    fcd_dtypes = {
        'linkId': str, 'roadLen': 'float32', 'dataTime': 'float64',
        'travelTime': 'float32', 'status': 'float32', 'carNum': 'float32'
    }
    invalid_roadlen = CONFIG['invalid_roadlen']
    min_speed = CONFIG['min_speed_kmh']
    max_speed = CONFIG['max_speed_kmh']
    
    # 过滤黑名单日期
    filtered_dates = [d for d in target_dates if d not in skip_dates]
    print(f"目标日期: {len(target_dates)} 天")
    print(f"剔除黑名单后: {len(filtered_dates)} 天")
    if skip_dates:
        print(f"已跳过日期: {skip_dates}")
    
    all_fcd_aggs = []
    
    for date_str in filtered_dates:
        date_dir = os.path.join(fcd_root, date_str)
        if not os.path.exists(date_dir):
            print(f"  ⚠️ 目录不存在: {date_dir}")
            continue
        
        csv_files = glob.glob(os.path.join(date_dir, "*.csv"))
        if not csv_files:
            print(f"  ⚠️ 无CSV文件: {date_dir}")
            continue
        
        day_records = []
        
        for f_path in tqdm(csv_files, desc=f"读取 {date_str}", unit="file", leave=False):
            try:
                df = pd.read_csv(f_path, usecols=fcd_cols, dtype=fcd_dtypes)
                if df.empty:
                    continue
                
                # 筛选目标路段
                df = df[df['linkId'].isin(target_links)]
                if df.empty:
                    continue
                
                # 清洗异常数据
                df = df[~df['roadLen'].isin(invalid_roadlen)]
                df = df[df['travelTime'] > 0]
                
                # 速度过滤
                speed_kmh = (df['roadLen'] / df['travelTime']) * 3.6
                df = df[(speed_kmh >= min_speed) & (speed_kmh <= max_speed)].copy()
                df['speed_kmh'] = speed_kmh.loc[df.index]
                
                if df.empty:
                    continue
                
                # 时区对齐 (+8h)
                dt_series = pd.to_datetime(df['dataTime'], unit='s') + pd.Timedelta(hours=8)
                df['hour_start'] = dt_series.dt.floor('h')
                
                day_records.append(df[['linkId', 'hour_start', 'carNum', 'speed_kmh', 'status']])
            
            except Exception as e:
                # 静默处理错误，不打断进度条
                continue
        
        if day_records:
            day_df = pd.concat(day_records)
            # 按 linkId + hour_start 聚合
            agg_df = day_df.groupby(['linkId', 'hour_start']).agg({
                'carNum': 'sum',
                'speed_kmh': 'mean',
                'status': 'mean',
                'linkId': 'count'
            }).rename(columns={
                'carNum': 'fcd_flow',
                'speed_kmh': 'fcd_speed',
                'status': 'fcd_status',
                'linkId': 'fcd_record_count'
            }).reset_index()
            all_fcd_aggs.append(agg_df)
            print(f"  ✓ {date_str}: {len(agg_df)} 条聚合记录")
    
    if not all_fcd_aggs:
        return pd.DataFrame()
    
    return pd.concat(all_fcd_aggs, ignore_index=True)


def merge_fcd(fcd_root=None):
    """
    融合浮动车数据
    """
    print("=" * 60)
    print("Stage 6: 融合浮动车数据")
    print("=" * 60)
    
    if fcd_root:
        CONFIG['fcd_root'] = fcd_root
    
    # 读取卡口数据
    ckpt_path = pm.get_processed_path(CONFIG['input_checkpoint'])
    print(f"读取卡口数据: {ckpt_path.name}")
    ckpt_df = pd.read_csv(ckpt_path)
    
    print(f"卡口数: {ckpt_df['卡口编号'].nunique()}")
    print(f"记录数: {len(ckpt_df)}")
    
    # 解析时间
    ckpt_df['start_dt'] = pd.to_datetime(ckpt_df['开始时间'])
    
    # 收集所有需要的 Link ID（展开多Link）
    all_link_ids = set()
    link_id_mapping = {}  # {原始Link_ID: [展开后的link_ids]}
    
    for link_id_raw in ckpt_df['Link_ID'].unique():
        link_ids = parse_link_ids(link_id_raw)
        link_id_mapping[str(link_id_raw)] = link_ids
        all_link_ids.update(link_ids)
    
    print(f"唯一Link数 (展开后): {len(all_link_ids)}")
    
    # 获取日期范围
    target_dates = ckpt_df['start_dt'].dt.strftime('%Y%m%d').unique().tolist()
    
    # 加载浮动车数据
    print(f"\n浮动车数据目录: {CONFIG['fcd_root']}")
    fcd_df = load_fcd_for_dates(
        CONFIG['fcd_root'],
        target_dates,
        all_link_ids,
        CONFIG['skip_dates']
    )
    
    if fcd_df.empty:
        print("❌ 未加载到任何浮动车数据！")
        return None
    
    print(f"\n浮动车数据总量: {len(fcd_df)} 条")
    
    # 同步剔除黑名单日期的卡口数据
    skip_dates = CONFIG['skip_dates']
    if skip_dates:
        skip_dt_list = pd.to_datetime(skip_dates, format='%Y%m%d')
        mask_keep = ~ckpt_df['start_dt'].dt.normalize().isin(skip_dt_list)
        removed_count = len(ckpt_df) - mask_keep.sum()
        ckpt_df = ckpt_df[mask_keep].copy()
        print(f"同步剔除黑名单日期的卡口记录: {removed_count} 条")
    
    # === 核心：为每条卡口记录匹配浮动车数据 ===
    print("\n正在匹配浮动车数据...")
    
    # 将浮动车数据转为字典便于快速查找
    # key: (linkId, hour_start)
    fcd_df['hour_start'] = pd.to_datetime(fcd_df['hour_start'])
    fcd_dict = {}
    for _, row in fcd_df.iterrows():
        key = (row['linkId'], row['hour_start'])
        fcd_dict[key] = {
            'fcd_flow': row['fcd_flow'],
            'fcd_speed': row['fcd_speed'],
            'fcd_status': row['fcd_status'],
            'fcd_record_count': row['fcd_record_count']
        }
    
    results = []
    
    for idx, row in tqdm(ckpt_df.iterrows(), total=len(ckpt_df), desc="匹配中"):
        link_id_raw = str(row['Link_ID'])
        hour_start = row['start_dt'].floor('h')
        
        # 获取该卡口对应的所有Link
        link_ids = link_id_mapping.get(link_id_raw, [link_id_raw])
        
        # 收集所有Link的浮动车数据
        fcd_values = []
        for lid in link_ids:
            key = (lid, hour_start)
            if key in fcd_dict:
                fcd_values.append(fcd_dict[key])
        
        new_row = row.to_dict()
        
        if fcd_values:
            # 聚合多个Link的浮动车数据
            new_row['fcd_flow'] = sum(v['fcd_flow'] for v in fcd_values)
            new_row['fcd_speed'] = np.mean([v['fcd_speed'] for v in fcd_values])
            new_row['fcd_status'] = np.mean([v['fcd_status'] for v in fcd_values])
            new_row['fcd_record_count'] = sum(v['fcd_record_count'] for v in fcd_values)
            new_row['fcd_matched'] = 1
        else:
            # 无匹配
            new_row['fcd_flow'] = 0
            new_row['fcd_speed'] = np.nan
            new_row['fcd_status'] = np.nan
            new_row['fcd_record_count'] = 0
            new_row['fcd_matched'] = 0
        
        results.append(new_row)
    
    result_df = pd.DataFrame(results)
    
    # 计算渗透率
    result_df['penetration_rate'] = result_df.apply(
        lambda r: r['fcd_flow'] / r['flow_std'] if r['flow_std'] > 0 else 0,
        axis=1
    )
    
    # === 统计覆盖率并剔除无效卡口 ===
    print("\n计算浮动车覆盖率...")
    
    coverage_stats = result_df.groupby('卡口编号').agg(
        total_hours=('fcd_matched', 'count'),
        matched_hours=('fcd_matched', 'sum'),
        卡口名称=('卡口名称', 'first'),
        Link_ID=('Link_ID', 'first')
    ).reset_index()
    
    coverage_stats['coverage_rate'] = coverage_stats['matched_hours'] / coverage_stats['total_hours']
    
    # 覆盖率阈值
    min_coverage = CONFIG['min_coverage_rate']
    
    # 找出不满足覆盖率要求的卡口
    low_coverage_ckpts = coverage_stats[coverage_stats['coverage_rate'] < min_coverage]['卡口编号'].tolist()
    valid_ckpts = coverage_stats[coverage_stats['coverage_rate'] >= min_coverage]['卡口编号'].tolist()
    
    # 细分：完全无数据 vs 覆盖率不足
    no_fcd_ckpts = coverage_stats[coverage_stats['matched_hours'] == 0]['卡口编号'].tolist()
    low_but_has_fcd = [c for c in low_coverage_ckpts if c not in no_fcd_ckpts]
    
    print(f"\n卡口覆盖情况 (阈值: {min_coverage:.0%}):")
    print(f"  - 满足要求 (≥{min_coverage:.0%}): {len(valid_ckpts)} 个")
    print(f"  - 完全无数据: {len(no_fcd_ckpts)} 个 (剔除)")
    print(f"  - 覆盖率不足: {len(low_but_has_fcd)} 个 (剔除)")
    print(f"  - 剔除总计: {len(low_coverage_ckpts)} 个")
    
    # 剔除不满足覆盖率要求的卡口
    final_df = result_df[result_df['卡口编号'].isin(valid_ckpts)].copy()
    
    # 清理临时列
    final_df.drop(columns=['start_dt', 'fcd_matched'], inplace=True, errors='ignore')
    
    # === 输出统计报告 ===
    
    # 1. 被剔除的卡口
    if low_coverage_ckpts:
        dropped_df = coverage_stats[coverage_stats['coverage_rate'] < min_coverage][
            ['卡口编号', '卡口名称', 'Link_ID', 'total_hours', 'matched_hours', 'coverage_rate']
        ].copy()
        dropped_df['原因'] = dropped_df.apply(
            lambda r: '完全无浮动车数据' if r['matched_hours'] == 0 else f"覆盖率不足({r['coverage_rate']:.1%}<{min_coverage:.0%})",
            axis=1
        )
        
        dropped_path = pm.get_processed_path(CONFIG['output_dropped'])
        dropped_df.to_csv(dropped_path, index=False, encoding='utf-8-sig')
        print(f"\n⚠️ 剔除卡口报告: {dropped_path.name}")
    
    # 2. 覆盖率报告 (只包含保留的卡口)
    valid_coverage = coverage_stats[coverage_stats['coverage_rate'] >= min_coverage].copy()
    valid_coverage = valid_coverage.sort_values('coverage_rate', ascending=True)
    
    coverage_path = pm.get_processed_path(CONFIG['output_coverage'])
    valid_coverage.to_csv(coverage_path, index=False, encoding='utf-8-sig')
    print(f"📊 覆盖率报告: {coverage_path.name}")
    
    # 覆盖率统计
    print("\n覆盖率分布:")
    bins = [0, 0.25, 0.5, 0.75, 0.9, 1.0]
    labels = ['0-25%', '25-50%', '50-75%', '75-90%', '90-100%']
    valid_coverage['coverage_bin'] = pd.cut(valid_coverage['coverage_rate'], bins=bins, labels=labels)
    bin_counts = valid_coverage['coverage_bin'].value_counts().sort_index()
    for label, count in bin_counts.items():
        print(f"  - {label}: {count} 个卡口")
    
    avg_coverage = valid_coverage['coverage_rate'].mean()
    print(f"\n平均覆盖率: {avg_coverage:.1%}")
    
    # 3. 保存最终数据
    output_path = pm.get_processed_path(CONFIG['output_final'])
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print("Stage 6 完成")
    print("=" * 60)
    print(f"最终卡口数: {final_df['卡口编号'].nunique()}")
    print(f"最终记录数: {len(final_df)}")
    print(f"✅ 输出文件: {output_path.name}")
    
    return final_df


def run(fcd_root=None):
    """执行流程"""
    return merge_fcd(fcd_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 6: 融合浮动车数据")
    parser.add_argument('--fcd-root', type=str, default=None,
                        help='浮动车数据根目录')
    args = parser.parse_args()
    
    run(fcd_root=args.fcd_root)
