"""
特征工程增强脚本
================

在 final_training_data.csv 基础上添加以下特征：

1. 物理特征：
   - fcd_flow_per_length: 单位长度浮动车累计量
   - theoretical_flow: 理论流量 (fcd_flow * fcd_speed / length)
   - density_proxy: 密度代理 (fcd_flow / (length * fcd_speed))

2. 时间编码：
   - hour: 小时 (0-23)
   - hour_sin, hour_cos: 小时周期编码
   - weekday: 星期几 (0=周一, 6=周日)
   - weekday_sin, weekday_cos: 星期周期编码
   - is_weekend: 是否周末
   - time_period: 时段分类 (夜间/早高峰/平峰/晚高峰等)

3. 道路类型编码：
   - kind_x: 原始道路类型代码
   - road_type_name: 道路类型名称
   - kind_01 ~ kind_06: 道路类型 One-Hot 编码

4. 车道数编码：
   - width: 原始车道分档值
   - lane_category: 车道数类别 (1车道/2-3车道/4车道及以上)
   - lane_1, lane_2_3, lane_4_plus: 车道数 One-Hot 编码

5. 路况特征：
   - fcd_status: 原始路况值 (聚合后的平均值)
   - status_level: 路况等级分类
   - is_congested: 是否拥堵 (status >= 2.5)

用法:
    python 7_feature_engineering.py
    python 7_feature_engineering.py --input final_training_data.csv
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path

# === 路径设置 ===
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from utils.path_manager import pm


# === 配置 ===
CONFIG = {
    'input_file': 'final_training_data.csv',
    'output_file': 'training_data_with_features_new.csv',
    'output_report': 'report_feature_engineering.txt',
    
    # 道路类型映射
    'road_kind_mapping': {
        '00': '高速公路',
        '01': '城市高速',
        '02': '国道',
        '03': '省道',
        '04': '县道',
        '06': '市镇村道'
    },
    
    # 车道数映射 (width值 -> 车道类别)
    'lane_mapping': {
        30: '1车道',
        55: '2-3车道',
        130: '4车道及以上'
    },
    
    # 路况等级映射
    'status_mapping': {
        0: '无路况',
        1: '畅通',
        2: '缓慢(轻度)',
        3: '缓慢(重度)',
        4: '拥堵',
        5: '严重拥堵'
    },
    
    # 时段划分
    'time_periods': {
        (0, 6): '夜间',
        (7, 9): '早高峰',
        (10, 11): '上午平峰',
        (12, 13): '午间',
        (14, 16): '下午平峰',
        (17, 19): '晚高峰',
        (20, 23): '晚间',
    }
}


def add_time_features(df):
    """添加时间相关特征"""
    print("\n[1/5] 添加时间特征...")
    
    # 解析时间
    df['start_time'] = pd.to_datetime(df['开始时间'])
    
    # 基础时间特征
    df['hour'] = df['start_time'].dt.hour
    df['weekday'] = df['start_time'].dt.weekday  # 0=周一, 6=周日
    df['date'] = df['start_time'].dt.date
    
    # 周期编码 (用于捕捉周期性)
    # 小时: 24小时周期
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # 星期: 7天周期
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    
    # 是否周末
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    
    # 时段分类
    def get_time_period(hour):
        for (start, end), period_name in CONFIG['time_periods'].items():
            if start <= hour <= end:
                return period_name
        return '其他'
    
    df['time_period'] = df['hour'].apply(get_time_period)
    
    # 时段 One-Hot 编码
    time_period_dummies = pd.get_dummies(df['time_period'], prefix='period')
    df = pd.concat([df, time_period_dummies], axis=1)
    
    print(f"  ✓ hour, weekday, 周期编码 (sin/cos)")
    print(f"  ✓ is_weekend")
    print(f"  ✓ time_period + One-Hot ({df['time_period'].nunique()} 类)")
    
    return df


def add_road_type_features(df):
    """添加道路类型特征"""
    print("\n[2/5] 添加道路类型特征...")
    
    # 确保 kind_x 是字符串格式
    df['kind_x'] = df['kind_x'].astype(str).str.zfill(2)
    
    # 道路类型名称
    df['road_type_name'] = df['kind_x'].map(CONFIG['road_kind_mapping']).fillna('未知')
    
    # One-Hot 编码
    kind_dummies = pd.get_dummies(df['kind_x'], prefix='kind')
    df = pd.concat([df, kind_dummies], axis=1)
    
    # 统计
    kind_dist = df['kind_x'].value_counts()
    print(f"  ✓ road_type_name")
    print(f"  ✓ kind One-Hot 编码")
    print(f"  道路类型分布:")
    for kind_code, count in kind_dist.items():
        kind_name = CONFIG['road_kind_mapping'].get(kind_code, '未知')
        print(f"    - {kind_code} ({kind_name}): {count} 条 ({count/len(df)*100:.1f}%)")
    
    return df


def add_lane_features(df):
    """添加车道数特征"""
    print("\n[3/5] 添加车道数特征...")
    
    # 车道类别
    df['lane_category'] = df['width'].map(CONFIG['lane_mapping']).fillna('未知')
    
    # One-Hot 编码
    df['lane_1'] = (df['width'] == 30).astype(int)
    df['lane_2_3'] = (df['width'] == 55).astype(int)
    df['lane_4_plus'] = (df['width'] == 130).astype(int)
    
    # 统计
    lane_dist = df['lane_category'].value_counts()
    print(f"  ✓ lane_category")
    print(f"  ✓ lane One-Hot 编码 (lane_1, lane_2_3, lane_4_plus)")
    print(f"  车道数分布:")
    for lane_cat, count in lane_dist.items():
        print(f"    - {lane_cat}: {count} 条 ({count/len(df)*100:.1f}%)")
    
    return df


def add_status_features(df):
    """添加路况特征"""
    print("\n[4/5] 添加路况特征...")
    
    # 路况等级分类 (基于聚合后的平均值)
    def get_status_level(status):
        if pd.isna(status):
            return '无数据'
        elif status < 0.5:
            return '无路况'
        elif status < 1.5:
            return '畅通'
        elif status < 2.5:
            return '缓慢(轻度)'
        elif status < 3.5:
            return '缓慢(重度)'
        elif status < 4.5:
            return '拥堵'
        else:
            return '严重拥堵'
    
    df['status_level'] = df['fcd_status'].apply(get_status_level)
    
    # 是否拥堵 (status >= 2.5，即缓慢重度及以上)
    df['is_congested'] = (df['fcd_status'] >= 2.5).astype(int)
    
    # 路况 One-Hot 编码
    status_dummies = pd.get_dummies(df['status_level'], prefix='status')
    df = pd.concat([df, status_dummies], axis=1)
    
    # 统计
    status_dist = df['status_level'].value_counts()
    print(f"  ✓ status_level")
    print(f"  ✓ is_congested")
    print(f"  ✓ status One-Hot 编码")
    print(f"  路况分布:")
    for status_name, count in status_dist.items():
        print(f"    - {status_name}: {count} 条 ({count/len(df)*100:.1f}%)")
    
    congested_ratio = df['is_congested'].mean()
    print(f"  拥堵比例: {congested_ratio:.1%}")
    
    return df


def add_physical_features(df):
    """添加物理/理论特征"""
    print("\n[5/5] 添加物理特征...")
    
    # 确保数值列
    df['fcd_flow'] = pd.to_numeric(df['fcd_flow'], errors='coerce').fillna(0)
    df['fcd_speed'] = pd.to_numeric(df['fcd_speed'], errors='coerce')
    df['length'] = pd.to_numeric(df['length'], errors='coerce')
    
    # 1. 单位长度浮动车累计量
    # fcd_flow 是"车·秒"累计，除以长度得到单位长度的累计
    df['fcd_flow_per_length'] = df.apply(
        lambda r: r['fcd_flow'] / r['length'] if r['length'] > 0 else 0,
        axis=1
    )
    
    # 2. 理论流量估计
    # 基于交通流理论: Q = N * v / L
    # 其中 N 是在途车辆数，v 是速度，L 是路段长度
    # fcd_flow ≈ N * T (T=3600秒)，所以 N ≈ fcd_flow / 3600
    # Q ≈ (fcd_flow / 3600) * v / L = fcd_flow * v / (3600 * L)
    df['theoretical_flow'] = df.apply(
        lambda r: (r['fcd_flow'] * r['fcd_speed'] / (3600 * r['length'])) 
                  if (r['length'] > 0 and pd.notna(r['fcd_speed']) and r['fcd_speed'] > 0) else 0,
        axis=1
    )
    
    # 3. 密度代理
    # 交通密度 K = N / L ≈ fcd_flow / (3600 * L)
    # 或者 K = Q / v，这里用 fcd_flow / (length * speed) 作为代理
    # df['density_proxy'] = df.apply(
    #     lambda r: r['fcd_flow'] / (r['length'] * r['fcd_speed'])
    #               if (r['length'] > 0 and pd.notna(r['fcd_speed']) and r['fcd_speed'] > 0) else 0,
    #     axis=1
    # )
    # 修改后：真正的密度 K (Veh/km)
    df['density_proxy'] = df.apply(
        lambda r: r['fcd_flow'] / (3600 * r['length'])
        if r['length'] > 0 else 0,
        axis=1
    )
    
    # 4. 速度-流量交互特征
    df['speed_flow_interaction'] = df['fcd_flow'] * df['fcd_speed']
    
    # 5. 比值特征 (用于分析)
    df['ratio'] = df.apply(
        lambda r: r['flow_std'] / r['fcd_flow'] if r['fcd_flow'] > 0 else np.nan,
        axis=1
    )
    
    # 统计
    print(f"  ✓ fcd_flow_per_length: 单位长度浮动车累计")
    print(f"  ✓ theoretical_flow: 理论流量 (fcd_flow * speed / (3600 * length))")
    print(f"  ✓ density_proxy: 密度代理")
    print(f"  ✓ speed_flow_interaction: 速度-流量交互")
    print(f"  ✓ ratio: 卡口/浮动车比值 (用于分析)")
    
    # 检查理论流量与实际流量的相关性
    valid_mask = (df['theoretical_flow'] > 0) & (df['flow_std'] > 0)
    if valid_mask.sum() > 100:
        corr = df.loc[valid_mask, 'theoretical_flow'].corr(df.loc[valid_mask, 'flow_std'])
        print(f"\n  theoretical_flow 与 flow_std 相关系数: {corr:.4f}")
    
    return df


def generate_report(df, output_path):
    """生成特征工程报告"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("特征工程报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"数据量: {len(df)} 条\n")
        f.write(f"卡口数: {df['卡口编号'].nunique()}\n")
        f.write(f"特征数: {len(df.columns)}\n\n")
        
        # 特征列表
        f.write("-" * 40 + "\n")
        f.write("特征列表\n")
        f.write("-" * 40 + "\n")
        
        feature_groups = {
            '原始特征': ['卡口编号', '卡口名称', 'Link_ID', '开始时间', '结束时间', 
                       'flow_large', 'flow_small', 'flow_std', 
                       'fcd_flow', 'fcd_speed', 'fcd_status', 'fcd_record_count',
                       'kind_x', 'width', 'length', 'penetration_rate'],
            '时间特征': ['hour', 'weekday', 'hour_sin', 'hour_cos', 
                       'weekday_sin', 'weekday_cos', 'is_weekend', 'time_period'],
            '道路类型': ['road_type_name'] + [c for c in df.columns if c.startswith('kind_')],
            '车道数': ['lane_category', 'lane_1', 'lane_2_3', 'lane_4_plus'],
            '路况': ['status_level', 'is_congested'] + [c for c in df.columns if c.startswith('status_')],
            '物理特征': ['fcd_flow_per_length', 'theoretical_flow', 'density_proxy', 
                       'speed_flow_interaction', 'ratio'],
        }
        
        for group_name, features in feature_groups.items():
            existing = [f for f in features if f in df.columns]
            f.write(f"\n{group_name} ({len(existing)}):\n")
            for feat in existing:
                f.write(f"  - {feat}\n")
        
        # 数值特征统计
        f.write("\n" + "-" * 40 + "\n")
        f.write("关键数值特征统计\n")
        f.write("-" * 40 + "\n\n")
        
        key_numeric = ['flow_std', 'fcd_flow', 'fcd_speed', 'fcd_status',
                       'theoretical_flow', 'density_proxy', 'ratio']
        
        for col in key_numeric:
            if col in df.columns:
                s = df[col].dropna()
                f.write(f"{col}:\n")
                f.write(f"  mean={s.mean():.4f}, median={s.median():.4f}, std={s.std():.4f}\n")
                f.write(f"  min={s.min():.4f}, max={s.max():.4f}\n\n")
        
        # 类别特征分布
        f.write("-" * 40 + "\n")
        f.write("类别特征分布\n")
        f.write("-" * 40 + "\n\n")
        
        cat_features = ['time_period', 'road_type_name', 'lane_category', 'status_level']
        for col in cat_features:
            if col in df.columns:
                f.write(f"{col}:\n")
                dist = df[col].value_counts()
                for val, count in dist.items():
                    f.write(f"  {val}: {count} ({count/len(df)*100:.1f}%)\n")
                f.write("\n")
    
    print(f"\n📄 特征报告: {output_path.name}")


def run(input_file=None):
    """执行特征工程"""
    print("=" * 60)
    print("特征工程增强")
    print("=" * 60)
    
    # 读取数据
    if input_file is None:
        input_file = CONFIG['input_file']
    
    input_path = pm.get_processed_path(input_file)
    print(f"读取数据: {input_path.name}")
    df = pd.read_csv(input_path, low_memory=False)
    
    print(f"原始数据: {len(df)} 条, {len(df.columns)} 列")
    
    # 添加各类特征
    df = add_time_features(df)
    df = add_road_type_features(df)
    df = add_lane_features(df)
    df = add_status_features(df)
    df = add_physical_features(df)
    
    # 清理临时列
    cols_to_drop = ['start_time', 'date']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    # 保存
    output_path = pm.get_processed_path(CONFIG['output_file'])
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    # 生成报告
    report_path = pm.get_processed_path(CONFIG['output_report'])
    generate_report(df, report_path)
    
    print("\n" + "=" * 60)
    print("特征工程完成")
    print("=" * 60)
    print(f"输出数据: {len(df)} 条, {len(df.columns)} 列")
    print(f"新增特征数: {len(df.columns) - 20} (约)")  # 粗略估计
    print(f"✅ 输出文件: {output_path.name}")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="特征工程增强")
    parser.add_argument('--input', type=str, default=None,
                        help='输入文件名 (默认: final_training_data.csv)')
    args = parser.parse_args()
    
    run(input_file=args.input)
