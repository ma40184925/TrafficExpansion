"""
Stage 5: 合并路网属性
将路网的道路类型、宽度、长度等属性合并到卡口数据

特殊处理:
- Link_ID 可能是单个值 (如 "17563396") 或多个值 (如 "100986499+60322527")
- 对于多Link的情况，需要匹配所有Link并保留完整信息

用法:
    python 5_merge_road_attrs.py
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


# === 配置参数 ===
CONFIG = {
    # 输入
    'input_checkpoint': 'checkpoint_merged.csv',     # Stage 4 输出
    'input_road_network': 'jinan_road_network.csv',  # 路网文件
    # 道路类型映射
    'road_kind_mapping': {
        '00': '高速公路',
        '01': '城市高速',
        '02': '国道',
        '03': '省道',
        '04': '县道',
        '06': '市镇村道'
    },
    # 输出
    'output_with_attrs': 'checkpoint_with_road_attrs.csv',
    'output_road_stats': 'report_road_distribution.csv',
    'output_dropped': 'report_dropped_checkpoints.csv',
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


def merge_road_attrs():
    """
    合并路网属性到卡口数据
    """
    print("=" * 50)
    print("Stage 5: 合并路网属性")
    print("=" * 50)
    
    # 读取卡口数据
    ckpt_path = pm.get_processed_path(CONFIG['input_checkpoint'])
    print(f"读取卡口数据: {ckpt_path.name}")
    ckpt_df = pd.read_csv(ckpt_path)
    
    # 读取路网数据
    road_path = pm.get_raw_path(CONFIG['input_road_network'])
    print(f"读取路网数据: {CONFIG['input_road_network']}")
    road_df = pd.read_csv(road_path)
    
    # 标准化路网ID
    road_df['id'] = normalize_id(road_df['id'])
    
    # 格式化 kind_x (确保是两位字符串，如 '06')
    if 'kind_x' in road_df.columns:
        road_df['kind_x'] = road_df['kind_x'].astype(str).str.zfill(2)
    
    # 构建路网属性字典，方便快速查找
    # 只保留需要的列
    road_cols = ['id', 'kind_x', 'width', 'length']
    road_cols = [c for c in road_cols if c in road_df.columns]
    road_dict = road_df.set_index('id')[road_cols[1:]].to_dict('index')
    
    print(f"\n路网路段数: {len(road_dict)}")
    print(f"卡口记录数: {len(ckpt_df)}")
    print(f"卡口数: {ckpt_df['卡口编号'].nunique()}")
    
    # 处理每条记录
    print("\n正在匹配路网属性...")
    
    results = []
    dropped_records = []
    
    for idx, row in ckpt_df.iterrows():
        link_id_raw = str(row['Link_ID'])
        link_ids = parse_link_ids(link_id_raw)
        
        # 查找每个 Link 的属性
        attrs_list = []
        missing_links = []
        
        for lid in link_ids:
            if lid in road_dict:
                attrs_list.append({
                    'link_id': lid,
                    **road_dict[lid]
                })
            else:
                missing_links.append(lid)
        
        # 如果所有Link都找不到，记录为dropped
        if not attrs_list:
            dropped_records.append({
                '卡口编号': row['卡口编号'],
                '卡口名称': row.get('卡口名称', ''),
                'Link_ID': link_id_raw,
                '原因': f"所有Link均未在路网中找到: {missing_links}"
            })
            continue
        
        # 如果部分Link找不到，仅警告但继续处理
        if missing_links:
            # 可以选择记录警告，这里先忽略
            pass
        
        # 合并属性
        # 对于多Link情况：
        # - kind_x: 取第一个（假设同一组的道路类型相同）
        # - width: 取平均
        # - length: 求和（双向道路总长度）
        
        new_row = row.to_dict()
        
        if len(attrs_list) == 1:
            # 单Link，直接使用
            attr = attrs_list[0]
            new_row['kind_x'] = attr.get('kind_x', '')
            new_row['width'] = attr.get('width', np.nan)
            new_row['length'] = attr.get('length', np.nan)
            new_row['link_count'] = 1
            new_row['matched_links'] = attr['link_id']
        else:
            # 多Link，需要聚合
            new_row['kind_x'] = attrs_list[0].get('kind_x', '')  # 取第一个

            new_row['width'] = attrs_list[0].get('width', '')
            
            lengths = [a.get('length') for a in attrs_list if pd.notna(a.get('length'))]
            new_row['length'] = sum(lengths) if lengths else np.nan
            
            new_row['link_count'] = len(attrs_list)
            new_row['matched_links'] = '+'.join([a['link_id'] for a in attrs_list])
        
        results.append(new_row)
    
    # 转换为DataFrame
    result_df = pd.DataFrame(results)
    
    # 统计
    print("\n" + "-" * 50)
    print("匹配结果统计")
    print("-" * 50)
    
    original_ckpts = ckpt_df['卡口编号'].nunique()
    matched_ckpts = result_df['卡口编号'].nunique() if not result_df.empty else 0
    dropped_ckpts = len(set(ckpt_df['卡口编号'].unique()) - set(result_df['卡口编号'].unique() if not result_df.empty else []))
    
    print(f"原始卡口数: {original_ckpts}")
    print(f"匹配成功: {matched_ckpts}")
    print(f"匹配失败: {dropped_ckpts}")
    
    # Link数量分布
    if not result_df.empty:
        link_count_dist = result_df.groupby('卡口编号')['link_count'].first().value_counts().sort_index()
        print("\nLink数量分布:")
        for count, num in link_count_dist.items():
            print(f"  - {count} 个Link: {num} 个卡口")
    
    # 道路类型分布
    if not result_df.empty and 'kind_x' in result_df.columns:
        print("\n道路类型分布:")
        kind_mapping = CONFIG['road_kind_mapping']
        kind_stats = result_df.groupby('卡口编号')['kind_x'].first().value_counts()
        total = kind_stats.sum()
        
        for kind_code, count in kind_stats.items():
            kind_name = kind_mapping.get(kind_code, '未知')
            pct = count / total * 100
            print(f"  - {kind_code} ({kind_name}): {count} ({pct:.1f}%)")
        
        # 保存道路类型统计
        stats_df = pd.DataFrame({
            '类型代码': kind_stats.index,
            '卡口数量': kind_stats.values
        })
        stats_df['道路类型'] = stats_df['类型代码'].map(kind_mapping).fillna('未知')
        stats_df['占比'] = (stats_df['卡口数量'] / total * 100).round(1).astype(str) + '%'
        
        stats_path = pm.get_processed_path(CONFIG['output_road_stats'])
        stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
        print(f"\n📊 道路类型统计: {stats_path.name}")
    
    # 保存被剔除的记录
    if dropped_records:
        dropped_df = pd.DataFrame(dropped_records).drop_duplicates(subset=['卡口编号'])
        dropped_path = pm.get_processed_path(CONFIG['output_dropped'])
        dropped_df.to_csv(dropped_path, index=False, encoding='utf-8-sig')
        print(f"⚠️ 剔除记录: {dropped_path.name} ({len(dropped_df)} 个卡口)")
    
    # 保存结果
    output_path = pm.get_processed_path(CONFIG['output_with_attrs'])
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 50)
    print("Stage 5 完成")
    print("=" * 50)
    print(f"输出记录数: {len(result_df)}")
    print(f"输出卡口数: {result_df['卡口编号'].nunique() if not result_df.empty else 0}")
    print(f"✅ 输出文件: {output_path.name}")
    
    return result_df


def run():
    """执行流程"""
    return merge_road_attrs()


if __name__ == "__main__":
    run()
