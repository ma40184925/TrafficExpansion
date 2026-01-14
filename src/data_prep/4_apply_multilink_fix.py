"""
Stage 4: 应用修正配置，合并多对一卡口
根据 manual_fix_config.csv 的配置，将多个卡口合并为通用卡口

处理逻辑:
- MEAN: 两个卡口是同一点的重复，流量取平均，生成一个通用卡口
- SUM: 双向卡口合并，流量求和，生成一个通用卡口
- Link_ID 完整保留（如 "100986499+60322527"），后续匹配路网时会用到

用法:
    python 4_apply_multilink_fix.py
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
    'input_flow': 'checkpoint_with_links_final.csv',  # Stage 2 输出
    'input_fix_config': 'manual_fix_config.csv',       # Stage 3 生成并人工编辑
    # 输出
    'output_merged': 'checkpoint_merged.csv',          # 合并后的数据
}


def normalize_id(series):
    """标准化ID字段"""
    return (series.astype(str)
            .str.replace(r'\.0$', '', regex=True)
            .str.strip())


def load_fix_config():
    """
    加载修正配置，解析为处理规则
    返回: 
        - ckpt_groups: {通用卡口ID: {'ckpt_ids': [...], 'link_id': '...', 'op': 'SUM/MEAN'}}
        - affected_ckpts: set of 所有涉及合并的卡口ID
    """
    fix_path = pm.get_processed_path(CONFIG['input_fix_config'])
    print(f"读取修正配置: {fix_path.name}")
    fix_df = pd.read_csv(fix_path)
    
    ckpt_groups = {}
    affected_ckpts = set()
    
    for _, row in fix_df.iterrows():
        link_id = str(row['Link_ID']).strip()
        op = str(row['建议操作']).strip().upper()
        
        # 解析卡口ID列表
        ckpt_str = str(row['卡口ID列表'])
        if '|' in ckpt_str:
            ckpt_ids = [c.strip() for c in ckpt_str.split('|')]
        else:
            ckpt_ids = [ckpt_str.strip()]
        
        # 解析卡口名称列表
        name_str = str(row['卡口名称列表'])
        if '|' in name_str:
            ckpt_names = [n.strip() for n in name_str.split('|')]
        else:
            ckpt_names = [name_str.strip()]
        
        # 生成通用卡口ID（使用第一个卡口ID + 后缀）
        merged_ckpt_id = f"{ckpt_ids[0]}_merged"
        
        # 生成通用卡口名称（合并两个名称的公共部分）
        merged_name = _generate_merged_name(ckpt_names)
        
        ckpt_groups[merged_ckpt_id] = {
            'ckpt_ids': ckpt_ids,
            'ckpt_names': ckpt_names,
            'merged_name': merged_name,
            'link_id': link_id,  # 可能是 "123+456" 这种格式
            'op': op
        }
        
        affected_ckpts.update(ckpt_ids)
    
    return ckpt_groups, affected_ckpts


def _generate_merged_name(names):
    """
    从多个卡口名称生成合并后的通用名称
    例如: ["龙驰路与西山东路路口北向南", "龙驰路与西山东路路口南向北"] 
        -> "龙驰路与西山东路路口(合并)"
    """
    if len(names) == 1:
        return names[0]
    
    # 尝试找公共前缀
    name1, name2 = names[0], names[1]
    
    # 去掉方向词找公共部分
    direction_words = ['北向南', '南向北', '东向西', '西向东', '卡口', '(礼让行人)', '（礼让）', '(礼让)']
    
    clean1 = name1
    clean2 = name2
    for dw in direction_words:
        clean1 = clean1.replace(dw, '')
        clean2 = clean2.replace(dw, '')
    
    # 如果清理后相同，用这个作为基础名
    if clean1.strip() == clean2.strip():
        return f"{clean1.strip()}(合并)"
    
    # 否则找最长公共前缀
    common_prefix = ""
    for c1, c2 in zip(name1, name2):
        if c1 == c2:
            common_prefix += c1
        else:
            break
    
    if len(common_prefix) > 5:
        return f"{common_prefix.strip()}(合并)"
    
    # 兜底：直接拼接
    return f"{names[0]} & {names[1]}"


def apply_fixes():
    """
    应用修正配置，合并多对一卡口
    """
    print("=" * 50)
    print("Stage 4: 应用修正配置，合并多对一卡口")
    print("=" * 50)
    
    # 读取流量数据
    flow_path = pm.get_processed_path(CONFIG['input_flow'])
    print(f"读取流量数据: {flow_path.name}")
    df = pd.read_csv(flow_path)
    
    df['卡口编号'] = normalize_id(df['卡口编号'])
    df['Link_ID'] = normalize_id(df['Link_ID'])
    
    print(f"原始数据: {len(df)} 行, {df['卡口编号'].nunique()} 个卡口")
    
    # 加载修正配置
    ckpt_groups, affected_ckpts = load_fix_config()
    print(f"\n修正配置: {len(ckpt_groups)} 组合并规则")
    print(f"涉及卡口数: {len(affected_ckpts)}")
    
    # 分离需要处理和不需要处理的数据
    df_to_merge = df[df['卡口编号'].isin(affected_ckpts)].copy()
    df_unchanged = df[~df['卡口编号'].isin(affected_ckpts)].copy()
    
    print(f"需要合并的记录: {len(df_to_merge)} 行")
    print(f"保持不变的记录: {len(df_unchanged)} 行")
    
    # 处理每一组合并
    merged_records = []
    
    for merged_id, group_info in ckpt_groups.items():
        ckpt_ids = group_info['ckpt_ids']
        op = group_info['op']
        link_id = group_info['link_id']
        merged_name = group_info['merged_name']
        
        # 筛选这组卡口的数据
        group_df = df_to_merge[df_to_merge['卡口编号'].isin(ckpt_ids)].copy()
        
        if group_df.empty:
            print(f"  ⚠️ 未找到卡口数据: {ckpt_ids}")
            continue
        
        # 按时间聚合
        # 聚合键：开始时间、结束时间
        group_keys = ['开始时间', '结束时间']
        
        # 确定聚合方式
        if op == 'SUM':
            agg_func = 'sum'
        else:  # MEAN
            agg_func = 'mean'
        
        # 执行聚合
        agg_dict = {
            'flow_large': agg_func,
            'flow_small': agg_func,
            'flow_std': agg_func,
        }
        
        # 保留其他列的第一个值（经纬度等）
        other_cols = [c for c in group_df.columns 
                      if c not in group_keys + list(agg_dict.keys()) + ['卡口编号', '卡口名称', 'Link_ID']]
        for col in other_cols:
            agg_dict[col] = 'first'
        
        aggregated = group_df.groupby(group_keys).agg(agg_dict).reset_index()
        
        # 添加新的标识列
        aggregated['卡口编号'] = merged_id
        aggregated['卡口名称'] = merged_name
        aggregated['Link_ID'] = link_id  # 保留完整的 Link_ID（可能是 "123+456"）
        aggregated['原始卡口'] = ' | '.join(ckpt_ids)  # 记录原始卡口
        aggregated['合并方式'] = op
        
        merged_records.append(aggregated)
        
        print(f"  ✓ {op}: {merged_name[:30]}... -> {len(aggregated)} 条记录")
    
    # 合并所有结果
    if merged_records:
        df_merged = pd.concat(merged_records, ignore_index=True)
    else:
        df_merged = pd.DataFrame()
    
    # 给未合并的数据添加标记列（保持一致）
    df_unchanged['原始卡口'] = df_unchanged['卡口编号']
    df_unchanged['合并方式'] = 'SINGLE'
    
    # 最终合并
    final_df = pd.concat([df_unchanged, df_merged], ignore_index=True)
    
    # 保存
    output_path = pm.get_processed_path(CONFIG['output_merged'])
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    # 统计
    print("\n" + "=" * 50)
    print("处理完成")
    print("=" * 50)
    print(f"原始卡口数: {df['卡口编号'].nunique()}")
    print(f"合并后卡口数: {final_df['卡口编号'].nunique()}")
    print(f"  - 未合并: {df_unchanged['卡口编号'].nunique()}")
    print(f"  - 已合并: {len(ckpt_groups)} 组 ({len(affected_ckpts)} 个原始卡口 → {len(ckpt_groups)} 个通用卡口)")
    print(f"总记录数: {len(final_df)}")
    print(f"\n✅ 输出文件: {output_path.name}")
    
    # 显示合并详情
    print("\n[合并详情]")
    print("-" * 80)
    for merged_id, group_info in list(ckpt_groups.items())[:5]:
        print(f"  {group_info['op']:4} | {group_info['link_id']:25} | {group_info['merged_name'][:40]}")
    if len(ckpt_groups) > 5:
        print(f"  ... 共 {len(ckpt_groups)} 组")
    
    return final_df


def run():
    """执行流程"""
    return apply_fixes()


if __name__ == "__main__":
    run()
