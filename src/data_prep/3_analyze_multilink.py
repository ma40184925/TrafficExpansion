"""
Stage 3: 多对一分析与模板生成
基于 checkpoint_with_links_final.csv 分析 Link-卡口 映射关系

功能:
1. 分析 Link-卡口 映射关系
2. 生成人工修正模板（自动推断 SUM/MEAN）

用法:
    python 3_analyze_multilink.py               # 分析并生成模板
    python 3_analyze_multilink.py --analyze     # 仅分析，不生成模板
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
    # 输入 (Stage 2 的输出)
    'input_file': 'checkpoint_with_links_final.csv',
    # 输出
    'output_mapping_report': 'report_link_checkpoint_mapping.csv',
    'output_fix_template': 'manual_fix_config.csv',
}


def normalize_id(series):
    """标准化ID字段"""
    return (series.astype(str)
            .str.replace(r'\.0$', '', regex=True)
            .str.strip())


def detect_direction(name):
    """从卡口名称检测方向"""
    if '东向西' in name:
        return 'EW'
    elif '西向东' in name:
        return 'WE'
    elif '南向北' in name:
        return 'SN'
    elif '北向南' in name:
        return 'NS'
    return 'UNKNOWN'


def infer_aggregation_action(names):
    """根据卡口名称推断聚合操作"""
    directions = [detect_direction(n) for n in names]
    unique_dirs = set(directions)

    # 规则1: 双向 -> 求和
    if ('NS' in unique_dirs and 'SN' in unique_dirs) or \
       ('EW' in unique_dirs and 'WE' in unique_dirs):
        return "SUM", "检测到双向名称，建议求和"

    # 规则2: 方向相同 -> 取平均
    if len(unique_dirs) == 1 and list(unique_dirs)[0] != 'UNKNOWN':
        return "MEAN", "方向相同，建议取平均"

    # 规则3: 名称相似 -> 取平均
    clean_names = [n.replace('卡口', '').replace('(礼让)', '').replace('（礼让）', '')
                   for n in names]
    if len(set(clean_names)) < len(names):
        return "MEAN", "名称高度相似，疑似重复"

    return "CHECK", "无法自动判断，请人工核实"


def analyze_mapping(df):
    """
    分析 Link-卡口 映射关系
    """
    print("=" * 50)
    print("分析 Link-卡口 映射关系")
    print("=" * 50)

    # 提取唯一映射 (只需要这三列)
    unique_mapping = df[['Link_ID', '卡口编号', '卡口名称']].drop_duplicates()

    # 聚合统计
    link_stats = unique_mapping.groupby('Link_ID').agg({
        '卡口编号': ['count', list],
        '卡口名称': list
    }).reset_index()

    link_stats.columns = ['Link_ID', '卡口数量', '卡口ID列表', '卡口名称列表']

    # 分布统计
    dist_stats = link_stats['卡口数量'].value_counts().sort_index()

    print(f"总路段数: {len(link_stats)}")
    print(f"总卡口数: {link_stats['卡口数量'].sum()}")
    print("\n映射关系分布:")
    for count, num_links in dist_stats.items():
        print(f"  - {count}个卡口 → {num_links} 条路段")

    # 多对一
    multi_links = link_stats[link_stats['卡口数量'] > 1].copy()

    if not multi_links.empty:
        print(f"\n🔍 发现 {len(multi_links)} 条路段挂载了多个卡口")
        print("\n[示例]")
        for _, row in multi_links.head(5).iterrows():
            names = ", ".join(row['卡口名称列表'][:2])
            if len(row['卡口名称列表']) > 2:
                names += f"... (+{len(row['卡口名称列表'])-2})"
            print(f"  Link {row['Link_ID']}: {names}")
    else:
        print("\n🎉 不存在多对一情况")

    # 保存映射报告
    report_df = link_stats.copy()
    report_df['卡口名称列表'] = report_df['卡口名称列表'].apply(lambda x: " | ".join(x))
    report_df['卡口ID列表'] = report_df['卡口ID列表'].apply(lambda x: " | ".join(x))

    report_path = pm.get_processed_path(CONFIG['output_mapping_report'])
    report_df.sort_values('卡口数量', ascending=False).to_csv(
        report_path, index=False, encoding='utf-8-sig'
    )
    print(f"\n📄 映射报告: {report_path.name}")

    return link_stats, multi_links


def generate_template(multi_links):
    """
    生成人工修正模板
    """
    print("\n" + "=" * 50)
    print("生成人工修正模板")
    print("=" * 50)

    if multi_links.empty:
        print("无多对一情况，无需生成模板")
        return

    recommendations = []

    for _, row in multi_links.iterrows():
        link_id = row['Link_ID']
        names = row['卡口名称列表']
        ckpt_ids = row['卡口ID列表']

        action, reason = infer_aggregation_action(names)

        recommendations.append({
            'Link_ID': link_id,
            '建议操作': action,
            '推断理由': reason,
            '卡口数量': len(names),
            '卡口名称列表': " | ".join(names),
            '卡口ID列表': " | ".join(ckpt_ids)
        })

    rec_df = pd.DataFrame(recommendations)

    # 统计
    action_counts = rec_df['建议操作'].value_counts()
    print("推断结果统计:")
    for action, count in action_counts.items():
        print(f"  - {action}: {count} 条")

    # 保存
    output_path = pm.get_processed_path(CONFIG['output_fix_template'])
    rec_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n✅ 模板已生成: {output_path.name}")
    print("\n【操作指南】")
    print("1. 用 Excel 打开该文件")
    print("2. 检查 '建议操作' 列:")
    print("   - SUM: 双向车流合并（求和）")
    print("   - MEAN: 重复数据（取平均）")
    print("   - CHECK: 需人工核实，请改为 SUM 或 MEAN")
    print("3. 修改后保存")
    print("4. 后续脚本会读取此文件进行聚合处理")


def run(analyze_only=False):
    """执行流程"""
    # 读取数据
    input_path = pm.get_processed_path(CONFIG['input_file'])
    print(f"读取数据: {input_path.name}")
    df = pd.read_csv(input_path)
    
    df['Link_ID'] = normalize_id(df['Link_ID'])
    df['卡口编号'] = normalize_id(df['卡口编号'])

    # 分析映射
    link_stats, multi_links = analyze_mapping(df)

    # 生成模板
    if not analyze_only and not multi_links.empty:
        generate_template(multi_links)

    return link_stats, multi_links


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: 多对一分析与模板生成")
    parser.add_argument('--analyze', action='store_true',
                        help='仅分析映射关系，不生成模板')
    args = parser.parse_args()

    run(analyze_only=args.analyze)
