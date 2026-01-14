"""
Stage 2: Link ID 匹配
合并原 3_a_match_links.py, 3_b_repair_matching.py, 3_b_manual_repair.py

功能:
1. 基于匹配表自动关联 卡口 → Link_ID
2. 空间最近邻修复未匹配卡口
3. 人工修复表合并（可选）

用法:
    python 2_link_matching.py                       # 自动匹配 + 空间修复
    python 2_link_matching.py --no-repair           # 仅自动匹配
    python 2_link_matching.py --manual FILE.csv     # 额外合并人工修复表
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
    # 输入
    'input_file': 'checkpoint_flow_std_high_quality.csv',
    'match_table': '交警卡口信息表0315-匹配表.xlsx',
    'road_network': 'jinan_road_network.csv',
    # 空间匹配阈值(米)
    'max_distance_meters': 50.0,
    # 输出
    'output_matched': 'checkpoint_with_links.csv',
    'output_unmatched': 'unmatched_checkpoints.csv',
    'output_repair_success': 'report_repair_success.csv',
    'output_repair_failed': 'report_repair_failed.csv',
    'output_final': 'checkpoint_with_links_final.csv',
}


def normalize_id(series):
    """标准化ID字段"""
    return (series.astype(str)
            .str.replace(r'\.0$', '', regex=True)
            .str.strip())


def step1_auto_match():
    """
    Step 1: 基于匹配表自动匹配
    返回: (matched_df, unmatched_df)
    """
    print("=" * 50)
    print("Step 1: 自动匹配 Link ID")
    print("=" * 50)

    # 读取流量数据
    flow_path = pm.get_processed_path(CONFIG['input_file'])
    print(f"读取流量数据: {flow_path.name}")
    flow_df = pd.read_csv(flow_path)

    # 读取匹配表
    match_path = pm.get_raw_path(CONFIG['match_table'])
    print(f"读取匹配表: {CONFIG['match_table']}")
    match_df = pd.read_excel(match_path)

    # 统一ID格式
    flow_df['卡口编号'] = normalize_id(flow_df['卡口编号'])
    match_df['卡口编号'] = normalize_id(match_df['卡口编号'])

    # 执行匹配 (Left Join)
    cols_to_use = ['卡口编号', 'Link_ID', 'lon_84', 'lat_84']
    missing_cols = [c for c in cols_to_use if c not in match_df.columns]
    if missing_cols:
        raise KeyError(f"匹配表缺少必要列: {missing_cols}")

    print("正在执行匹配...")
    merged_df = pd.merge(flow_df, match_df[cols_to_use], on='卡口编号', how='left')

    # 分离成功/失败
    success_mask = merged_df['Link_ID'].notna()
    matched_df = merged_df[success_mask].copy()
    unmatched_df = merged_df[~success_mask].copy()

    # 格式化 Link_ID
    matched_df['Link_ID'] = matched_df['Link_ID'].astype(int).astype(str)

    # 统计
    matched_count = matched_df['卡口编号'].nunique()
    unmatched_count = unmatched_df['卡口编号'].nunique()

    print("-" * 40)
    print(f"原始卡口数: {flow_df['卡口编号'].nunique()}")
    print(f"匹配成功: {matched_count}")
    print(f"匹配失败: {unmatched_count}")
    print("-" * 40)

    # 保存成功数据
    output_success = pm.get_processed_path(CONFIG['output_matched'])
    matched_df.to_csv(output_success, index=False, encoding='utf-8-sig')
    print(f"✅ 匹配成功: {output_success.name}")

    # 保存失败数据
    if not unmatched_df.empty:
        unmatched_unique = unmatched_df[['卡口编号', '卡口名称']].drop_duplicates()
        for col in ['lon_84', 'lat_84']:
            if col in unmatched_df.columns:
                first_vals = unmatched_df.groupby('卡口编号')[col].first()
                unmatched_unique = unmatched_unique.merge(
                    first_vals.reset_index(), on='卡口编号', how='left'
                )

        output_fail = pm.get_processed_path(CONFIG['output_unmatched'])
        unmatched_unique.to_csv(output_fail, index=False, encoding='utf-8-sig')
        print(f"⚠️ 未匹配名单: {output_fail.name}")
    else:
        print("🎉 所有卡口都匹配成功！")

    return matched_df, unmatched_df


def step2_spatial_repair(unmatched_df):
    """
    Step 2: 空间最近邻修复
    """
    print("\n" + "=" * 50)
    print("Step 2: 空间最近邻修复")
    print("=" * 50)

    try:
        import geopandas as gpd
        from shapely import wkt
    except ImportError:
        print("⚠️ 缺少 geopandas/shapely，跳过空间修复")
        print("   安装: pip install geopandas shapely")
        return None

    # 读取路网
    road_path = pm.get_raw_path(CONFIG['road_network'])
    print(f"读取路网: {CONFIG['road_network']}")
    road_df = pd.read_csv(road_path)

    # 几何转换
    print("处理几何投影 (WGS84 → UTM Zone 50N)...")
    road_df['geometry'] = road_df['geometry'].apply(wkt.loads)
    gdf_roads = gpd.GeoDataFrame(road_df, geometry='geometry', crs="EPSG:4326")

    # 提取未匹配卡口唯一坐标
    unmatched_unique = unmatched_df[['卡口编号', '卡口名称', 'lon_84', 'lat_84']].drop_duplicates()

    if 'lon_84' not in unmatched_unique.columns or unmatched_unique['lon_84'].isna().all():
        print("⚠️ 未匹配卡口缺少有效经纬度，跳过空间修复")
        return None

    gdf_points = gpd.GeoDataFrame(
        unmatched_unique,
        geometry=gpd.points_from_xy(unmatched_unique.lon_84, unmatched_unique.lat_84),
        crs="EPSG:4326"
    )

    # 投影到米
    gdf_roads_meter = gdf_roads.to_crs("EPSG:32650")
    gdf_points_meter = gdf_points.to_crs("EPSG:32650")

    # 空间匹配
    max_dist = CONFIG['max_distance_meters']
    print(f"计算最近邻 (阈值: {max_dist}m)...")

    matched_repair = gpd.sjoin_nearest(
        gdf_points_meter,
        gdf_roads_meter[['id', 'geometry']],
        how='left',
        distance_col='dist_meters'
    )

    # 划分成功/失败
    success_mask = matched_repair['dist_meters'] <= max_dist
    success_repair = matched_repair[success_mask].copy()
    failed_repair = matched_repair[~success_mask].copy()

    print("-" * 40)
    print(f"尝试修复数: {len(gdf_points)}")
    print(f"成功 (≤{max_dist}m): {len(success_repair)}")
    print(f"失败 (>{max_dist}m): {len(failed_repair)}")
    print("-" * 40)

    # 保存报告
    if not success_repair.empty:
        success_repair.rename(columns={'id': 'Matched_LinkID'}, inplace=True)
        report_path = pm.get_processed_path(CONFIG['output_repair_success'])
        success_repair[['卡口编号', '卡口名称', 'Matched_LinkID', 'dist_meters']].to_csv(
            report_path, index=False, encoding='utf-8-sig'
        )
        print(f"✅ 修复成功报告: {report_path.name}")

    if not failed_repair.empty:
        failed_repair.rename(columns={'id': 'Nearest_LinkID'}, inplace=True)
        report_path = pm.get_processed_path(CONFIG['output_repair_failed'])
        failed_repair[['卡口编号', '卡口名称', 'Nearest_LinkID', 'dist_meters']].to_csv(
            report_path, index=False, encoding='utf-8-sig'
        )
        print(f"⚠️ 修复失败报告: {report_path.name}")

    if success_repair.empty:
        return None

    # 回捞流量数据
    repair_map = dict(zip(success_repair['卡口编号'], success_repair['Matched_LinkID']))

    raw_path = pm.get_processed_path(CONFIG['input_file'])
    raw_df = pd.read_csv(raw_path)
    raw_df['卡口编号'] = normalize_id(raw_df['卡口编号'])

    repaired_rows = raw_df[raw_df['卡口编号'].isin(repair_map.keys())].copy()
    repaired_rows['Link_ID'] = repaired_rows['卡口编号'].map(repair_map)
    repaired_rows['Link_ID'] = repaired_rows['Link_ID'].astype(int).astype(str)

    print(f"回捞流量记录: {len(repaired_rows)} 条")

    return repaired_rows


def step3_manual_repair(main_df, repair_file):
    """
    Step 3: 合并人工修复表
    """
    print("\n" + "=" * 50)
    print("Step 3: 合并人工修复表")
    print("=" * 50)

    # 尝试多个路径
    repair_path = pm.get_processed_path(repair_file)
    if not repair_path.exists():
        repair_path = pm.get_raw_path(repair_file)

    if not repair_path.exists():
        print(f"⚠️ 人工修复文件不存在: {repair_file}，跳过")
        return main_df

    print(f"读取人工修复表: {repair_path.name}")
    repair_df = pd.read_csv(repair_path)

    # 标准化
    repair_df['卡口编号'] = normalize_id(repair_df['卡口编号'])
    if 'LinkID' in repair_df.columns:
        repair_df.rename(columns={'LinkID': 'Link_ID'}, inplace=True)
    repair_df['Link_ID'] = normalize_id(repair_df['Link_ID'])

    # 构建映射
    repair_valid = repair_df[repair_df['Link_ID'].notna() & (repair_df['Link_ID'] != '')]
    repair_map = dict(zip(repair_valid['卡口编号'], repair_valid['Link_ID']))

    print(f"有效修复卡口数: {len(repair_map)}")

    # 回捞流量
    raw_path = pm.get_processed_path(CONFIG['input_file'])
    raw_df = pd.read_csv(raw_path)
    raw_df['卡口编号'] = normalize_id(raw_df['卡口编号'])

    repaired_rows = raw_df[raw_df['卡口编号'].isin(repair_map.keys())].copy()
    repaired_rows['Link_ID'] = repaired_rows['卡口编号'].map(repair_map)

    # 去重合并
    main_df['卡口编号'] = normalize_id(main_df['卡口编号'])
    main_df_safe = main_df[~main_df['卡口编号'].isin(repair_map.keys())]

    final_df = pd.concat([main_df_safe, repaired_rows], ignore_index=True)

    print("-" * 40)
    print(f"主表卡口数: {main_df['卡口编号'].nunique()}")
    print(f"人工修复数: {len(repair_map)}")
    print(f"合并后卡口数: {final_df['卡口编号'].nunique()}")
    print("-" * 40)

    return final_df


def run(auto_repair=True, manual_repair_file=None):
    """执行完整流程"""
    # Step 1: 自动匹配
    matched_df, unmatched_df = step1_auto_match()

    # Step 2: 空间修复
    if auto_repair and not unmatched_df.empty:
        repaired_df = step2_spatial_repair(unmatched_df)
        if repaired_df is not None and not repaired_df.empty:
            matched_df = pd.concat([matched_df, repaired_df], ignore_index=True)

    # Step 3: 人工修复
    if manual_repair_file:
        matched_df = step3_manual_repair(matched_df, manual_repair_file)

    # 保存最终结果
    output_path = pm.get_processed_path(CONFIG['output_final'])
    matched_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 50)
    print("Stage 2 完成")
    print("=" * 50)
    print(f"最终卡口数: {matched_df['卡口编号'].nunique()}")
    print(f"输出文件: {output_path.name}")

    return matched_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Link ID 匹配")
    parser.add_argument('--no-repair', action='store_true',
                        help='不执行空间修复')
    parser.add_argument('--manual', type=str, default=None,
                        help='人工修复文件名')
    args = parser.parse_args()

    run(auto_repair=not args.no_repair, manual_repair_file=args.manual)
