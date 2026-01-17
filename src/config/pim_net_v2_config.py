"""
PIM-Net v2 配置文件
===================

包含数据处理、模型架构、训练参数的完整配置
"""

# =============================================================================
# 数据配置
# =============================================================================
DATA_CONFIG = {
    # 输入文件
    'input_file': 'training_data_with_features.csv',
    
    # 数据划分比例
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'random_seed': 42,
    
    # 目标变量
    'target_col': 'flow_std',
    
    # 浮动车特征
    'fcd_features': ['fcd_flow', 'fcd_speed', 'fcd_status'],
    
    # 物理特征
    'physics_features': ['theoretical_flow', 'density_proxy'],
    
    # 时间特征
    'time_features': ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'is_weekend'],
    
    # 类别特征列
    'road_type_col': 'kind_x',
    'lane_col': 'width',
    'time_period_col': 'time_period',
    'length_col': 'length',
    
    # 分组列（用于部分实验的按卡口划分）
    'group_col': '卡口编号',
}


# =============================================================================
# 类别映射
# =============================================================================
CATEGORY_CONFIG = {
    # 道路类型映射
    # '00'高速公路映射到与'01'城市高速相同的索引（代理策略）
    'road_type_mapping': {
        '00': 0,  # 高速公路 → 城市高速
        '01': 0,  # 城市高速
        '02': 1,  # 国道
        '03': 2,  # 省道
        '04': 3,  # 县道
        '06': 4,  # 市镇村道
    },
    
    # 车道数映射
    'lane_mapping': {
        30: 0,   # 1车道
        55: 1,   # 2-3车道
        130: 2,  # 4车道及以上
    },
    
    # 时段映射
    'time_period_mapping': {
        '夜间': 0,      # 0:00-6:00
        '早高峰': 1,    # 7:00-9:00
        '上午平峰': 2,  # 9:00-11:00
        '午间': 3,      # 11:00-14:00
        '下午平峰': 4,  # 14:00-17:00
        '晚高峰': 5,    # 17:00-19:00
        '晚间': 6,      # 19:00-24:00
    },
}


# =============================================================================
# 模型配置 (PIM-Net v2)
# =============================================================================
MODEL_CONFIG = {
    # --- 语义嵌入维度 ---
    'road_type_emb_dim': 16,
    'lane_emb_dim': 8,
    'time_period_emb_dim': 16,
    
    # --- 类别数量 ---
    'num_road_types': 5,
    'num_lanes': 3,
    'num_time_periods': 7,
    
    # --- SE-Block配置 ---
    'se_reduction': 4,  # SE降维比例
    
    # --- DCN配置 ---
    'cross_layers': 2,  # 交叉层数
    
    # --- Deep Network配置 ---
    'deep_dims': [64, 32],  # Deep塔隐藏层
    
    # --- 物理分支配置 ---
    'physics_hidden_dims': [32, 16],
    'alpha_min': 0.1,
    'alpha_max': 20.0,
    
    # --- 残差分支配置 ---
    'residual_hidden_dims': [64, 32],
    
    # --- 门控融合配置 ---
    'gate_hidden_dim': 32,
    
    # --- 正则化 ---
    'dropout': 0.2,
}


# =============================================================================
# 训练配置
# =============================================================================
TRAIN_CONFIG = {
    # 批大小
    'batch_size': 256,
    
    # 学习率
    'learning_rate': 1e-3,
    
    # 权重衰减 (L2正则化)
    'weight_decay': 1e-3,
    
    # 训练轮数
    'epochs': 100,
    
    # 早停耐心值
    'patience': 15,
    
    # 学习率调度
    'lr_scheduler': 'plateau',
    'lr_factor': 0.5,
    'lr_patience': 5,
    
    # 梯度裁剪
    'grad_clip': 1.0,
    
    # 设备
    'device': 'cuda',
    
    # 保存路径
    'save_dir': 'checkpoints',
    'model_name': 'pim_net_v2',
    
    # Log变换
    'use_log_transform': True,
}


# =============================================================================
# 评估配置
# =============================================================================
EVAL_CONFIG = {
    # 输出目录
    'output_dir': 'evaluation_results',
    
    # 可视化设置
    'figure_dpi': 150,
    'figure_format': 'png',
}
