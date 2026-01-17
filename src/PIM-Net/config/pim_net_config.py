"""
PIM-Net 配置文件
Physics-Informed Mapping Network for Traffic Flow Estimation
"""

# === 数据配置 ===
DATA_CONFIG = {
    # 输入文件 (特征工程后的数据)
    'input_file': 'training_data_with_features.csv',
    
    # 数据划分比例
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    
    # 随机种子
    'random_seed': 42,
    
    # 目标列
    'target_col': 'flow_std',
    
    # 浮动车特征列
    'fcd_features': ['fcd_flow', 'fcd_speed', 'fcd_status'],
    
    # 物理特征列
    # 'physics_features': ['theoretical_flow', 'density_proxy', 'fcd_flow_per_length'],
    'physics_features': ['theoretical_flow', 'density_proxy'],

    # 时间特征列
    'time_features': ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'is_weekend'],
    
    # 道路属性特征列 (用于Embedding的原始类别)
    'road_type_col': 'kind_x',  # 道路类型: 01, 02, 03, 04, 06
    'lane_col': 'width',        # 车道数: 30, 55, 130
    'length_col': 'length',     # 路段长度 (连续值)
    
    # 路况类别 (用于Embedding)
    'status_col': 'fcd_status',  # 连续值，模型内部离散化
    
    # 时段类别 (用于Embedding)
    'time_period_col': 'time_period',
    
    # 用于分组的列 (数据划分时按卡口分组)
    'group_col': '卡口编号',
}

# === 类别映射 ===
CATEGORY_CONFIG = {
    # 道路类型映射 (kind_x -> index)
    # 注意：'00'高速公路映射到与'01'城市高速相同的索引
    # 因为卡口数据中没有高速公路样本，使用城市高速作为代理
    'road_type_mapping': {
        '00': 0,  # 高速公路 → 映射到城市高速（代理策略）
        '01': 0,  # 城市高速
        '02': 1,  # 国道
        '03': 2,  # 省道
        '04': 3,  # 县道
        '06': 4,  # 市镇村道
    },
    
    # 车道数映射 (width -> index)
    'lane_mapping': {
        30: 0,   # 1车道
        55: 1,   # 2-3车道
        130: 2,  # 4车道及以上
    },
    
    # 时段映射
    'time_period_mapping': {
        '夜间': 0,
        '早高峰': 1,
        '上午平峰': 2,
        '午间': 3,
        '下午平峰': 4,
        '晚高峰': 5,
        '晚间': 6,
    },
}

# === 模型配置 ===
MODEL_CONFIG = {
    # 嵌入维度
    'road_type_emb_dim': 16,
    'lane_emb_dim': 8,
    'time_period_emb_dim': 16,
    
    # 类别数量
    'num_road_types': 5,
    'num_lanes': 3,
    'num_time_periods': 7,
    
    # 物理分支配置
    'physics_hidden_dims': [32, 16],  # α系数MLP的隐藏层
    'alpha_min': 0.1,   # α系数最小值
    'alpha_max': 20.0,  # α系数最大值
    
    # 残差分支配置 (简化：减少层数防止过拟合)
    'residual_hidden_dims': [64, 32],  # 简化为两层
    
    # Dropout (增强正则化)
    'dropout': 0.2,
    
    # 是否使用物理分支
    'use_physics_branch': True,
    
    # 是否使用语义嵌入
    'use_semantic_embedding': True,
    
    # 是否使用残差连接
    'use_residual': True,
}

# === 训练配置 ===
TRAIN_CONFIG = {
    # 批大小
    'batch_size': 256,
    
    # 学习率
    'learning_rate': 1e-3,
    
    # 权重衰减 (增强L2正则化)
    'weight_decay': 1e-3,
    
    # 训练轮数
    'epochs': 100,
    
    # 早停耐心值
    'patience': 15,
    
    # 学习率调度
    'lr_scheduler': 'plateau',  # 'plateau' or 'cosine'
    'lr_factor': 0.5,
    'lr_patience': 5,
    
    # 梯度裁剪
    'grad_clip': 1.0,
    
    # 设备
    'device': 'cuda',  # 'cuda' or 'cpu', 会自动检测
    
    # 模型保存路径
    'save_dir': 'checkpoints',
    'model_name': 'pim_net',
    
    # 是否使用Log变换目标值
    'use_log_transform': True,
}

# === 评估配置 ===
EVAL_CONFIG = {
    # 评估指标
    'metrics': ['mae', 'rmse', 'mape', 'r2'],
    
    # 分组评估
    'eval_by_road_type': True,
    'eval_by_time_period': True,
    
    # 可视化
    'plot_scatter': True,
    'plot_by_hour': True,
    'plot_residual': True,
}
