"""
PIM-Net v2: Physics-Informed Mapping Network with Deep Feature Interaction
===========================================================================

升级版物理感知映射网络，融合以下创新模块：
1. 物理感知的灰箱建模架构 (Physics-Informed Grey-Box Modeling)
2. 基于DCN的高阶特征交互 (High-Order Feature Interaction via Deep Cross Network)
3. 轻量级通道注意力机制 (Channel-wise Attention via SE-Block)
4. 道路语义嵌入机制 (Road Semantic Embedding)
5. 自适应门控融合策略 (Adaptive Gated Fusion Strategy)

Author: Jason
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# 路径设置
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from config import MODEL_CONFIG


# =============================================================================
# 核心模块定义
# =============================================================================

class SemanticEmbedding(nn.Module):
    """
    语义嵌入模块
    ============
    将离散的道路属性（类型、车道数、时段）映射到连续的低维语义空间，
    使模型能够学习并表达不同类别间的隐式语义关联。
    
    与One-Hot编码相比的优势：
    - 维度更低、更稠密
    - 可学习类别间的相似性
    - 支持对未见类别的泛化
    """
    
    def __init__(self, config=None):
        super().__init__()
        config = config or MODEL_CONFIG
        
        # 道路类型嵌入: 5类 → 16维
        self.road_type_emb = nn.Embedding(
            num_embeddings=config['num_road_types'],
            embedding_dim=config['road_type_emb_dim']
        )
        
        # 车道数嵌入: 3类 → 8维
        self.lane_emb = nn.Embedding(
            num_embeddings=config['num_lanes'],
            embedding_dim=config['lane_emb_dim']
        )
        
        # 时段嵌入: 7类 → 16维
        self.time_period_emb = nn.Embedding(
            num_embeddings=config['num_time_periods'],
            embedding_dim=config['time_period_emb_dim']
        )
        
        # 总输出维度
        self.output_dim = (
            config['road_type_emb_dim'] + 
            config['lane_emb_dim'] + 
            config['time_period_emb_dim']
        )
    
    def forward(self, road_type, lane, time_period):
        """
        Args:
            road_type: [B] 道路类型索引
            lane: [B] 车道数索引
            time_period: [B] 时段索引
        
        Returns:
            semantic_emb: [B, output_dim] 拼接后的语义嵌入向量
        """
        road_emb = self.road_type_emb(road_type)
        lane_emb = self.lane_emb(lane)
        time_emb = self.time_period_emb(time_period)
        
        return torch.cat([road_emb, lane_emb, time_emb], dim=-1)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (通道注意力)
    ==========================================
    
    实现特征重要性的动态重校准（Feature Recalibration）。
    通过学习每个特征通道的权重，自适应地增强重要特征、抑制次要特征。
    
    原理：
    1. Squeeze: 全局平均池化，获取特征的全局统计信息
    2. Excitation: 通过两层FC学习通道间的非线性关系
    3. Reweight: 用学习到的权重对原始特征进行重标定
    
    参考: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    """
    
    def __init__(self, num_features, reduction=4):
        """
        Args:
            num_features: 输入特征维度
            reduction: 降维比例，控制中间层大小
        """
        super().__init__()
        
        hidden_dim = max(num_features // reduction, 8)
        
        self.squeeze = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        
        self.excitation = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_features),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C] 输入特征
        
        Returns:
            out: [B, C] 重校准后的特征
            weights: [B, C] 学习到的通道权重（用于可视化分析）
        """
        # Squeeze: 对于[B, C]输入，直接使用特征本身计算权重
        # 这里我们用batch统计来增强稳定性
        weights = self.excitation(x)  # [B, C]
        
        # Reweight
        out = x * weights
        
        return out, weights


class CrossNetwork(nn.Module):
    """
    Deep Cross Network (DCN) - 显式特征交叉
    ========================================
    
    通过显式的特征交叉操作捕捉高阶特征交互，解决传统MLP难以学习
    乘性模式（Multiplicative Patterns）的问题。
    
    核心公式：
        x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l
    
    其中 ⊙ 表示逐元素乘法，实现了x_0与x_l的显式交叉。
    
    学术意义：
    - 显式建模"时间 × 空间 × 状态"的多阶耦合关系
    - 相比MLP，参数效率更高（O(d)而非O(d²)）
    - 每一层都增加一阶特征交互
    
    参考: Wang et al., "DCN V2: Improved Deep & Cross Network", WWW 2021
    """
    
    def __init__(self, input_dim, num_layers=2):
        """
        Args:
            input_dim: 输入特征维度
            num_layers: 交叉层数，每层增加一阶交互
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.input_dim = input_dim
        
        # 每层的权重和偏置
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, 1) * 0.01)
            for _ in range(num_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_layers)
        ])
        
    def forward(self, x0):
        """
        Args:
            x0: [B, D] 输入特征（同时作为交叉的基底）
        
        Returns:
            x: [B, D] 经过多层交叉后的特征
        """
        x = x0
        for i in range(self.num_layers):
            # x_{l+1} = x_0 * (x_l^T * W_l) + b_l + x_l
            xw = torch.matmul(x, self.weights[i])  # [B, 1]
            x = x0 * xw + self.biases[i] + x       # [B, D]
        return x


class DeepNetwork(nn.Module):
    """
    Deep Network (MLP) - 隐式特征交互
    ==================================
    
    通过多层感知机学习特征的隐式非线性变换。
    与CrossNetwork配合，形成"显式+隐式"的双塔结构。
    """
    
    def __init__(self, input_dim, hidden_dims=None, dropout=0.2):
        """
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            dropout: Dropout比率
        """
        super().__init__()
        
        hidden_dims = hidden_dims or [64, 32]
        
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        
    def forward(self, x):
        """
        Args:
            x: [B, D] 输入特征
        
        Returns:
            out: [B, hidden_dims[-1]] 变换后的特征
        """
        return self.network(x)


class FeatureInteractionModule(nn.Module):
    """
    特征交互模块（双塔结构）
    ========================
    
    结合CrossNetwork（显式交叉）和DeepNetwork（隐式变换），
    全面捕捉特征间的高阶交互关系。
    
    架构：
        输入 → [CrossNetwork] → Cross Output
            ↘ [DeepNetwork]  → Deep Output
                              ↓
                         Concat → 输出
    """
    
    def __init__(self, input_dim, cross_layers=2, deep_dims=None, dropout=0.2):
        """
        Args:
            input_dim: 输入特征维度
            cross_layers: DCN层数
            deep_dims: Deep网络隐藏层维度
            dropout: Dropout比率
        """
        super().__init__()
        
        deep_dims = deep_dims or [64, 32]
        
        # Cross Tower
        self.cross_net = CrossNetwork(input_dim, cross_layers)
        
        # Deep Tower
        self.deep_net = DeepNetwork(input_dim, deep_dims, dropout)
        
        # 输出维度 = Cross输出 + Deep输出
        self.output_dim = input_dim + self.deep_net.output_dim
        
    def forward(self, x):
        """
        Args:
            x: [B, D] 输入特征
        
        Returns:
            out: [B, output_dim] Cross和Deep输出的拼接
        """
        cross_out = self.cross_net(x)       # [B, D]
        deep_out = self.deep_net(x)         # [B, deep_dims[-1]]
        
        return torch.cat([cross_out, deep_out], dim=-1)


class PhysicsGuidedBranch(nn.Module):
    """
    物理引导分支
    ============
    
    基于交通流基本关系 Q = K × V，学习动态校正系数α，
    将理论流量映射为基础估计值。
    
    核心公式：
        Q_phy = α(context) × Q_theo + bias
    
    物理意义：
    - α可理解为"动态渗透率校正系数"
    - 不同道路类型、时段、路况下，α应有所不同
    - 通过Sigmoid将α约束在合理范围内，保证数值稳定性
    """
    
    def __init__(self, condition_dim, hidden_dims=None, dropout=0.2,
                 alpha_min=0.1, alpha_max=20.0):
        """
        Args:
            condition_dim: 条件特征维度
            hidden_dims: MLP隐藏层维度
            dropout: Dropout比率
            alpha_min: α最小值
            alpha_max: α最大值
        """
        super().__init__()
        
        hidden_dims = hidden_dims or [32, 16]
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_range = alpha_max - alpha_min
        
        # Alpha估计网络
        layers = []
        in_dim = condition_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, 1))
        self.alpha_net = nn.Sequential(*layers)
        
        # 可学习偏置
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, theoretical_flow, conditions):
        """
        Args:
            theoretical_flow: [B] 理论流量（标准化后）
            conditions: [B, condition_dim] 条件特征
        
        Returns:
            q_phy: [B] 物理分支输出
            alpha: [B] 学习到的校正系数
        """
        # 估计α
        alpha_raw = self.alpha_net(conditions)
        
        # Sigmoid约束到[alpha_min, alpha_max]
        alpha = torch.sigmoid(alpha_raw) * self.alpha_range + self.alpha_min
        alpha = alpha.squeeze(-1)  # [B]
        
        # Q_phy = α × Q_theo + bias
        q_phy = alpha * theoretical_flow + self.bias
        
        return q_phy, alpha


class ResidualCorrectionBranch(nn.Module):
    """
    残差修正分支
    ============
    
    学习物理模型无法解释的非线性偏差，对物理分支的输出进行补充修正。
    """
    
    def __init__(self, input_dim, hidden_dims=None, dropout=0.2):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dims: MLP隐藏层维度
            dropout: Dropout比率
        """
        super().__init__()
        
        hidden_dims = hidden_dims or [64, 32]
        
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        
        # 输出残差值
        layers.append(nn.Linear(in_dim, 1))
        self.residual_net = nn.Sequential(*layers)
        
    def forward(self, features):
        """
        Args:
            features: [B, input_dim] 融合特征
        
        Returns:
            residual: [B] 残差修正值
        """
        return self.residual_net(features).squeeze(-1)


class AdaptiveGatingUnit(nn.Module):
    """
    自适应门控融合单元 (AGU)
    ========================
    
    动态学习物理分支和残差分支的融合权重，实现场景自适应的预测。
    
    核心公式：
        z = σ(MLP(context))
        Q_final = (1 - z) × Q_phy + z × Q_res
    
    物理意义：
    - z → 0: 更信任物理模型（正常交通流状态）
    - z → 1: 更信任数据残差（极端路况、物理失效）
    
    这是一种"元学习"思想：模型不仅学流量，还在学
    "当前场景下，应该更相信物理公式还是数据残差"
    
    改进：初始化时偏向物理分支（z≈0.3-0.4），防止残差分支"走捷径"
    """
    
    def __init__(self, context_dim, hidden_dim=32, init_gate_bias=-0.5):
        """
        Args:
            context_dim: 上下文特征维度
            hidden_dim: 隐藏层维度
            init_gate_bias: 初始门控偏置，负值使初始gate偏向0（物理分支）
                           sigmoid(-0.5) ≈ 0.38, sigmoid(-1.0) ≈ 0.27
        """
        super().__init__()
        
        self.gate_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 初始化最后一层的偏置，使初始输出偏向物理分支
        # gate_net[-2]是最后一个Linear层（Sigmoid是[-1]）
        nn.init.zeros_(self.gate_net[-2].weight)
        nn.init.constant_(self.gate_net[-2].bias, init_gate_bias)
        
    def forward(self, context, q_phy, q_res):
        """
        Args:
            context: [B, context_dim] 门控上下文特征
            q_phy: [B] 物理分支输出
            q_res: [B] 残差分支输出
        
        Returns:
            q_final: [B] 融合后的最终输出
            gate: [B] 门控系数（用于可解释性分析）
        """
        gate = self.gate_net(context).squeeze(-1)  # [B]
        
        # 加权融合
        q_final = (1 - gate) * q_phy + gate * q_res
        
        return q_final, gate


# =============================================================================
# PIM-Net v2 完整模型
# =============================================================================

class PIMNetV2(nn.Module):
    """
    PIM-Net v2: Physics-Informed Mapping Network with Deep Feature Interaction
    ===========================================================================
    
    完整架构：
    
    输入特征
        │
        ▼
    ┌─────────────────────┐
    │  Semantic Embedding │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │      SE-Block       │  ← 通道注意力
    └──────────┬──────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
    ┌────────┐   ┌────────┐
    │  DCN   │   │  Deep  │  ← 双塔特征交互
    └───┬────┘   └───┬────┘
        └──────┬─────┘
               │
        ┌──────┴──────┐
        ▼             ▼
    ┌────────┐   ┌────────┐
    │Physics │   │Residual│  ← 双流预测
    │ Branch │   │ Branch │
    └───┬────┘   └───┬────┘
        │             │
        └──────┬──────┘
               ▼
    ┌─────────────────────┐
    │  Adaptive Gating    │  ← 自适应融合
    └──────────┬──────────┘
               │
               ▼
          最终预测 Q
    """
    
    def __init__(self, fcd_dim, physics_dim, time_dim, config=None):
        """
        Args:
            fcd_dim: 浮动车特征维度 (3: flow, speed, status)
            physics_dim: 物理特征维度 (2: Q_theo, K_proxy)
            time_dim: 时间特征维度 (5: sin/cos编码 + weekend)
            config: 模型配置字典
        """
        super().__init__()
        
        self.config = config or MODEL_CONFIG
        
        # === 1. 语义嵌入模块 ===
        self.semantic_emb = SemanticEmbedding(self.config)
        semantic_dim = self.semantic_emb.output_dim  # 40
        
        # === 2. 特征拼接后的总维度 ===
        # FCD特征 + 物理特征 + 时间特征 + 语义嵌入 + 路段长度
        total_feature_dim = fcd_dim + physics_dim + time_dim + semantic_dim + 1
        
        # === 3. SE-Block 通道注意力 ===
        self.se_block = SEBlock(
            num_features=total_feature_dim,
            reduction=self.config.get('se_reduction', 4)
        )
        
        # === 4. 特征交互模块 (DCN + Deep双塔) ===
        self.feature_interaction = FeatureInteractionModule(
            input_dim=total_feature_dim,
            cross_layers=self.config.get('cross_layers', 2),
            deep_dims=self.config.get('deep_dims', [64, 32]),
            dropout=self.config.get('dropout', 0.2)
        )
        interaction_output_dim = self.feature_interaction.output_dim
        
        # === 5. 物理引导分支 ===
        # 条件特征 = 交互后的特征
        self.physics_branch = PhysicsGuidedBranch(
            condition_dim=interaction_output_dim,
            hidden_dims=self.config.get('physics_hidden_dims', [32, 16]),
            dropout=self.config.get('dropout', 0.2),
            alpha_min=self.config.get('alpha_min', 0.1),
            alpha_max=self.config.get('alpha_max', 20.0)
        )
        
        # === 6. 残差修正分支 ===
        # 输入 = 交互特征 + 物理分支输出
        residual_input_dim = interaction_output_dim + 1
        self.residual_branch = ResidualCorrectionBranch(
            input_dim=residual_input_dim,
            hidden_dims=self.config.get('residual_hidden_dims', [64, 32]),
            dropout=self.config.get('dropout', 0.2)
        )
        
        # === 7. 自适应门控融合 ===
        # 上下文 = 交互特征
        self.adaptive_gate = AdaptiveGatingUnit(
            context_dim=interaction_output_dim,
            hidden_dim=self.config.get('gate_hidden_dim', 32)
        )
        
        # 保存维度信息
        self.total_feature_dim = total_feature_dim
        self.interaction_output_dim = interaction_output_dim
        
    def forward(self, batch):
        """
        前向传播
        
        Args:
            batch: 字典，包含以下键：
                - fcd_features: [B, 3] 浮动车特征
                - physics_features: [B, 2] 物理特征
                - time_features: [B, 5] 时间特征
                - road_type: [B] 道路类型索引
                - lane: [B] 车道数索引
                - time_period: [B] 时段索引
                - length: [B] 路段长度
                - theoretical_flow: [B] 理论流量
        
        Returns:
            output: 字典，包含：
                - prediction: [B] 最终预测流量
                - q_phy: [B] 物理分支输出
                - q_res: [B] 残差分支输出
                - alpha: [B] 物理系数
                - gate: [B] 门控系数
                - se_weights: [B, C] SE注意力权重
        """
        # === 提取输入 ===
        fcd_features = batch['fcd_features']      # [B, 3]
        physics_features = batch['physics_features']  # [B, 2]
        time_features = batch['time_features']    # [B, 5]
        road_type = batch['road_type']            # [B]
        lane = batch['lane']                      # [B]
        time_period = batch['time_period']        # [B]
        length = batch['length']                  # [B]
        theoretical_flow = batch['theoretical_flow']  # [B]
        
        # === 1. 语义嵌入 ===
        semantic_emb = self.semantic_emb(road_type, lane, time_period)  # [B, 40]
        
        # === 2. 特征拼接 ===
        combined_features = torch.cat([
            fcd_features,
            physics_features,
            time_features,
            semantic_emb,
            length.unsqueeze(-1)
        ], dim=-1)  # [B, total_feature_dim]
        
        # === 3. SE-Block 通道注意力 ===
        attended_features, se_weights = self.se_block(combined_features)
        
        # === 4. 特征交互 (DCN + Deep) ===
        interaction_output = self.feature_interaction(attended_features)
        
        # === 5. 物理分支 ===
        q_phy, alpha = self.physics_branch(theoretical_flow, interaction_output)
        
        # === 6. 残差分支 ===
        residual_input = torch.cat([
            interaction_output,
            q_phy.unsqueeze(-1)
        ], dim=-1)
        q_res = self.residual_branch(residual_input)
        
        # === 7. 自适应门控融合 ===
        prediction, gate = self.adaptive_gate(interaction_output, q_phy, q_res)
        
        return {
            'prediction': prediction,
            'q_phy': q_phy,
            'q_res': q_res,
            'alpha': alpha,
            'gate': gate,
            'se_weights': se_weights
        }


# =============================================================================
# 消融实验变体
# =============================================================================

class PIMNetV2Ablation(nn.Module):
    """
    PIM-Net v2 消融实验变体
    
    支持选择性关闭各模块以验证其有效性
    """
    
    def __init__(self, fcd_dim, physics_dim, time_dim, config=None,
                 use_se=True, use_dcn=True, use_gate=True, use_physics=True):
        """
        Args:
            use_se: 是否使用SE-Block
            use_dcn: 是否使用DCN（否则只用Deep）
            use_gate: 是否使用门控融合（否则用加法）
            use_physics: 是否使用物理分支
        """
        super().__init__()
        
        self.config = config or MODEL_CONFIG
        self.use_se = use_se
        self.use_dcn = use_dcn
        self.use_gate = use_gate
        self.use_physics = use_physics
        
        # 语义嵌入
        self.semantic_emb = SemanticEmbedding(self.config)
        semantic_dim = self.semantic_emb.output_dim
        
        total_feature_dim = fcd_dim + physics_dim + time_dim + semantic_dim + 1
        
        # SE-Block (可选)
        if use_se:
            self.se_block = SEBlock(total_feature_dim)
        
        # 特征交互
        if use_dcn:
            self.feature_interaction = FeatureInteractionModule(
                input_dim=total_feature_dim,
                cross_layers=2,
                deep_dims=[64, 32],
                dropout=self.config.get('dropout', 0.2)
            )
            interaction_output_dim = self.feature_interaction.output_dim
        else:
            # 只用Deep网络
            self.deep_only = DeepNetwork(total_feature_dim, [64, 32])
            interaction_output_dim = self.deep_only.output_dim
        
        # 物理分支 (可选)
        if use_physics:
            self.physics_branch = PhysicsGuidedBranch(
                condition_dim=interaction_output_dim,
                hidden_dims=[32, 16],
                dropout=self.config.get('dropout', 0.2)
            )
        
        # 残差分支
        residual_input_dim = interaction_output_dim + (1 if use_physics else 0)
        self.residual_branch = ResidualCorrectionBranch(
            input_dim=residual_input_dim,
            hidden_dims=[64, 32],
            dropout=self.config.get('dropout', 0.2)
        )
        
        # 门控融合 (可选)
        if use_gate and use_physics:
            self.adaptive_gate = AdaptiveGatingUnit(interaction_output_dim)
        
        # 如果不用物理分支，需要一个直接输出头
        if not use_physics:
            self.direct_head = nn.Linear(interaction_output_dim, 1)
        
        self.interaction_output_dim = interaction_output_dim
        
    def forward(self, batch):
        fcd_features = batch['fcd_features']
        physics_features = batch['physics_features']
        time_features = batch['time_features']
        road_type = batch['road_type']
        lane = batch['lane']
        time_period = batch['time_period']
        length = batch['length']
        theoretical_flow = batch['theoretical_flow']
        
        # 语义嵌入
        semantic_emb = self.semantic_emb(road_type, lane, time_period)
        
        # 特征拼接
        combined = torch.cat([
            fcd_features, physics_features, time_features,
            semantic_emb, length.unsqueeze(-1)
        ], dim=-1)
        
        # SE-Block
        se_weights = None
        if self.use_se:
            combined, se_weights = self.se_block(combined)
        
        # 特征交互
        if self.use_dcn:
            interaction_output = self.feature_interaction(combined)
        else:
            interaction_output = self.deep_only(combined)
        
        # 分支计算
        output = {'se_weights': se_weights}
        
        if self.use_physics:
            q_phy, alpha = self.physics_branch(theoretical_flow, interaction_output)
            output['q_phy'] = q_phy
            output['alpha'] = alpha
            
            residual_input = torch.cat([interaction_output, q_phy.unsqueeze(-1)], dim=-1)
            q_res = self.residual_branch(residual_input)
            output['q_res'] = q_res
            
            if self.use_gate:
                prediction, gate = self.adaptive_gate(interaction_output, q_phy, q_res)
                output['gate'] = gate
            else:
                prediction = q_phy + q_res
                output['gate'] = None
        else:
            q_res = self.residual_branch(interaction_output)
            prediction = q_res
            output['q_phy'] = None
            output['q_res'] = q_res
            output['alpha'] = None
            output['gate'] = None
        
        output['prediction'] = prediction
        return output


# =============================================================================
# 工厂函数
# =============================================================================

def build_model_v2(fcd_dim, physics_dim, time_dim, config=None):
    """
    构建PIM-Net v2模型
    
    Args:
        fcd_dim: 浮动车特征维度
        physics_dim: 物理特征维度
        time_dim: 时间特征维度
        config: 模型配置
    
    Returns:
        model: PIMNetV2实例
    """
    model = PIMNetV2(fcd_dim, physics_dim, time_dim, config)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "=" * 60)
    print("PIM-Net v2 模型构建完成")
    print("=" * 60)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  特征维度: {model.total_feature_dim}")
    print(f"  交互输出维度: {model.interaction_output_dim}")
    print("\n  模块配置:")
    print(f"    ✓ 语义嵌入 (Semantic Embedding)")
    print(f"    ✓ 通道注意力 (SE-Block)")
    print(f"    ✓ 深度交叉网络 (DCN)")
    print(f"    ✓ 物理引导分支 (Physics Branch)")
    print(f"    ✓ 残差修正分支 (Residual Branch)")
    print(f"    ✓ 自适应门控融合 (Adaptive Gating)")
    
    return model


if __name__ == "__main__":
    # 测试模型
    print("测试 PIM-Net v2...")
    
    batch_size = 32
    fcd_dim = 3
    physics_dim = 2
    time_dim = 5
    
    # 构建模型
    model = build_model_v2(fcd_dim, physics_dim, time_dim)
    
    # 模拟输入
    batch = {
        'fcd_features': torch.randn(batch_size, fcd_dim),
        'physics_features': torch.randn(batch_size, physics_dim),
        'time_features': torch.randn(batch_size, time_dim),
        'road_type': torch.randint(0, 5, (batch_size,)),
        'lane': torch.randint(0, 3, (batch_size,)),
        'time_period': torch.randint(0, 7, (batch_size,)),
        'length': torch.randn(batch_size),
        'theoretical_flow': torch.randn(batch_size),
    }
    
    # 前向传播
    output = model(batch)
    
    print("\n输出信息:")
    for key, value in output.items():
        if value is not None:
            print(f"  {key}: {value.shape}")
    
    print(f"\n门控系数范围: [{output['gate'].min():.3f}, {output['gate'].max():.3f}]")
    print(f"Alpha系数范围: [{output['alpha'].min():.3f}, {output['alpha'].max():.3f}]")
