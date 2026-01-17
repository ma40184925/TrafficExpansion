"""
PIM-Net: Physics-Informed Mapping Network
==========================================

基于物理感知的交通流量映射网络

核心架构:
- 物理引导分支 (Physics-Guided Branch): 基于交通流理论估计基础流量
- 残差修正分支 (Residual Correction Branch): 学习非线性偏差
- 语义嵌入模块 (Semantic Embedding): 捕捉道路类型/时段的隐式语义关系

关键改进:
1. Alpha系数范围约束 [alpha_min, alpha_max]，防止发散
2. 简化残差分支，减少过拟合风险
3. 输入特征标准化
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


class SemanticEmbedding(nn.Module):
    """
    语义嵌入模块
    将离散的道路属性映射到连续的语义空间
    """
    
    def __init__(self, config=None):
        super().__init__()
        config = config or MODEL_CONFIG
        
        # 道路类型嵌入
        self.road_type_emb = nn.Embedding(
            num_embeddings=config['num_road_types'],
            embedding_dim=config['road_type_emb_dim']
        )
        
        # 车道数嵌入
        self.lane_emb = nn.Embedding(
            num_embeddings=config['num_lanes'],
            embedding_dim=config['lane_emb_dim']
        )
        
        # 时段嵌入
        self.time_period_emb = nn.Embedding(
            num_embeddings=config['num_time_periods'],
            embedding_dim=config['time_period_emb_dim']
        )
        
        self.output_dim = (
            config['road_type_emb_dim'] + 
            config['lane_emb_dim'] + 
            config['time_period_emb_dim']
        )
    
    def forward(self, road_type, lane, time_period):
        """
        Args:
            road_type: [batch_size] 道路类型索引
            lane: [batch_size] 车道数索引
            time_period: [batch_size] 时段索引
        
        Returns:
            semantic_emb: [batch_size, output_dim] 语义嵌入向量
        """
        road_emb = self.road_type_emb(road_type)
        lane_emb = self.lane_emb(lane)
        time_emb = self.time_period_emb(time_period)
        
        return torch.cat([road_emb, lane_emb, time_emb], dim=-1)


class PhysicsGuidedBranch(nn.Module):
    """
    物理引导分支
    学习动态系数 α，将理论流量映射为基础估计值
    
    Q_base = α(conditions) × Q_theoretical + bias
    
    关键改进：使用Sigmoid将α约束在[alpha_min, alpha_max]范围内
    """
    
    def __init__(self, condition_dim, hidden_dims=None, dropout=0.1, 
                 alpha_min=0.1, alpha_max=20.0):
        """
        Args:
            condition_dim: 条件特征维度 (语义嵌入 + 时间特征 + 速度等)
            hidden_dims: MLP隐藏层维度列表
            dropout: Dropout比率
            alpha_min: α系数最小值
            alpha_max: α系数最大值
        """
        super().__init__()
        
        hidden_dims = hidden_dims or [32, 16]
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_range = alpha_max - alpha_min
        
        # 构建MLP用于估计α系数
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
        
        # 输出层
        layers.append(nn.Linear(in_dim, 1))
        
        self.alpha_net = nn.Sequential(*layers)
        
        # 可学习的偏置项
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, theoretical_flow, conditions):
        """
        Args:
            theoretical_flow: [batch_size] 理论流量 (已标准化)
            conditions: [batch_size, condition_dim] 条件特征
        
        Returns:
            base_estimate: [batch_size] 基础估计值
            alpha: [batch_size] 学习到的系数 (用于可解释性分析)
        """
        # 估计α系数
        alpha_raw = self.alpha_net(conditions)
        
        # 【关键改进】使用Sigmoid将α约束在[alpha_min, alpha_max]范围内
        alpha = torch.sigmoid(alpha_raw) * self.alpha_range + self.alpha_min
        alpha = alpha.squeeze(-1)
        
        # 基础估计 = α × 理论流量 + 偏置
        base_estimate = alpha * theoretical_flow + self.bias
        
        return base_estimate, alpha


class ResidualCorrectionBranch(nn.Module):
    """
    残差修正分支
    学习物理模型无法解释的非线性偏差
    
    简化版：减少层数以防止过拟合
    """
    
    def __init__(self, input_dim, hidden_dims=None, dropout=0.2):
        """
        Args:
            input_dim: 输入特征总维度
            hidden_dims: MLP隐藏层维度列表
            dropout: Dropout比率
        """
        super().__init__()
        
        hidden_dims = hidden_dims or [64, 32]
        
        # 构建MLP (简化为2层)
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
        
        # 输出层: 输出残差修正值
        layers.append(nn.Linear(in_dim, 1))
        
        self.residual_net = nn.Sequential(*layers)
        
    def forward(self, features):
        """
        Args:
            features: [batch_size, input_dim] 融合特征
        
        Returns:
            residual: [batch_size] 残差修正值
        """
        return self.residual_net(features).squeeze(-1)


class FeatureInteraction(nn.Module):
    """
    特征交互模块
    显式计算重要的二阶交互特征
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, fcd_flow, fcd_speed, fcd_status, length):
        """
        Args:
            fcd_flow: [batch_size] 浮动车流量 (已标准化)
            fcd_speed: [batch_size] 浮动车速度 (已标准化)
            fcd_status: [batch_size] 路况 (已标准化)
            length: [batch_size] 路段长度 (已标准化)
        
        Returns:
            interactions: [batch_size, num_interactions] 交互特征
        """
        # 流量-速度交互
        flow_speed = fcd_flow * fcd_speed
        
        # 流量-路况交互 (拥堵时的流量积压)
        flow_status = fcd_flow * fcd_status
        
        # 速度-路况交互
        speed_status = fcd_speed * fcd_status
        
        # 流量-长度交互
        flow_length = fcd_flow * length
        
        interactions = torch.stack([
            flow_speed,
            flow_status,
            speed_status,
            flow_length,
        ], dim=-1)
        
        return interactions


class PIMNet(nn.Module):
    """
    PIM-Net: Physics-Informed Mapping Network
    
    完整的双流架构模型
    """
    
    def __init__(self, fcd_dim, physics_dim, time_dim, config=None):
        """
        Args:
            fcd_dim: 浮动车特征维度
            physics_dim: 物理特征维度
            time_dim: 时间特征维度
            config: 模型配置
        """
        super().__init__()
        
        self.config = config or MODEL_CONFIG
        
        # === 模块初始化 ===
        
        # 1. 语义嵌入模块
        self.use_semantic_embedding = self.config['use_semantic_embedding']
        if self.use_semantic_embedding:
            self.semantic_emb = SemanticEmbedding(self.config)
            semantic_dim = self.semantic_emb.output_dim
        else:
            semantic_dim = (
                self.config['num_road_types'] + 
                self.config['num_lanes'] + 
                self.config['num_time_periods']
            )
        
        # 2. 特征交互模块
        self.feature_interaction = FeatureInteraction()
        interaction_dim = 4  # 4个交互特征
        
        # 3. 物理引导分支
        self.use_physics_branch = self.config['use_physics_branch']
        if self.use_physics_branch:
            # 条件特征: 语义嵌入 + 时间特征 + FCD特征
            physics_condition_dim = semantic_dim + time_dim + fcd_dim
            self.physics_branch = PhysicsGuidedBranch(
                condition_dim=physics_condition_dim,
                hidden_dims=self.config['physics_hidden_dims'],
                dropout=self.config['dropout'],
                alpha_min=self.config.get('alpha_min', 0.1),
                alpha_max=self.config.get('alpha_max', 20.0),
            )
        
        # 4. 残差修正分支
        self.use_residual = self.config['use_residual']
        # 残差分支输入: 所有特征的拼接
        residual_input_dim = (
            fcd_dim +           # 浮动车原始特征
            physics_dim +       # 物理特征
            time_dim +          # 时间特征
            semantic_dim +      # 语义嵌入
            interaction_dim +   # 交互特征
            1                   # 路段长度
        )
        
        if self.use_physics_branch:
            residual_input_dim += 1  # 加上物理分支的基础估计值
        
        self.residual_branch = ResidualCorrectionBranch(
            input_dim=residual_input_dim,
            hidden_dims=self.config['residual_hidden_dims'],
            dropout=self.config['dropout']
        )
        
        # 5. 如果不使用物理分支，需要一个直接预测头
        if not self.use_physics_branch:
            self.direct_head = nn.Linear(residual_input_dim, 1)
        
    def forward(self, batch):
        """
        前向传播
        
        Args:
            batch: 包含各类特征的字典
        
        Returns:
            output: dict containing
                - prediction: [batch_size] 预测流量
                - alpha: [batch_size] 物理系数 (如果使用物理分支)
                - base: [batch_size] 基础估计 (如果使用物理分支)
                - residual: [batch_size] 残差修正值
        """
        # === 提取输入 ===
        fcd_features = batch['fcd_features']  # [B, fcd_dim]
        physics_features = batch['physics_features']  # [B, physics_dim]
        time_features = batch['time_features']  # [B, time_dim]
        road_type = batch['road_type']  # [B]
        lane = batch['lane']  # [B]
        time_period = batch['time_period']  # [B]
        length = batch['length']  # [B]
        theoretical_flow = batch['theoretical_flow']  # [B] (已标准化)
        
        # 提取浮动车特征中的各分量 (用于交互)
        fcd_flow = fcd_features[:, 0]
        fcd_speed = fcd_features[:, 1]
        fcd_status = fcd_features[:, 2]
        
        # === 语义嵌入 ===
        if self.use_semantic_embedding:
            semantic_emb = self.semantic_emb(road_type, lane, time_period)
        else:
            # 使用One-hot编码
            road_onehot = F.one_hot(road_type, self.config['num_road_types']).float()
            lane_onehot = F.one_hot(lane, self.config['num_lanes']).float()
            time_onehot = F.one_hot(time_period, self.config['num_time_periods']).float()
            semantic_emb = torch.cat([road_onehot, lane_onehot, time_onehot], dim=-1)
        
        # === 特征交互 ===
        interaction_features = self.feature_interaction(
            fcd_flow, fcd_speed, fcd_status, length
        )
        
        # === 物理引导分支 ===
        output = {}
        
        if self.use_physics_branch:
            # 构建条件特征
            physics_conditions = torch.cat([
                semantic_emb,
                time_features,
                fcd_features,  # 使用完整的FCD特征
            ], dim=-1)
            
            # 物理分支前向
            base_estimate, alpha = self.physics_branch(
                theoretical_flow, 
                physics_conditions
            )
            
            output['base'] = base_estimate
            output['alpha'] = alpha
        
        # === 残差修正分支 ===
        # 拼接所有特征
        residual_input = [
            fcd_features,
            physics_features,
            time_features,
            semantic_emb,
            interaction_features,
            length.unsqueeze(-1),
        ]
        
        if self.use_physics_branch:
            residual_input.append(base_estimate.unsqueeze(-1))
        
        residual_features = torch.cat(residual_input, dim=-1)
        
        # 残差分支前向
        residual = self.residual_branch(residual_features)
        output['residual'] = residual
        
        # === 最终预测 ===
        if self.use_physics_branch:
            if self.use_residual:
                prediction = base_estimate + residual
            else:
                prediction = base_estimate
        else:
            prediction = self.direct_head(residual_features).squeeze(-1)
        
        output['prediction'] = prediction
        
        return output


class PIMNetLite(nn.Module):
    """
    PIM-Net 轻量版
    用于消融实验，可选择性关闭各模块
    """
    
    def __init__(self, fcd_dim, physics_dim, time_dim, 
                 use_physics=True, use_embedding=True, use_residual=True):
        super().__init__()
        
        config = MODEL_CONFIG.copy()
        config['use_physics_branch'] = use_physics
        config['use_semantic_embedding'] = use_embedding
        config['use_residual'] = use_residual
        
        self.model = PIMNet(fcd_dim, physics_dim, time_dim, config)
    
    def forward(self, batch):
        return self.model(batch)


def build_model(fcd_dim, physics_dim, time_dim, config=None):
    """
    构建模型的工厂函数
    
    Args:
        fcd_dim: 浮动车特征维度
        physics_dim: 物理特征维度
        time_dim: 时间特征维度
        config: 模型配置
    
    Returns:
        model: PIMNet模型实例
    """
    model = PIMNet(fcd_dim, physics_dim, time_dim, config)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型构建完成:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  物理分支: {'启用' if model.use_physics_branch else '禁用'}")
    print(f"  语义嵌入: {'启用' if model.use_semantic_embedding else '禁用'}")
    print(f"  残差连接: {'启用' if model.use_residual else '禁用'}")
    if model.use_physics_branch:
        alpha_min = model.config.get('alpha_min', 0.1)
        alpha_max = model.config.get('alpha_max', 20.0)
        print(f"  Alpha范围: [{alpha_min}, {alpha_max}]")
    
    return model


if __name__ == "__main__":
    # 测试模型
    batch_size = 32
    fcd_dim = 3
    physics_dim = 3
    time_dim = 5
    
    # 构建模型
    model = build_model(fcd_dim, physics_dim, time_dim)
    
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
        'target': torch.rand(batch_size) * 1000,
    }
    
    # 前向传播
    output = model(batch)
    
    print("\n输出信息:")
    for key, value in output.items():
        print(f"  {key}: {value.shape}")
    
    # 验证Alpha范围
    print(f"\nAlpha范围检查: [{output['alpha'].min():.3f}, {output['alpha'].max():.3f}]")
