"""
基线模型对比实验
================

包含以下基线模型:
1. 线性回归 (Linear Regression)
2. 纯物理公式 (Pure Physics: Q = α × Q_theo)
3. XGBoost
4. 随机森林 (Random Forest)
5. MLP (多层感知机)

用法:
    python baselines.py
    python baselines.py --models lr xgb mlp
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 尝试导入XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("警告: XGBoost未安装，将跳过XGBoost基线")

# 路径设置
current_file = Path(__file__).resolve()
src_dir = current_file.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from config import DATA_CONFIG
from models.data_loader import DataProcessor
from utils.path_manager import pm


def compute_metrics(y_true, y_pred):
    """计算评估指标"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE
    mask = y_true > 1
    if mask.sum() > 0:
        mape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100
    else:
        mape = np.nan
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
    }


def prepare_sklearn_data(data_path):
    """为sklearn模型准备数据"""
    print(f"加载数据: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    
    # 预处理
    df['kind_x'] = df['kind_x'].astype(str).str.zfill(2)
    df = df[df['flow_std'] > 0].copy()
    df = df[df['fcd_flow'] > 0].copy()
    
    # 特征列
    feature_cols = [
        # 浮动车特征
        'fcd_flow', 'fcd_speed', 'fcd_status',
        # 物理特征
        'theoretical_flow', 'density_proxy', 'fcd_flow_per_length',
        # 时间特征
        'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'is_weekend',
        # 路段属性
        'length',
    ]
    
    # One-hot编码的类别特征
    kind_cols = [c for c in df.columns if c.startswith('kind_')]
    lane_cols = ['lane_1', 'lane_2_3', 'lane_4_plus']
    period_cols = [c for c in df.columns if c.startswith('period_')]
    
    all_feature_cols = feature_cols + kind_cols + lane_cols + period_cols
    
    # 确保所有列存在
    available_cols = [c for c in all_feature_cols if c in df.columns]
    print(f"使用特征: {len(available_cols)} 个")
    
    X = df[available_cols].values
    y = df['flow_std'].values
    
    # 处理NaN
    X = np.nan_to_num(X, nan=0)
    
    # 按卡口划分数据集
    checkpoints = df['卡口编号'].unique()
    np.random.seed(42)
    np.random.shuffle(checkpoints)
    
    n_train = int(len(checkpoints) * 0.7)
    n_val = int(len(checkpoints) * 0.15)
    
    train_ckpts = set(checkpoints[:n_train])
    val_ckpts = set(checkpoints[n_train:n_train+n_val])
    test_ckpts = set(checkpoints[n_train+n_val:])
    
    train_mask = df['卡口编号'].isin(train_ckpts).values
    val_mask = df['卡口编号'].isin(val_ckpts).values
    test_mask = df['卡口编号'].isin(test_ckpts).values
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # 同时返回理论流量用于纯物理基线
    theo_idx = available_cols.index('theoretical_flow')
    theo_train = X_train[:, theo_idx]
    theo_test = X_test[:, theo_idx]
    
    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'theo_train': theo_train, 'theo_test': theo_test,
        'feature_cols': available_cols,
    }


class PurePhysicsBaseline:
    """纯物理公式基线: Q = α × Q_theoretical"""
    
    def __init__(self):
        self.alpha = 1.0
    
    def fit(self, X_theo, y):
        # 最小二乘估计 α
        mask = X_theo > 0
        if mask.sum() > 0:
            self.alpha = np.sum(y[mask] * X_theo[mask]) / np.sum(X_theo[mask] ** 2)
        return self
    
    def predict(self, X_theo):
        return self.alpha * X_theo
    
    def get_params(self):
        return {'alpha': self.alpha}


def run_baselines(data_path, models_to_run=None):
    """运行基线模型对比"""
    
    # 准备数据
    data = prepare_sklearn_data(data_path)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    theo_train, theo_test = data['theo_train'], data['theo_test']
    
    # 默认运行所有模型
    if models_to_run is None:
        models_to_run = ['lr', 'physics', 'rf', 'mlp']
        if HAS_XGB:
            models_to_run.append('xgb')
    
    results = {}
    
    # === 1. 线性回归 ===
    if 'lr' in models_to_run:
        print("\n[1] 线性回归 (Linear Regression)")
        print("-" * 40)
        start = time.time()
        
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)  # 确保非负
        
        metrics = compute_metrics(y_test, y_pred)
        metrics['Time'] = time.time() - start
        results['Linear Regression'] = metrics
        
        print(f"  MAE: {metrics['MAE']:.2f}")
        print(f"  RMSE: {metrics['RMSE']:.2f}")
        print(f"  R²: {metrics['R2']:.4f}")
        print(f"  训练时间: {metrics['Time']:.2f}s")
    
    # === 2. 纯物理公式 ===
    if 'physics' in models_to_run:
        print("\n[2] 纯物理公式 (Q = α × Q_theo)")
        print("-" * 40)
        start = time.time()
        
        model = PurePhysicsBaseline()
        model.fit(theo_train, y_train)
        y_pred = model.predict(theo_test)
        y_pred = np.maximum(y_pred, 0)
        
        metrics = compute_metrics(y_test, y_pred)
        metrics['Time'] = time.time() - start
        metrics['alpha'] = model.alpha
        results['Pure Physics'] = metrics
        
        print(f"  学习到的α: {model.alpha:.4f}")
        print(f"  MAE: {metrics['MAE']:.2f}")
        print(f"  RMSE: {metrics['RMSE']:.2f}")
        print(f"  R²: {metrics['R2']:.4f}")
    
    # === 3. 随机森林 ===
    if 'rf' in models_to_run:
        print("\n[3] 随机森林 (Random Forest)")
        print("-" * 40)
        start = time.time()
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = compute_metrics(y_test, y_pred)
        metrics['Time'] = time.time() - start
        results['Random Forest'] = metrics
        
        print(f"  MAE: {metrics['MAE']:.2f}")
        print(f"  RMSE: {metrics['RMSE']:.2f}")
        print(f"  R²: {metrics['R2']:.4f}")
        print(f"  训练时间: {metrics['Time']:.2f}s")
        
        # 特征重要性
        importances = model.feature_importances_
        top_k = 5
        top_idx = np.argsort(importances)[-top_k:][::-1]
        print(f"  Top-{top_k} 特征重要性:")
        for i in top_idx:
            print(f"    {data['feature_cols'][i]}: {importances[i]:.4f}")
    
    # === 4. XGBoost ===
    if 'xgb' in models_to_run and HAS_XGB:
        print("\n[4] XGBoost")
        print("-" * 40)
        start = time.time()
        
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)],
                  verbose=False)
        y_pred = model.predict(X_test)
        
        metrics = compute_metrics(y_test, y_pred)
        metrics['Time'] = time.time() - start
        results['XGBoost'] = metrics
        
        print(f"  MAE: {metrics['MAE']:.2f}")
        print(f"  RMSE: {metrics['RMSE']:.2f}")
        print(f"  R²: {metrics['R2']:.4f}")
        print(f"  训练时间: {metrics['Time']:.2f}s")
        
        # 特征重要性
        importances = model.feature_importances_
        top_k = 5
        top_idx = np.argsort(importances)[-top_k:][::-1]
        print(f"  Top-{top_k} 特征重要性:")
        for i in top_idx:
            print(f"    {data['feature_cols'][i]}: {importances[i]:.4f}")
    
    # === 5. MLP ===
    if 'mlp' in models_to_run:
        print("\n[5] MLP (多层感知机)")
        print("-" * 40)
        start = time.time()
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=False
        )
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred = np.maximum(y_pred, 0)
        
        metrics = compute_metrics(y_test, y_pred)
        metrics['Time'] = time.time() - start
        results['MLP'] = metrics
        
        print(f"  MAE: {metrics['MAE']:.2f}")
        print(f"  RMSE: {metrics['RMSE']:.2f}")
        print(f"  R²: {metrics['R2']:.4f}")
        print(f"  训练时间: {metrics['Time']:.2f}s")
    
    # === 汇总 ===
    print("\n" + "=" * 60)
    print("基线模型对比汇总")
    print("=" * 60)
    print(f"{'模型':<20} {'MAE':>10} {'RMSE':>10} {'MAPE':>10} {'R²':>10}")
    print("-" * 60)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['MAE']:>10.2f} {metrics['RMSE']:>10.2f} "
              f"{metrics['MAPE']:>10.2f}% {metrics['R2']:>10.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Baseline Models Comparison")
    parser.add_argument('--data_path', type=str, default=None, help='数据路径')
    parser.add_argument('--models', nargs='+', default=None,
                        choices=['lr', 'physics', 'rf', 'xgb', 'mlp'],
                        help='要运行的模型')
    args = parser.parse_args()
    
    # 数据路径
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = pm.get_processed_path(DATA_CONFIG['input_file'])
    
    print("=" * 60)
    print("基线模型对比实验")
    print("=" * 60)
    
    run_baselines(data_path, models_to_run=args.models)


if __name__ == "__main__":
    main()
