# src/utils.py
import numpy as np
import torch
import random


def set_seed(seed=42):
    """
    设置随机种子，保证结果可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_metrics(y_true, y_pred):
    """
    计算评估指标

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        rmse: 均方根误差
        mae: 平均绝对误差
        mape: 平均绝对百分比误差
        r2: R²决定系数
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    # 避免除零
    y_true_nonzero = y_true.copy()
    y_true_nonzero[y_true_nonzero == 0] = 1e-6
    mape = np.mean(np.abs((y_true - y_pred) / y_true_nonzero)) * 100

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }


def save_metrics(metrics, filepath):
    """保存评估指标到文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("========== Model Metrics ==========\n")
        f.write(f"RMSE: {metrics['rmse']:.6f} million km2\n")
        f.write(f"MAE: {metrics['mae']:.6f} million km2\n")
        f.write(f"MAPE: {metrics['mape']:.2f}%\n")
        f.write(f"R2: {metrics['r2']:.6f}\n")

        if 'monthly_rmse' in metrics:
            f.write("\n========== Monthly RMSE ==========\n")
            for m, rmse in enumerate(metrics['monthly_rmse'], 1):
                f.write(f"Month {m}: {rmse:.6f} million km2\n")

    print(f"Metrics saved to: {filepath}")