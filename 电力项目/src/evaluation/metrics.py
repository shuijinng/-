# src/evaluation/metrics.py
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def calculate_mape(y_true, y_pred):
    """计算平均绝对百分比误差 (MAPE)。"""
    # 处理 y_true 为 0 的情况
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if not non_zero_mask.any():
        return np.inf # 或者返回一个很大的数
    return (np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])).mean() * 100