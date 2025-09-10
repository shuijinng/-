# src/models/train.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

def train_model(X_train: pd.DataFrame, y_train: pd.Series,
               model_type: str = "XGBoost") -> object:
    """
    训练指定类型的模型。

    Args:
        X_train (pd.DataFrame): 训练特征。
        y_train (pd.Series): 训练标签。
        model_type (str): 模型类型 ("Linear", "RandomForest", "GradientBoosting", "XGBoost")。

    Returns:
        object: 训练好的模型。
    """
    logger.info(f"开始训练 {model_type} 模型...")

    if model_type == "Linear":
        model = LinearRegression()
    elif model_type == "RandomForest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "GradientBoosting":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_type == "XGBoost":
        model = xgb.XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    model.fit(X_train, y_train)
    logger.info(f"{model_type} 模型训练完成。")
    return model


def evaluate_model(model: object, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """
    在验证集上评估模型。

    Args:
        model (object): 训练好的模型。
        X_val (pd.DataFrame): 验证集特征。
        y_val (pd.Series): 验证集标签。

    Returns:
        dict: 包含评估指标的字典。
    """
    y_pred = model.predict(X_val)

    # 计算各个指标
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)  # <-- 添加这一行！计算 MAE
    mape = (abs(y_val - y_pred) / y_val).mean() * 100  # 注意：避免 y_val 为0

    logger.info(f"验证集评估结果:")
    logger.info(f"  MAE: {mae:.4f}")  # 现在 'mae' 已定义
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  MAPE: {mape:.4f}%")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}  # 现在 'mae' 已定义

def save_model(model: object, model_path: str):
    """保存训练好的模型。"""
    try:
        # 确保模型目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"模型已保存至: {model_path}")
    except Exception as e:
        logger.error(f"保存模型时出错: {e}")
        raise