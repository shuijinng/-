# src/data/processor.py
import pandas as pd
from typing import Tuple
import logging
from datetime import datetime
import holidays # 可能需要安装: pip install holidays

logger = logging.getLogger(__name__)

def _extract_datetime_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """从时间戳列提取日期时间特征。"""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df['hour'] = df[time_col].dt.hour
    df['day_of_week'] = df[time_col].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_month'] = df[time_col].dt.day
    df['month'] = df[time_col].dt.month
    df['year'] = df[time_col].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int) # 5=Saturday, 6=Sunday
    # df['is_holiday'] = df[time_col].apply(lambda x: int(x in holidays.US())) # 示例，需根据国家调整
    return df

def _create_lag_features(df: pd.DataFrame, target_col: str, lags: list) -> pd.DataFrame:
    """创建滞后特征。"""
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

def _create_rolling_features(df: pd.DataFrame, target_col: str, windows: list) -> pd.DataFrame:
    """创建滑动窗口统计特征。"""
    df = df.copy()
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
    return df

def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   time_col: str, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对训练集和测试集进行预处理（特征工程）。

    Args:
        train_df (pd.DataFrame): 原始训练集。
        test_df (pd.DataFrame): 原始测试集。
        time_col (str): 时间列名。
        target_col (str): 目标列名（负荷）。

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 处理后的训练集和测试集。
    """
    logger.info("开始数据预处理（特征工程）...")

    # 合并数据以便进行一致的特征工程 (注意：测试集不应包含未来信息)
    # 这里假设测试集是紧接着训练集之后的时间段
    # 为了创建滞后和滑动特征，我们需要将训练集的最后部分与测试集拼接
    # 但最终只保留测试集部分用于预测
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # 1. 提取日期时间特征
    combined_df = _extract_datetime_features(combined_df, time_col)

    # 2. 创建滞后特征 (例如，前1, 6, 24小时的负荷)
    lags = [1, 6, 24]
    combined_df = _create_lag_features(combined_df, target_col, lags)

    # 3. 创建滑动窗口特征 (例如，过去3, 12, 24小时的均值和标准差)
    windows = [3, 12, 24]
    combined_df = _create_rolling_features(combined_df, target_col, windows)

    # 4. 处理缺失值 (滞后和滑动特征会产生NaN)
    # 对于训练集，可以删除含有NaN的行
    # 对于测试集，如果特征依赖于训练集的最后值，需要特殊处理或填充
    # 这里简化处理：删除含有NaN的行 (在训练前)
    # 注意：测试集的预测不能依赖未来的值，所以确保特征工程逻辑正确
    # 更严谨的做法是在训练集上训练特征生成器，并应用于测试集

    # 分离处理后的数据
    processed_train_df = combined_df.iloc[:len(train_df)].copy()
    processed_test_df = combined_df.iloc[len(train_df):].copy()

    # 删除训练集中因特征工程产生的NaN行
    initial_train_len = len(processed_train_df)
    processed_train_df.dropna(inplace=True)
    logger.info(f"训练集处理后，删除了 {initial_train_len - len(processed_train_df)} 行缺失值。")

    # 对于测试集，如果仍有缺失（例如，第一个样本），可以考虑用训练集的最后一个有效值填充
    # 或者，如果模型允许，保留并让模型处理（如XGBoost可以处理缺失值）
    # 这里选择用训练集最后一个有效值填充测试集的初始缺失
    if processed_test_df.isnull().any().any():
        last_valid_values = processed_train_df.iloc[-1] # 获取训练集最后一行
        processed_test_df.fillna(last_valid_values, inplace=True)
        logger.warning("测试集存在缺失值，已用训练集最后一行的值填充。")

    logger.info(f"数据预处理完成！处理后训练集: {processed_train_df.shape}, 处理后测试集: {processed_test_df.shape}")
    return processed_train_df, processed_test_df