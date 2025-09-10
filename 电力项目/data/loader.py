# src/data/loader.py
import pandas as pd
from typing import Tuple
import logging

# 配置日志
logger = logging.getLogger(__name__)

def load_data(train_path: str, test_path: str, time_col: str = 'time') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载训练集和测试集数据，并解析时间列。

    Args:
        train_path (str): 训练集文件路径。
        test_path (str): 测试集文件路径。
        time_col (str): 时间列的列名，默认为 'time'。

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 训练集和测试集 DataFrame（time 已解析为 datetime）。
    """
    try:
        # 显式指定 date_parser 来解析时间
        parse_dates = [time_col]  # 告诉 pandas 把这一列当作日期解析

        train_df = pd.read_csv(train_path, parse_dates=parse_dates)
        test_df = pd.read_csv(test_path, parse_dates=parse_dates)

        # ✅ 确保 time 列是 datetime 类型（以防 parse_dates 失败）
        if not pd.api.types.is_datetime64_any_dtype(train_df[time_col]):
            train_df[time_col] = pd.to_datetime(train_df[time_col])
        if not pd.api.types.is_datetime64_any_dtype(test_df[time_col]):
            test_df[time_col] = pd.to_datetime(test_df[time_col])

        logger.info(f"数据加载成功！训练集: {train_df.shape}, 测试集: {test_df.shape}")
        logger.info(f"时间列 '{time_col}' 已解析为 datetime 类型。")
        logger.info(f"训练集时间范围: {train_df[time_col].min()} 到 {train_df[time_col].max()}")
        logger.info(f"测试集时间范围: {test_df[time_col].min()} 到 {test_df[time_col].max()}")

        return train_df, test_df

    except FileNotFoundError as e:
        logger.error(f"文件未找到: {e}")
        raise
    except Exception as e:
        logger.error(f"加载数据时发生错误: {e}")
        raise