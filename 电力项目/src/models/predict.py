# src/models/predict.py

import joblib
import pandas as pd
import logging
import numpy as np
logger = logging.getLogger(__name__)

def load_model(model_path: str):
    """加载训练好的模型。"""
    try:
        model = joblib.load(model_path)
        logger.info(f"模型已从 {model_path} 加载。")
        return model
    except Exception as e:
        logger.error(f"加载模型时出错: {e}")
        raise

# src/models/predict.py

def make_predictions(model, X_test: pd.DataFrame) -> np.ndarray:  # ← 修改返回类型
    """
    使用模型进行预测。

    Returns:
        np.ndarray: 预测结果数组。
    """
    logger.info("开始进行预测...")
    try:
        # 1. 执行模型预测
        raw_predictions = model.predict(X_test)  # shape: (768,)
        logger.info(f"模型预测完成。输出形状: {raw_predictions.shape}")

        # 2. 调试信息
        logger.info(f"X_test index type: {type(X_test.index)}")
        logger.info(f"X_test index (前5个): {X_test.index[:5]}")
        logger.info(f"X_test shape inside function: {X_test.shape}")

        # 3. ✅ 直接返回 numpy array，不要包装成 Series
        return raw_predictions  # ← 只返回数组

    except Exception as e:
        logger.error(f"在 make_predictions 函数中发生未预期的错误: {type(e).__name__}: {e}")
        logger.error(f"X_test 的形状: {X_test.shape}")
        logger.error(f"X_test 的列: {list(X_test.columns)}")
        raise