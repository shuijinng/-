# main.py
import logging

import pandas as pd

from config import TRAIN_FILE, TEST_FILE, MODEL_FILE, TIME_COL, TARGET_COL
from 慧与.课上代码.电力项目.data.loader import load_data
from 慧与.课上代码.电力项目.data.processor import preprocess_data
from src.models.train import train_model, evaluate_model, save_model
from src.models.predict import make_predictions
# 可选导入
from src.visualization.plotter import plot_time_series, plot_predictions
from sklearn.model_selection import train_test_split # <-- 添加这一行！


# 配置根日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=== 电力负荷预测项目启动 ===")

    # 1. 加载数据
    train_df, test_df = load_data(TRAIN_FILE, TEST_FILE)

    # 2. 数据预处理 (特征工程)
    processed_train_df, processed_test_df = preprocess_data(train_df, test_df, TIME_COL, TARGET_COL)

    # 3. 准备训练和验证数据
    # 假设目标列是 'load'，特征是除目标列和时间列外的所有列
    feature_cols = [col for col in processed_train_df.columns if col not in [TIME_COL, TARGET_COL]]
    X = processed_train_df[feature_cols]
    y = processed_train_df[TARGET_COL]

    # 划分训练集和验证集 (注意：时间序列数据最好用时间划分，这里简化)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False) # shuffle=False 保持时间顺序

    # 4. 训练模型
    model = train_model(X_train, y_train, model_type="XGBoost") # 可以尝试其他模型

    # 5. 验证模型
    val_metrics = evaluate_model(model, X_val, y_val)

    # 6. 保存模型
    save_model(model, MODEL_FILE)

    # main.py

    # 7. 在测试集上进行预测
    X_test = processed_test_df[feature_cols]
    predictions_array = make_predictions(model, X_test)  # ← 接收 numpy array

    # ✅ 关键：用 test_df 的 time 列创建带时间索引的 Series
    predictions = pd.Series(predictions_array, index=test_df[TIME_COL], name="predicted_load")

    # 输出验证
    print("预测结果 (前5个):")
    print(predictions.head())

    # 8. 可视化
    if TARGET_COL in test_df.columns:
        y_test_true = test_df.set_index(TIME_COL)[TARGET_COL]  # 真实值，datetime 索引
        plot_predictions(y_test_true, predictions)  # 传入带时间索引的 Series

    return predictions

if __name__ == "__main__":
    predictions = main()
    print("预测结果 (前5个):")
    print(predictions.head())

