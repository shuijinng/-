# requirements.txt
import os

# 项目根目录 (假设 main.py 所在的目录就是根目录)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # 这会指向 main.py 所在的目录

# 数据路径
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# 确保这些路径是正确的，请检查文件是否真的存在
TRAIN_FILE = os.path.join(RAW_DATA_DIR, "train.csv") # 应该指向 D:\111huiyu\慧与\课上代码\电力项目\data\raw\train.csv
TEST_FILE = os.path.join(RAW_DATA_DIR, "test.csv")

# 模型路径
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_FILE = os.path.join(MODEL_DIR, "best_model.pkl")

# 特征和目标列名 (需要根据你的数据实际情况修改)
# 请打开 train.csv 文件，查看第一行，确认时间列和负荷列的准确列名
TIME_COL = "time"  # <-- 修改为你的数据中时间列的实际名称
TARGET_COL = "power_load" # <-- 修改为你的数据中负荷列的实际名称

# 其他配置
RANDOM_STATE = 42
TEST_SIZE = 0.2