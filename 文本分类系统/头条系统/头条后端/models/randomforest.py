# train_random_forest.py
import pandas as pd
import jieba
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# =================== 配置路径 ===================
# 推荐：将数据复制到无中文路径，避免兼容性问题
DATA_PATH = r"D:\111huiyu\慧与\题\作业集合\8月25日-吕雨阳\01.源代码\头条后端\data\train_new.csv"
STOPWORDS_PATH = r"D:\111huiyu\慧与\题\作业集合\8月25日-吕雨阳\01.源代码\头条后端\data\stopwords.txt"
MODEL_SAVE_PATH = r"D:\111huiyu\慧与\题\作业集合\8月25日-吕雨阳\01.源代码\头条后端\model\rf_model.pkl"
VECTORIZER_SAVE_PATH = r"D:\111huiyu\慧与\题\作业集合\8月25日-吕雨阳\01.源代码\头条后端\model\tfidf_vectorizer.pkl"

# =================== 加载停用词 ===================
def load_stopwords(filepath):
    if not os.path.exists(filepath):
        print(f"⚠️ 停用词文件未找到: {filepath}")
        return set()
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

stopwords = load_stopwords(STOPWORDS_PATH)
print(f"✅ 加载停用词: {len(stopwords)} 个")

# =================== 中文文本预处理 ===================
def preprocess(text):
    """分词 + 去停用词 + 过滤短词"""
    words = jieba.lcut(str(text))
    return ' '.join(w for w in words if w not in stopwords and len(w) > 1 and w.isalnum())

# =================== 加载数据 ===================
print("📊 正在加载数据...")
df = pd.read_csv(DATA_PATH)

# 检查必要列
if 'sentence' not in df.columns or 'label' not in df.columns:
    raise ValueError("❌ 数据文件必须包含 'sentence' 和 'label' 列")

# 预处理
print("🧹 正在预处理文本...")
df['text_clean'] = df['sentence'].apply(preprocess)

# 过滤空文本
before_count = len(df)
df = df[df['text_clean'].str.strip() != '']
after_count = len(df)
print(f"📉 过滤空文本: {before_count} → {after_count}")

# =================== 划分训练集（可选：小样本调试） ===================
# df_sample = df.sample(n=50000, random_state=42)  # 调试用
# X = df_sample['text_clean']
# y = df_sample['label']

X = df['text_clean']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ 训练集: {len(X_train)}, 测试集: {len(X_test)}, 类别数: {y.nunique()}")

# =================== 构建 Pipeline（推荐做法） ===================
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=50000,      # 控制特征维度
        ngram_range=(1, 2),      # 使用 unigram + bigram
        sublinear_tf=True,       # 使用 sublinear scaling: 1 + log(tf)
        max_df=0.95,             # 忽略出现频率过高（>95%）的词
        min_df=3                 # 忽略出现次数太少（<3）的词
    )),
    ('rf', RandomForestClassifier(
        n_estimators=200,        # 树的数量
        max_depth=20,            # 控制过拟合
        min_samples_split=5,     # 内部节点再划分所需最小样本数
        min_samples_leaf=2,      # 叶子节点最少样本数
        random_state=42,
        n_jobs=-1,               # 使用所有CPU核心
        verbose=1                # 显示训练进度
    ))
])

# =================== 训练模型 ===================
print("🚀 开始训练随机森林...")
pipeline.fit(X_train, y_train)

# =================== 评估模型 ===================
print("🔍 模型评估...")
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ 准确率: {acc:.4f}")
print("\n📋 分类报告:")
print(classification_report(y_test, y_pred))

# =================== 保存模型 ===================
print(f"💾 保存模型到: {MODEL_SAVE_PATH}")
joblib.dump(pipeline, MODEL_SAVE_PATH)

# 也可以单独保存 TF-IDF 向量化器（可选）
joblib.dump(pipeline.named_steps['tfidf'], VECTORIZER_SAVE_PATH)
print("🎉 模型保存成功！")

# =================== 使用示例（可选） ===================
"""
# 加载模型
pipeline = joblib.load(MODEL_SAVE_PATH)
pred = pipeline.predict(["这是一个关于科技的新闻"])
print(pred)
"""