# train_fasttext.py - 带进度条 + 文件验证 + 模型保存
import os
from tqdm import tqdm
import fasttext

# ==================== 路径配置 ====================
# 你指定的训练数据文件路径
TEMP_FILE = r"D:\111huiyu\train_fasttext.txt"

# 模型保存路径
MODEL_DIR = r"D:\111huiyu\model"
MODEL_PATH = os.path.join(MODEL_DIR, "fasttext_model.bin")

# 创建目录（如果不存在）
os.makedirs(MODEL_DIR, exist_ok=True)

# ==================== 检查训练文件 ====================
print(f"🔍 正在检查训练文件: {TEMP_FILE}")

# 检查文件是否存在
if not os.path.exists(TEMP_FILE):
    raise FileNotFoundError(f"❌ 文件不存在！请确认路径是否正确。\n👉 {TEMP_FILE}")

# 检查文件是否为空
file_size = os.path.getsize(TEMP_FILE)
if file_size == 0:
    raise ValueError(f"❌ 训练文件为空！请重新生成数据。\n👉 {TEMP_FILE}")

print(f"✅ 文件存在，大小: {file_size / 1024:.2f} KB")

# 读取前3行做格式检查
with open(TEMP_FILE, 'r', encoding='utf-8') as f:
    sample_lines = [next(f).strip() for _ in range(3)]

print("📄 文件前3行示例:")
for i, line in enumerate(sample_lines, 1):
    print(f"   {i}: {line}")

# 简单验证格式
for line in sample_lines:
    if not line.startswith("__label__"):
        print("⚠️  警告：行不以 '__label__' 开头，fastText 可能无法识别！")
        break

# ==================== 开始训练（带进度条）====================
print("\n🚀 开始训练 fastText 模型...")

try:
    model = fasttext.train_supervised(
        input=TEMP_FILE,
        epoch=25,           # 训练轮数
        lr=1.0,             # 学习率
        wordNgrams=2,       # 二元语法，提升效果
        dim=100,            # 词向量维度
        minCount=1,         # 最小词频
        verbose=2,          # 显示详细日志（相当于进度）
        loss="softmax"      # 损失函数
    )

    # 保存模型
    model.save_model(MODEL_PATH)
    print(f"\n✅ 模型训练完成！已保存到：\n👉 {MODEL_PATH}")

    # 可选：打印一些模型信息
    print(f"📊 模型维度: {model.f.dim}")
    print(f"📚 词汇量: {model.f.words}")
    print(f"🏷️  标签数: {model.f.labels}")

except Exception as e:
    print(f"❌ 训练失败！错误信息：{e}")
    raise

# ==================== 简单测试 ====================
print("\n🧪 模型测试：")
test_texts = [
    "中华女子学院 招收 男生",
    "手机 钱包 科技 创新",
    "李永波 谢杏芳 林丹 羽毛球"
]

for text in test_texts:
    label_pred, prob = model.predict(text.replace(" ", ""))  # 原始模型接受未分词文本（但你训练时是分词的）
    print(f"  '{text}' → 预测: {label_pred[0]} (置信度: {prob[0]:.4f})")