# train_bert_fast.py - 使用本地 BERT 模型快速训练（4060 显卡优化版）
import os
import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
from torch.utils.data import Dataset
import pandas as pd

# ==================== 环境设置 ====================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 国内镜像
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# 设置随机种子
set_seed(42)

# ==================== 路径配置 ====================
BERT_LOCAL_PATH = r"D:\111huiyu\慧与\题\作业集合\8月25日-吕雨阳\01.源代码\头条后端\data\bert_pretrain"
MODEL_DIR = r"D:\111huiyu\慧与\题\作业集合\8月25日-吕雨阳\01.源代码\头条后端\model"
DATA_PATH = r"D:\111huiyu\慧与\题\作业集合\8月25日-吕雨阳\01.源代码\头条后端\data\train_new.csv"
SAVE_PATH = os.path.join(MODEL_DIR, "bert_model")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)

print(f"📁 BERT 本地模型路径: {BERT_LOCAL_PATH}")
print(f"📁 模型保存路径: {SAVE_PATH}")
print(f"📄 数据文件: {DATA_PATH}")

# ==================== 检查文件 ====================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ 数据文件不存在: {DATA_PATH}")

if not os.path.exists(BERT_LOCAL_PATH):
    raise FileNotFoundError(f"❌ 本地 BERT 模型不存在！请检查路径:\n👉 {BERT_LOCAL_PATH}")

required_files = ["pytorch_model.bin", "config.json", "vocab.txt"]
for f in required_files:
    if not os.path.exists(os.path.join(BERT_LOCAL_PATH, f)):
        raise FileNotFoundError(f"❌ 缺少必要文件: {f}，请确认 BERT 模型完整")

# ==================== 加载数据 ====================
print("📊 正在加载数据...")
df = pd.read_csv(DATA_PATH)

if 'sentence' not in df.columns or 'label' not in df.columns:
    raise ValueError(f"❌ CSV 文件必须包含 'sentence' 和 'label' 列！现有列: {list(df.columns)}")

# 🔥 快速模式：只用 5 万条数据（可注释掉这行跑全量）
df = df.head(50000).reset_index(drop=True)
num_labels = df['label'].nunique()
print(f"✅ 数据加载成功，共 {len(df)} 条，{num_labels} 个类别")

# ==================== 自定义 Dataset ====================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):  # 🔥 max_len 从 128 → 64
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors=None
        )

        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ==================== 加载分词器和模型 ====================
print("🔍 正在加载本地 BERT 模型...")

try:
    tokenizer = BertTokenizer.from_pretrained(BERT_LOCAL_PATH)
    print("✅ 分词器加载成功")

    model = BertForSequenceClassification.from_pretrained(
        BERT_LOCAL_PATH,
        num_labels=num_labels
    )
    print(f"✅ BERT 模型加载成功，类别数: {num_labels}")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    raise

# ==================== 训练参数（4060 优化版）====================
print("⚙️  配置训练参数...")

training_args = TrainingArguments(
    output_dir=SAVE_PATH,
    num_train_epochs=1,                    # 🔥 只训练 1 轮，足够！
    per_device_train_batch_size=32,        # 🔥 4060 支持大 batch（原 16）
    warmup_steps=200,                      # 少量 warmup
    weight_decay=0.01,
    logging_dir=os.path.join(SAVE_PATH, "logs"),
    logging_steps=50,                      # 每 50 步输出 loss
    save_steps=1000,
    save_total_limit=2,
    learning_rate=2e-5,
    seed=42,
    disable_tqdm=False,                    # 显示进度条
    report_to=[],                          # 不上报
    fp16=True,                             # ⚡ 启用半精度（4060 支持）
    dataloader_num_workers=4,              # 加快数据加载
    remove_unused_columns=True,
    # 注意：旧版不支持 evaluation_strategy 或 device
)

# ==================== 构建训练器 ====================
print("🧠 构建训练器...")
try:
    train_dataset = TextDataset(
        df['sentence'].astype(str).tolist(),
        df['label'].tolist(),
        tokenizer,
        max_len=64  # 🔥 与上面保持一致
    )
    print(f"✅ 数据集构建完成，共 {len(train_dataset)} 个样本")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    print("✅ 训练器构建成功")
except Exception as e:
    print(f"❌ 构建训练器失败: {e}")
    raise
# ==================== 开始训练 ====================
if __name__ == "__main__":
    print("🚀 开始训练 BERT 模型（1 轮，极速模式）...")
    print(f"💡 使用设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        trainer.train()
        print("✅ 训练完成")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        raise

    # ==================== 保存微调后的模型 ====================
    print("💾 正在保存微调后的模型...")
    try:
        model.save_pretrained(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)
        print(f"🎉 模型已成功保存至:\n👉 {SAVE_PATH}")
    except Exception as e:
        print(f"❌ 模型保存失败: {e}")
        raise

    # ==================== 使用说明 ====================
    print("\n" + "="*60)
    print("📌 BERT 模型微调完成！（快速版）")
    print("💡 加载模型方法：")
    print(f"  model = BertForSequenceClassification.from_pretrained(r'{SAVE_PATH}')")
    print(f"  tokenizer = BertTokenizer.from_pretrained(r'{SAVE_PATH}')")
    print("="*60)