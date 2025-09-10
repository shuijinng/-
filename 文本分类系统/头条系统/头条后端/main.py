# main.py
#终端启动 uvicorn main:app --reload
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Dict, Any
import pickle
import os
import jieba
import numpy as np
import fasttext

# ------------------ 路径配置 ------------------
MODEL_DIR = r"D:\111huiyu\model"
STOPWORDS_PATH = r"D:\111huiyu\慧与\课上代码\头条满分\data\stopwords.txt"

# ------------------ 加载停用词 ------------------
def load_stopwords(filepath: str) -> set:
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return {line.strip() for line in f if line.strip()}
    except Exception as e:
        print(f"⚠️ 警告：加载停用词失败 {filepath}: {e}")
    return set()

stopwords = load_stopwords(STOPWORDS_PATH)

def preprocess_text(text: str) -> str:
    words = jieba.lcut(text)
    return ' '.join(w for w in words if w not in stopwords and len(w.strip()) > 1)

# ------------------ 分类标签 ------------------
CATEGORIES = {
    0: "finance",      1: "realty",     2: "stocks",
    3: "education",    4: "science",    5: "society",
    6: "politics",     7: "sports",     8: "game",
    9: "entertainment"
}

# ------------------ 模型服务 ------------------
class ModelService:
    def __init__(self):
        self.rf_model = None
        self.ft_model = None
        self.bert_model = None
        self.bert_tokenizer = None

    def load_models(self):
        """加载所有模型"""
        # 1. 随机森林
        try:
            rf_path = os.path.join(MODEL_DIR, "rf_model.pkl")
            if not os.path.exists(rf_path):
                print(f"❌ RF 模型文件不存在: {rf_path}")
            else:
                with open(rf_path, "rb") as f:
                    self.rf_model = pickle.load(f)
                print("✅ RF 模型加载成功")
        except Exception as e:
            print(f"❌ RF 加载失败: {e}")

        # 2. fastText
        try:
            ft_path = os.path.join(MODEL_DIR, "fasttext_model.bin")
            if not os.path.exists(ft_path):
                print(f"❌ fastText 模型文件不存在: {ft_path}")
            else:
                self.ft_model = fasttext.load_model(ft_path)
                print("✅ fastText 模型加载成功")
        except Exception as e:
            print(f"❌ fastText 加载失败: {e}")

        # 3. BERT
        try:
            from transformers import BertForSequenceClassification, BertTokenizer
            bert_path = os.path.join(MODEL_DIR, "bert_model")
            if not os.path.exists(bert_path):
                print(f"❌ BERT 模型目录不存在: {bert_path}")
            else:
                self.bert_model = BertForSequenceClassification.from_pretrained(bert_path)
                self.bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
                self.bert_model.eval()  # 推理模式
                print("✅ BERT 模型加载成功")
        except Exception as e:
            print(f"❌ BERT 加载失败: {e}")

    def predict_rf(self, text: str) -> Dict[str, Any]:
        if not self.rf_model:
            return {"category": "unknown", "confidence": 0.0}
        try:
            cleaned = preprocess_text(text)
            pred = self.rf_model.predict([cleaned])[0]
            prob = self.rf_model.predict_proba([cleaned])[0].max()
            return {"category": CATEGORIES.get(pred, "unknown"), "confidence": float(prob)}
        except Exception as e:
            print(f"|RF 预测失败: {e}")
            return {"category": "unknown", "confidence": 0.0}

    def predict_fasttext(self, text: str) -> Dict[str, Any]:
        if not self.ft_model:
            return {"category": "unknown", "confidence": 0.0}
        try:
            cleaned = preprocess_text(text)
            if not cleaned or len(cleaned.strip()) == 0:
                print(f"⚠️ fastText 输入为空: '{text}' -> '{cleaned}'")
                return {"category": "unknown", "confidence": 0.0}

            cleaned = str(cleaned).strip()
            if not cleaned:
                return {"category": "unknown", "confidence": 0.0}

            print(f"fastText 输入: '{cleaned}'")

            pred_label, pred_prob = self.ft_model.predict(cleaned)

            # ✅ 确保 label_str 是 str
            label_str = str(pred_label[0]) if isinstance(pred_label, (list, tuple)) and len(pred_label) > 0 else ""

            # ✅ 将 numpy 概率转为 Python float
            conf = float(pred_prob[0]) if isinstance(pred_prob, (list, tuple, np.ndarray)) and len(
                pred_prob) > 0 else 0.0
            conf = float(conf)  # 确保是 float

            # ✅ 关键修复：裁剪置信度到 [0.0, 1.0]
            conf = max(0.0, min(1.0, conf))

            print(f"原始 pred_prob 类型: {type(pred_prob[0])}, 值: {pred_prob[0]}")
            print(f"裁剪后 conf 类型: {type(conf)}, 值: {conf}")

            # 解析标签
            if label_str.startswith('__label__'):
                try:
                    label = int(label_str.replace('__label__', ''))
                except ValueError:
                    label = -1
            else:
                label = -1

            category = CATEGORIES.get(label, "unknown")

            result = {"category": category, "confidence": conf}
            print(f"fastText 输出: {result}")
            return result

        except Exception as e:
            print(f"❌ fastText 预测失败: {e}")
            import traceback
            traceback.print_exc()
            return {"category": "unknown", "confidence": 0.0}

    def predict_bert(self, text: str) -> Dict[str, Any]:
        if not self.bert_model or not self.bert_tokenizer:
            return {"category": "unknown", "confidence": 0.0}
        try:
            import torch
            inputs = self.bert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                conf, pred = torch.max(probs, dim=1)
                label = pred.item()
                return {"category": CATEGORIES.get(label, "unknown"), "confidence": conf.item()}
        except Exception as e:
            print(f"BERT 预测失败: {e}")
            return {"category": "unknown", "confidence": 0.0}


# ------------------ FastAPI 应用 ------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 服务启动中...")
    model_service.load_models()
    print("✅ 所有模型加载完成")
    yield
    print("🛑 服务关闭")

app = FastAPI(title="多模型文本分类 API", lifespan=lifespan)

# CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请改为具体域名
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化模型服务
model_service = ModelService()

# 请求体
class TextRequest(BaseModel):
    text: str

# 响应体（每个模型的预测结果）
class ModelResult(BaseModel):
    category: str
    confidence: float

# 综合响应
class MultiModelResponse(BaseModel):
    text: str
    random_forest: ModelResult
    fasttext: ModelResult
    bert: ModelResult

# ------------------ API 接口 ------------------

@app.post("/predict", response_model=MultiModelResponse)
async def predict(request: TextRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="文本不能为空")

    try:
        result = MultiModelResponse(
            text=text,
            random_forest=model_service.predict_rf(text),
            fasttext=model_service.predict_fasttext(text),
            bert=model_service.predict_bert(text)
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models": {
            "random_forest": model_service.rf_model is not None,
            "fasttext": model_service.ft_model is not None,
            "bert": model_service.bert_model is not None
        },
        "message": "服务正常运行"
    }