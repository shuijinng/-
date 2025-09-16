# backend/app.py - Flask 版本的后端服务
from flask import Flask, request, jsonify
from openai import OpenAI
from flask_cors import CORS  # 用于处理跨域请求

# 初始化 Flask 应用
app = Flask(__name__)

# 启用跨域支持（允许前端 http://localhost:8000 访问）
CORS(app, origins=[
    "http://localhost:8000",
    "http://127.0.0.1:8848"  # ← 添加这一行！
])

# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key="sk-bfc0f0aec8aa42ea9ce583fe798afa89",  # 🔑 替换为你的 DeepSeek API Key
    base_url="https://api.deepseek.com/v1"
)

@app.route('/ask', methods=['POST'])
def ask():

    data = request.get_json()
    question = data.get("question", "").strip()

    # 检查输入
    if not question:
        return jsonify({"answer": "请输入一个问题哦～"}), 400

    try:
        # 调用 DeepSeek 模型
        response = client.chat.completions.create(
            model="deepseek-chat",  # 可用模型：deepseek-chat 或 deepseek-coder（后者适合技术问题）
            messages=[
                {"role": "system", "content": "你是一个专业的旅游助手，用中文简洁、友好地回答旅行问题。"},
                {"role": "user", "content": question}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        return jsonify({"answer": answer})

    except Exception as e:
        print("调用 DeepSeek 时出错：", str(e))  # 打印错误日志
        return jsonify({"answer": f"抱歉，机器人暂时无法响应：{str(e)}"}), 500


# 可选：加一个健康检查接口
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "旅游机器人服务正常运行中"})


if __name__ == '__main__':
    print("🌍 旅游问答机器人后端已启动！")
    print("📍 访问 http://localhost:8001/health 检查服务状态")
    print("📩 POST 请求发送到 http://localhost:8001/ask")
    app.run(host='0.0.0.0', port=8001, debug=True)