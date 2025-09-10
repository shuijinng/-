// frontend/api.js
class TextClassifierAPI {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };

        try {
            const response = await fetch(url, { ...defaultOptions, ...options });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API请求失败:', error);
            throw error;
        }
    }

    // 获取健康状态
    async health() {
        return this.request('/health');
    }

    // 获取所有分类
    async getCategories() {
        return this.request('/categories');
    }

    // 单个文本预测
    async predict(text) {
        return this.request('/predict', {
            method: 'POST',
            body: JSON.stringify({ text })
        });
    }

    // 批量预测
    async predictBatch(texts) {
        return this.request('/predict/batch', {
            method: 'POST',
            body: JSON.stringify({ texts })
        });
    }
}

// 使用示例
const api = new TextClassifierAPI();

// 在您的前端代码中使用
async function classifyText() {
    try {
        // 检查服务状态
        const health = await api.health();
        if (!health.model_loaded) {
            alert('模型尚未加载，请稍后重试');
            return;
        }

        // 获取分类列表
        const categories = await api.getCategories();
        console.log('可用分类:', categories);

        // 单个文本预测
        const result = await api.predict('中华女子学院：本科层次仅1专业招男生');
        console.log('分类结果:', result);

        // 批量预测
        const batchResult = await api.predictBatch([
            '东5环海棠公社230-290平2居准现房98折优惠',
            '卡佩罗：告诉你德国脚生猛的原因 不希望英德战踢点球'
        ]);
        console.log('批量结果:', batchResult);

    } catch (error) {
        console.error('分类失败:', error);
    }
}