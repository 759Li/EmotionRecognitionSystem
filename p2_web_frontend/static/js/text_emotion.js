// static/js/text_emotion.js
/**
 * 文本情绪分析模块
 * 负责与后端API交互，处理文本情绪分析功能
 */
class TextEmotionAnalyzer {
    constructor() {
        this.apiBaseUrl = '/api/text';
        this.initEventListeners();
    }
    
    initEventListeners() {
        // 绑定分析按钮点击事件
        document.getElementById('analyze-btn').addEventListener('click', this.analyzeText.bind(this));
        
        // 绑定文本框回车事件
        document.getElementById('text-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.analyzeText();
            }
        });
    }
    
    async analyzeText() {
        const textInput = document.getElementById('text-input');
        const text = textInput.value.trim();
        
        if (!text) {
            this.showResult({
                success: false,
                message: '请输入要分析的文本'
            });
            return;
        }
        
        // 显示加载状态
        this.showLoading(true);
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/emotion`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });
            
            if (!response.ok) {
                throw new Error(`API请求失败: ${response.status}`);
            }
            
            const data = await response.json();
            
            this.showResult({
                success: data.code === 200,
                data: data.data,
                message: data.message
            });
            
        } catch (error) {
            this.showResult({
                success: false,
                message: `分析出错: ${error.message}`
            });
        } finally {
            this.showLoading(false);
        }
    }
    
    showResult(result) {
        const resultContainer = document.getElementById('emotion-result');
        const messageElement = document.getElementById('result-message');
        const labelElement = document.getElementById('emotion-label');
        const confidenceElement = document.getElementById('confidence');
        const keywordsElement = document.getElementById('keywords');
        
        if (result.success) {
            resultContainer.classList.remove('hidden');
            resultContainer.classList.add('show');
            
            messageElement.textContent = result.message;
            labelElement.textContent = result.data.label;
            confidenceElement.textContent = `${result.data.confidence * 100}%`;
            
            // 显示关键词
            keywordsElement.innerHTML = '';
            if (result.data.keywords && result.data.keywords.length > 0) {
                result.data.keywords.forEach(keyword => {
                    const span = document.createElement('span');
                    span.className = 'keyword';
                    span.textContent = keyword;
                    keywordsElement.appendChild(span);
                });
            } else {
                keywordsElement.innerHTML = '未识别到情绪关键词';
            }
            
            // 根据情绪标签设置样式
            this.setEmotionStyle(result.data.label);
        } else {
            resultContainer.classList.remove('show');
            resultContainer.classList.add('hidden');
            messageElement.textContent = result.message;
        }
    }
    
    setEmotionStyle(emotionLabel) {
        const emotionStyles = {
            "喜": "text-happy bg-happy-light",
            "怒": "text-angry bg-angry-light",
            "哀": "text-sad bg-sad-light",
            "惧": "text-fear bg-fear-light",
            "爱": "text-love bg-love-light",
            "恶": "text-disgust bg-disgust-light",
            "惊": "text-surprise bg-surprise-light"
        };
        
        const resultBox = document.getElementById('result-box');
        const labelElement = document.getElementById('emotion-label');
        
        // 清除原有样式
        resultBox.className = 'result-box p-4 rounded-lg mb-4';
        labelElement.className = 'emotion-label text-xl font-bold';
        
        // 应用新样式
        const styleClass = emotionStyles[emotionLabel] || '';
        if (styleClass) {
            resultBox.classList.add(styleClass.split(' ')[1]);
            labelElement.classList.add(styleClass.split(' ')[0]);
        }
    }
    
    showLoading(show) {
        const loadingElement = document.getElementById('loading');
        if (show) {
            loadingElement.classList.remove('hidden');
        } else {
            loadingElement.classList.add('hidden');
        }
    }
}

// 页面加载完成后初始化文本情绪分析模块
document.addEventListener('DOMContentLoaded', () => {
    const textEmotionAnalyzer = new TextEmotionAnalyzer();
});