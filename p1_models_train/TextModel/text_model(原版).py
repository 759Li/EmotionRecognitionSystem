"""
文件功能：文本情感分析模块

更新记录：
- 2025年07月 初始版本，实现基于BERT的文本情感分类功能
- 2025年07月 补充情绪类别映射逻辑和概率计算功能

依赖库：
- torch (PyTorch 深度学习框架)
- transformers (HuggingFace 提供的预训练模型接口)

功能描述：
该模块提供了一个完整的文本情感分析解决方案，使用BERT预训练模型进行
情绪预测，并将原始情绪分类结果映射到标准的情绪类别体系中。

主要函数：
analyze_text_emotion(text, model_path) - 执行文本情感分析的核心函数

编写日期：2025年07月
班级：物联一班
学号：202378040109
作者：李正标
"""
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

def analyze_text_emotion(text, model_path):
    # 加载模型和分词器
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # 对输入文本进行分词和编码
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    # 模型推理
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # 计算softmax得到各类别的概率分布
        probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
    # 原始模型的情绪类别
    original_emotions = ["恐惧", "愤怒", "厌恶", "喜好", "悲伤", "高兴", "惊讶"]
    # 目标情绪类别（中英文对照）
    target_emotions_cn = ["恐惧", "愤怒", "厌恶", "中性", "悲伤", "高兴", "惊讶"]
    target_emotions_en = ["fear", "anger", "disgust", "neutral", "sadness", "happy", "surprise"]
    # 中英文情绪映射字典
    emotion_translation = {
        "fear": "恐惧",
        "anger": "愤怒",
        "disgust": "厌恶",
        "neutral": "中性",
        "sadness": "悲伤",
        "happy": "高兴",
        "surprise": "惊讶"
    }
    # 映射关系（原始情绪 -> 目标情绪）
    mapping = {
        "高兴": "happy",
        "悲伤": "sadness",
        "厌恶": "disgust",
        "喜好": "happy",  # 喜好映射为高兴
        "恐惧": "fear",
        "惊讶": "surprise",
        "愤怒": "anger"
    }
    # 初始化目标情绪概率字典
    target_probs = {emotion: 0.0 for emotion in target_emotions_en}
    # 合并概率
    for orig_emotion, prob in zip(original_emotions, probabilities):
        target_emotion = mapping[orig_emotion]
        target_probs[target_emotion] += prob
    # 提取除中性外的其他六种情绪概率（过滤"neutral"）
    other_emotions_probs = [
        target_probs[emotion]
        for emotion in target_emotions_en
        if emotion != "neutral"
    ]
    # 计算非中性情绪的概率极值与差值（处理空列表极端情况）
    if other_emotions_probs:
        max_prob = max(other_emotions_probs)
        min_prob = min(other_emotions_probs)
        diff = max_prob - min_prob
    else:
        diff = 0.0  # 极端情况：非中性情绪全为0
    # 动态计算中性情绪概率（归一化处理，max_diff为概率差的理论最大值1.0）
    max_diff = 1.0  # 概率差的理论最大值（如1和0的差）
    neutral_prob = 1.0 - (diff / max_diff) if max_diff != 0 else 1.0
    target_probs["neutral"] = max(0.0, min(1.0, neutral_prob))  # 约束范围
    # 计算剩余概率（非中性情绪的总概率）
    remaining_prob = 1.0 - target_probs["neutral"]
    # 分配剩余概率到非中性情绪（处理除零错误）
    sum_other = sum(other_emotions_probs)  # 理论上sum_other=1（合并后非中性情绪和为1）
    for emotion in target_emotions_en:
        if emotion != "neutral":
            if sum_other > 0:
                # 按比例分配（非中性情绪内部占比）
                target_probs[emotion] = (target_probs[emotion] / sum_other) * remaining_prob
            else:
                # 极端情况：非中性情绪全为0 → 均匀分配
                target_probs[emotion] = remaining_prob / 6  # 6种非中性情绪
    # ========== 新增：修正浮点显示误差 ==========
    # 格式化概率（保留8位小数）
    formatted_target_probs = {
        emotion: f"{prob:.8f}"
        for emotion, prob in target_probs.items()
    }
    # 计算格式化后的总和，修正误差
    sum_formatted = sum(float(prob) for prob in formatted_target_probs.values())
    error = 1.0 - sum_formatted
    if abs(error) > 1e-9:  # 误差超过1e-9时调整
        # 找到概率最大的情绪（优先调整影响最小的项）
        max_emotion = max(formatted_target_probs.keys(), key=lambda x: float(formatted_target_probs[x]))
        # 调整该情绪的概率
        adjusted_prob = float(formatted_target_probs[max_emotion]) + error
        adjusted_prob = max(0.0, min(1.0, adjusted_prob))  # 确保范围合法
        formatted_target_probs[max_emotion] = f"{adjusted_prob:.8f}"
    # 计算置信度（最高概率，保留8位小数）
    confidence = round(max(float(prob) for prob in formatted_target_probs.values()), 8)
    # 确定情绪类别
    emotion_class = max(target_probs, key=target_probs.get)
    # 构建输出结果
    result = {
        "emotions": formatted_target_probs,
        "keywords": [],  # 由于没有关键词提取逻辑，这里返回空列表
        "confidence": confidence,
        "face_count": 0,  # 文本分析没有人脸信息
        "face_details": [],
        "emotion_class": emotion_class
    }

    return result

#========================================================测试============================================================

# 修正贴合于本项目的输出结果：
# if __name__ == "__main__":
#     # 示例调用（请替换为实际的模型路径）
#     text = "就这样了吧"
#     model_path = "nlp_structbert_emotion_classification_chinese_base"
#     result = analyze_text_emotion(text, model_path)
#     # 打印结果
#     print(f"情绪类别: {result['emotion_class']}")
#     print(f"置信度: {result['confidence']}")
#     print("情绪概率分布:")
#     add_result = 0.0
#     for emotion, prob in result['emotions'].items():
#         add_result += float(prob)
#         print(f"  {emotion}: {prob}")
#     print(f"概率和: {add_result}")

# 原始模型输出结果：["恐惧", "愤怒", "厌恶", "喜好", "悲伤", "高兴", "惊讶"]
# if __name__ == "__main__":
#     text = "你好吗"
#     model_path = "nlp_structbert_emotion_classification_chinese_base"
#     tokenizer = BertTokenizer.from_pretrained('nlp_structbert_emotion_classification_chinese_base')
#     model = BertForSequenceClassification.from_pretrained(model_path)
#     model.eval()
#     # 对输入文本进行分词和编码
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
#     # 模型推理
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits
#         # 计算softmax得到各类别的概率分布
#         probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
#     # 原始模型的情绪类别
#     original_emotions = ["恐惧", "愤怒", "厌恶", "喜好", "悲伤", "高兴", "惊讶"]
#     # 【害怕，，厌恶，，悲伤】
#     print([{x: y} for x, y in zip(original_emotions, probabilities)])
#     add=0
#     for x, y in zip(original_emotions, probabilities):
#         add+=y
#     print( add)