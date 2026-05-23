"""
文件功能：文本情感分析模块（最终生产版）

更新记录：
- 2025年07月 初始版本，实现基于BERT的文本情感分类功能
- 2025年07月 补充情绪类别映射逻辑和概率计算功能
- 2026年05月 增强：集成DeepSeek大模型API，实现BERT+DeepSeek双重分析
- 2026年05月 优化：添加DeepSeek中性优先规则，以及融合后概率差值检测强制中性

依赖库：
- torch (PyTorch 深度学习框架)
- transformers (HuggingFace 提供的预训练模型接口)
- openai (DeepSeek API调用，可选)

环境变量：
- DEEPSEEK_API_KEY: DeepSeek API密钥（若不设置则仅使用BERT模型）

功能描述：
该模块提供了一个增强的文本情感分析解决方案，使用BERT预训练模型进行快速情绪分类，
并可选地调用DeepSeek大模型API进行二次校验和融合。
当DeepSeek高置信度判定为中性时，直接采用其结果；否则融合后若最高概率与次高概率差距过小，
则强制输出中性，避免随机分类。最终输出与原始接口完全兼容。

主要函数：
analyze_text_emotion(text, model_path) - 执行文本情感分析的核心函数（接口不变）

编写日期：2025年07月（最终生产版本：2026年05月）
班级：物联一班
学号：202378040109
作者：李正标
"""

import os
import json
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ======================== 全局配置 ========================
# 是否启用DeepSeek双重分析（默认启用，若未设置API Key则自动禁用）
ENABLE_DEEPSEEK = True
# 调试模式：生产环境建议设为False
DEBUG_MODE = True

# DeepSeek API配置（从环境变量读取）
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not DEEPSEEK_API_KEY:
    if DEBUG_MODE:
        print("[警告] 未设置环境变量 DEEPSEEK_API_KEY，DeepSeek双重分析功能将被禁用")
    ENABLE_DEEPSEEK = False

# 目标情绪类别（英文键，与前端保持一致）
TARGET_EMOTIONS = ["fear", "anger", "disgust", "neutral", "sadness", "happy", "surprise"]
# 情绪中文映射（仅用于调试显示）
EMOTION_CN_MAP = {
    "fear": "恐惧",
    "anger": "愤怒",
    "disgust": "厌恶",
    "neutral": "中性",
    "sadness": "悲伤",
    "happy": "高兴",
    "surprise": "惊讶",
}

# 强制中性相关阈值
NEUTRAL_CONFIDENCE_THRESHOLD = 0.7      # DeepSeek中性置信度阈值
PROB_DIFF_THRESHOLD = 0.10              # 融合后最高与次高概率的最小差值阈值
FORCED_NEUTRAL_PROB = 0.85              # 强制输出中性时分配给 neutral 的概率

# 结果缓存字典
_RESULT_CACHE = {}

# DeepSeek客户端（懒初始化）
_ds_client = None


def _get_ds_client():
    """获取DeepSeek客户端单例"""
    global _ds_client
    if _ds_client is None and ENABLE_DEEPSEEK and DEEPSEEK_API_KEY:
        _ds_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com", timeout=15.0)
    return _ds_client


# ======================== BERT模型加载（懒加载） ========================
_BERT_TOKENIZER = None
_BERT_MODEL = None


def _load_bert_model(model_path: str):
    """加载BERT模型和分词器（懒加载，全局单例）"""
    global _BERT_TOKENIZER, _BERT_MODEL
    if _BERT_TOKENIZER is None or _BERT_MODEL is None:
        if DEBUG_MODE:
            print(f"[BERT] 正在加载模型: {model_path}")
        _BERT_TOKENIZER = BertTokenizer.from_pretrained(model_path)
        _BERT_MODEL = BertForSequenceClassification.from_pretrained(model_path)
        _BERT_MODEL.eval()
        if DEBUG_MODE:
            print("[BERT] 模型加载完成")
    return _BERT_TOKENIZER, _BERT_MODEL


def _analyze_bert(text: str, model_path: str) -> dict:
    """
    使用BERT模型进行情绪分析（内部函数）
    返回：包含概率分布、情绪类别、置信度的字典
    """
    tokenizer, model = _load_bert_model(model_path)

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).squeeze().tolist()

    # BERT模型原始输出的7个类别（固定顺序）
    bert_emotions = ["恐惧", "愤怒", "厌恶", "喜好", "悲伤", "高兴", "惊讶"]
    bert_probs = probabilities

    # 映射到目标情绪体系
    mapping = {
        "高兴": "happy",
        "悲伤": "sadness",
        "厌恶": "disgust",
        "喜好": "happy",    # 喜好映射为高兴
        "恐惧": "fear",
        "惊讶": "surprise",
        "愤怒": "anger",
    }

    target_probs = {emotion: 0.0 for emotion in TARGET_EMOTIONS}
    for orig_emotion, prob in zip(bert_emotions, bert_probs):
        target_emotion = mapping[orig_emotion]
        target_probs[target_emotion] += prob

    # 归一化
    total = sum(target_probs.values())
    if total > 0:
        target_probs = {k: v / total for k, v in target_probs.items()}

    emotion_class = max(target_probs, key=target_probs.get)
    confidence = target_probs[emotion_class]

    return {
        "emotions": target_probs,
        "emotion_class": emotion_class,
        "confidence": confidence,
    }


# ======================== DeepSeek分析 ========================
def _analyze_deepseek(text: str) -> dict:
    """调用DeepSeek API进行情绪分析，返回与BERT相同结构的字典，失败返回None"""
    client = _get_ds_client()
    if client is None:
        if DEBUG_MODE:
            print("[DeepSeek] 客户端未初始化，跳过调用")
        return None

    prompt = f"""你是一个专业的情绪分析模型。请分析以下文本的情感。

文本：{text}

要求：
1. 分析文本的情绪标签，从以下标签中选择一个作为主要情绪标签：{', '.join(TARGET_EMOTIONS)}
2. 输出一个JSON对象，包含每种情绪的概率，格式如下：
{{
    "emotions": {{
        "fear": 概率值（0~1之间的小数，所有情绪概率之和应为1）,
        "anger": 概率值,
        "disgust": 概率值,
        "neutral": 概率值,
        "sadness": 概率值,
        "happy": 概率值,
        "surprise": 概率值
    }},
    "emotion_class": 主要情绪标签（字符串）,
    "confidence": 置信度（最高概率值）
}}
只输出JSON，不要有其他解释文字。
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个情绪分析助手，只输出JSON格式的结果。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
            max_tokens=500,
        )
        result_json = json.loads(response.choices[0].message.content)

        if "emotions" not in result_json or "emotion_class" not in result_json:
            raise ValueError("DeepSeek返回的JSON缺少必要字段")

        # 确保所有目标情绪都存在
        for emotion in TARGET_EMOTIONS:
            if emotion not in result_json["emotions"]:
                result_json["emotions"][emotion] = 0.0

        # 归一化概率和
        prob_sum = sum(result_json["emotions"].values())
        if abs(prob_sum - 1.0) > 1e-6:
            max_emotion = result_json["emotion_class"]
            result_json["emotions"][max_emotion] += (1.0 - prob_sum)

        result_json["confidence"] = result_json["emotions"][result_json["emotion_class"]]

        if DEBUG_MODE:
            print(f"[DeepSeek] 分析完成，主情绪: {result_json['emotion_class']}, 置信度: {result_json['confidence']:.4f}")

        return result_json

    except Exception as e:
        if DEBUG_MODE:
            print(f"[DeepSeek] 调用失败: {str(e)}")
        return None


# ======================== 融合与强制中性逻辑 ========================
def _fuse_results(bert_result: dict, ds_result: dict) -> dict:
    """
    将BERT和DeepSeek结果融合，并应用强制中性规则：
    1. 如果DeepSeek存在且主情绪为neutral且置信度>=NEUTRAL_CONFIDENCE_THRESHOLD，直接返回DeepSeek结果。
    2. 否则进行置信度加权融合，若融合后最高概率与次高概率之差小于PROB_DIFF_THRESHOLD，则强制输出中性。
    """
    # 规则1: DeepSeek 高置信度中性优先
    if ds_result is not None and ds_result.get("confidence", 0) >= NEUTRAL_CONFIDENCE_THRESHOLD:
        if ds_result["emotion_class"] == "neutral":
            if DEBUG_MODE:
                print(f"[规则触发] DeepSeek高置信度中性 (conf={ds_result['confidence']:.3f})，直接采用")
            return ds_result

    # 若DeepSeek结果无效，仅使用BERT
    if ds_result is None or ds_result.get("confidence", 0) <= 0:
        return bert_result

    # 置信度加权融合
    bert_conf = bert_result["confidence"]
    ds_conf = ds_result["confidence"]
    total_conf = bert_conf + ds_conf
    if total_conf > 0:
        bert_weight = bert_conf / total_conf
        ds_weight = ds_conf / total_conf
    else:
        bert_weight = ds_weight = 0.5

    fused_probs = {}
    for emotion in TARGET_EMOTIONS:
        bert_prob = bert_result["emotions"].get(emotion, 0.0)
        ds_prob = ds_result["emotions"].get(emotion, 0.0)
        fused_probs[emotion] = bert_weight * bert_prob + ds_weight * ds_prob

    # 归一化
    total = sum(fused_probs.values())
    if total > 0:
        fused_probs = {k: v / total for k, v in fused_probs.items()}
    else:
        fused_probs = {k: 1.0 / len(TARGET_EMOTIONS) for k in TARGET_EMOTIONS}

    # 按概率降序排序
    sorted_emotions = sorted(fused_probs.items(), key=lambda x: x[1], reverse=True)
    top_emotion, top_prob = sorted_emotions[0]
    second_prob = sorted_emotions[1][1] if len(sorted_emotions) > 1 else 0.0
    diff = top_prob - second_prob

    if DEBUG_MODE:
        print(f"[融合] BERT权重={bert_weight:.3f}, DeepSeek权重={ds_weight:.3f}")
        print(f"[融合] 主情绪={top_emotion}, 置信度={top_prob:.4f}, 与次高差值={diff:.4f}")

    # 规则2: 最高与次高概率差值过小 -> 强制中性
    if diff < PROB_DIFF_THRESHOLD:
        if DEBUG_MODE:
            print(f"[规则触发] 融合后概率差值 {diff:.4f} < {PROB_DIFF_THRESHOLD}，强制输出中性")
        # 构造强制中性分布
        other_prob = (1.0 - FORCED_NEUTRAL_PROB) / (len(TARGET_EMOTIONS) - 1)
        forced_probs = {emotion: other_prob for emotion in TARGET_EMOTIONS}
        forced_probs["neutral"] = FORCED_NEUTRAL_PROB
        total_forced = sum(forced_probs.values())
        forced_probs = {k: v / total_forced for k, v in forced_probs.items()}
        return {
            "emotions": forced_probs,
            "emotion_class": "neutral",
            "confidence": forced_probs["neutral"],
        }

    # 正常返回融合结果
    return {
        "emotions": fused_probs,
        "emotion_class": top_emotion,
        "confidence": top_prob,
    }


# ======================== 公开接口 ========================
def analyze_text_emotion(text: str, model_path: str) -> dict:
    """
    执行文本情感分析的核心函数（增强版：BERT + DeepSeek双重分析 + 强制中性逻辑）
    
    参数：
        text (str): 输入文本
        model_path (str): BERT模型路径（本地目录或HuggingFace模型名）
    
    返回：
        dict: 包含以下字段的结果字典
            - emotions (dict): 7种情绪的概率，键为英文情绪名，值为字符串格式（保留8位小数）
            - keywords (list): 关键词列表（本模块不提取关键词，始终为空列表）
            - confidence (float): 最高概率值
            - face_count (int): 文本分析固定为0
            - face_details (list): 文本分析固定为空列表
            - emotion_class (str): 概率最高的情绪类别（英文）
    """
    cache_key = f"{text}_{model_path}_{ENABLE_DEEPSEEK}"
    if cache_key in _RESULT_CACHE:
        if DEBUG_MODE:
            print("[缓存] 命中缓存，直接返回结果")
        return _RESULT_CACHE[cache_key].copy()

    # BERT分析
    bert_result = _analyze_bert(text, model_path)

    # DeepSeek分析（可选）
    ds_result = None
    if ENABLE_DEEPSEEK and DEEPSEEK_API_KEY:
        ds_result = _analyze_deepseek(text)
        if ds_result is None and DEBUG_MODE:
            print("[回退] DeepSeek分析失败，将仅使用BERT结果")

    # 融合（内部包含强制中性规则）
    final_result = _fuse_results(bert_result, ds_result)

    # 格式化输出（保持与原代码一致）
    formatted_probs = {k: f"{v:.8f}" for k, v in final_result["emotions"].items()}
    # 修正浮点误差
    sum_formatted = sum(float(v) for v in formatted_probs.values())
    if abs(sum_formatted - 1.0) > 1e-8:
        max_emotion = max(formatted_probs.keys(), key=lambda x: float(formatted_probs[x]))
        adjusted = float(formatted_probs[max_emotion]) + (1.0 - sum_formatted)
        formatted_probs[max_emotion] = f"{adjusted:.8f}"

    result = {
        "emotions": formatted_probs,
        "keywords": [],
        "confidence": round(final_result["confidence"], 8),
        "face_count": 0,
        "face_details": [],
        "emotion_class": final_result["emotion_class"],
    }

    _RESULT_CACHE[cache_key] = result.copy()
    return result


def set_deepseek_api_key(api_key: str, enable: bool = True):
    """手动设置DeepSeek API Key（优先级高于环境变量）"""
    global DEEPSEEK_API_KEY, ENABLE_DEEPSEEK, _ds_client
    DEEPSEEK_API_KEY = api_key
    ENABLE_DEEPSEEK = enable
    _ds_client = None
    if DEBUG_MODE:
        print(f"[配置] DeepSeek API Key已设置，双重分析启用: {enable}")


# # ======================== 测试代码 ========================
# if __name__ == "__main__":
#     test_texts = [
#         "今天天气真好，心情很愉快！",
#         "我真的很生气，怎么会这样！",
#         "这事情让我感到非常恐惧。",
#         "就这样吧，没什么特别的感觉。",
#         "我好难过，眼泪都要掉下来了。",
#         "哇！这太令人惊讶了！",
#         "这个味道真让人恶心。",
#         "今天星期三",
#         "嗯，知道了",
#     ]

#     MODEL_PATH = "nlp_structbert_emotion_classification_chinese_base"

#     print("=" * 60)
#     print("文本情绪分析测试（最终生产版）")
#     print(f"DeepSeek双重分析: {'启用' if ENABLE_DEEPSEEK else '禁用'}")
#     print(f"中性优先阈值: {NEUTRAL_CONFIDENCE_THRESHOLD}")
#     print(f"概率差值阈值: {PROB_DIFF_THRESHOLD}")
#     print("=" * 60)

#     for text in test_texts:
#         print(f"\n输入文本: {text}")
#         result = analyze_text_emotion(text, MODEL_PATH)
#         print(f"  主情绪: {result['emotion_class']} ({EMOTION_CN_MAP.get(result['emotion_class'], '未知')})")
#         print(f"  置信度: {result['confidence']:.6f}")
#         print("  情绪概率:")
#         for emotion, prob_str in result["emotions"].items():
#             prob = float(prob_str)
#             if prob > 0.01:
#                 print(f"    {emotion}: {prob:.4f}")
#         print("-" * 40)

#     # 测试缓存
#     print("\n[缓存测试] 重复分析同一文本...")
#     first = analyze_text_emotion(test_texts[0], MODEL_PATH)
#     second = analyze_text_emotion(test_texts[0], MODEL_PATH)
#     print(f"两次结果是否相同: {first['emotions'] == second['emotions']}")