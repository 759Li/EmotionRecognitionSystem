"""
多模态情绪识别结果数据模型模块

功能说明：
提供统一的数据结构定义，支持文本、图像、视频三种识别类型的标准化输出。
包含情绪类别枚举定义、基础情绪结果类、综合识别结果类及工厂创建模式实现。

功能特点：
1. 支持七种标准情绪分类（愤怒/厌恶/恐惧/喜悦/中性/悲伤/惊讶）
2. 统一处理文本/图像/视频多模态输入
3. 提供数据库存储格式转换方法
4. 包含媒体路径规范化处理
5. 支持人脸检测结果的结构化表示

编写日期：2025年07月
班级：物联一班
学号：202378040109
作者：李正标
"""
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any

class EmotionClass(Enum):
    """
    情绪类别枚举（与七类标准对应）
    该枚举类定义了情绪类别，与系统中的七类标准情绪相对应。
    每个枚举值都有对应的英文和中文描述，方便在不同场景下使用。
    """
    ANGER = "anger"        # 愤怒
    DISGUST = "disgust"    # 厌恶
    FEAR = "fear"          # 恐惧
    HAPPY = "happy"        # 高兴
    NEUTRAL = "neutral"    # 中性
    SADNESS = "sadness"    # 悲伤
    SURPRISE = "surprise"  # 惊讶
    UNKNOWN = "unknown"    # 未定义
    @classmethod
    def get_display_name(cls, emotion_class: str) -> str:
        """
        根据情绪类别的英文标识，获取对应的中文显示名称。
        :param emotion_class: 情绪类别的英文标识
        :return: 情绪类别的中文显示名称
        """
        mapping = {
            "anger": "愤怒",
            "disgust": "厌恶",
            "fear": "恐惧",
            "happy": "喜悦",
            "neutral": "中性",
            "sadness": "悲伤",
            "surprise": "惊讶"
        }
        return mapping.get(emotion_class, emotion_class)
@dataclass
class EmotionResult:
    """
    该类定义了情绪识别结果的数据结构，包含了主情绪类别、置信度、全情绪概率分布等信息。
    可以将该类的实例转换为 JSON 格式，方便存储和传输。
    """
    emotion_class: str       # 主情绪类别
    confidence: float        # 置信度(0-1)
    emotions: Dict[str, float]  # 全情绪概率分布
    keywords: List[str]      # 关键词（文本专用）
    face_count: int = 0      # 人脸数量（图像/视频专用）
    face_details: List[Dict] = None  # 人脸细节（坐标+情绪）

    def to_json(self) -> Dict[str, Any]:
        """
        将情绪识别结果转换为 JSON 格式的字典，方便存储到数据库或进行网络传输。
        如果 face_details 为 None，则将其初始化为空列表。
        :return: 包含情绪识别结果的 JSON 格式字典
        """
        result = asdict(self)
        # 处理特殊字段的序列化问题
        # 1. 确保枚举值转换为字符串表示
        if 'emotion_class' in result and isinstance(result['emotion_class'], EmotionClass):
            result['emotion_class'] = result['emotion_class'].value
        # 2. 处理嵌套对象的序列化（如face_details中的枚举值）
        if 'face_details' in result and result['face_details'] is not None:
            for face in result['face_details']:
                if 'emotion' in face and isinstance(face['emotion'], EmotionClass):
                    face['emotion'] = face['emotion'].value
        if "face_details" in result and result["face_details"] is None:
            result["face_details"] = []
        # 确保 emotions 字段中的值是 float 类型，避免出现 numpy 类型或其他非标准类型
        if "emotions" in result:
            result["emotions"] = {k: float(v) for k, v in result["emotions"].items()}
        return result

@dataclass
class RecognitionResult:
    """
    该类定义了识别结果的通用数据结构，包含了用户 ID、识别类型、情绪识别结果等信息。
    可以将该类的实例转换为适合数据库存储的字典格式，同时提供了获取识别类型 ID 和绝对媒体路径的方法。
    """
    user_id: int                  # 用户 ID
    recognition_type: str         # 类型：text/image/video
    emotion_result: EmotionResult  # 情绪结果
    created_at: datetime          # 识别时间
    raw_content: Optional[str] = None  # 原始文本
    media_path: Optional[str] = None    # 媒体文件路径

    def to_database_dict(self) -> Dict[str, Any]:
        """
        将识别结果转换为适合数据库存储的字典格式。
        根据识别类型，将媒体路径和原始文本存储到对应的字段中。
        :return: 适合数据库存储的字典格式
        """
        return {
            "user_id": self.user_id,
            "type_id": self.get_recognition_type_id(self.recognition_type),
            "result": self.emotion_result.to_json(),
            "confidence": self.emotion_result.confidence,
            "video_path": self.media_path if self.recognition_type == "video" else None,
            "image_path": self.media_path if self.recognition_type == "image" else None,
            "text_content": self.raw_content if self.recognition_type == "text" else None,
            "created_at": self.created_at
        }

    @staticmethod
    def get_recognition_type_id(type_name: str) -> int:
        """
        根据识别类型的名称，获取对应的类型 ID，与数据库中的 RECOGNITION_TYPES 表严格同步。
        :param type_name: 识别类型的名称
        :return: 对应的类型 ID
        """
        type_mapping = {
            "text": 3,
            "image": 2,
            "video": 1
        }
        return type_mapping.get(type_name.lower(), 1)

    def get_absolute_media_path(self) -> str:
        """
        获取基于项目根目录的绝对媒体路径。
        如果媒体路径为空，则返回空字符串；如果媒体路径不以 'resources/' 开头，则添加该前缀。
        :return: 绝对媒体路径
        """
        if not self.media_path:
            return ""
        # 确保路径以 resources/ 开头（项目规定用户资源存储路径）
        if not self.media_path.startswith("resources/"):
            return f"resources/{self.media_path}"
        return self.media_path

class RecognitionResultFactory:
    """
    该类是多模态识别结果的工厂类，提供了创建文本、图像和视频识别结果的静态方法。
    严格按照项目要求的格式和规范生成识别结果对象。
    """
    @staticmethod
    def create_text_result(
            user_id: int,
            text_content: str,
            emotion_class: str,
            confidence: float,
            emotions: Dict[str, float],
            keywords: List[str]
    ) -> RecognitionResult:
        """
        创建文本识别结果对象。
        项目要求文本输入不超过 100 字，如果超过则截取前 100 字。
        :param user_id: 用户 ID
        :param text_content: 原始文本内容
        :param emotion_class: 主情绪类别
        :param confidence: 置信度
        :param emotions: 全情绪概率分布
        :param keywords: 关键词列表
        :return: 文本识别结果对象
        """
        if len(text_content) > 100:
            text_content = text_content[:100]  # 项目要求文本输入不超过 100 字
        emotion_result = EmotionResult(
            emotion_class=emotion_class,
            confidence=confidence,
            emotions=emotions,
            keywords=keywords
        )
        return RecognitionResult(
            user_id=user_id,
            recognition_type="text",
            emotion_result=emotion_result,
            created_at=datetime.now(),
            raw_content=text_content
        )

    @staticmethod
    def create_image_result(
            user_id: int,
            image_path: str,  # 存储于 resources/users_images/ 下
            emotion_class: str,
            confidence: float,
            emotions: Dict[str, float],
            keywords: List[str],
            face_count: int = 1,
            face_details: List[Dict] = None
    ) -> RecognitionResult:
        """
        创建图像识别结果对象。
        严格遵循 resources/ 的存储规范，将图像路径添加该前缀。
        :param user_id: 用户 ID
        :param image_path: 图像文件路径
        :param emotion_class: 主情绪类别
        :param confidence: 置信度
        :param emotions: 全情绪概率分布
        :param keywords: 关键词列表
        :param face_count: 人脸数量
        :param face_details: 人脸细节列表
        :return: 图像识别结果对象
        """
        emotion_result = EmotionResult(
            emotion_class=emotion_class,
            confidence=confidence,
            emotions=emotions,
            keywords=keywords,
            face_count=face_count,
            face_details=face_details or []
        )
        return RecognitionResult(
            user_id=user_id,
            recognition_type="image",
            emotion_result=emotion_result,
            created_at=datetime.now(),
            media_path=f"{image_path}"  # 严格遵循 resources/ 存储规范
        )

    @staticmethod
    def create_video_result(
            user_id: int,
            video_path: str,  # 存储于 resources/ 下
            emotion_class: str,
            confidence: float,
            emotions: Dict[str, float],
            keywords: List[str],
            face_count: int = 1,
            face_details: List[Dict] = None
    ) -> RecognitionResult:
        """
        创建视频识别结果对象。
        严格遵循 resources/ 的存储规范，将视频路径添加该前缀。
        :param user_id: 用户 ID
        :param video_path: 视频文件路径
        :param emotion_class: 主情绪类别
        :param confidence: 置信度
        :param emotions: 全情绪概率分布
        :param keywords: 关键词列表
        :param face_count: 人脸数量
        :param face_details: 人脸细节列表
        :return: 视频识别结果对象
        """
        emotion_result = EmotionResult(
            emotion_class=emotion_class,
            confidence=confidence,
            emotions=emotions,
            keywords=keywords,
            face_count=face_count,
            face_details=face_details or []
        )
        return RecognitionResult(
            user_id=user_id,
            recognition_type="video",
            emotion_result=emotion_result,
            created_at=datetime.now(),
            media_path=f"{video_path}"  # 严格遵循 resources/ 存储规范
        )