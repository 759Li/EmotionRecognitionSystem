"""
功能描述：
    实现基于LightGBM的微表情识别模型应用模块，包含：
    - 静态图片表情分析
    - 视频流实时表情检测
    - dlib关键点特征提取
    - 特征标准化与选择处理

编写日期：2025年07月
班级：物联一班
学号：202378040109
作者：李正标
"""

import os
import cv2
import numpy as np
import lightgbm as lgb
from imutils import face_utils
import dlib
from sklearn.preprocessing import StandardScaler
import sys
from dotenv import load_dotenv

load_dotenv()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
# 从项目根目录开始导入目标模块
from p1_models_train.LightGBM.LightGBM_MicroExpression_Analysis_Pipeline import _extract_geometric_features
# 解决情绪标签？？？问题
from PIL import ImageFont, ImageDraw, Image
import logging
# 初始化日志器，与app.py保持一致的命名空间
logger = logging.getLogger(__name__)

# 确保LightGBM版本兼容性
assert lgb.__version__ >= "4.6.0", f"需LightGBM 4.6.0+，当前版本：{lgb.__version__}"


def put_chinese_text(img, text, position, font_size=20, color=(0, 255, 0)):
    """
    在图片上绘制中文（解决OpenCV默认字体不支持中文的问题）
    :param img: 原始图像（BGR格式）
    :param text: 要绘制的中文文本
    :param position: 绘制位置 (x, y)
    :param font_size: 字体大小
    :param color: 字体颜色 (B, G, R)
    :return: 绘制后的图像
    """
    # 转换为PIL图像（RGB格式）
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 加载中文字体（需确保字体文件存在，这里使用系统默认的SimHei字体）
    try:
        # 尝试加载系统中的SimHei字体（Windows系统通常自带）
        font = ImageFont.truetype("simhei.ttf", font_size)
    except IOError:
        # 如果找不到SimHei，尝试其他常见中文字体
        try:
            font = ImageFont.truetype("msyh.ttc", font_size)  # 微软雅黑
        except IOError:
            # 若仍找不到，提示用户安装字体
            raise FileNotFoundError("未找到支持中文的字体文件，请安装SimHei或微软雅黑字体")

    # 绘制文本
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))  # PIL的颜色是(R, G, B)

    # 转换回OpenCV格式（BGR）
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

class BaseMicroExpressionModel:
    """基础微表情模型类，封装共享资源和通用方法"""
    EMOTION_LABELS = ["愤怒", "厌恶", "恐惧", "高兴", "中性", "悲伤", "惊讶"]  # 与训练标签一致

    def __init__(self, model_dir=None):
        self.base_dir = os.path.dirname(__file__)
        self._project_root = os.path.abspath(os.path.join(self.base_dir, "..", ".."))

        env_model_dir = os.environ.get("IMAGE_MODEL_DIR")
        if model_dir:
            self.model_dir = model_dir
        elif env_model_dir:
            self.model_dir = env_model_dir if os.path.isabs(env_model_dir) else os.path.abspath(os.path.join(self._project_root, env_model_dir))
        else:
            self.model_dir = os.path.join(self.base_dir, "model")

        env_landmark = os.environ.get("LANDMARK_PATH")
        if env_landmark:
            self.landmark_path = env_landmark if os.path.isabs(env_landmark) else os.path.abspath(os.path.join(self._project_root, env_landmark))
        else:
            self.landmark_path = os.path.join(self.base_dir, "shape_predictor_68_face_landmarks.dat")
        # LightGBM模型文件路径
        self.model_path = os.path.join(self.model_dir, "micro_expression_model.txt")
        # 标准化器均值参数路径
        self.scaler_mean = os.path.join(self.model_dir, "scaler_mean.npy")
        # 标准化器方差参数路径
        self.scaler_scale = os.path.join(self.model_dir, "scaler_scale.npy")
        # 特征选择索引文件路径
        self.features_idx = os.path.join(self.model_dir, "selected_features.npy")

        # 核心组件初始化
        self.predictor = None  # dlib关键点检测器
        self.model = None  # LightGBM模型
        self.scaler = StandardScaler()  # 标准化器
        self.selected_features = None  # 筛选特征索引

        # 加载资源
        self._load_resources()

    def _load_resources(self):
        """加载模型和依赖资源（复用训练代码的模型路径逻辑）"""
        try:
            # 加载dlib关键点模型
            if not os.path.exists(self.landmark_path):
                raise FileNotFoundError(f"关键点模型缺失: {self.landmark_path}")
            self.predictor = dlib.shape_predictor(self.landmark_path)

            # 加载LightGBM模型
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件缺失: {self.model_path}")
            self.model = lgb.Booster(model_file=self.model_path)

            # 加载标准化参数
            # 注意：sklearn的StandardScaler需要mean_和scale_两个参数
            self.scaler.mean_ = np.load(self.scaler_mean)
            self.scaler.scale_ = np.load(self.scaler_scale)

            # 加载特征选择索引
            self.selected_features = np.load(self.features_idx)
            print(f"[初始化] 成功加载所有资源（特征维度: {len(self.selected_features)}）")

        except Exception as e:
            print(f"[初始化失败] {str(e)}")
            raise

    def _extract_features(self, image):
        """复用训练代码的特征提取逻辑（单帧处理）"""
        # 图像预处理 - 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 创建人脸检测器并检测人脸
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(40, 40))
        
        # 如果未检测到人脸返回None
        if len(faces) == 0:
            return None
        
        # 选取最大人脸（根据面积）
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        face_rect = dlib.rectangle(x, y, x + w, y + h)  # 构建dlib矩形区域

        # 关键点检测
        shape = self.predictor(gray, face_rect)
        shape_np = face_utils.shape_to_np(shape)  # 转换为numpy数组
        face_width = np.linalg.norm(shape_np[16] - shape_np[0])  # 计算面部宽度用于归一化
        # 几何特征提取（将列表转换为numpy数组）
        features_list = _extract_geometric_features(shape_np, face_width)
        return np.array(features_list)


class StaticImageProcessor(BaseMicroExpressionModel):
    """静态图片处理模块"""

    def predict_image(self, image_path, save_path=None, show_landmarks=False):
        """
        预测单张图片的微表情
        :param image_path: 图片路径
        :param save_path: 保存路径，为None则不保存
        :param show_landmarks: 是否显示面部68关键点
        :return: 预测结果字典
        """
        # 图像读取
        image = cv2.imread(image_path)
        if image is None:
            return {
                "success": False,
                "error": "图片无法读取",
                "emotions": {},
                "keywords": [],
                "confidence": 0.0,
                "face_count": 0,
                "face_details": [],
                "emotion_class": ""
            }

        # 特征提取
        features = self._extract_features(image)
        if features is None:
            return {
                "success": False,
                "error": "未检测到有效人脸",
                "emotions": {},
                "keywords": [],
                "confidence": 0.0,
                "face_count": 0,
                "face_details": [],
                "emotion_class": ""
            }

        # 特征预处理及预测
        try:
            # 特征标准化
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            # 选择重要特征
            features_selected = features_scaled[:, self.selected_features]

            # 模型预测
            probs = self.model.predict(features_selected)[0]
            # 获取预测标签
            pred_label = self.EMOTION_LABELS[np.argmax(probs)]

            # 构建情绪概率字典，使用英文键名
            emotion_probs = {
                'fear': probs[self.EMOTION_LABELS.index("恐惧")],
                'anger': probs[self.EMOTION_LABELS.index("愤怒")],
                'disgust': probs[self.EMOTION_LABELS.index("厌恶")],
                'neutral': probs[self.EMOTION_LABELS.index("中性")],
                'sadness': probs[self.EMOTION_LABELS.index("悲伤")],
                'happy': probs[self.EMOTION_LABELS.index("高兴")],
                'surprise': probs[self.EMOTION_LABELS.index("惊讶")]
            }

            # 获取人脸检测器并检测人脸
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(40, 40))

            # 初始化人脸详细信息
            face_details = []
            output_image = image.copy()  # 用于绘制结果的副本

            # 如果检测到人脸
            if len(faces) > 0:
                # 选取最大人脸（根据面积）
                (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                # 绘制边界框
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 获取情绪标签（英文转中文）
                emotion_map = {
                    'fear': '恐惧',
                    'anger': '愤怒',
                    'disgust': '厌恶',
                    'neutral': '中性',
                    'sadness': '悲伤',
                    'happy': '高兴',
                    'surprise': '惊讶'
                }
                emotion_label = emotion_map.get(next(k for k, v in emotion_probs.items() if v == np.max(probs)), "未知")
                # 仅添加边界框信息
                face_details = [
                    {
                        "bounding_box": [int(x) for x in [x, y, w, h]],
                        "emotion": emotion_label,
                    }
                ]
                # 绘制情绪标签
                label_text = f"{emotion_label} ({np.max(probs):.2f})"
                # cv2.putText(output_frame, label_text, (x, y - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # 替换为：使用支持中文的函数
                output_image = put_chinese_text(
                    output_image,
                    label_text,
                    position=(x, y - 10),  # 绘制位置
                    font_size=32,
                    color=(0, 255, 0)  # 绿色
                )

                # 如果需要显示关键点
                if show_landmarks:
                    # 重新检测关键点用于绘制
                    face_rect = dlib.rectangle(x, y, x + w, y + h)
                    shape = self.predictor(gray, face_rect)
                    shape_np = face_utils.shape_to_np(shape)

                    # 绘制68个关键点
                    for (x_point, y_point) in shape_np:
                        cv2.circle(output_image, (x_point, y_point), 2, (0, 0, 255), -1)


            # 保存处理后的图片
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, output_image)

            # 返回预测结果
            return {
                "success": True,
                "error": "",
                "emotions": {k: f"{v:.8f}" for k, v in emotion_probs.items()},
                "keywords": [],
                "confidence": float(np.max(probs)),
                "face_count": len(faces),  # 已检测到的人脸数量
                "face_details": face_details,
                "emotion_class": next(k for k, v in emotion_probs.items() if v == np.max(probs))
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"预测失败: {str(e)}",
                "emotions": {},
                "keywords": [],
                "confidence": 0.0,
                "face_count": 0,
                "face_details": [],
                "emotion_class": ""
            }


class VideoProcessor(BaseMicroExpressionModel):
    """动态视频处理模块"""

    def process_video(self, video_path, output_path=None, skip_frames=5, show_landmarks=False):
        """
        处理视频流并输出微表情分析结果
        :param video_path: 视频路径
        :param output_path: 结果视频保存路径（可选）
        :param skip_frames: 跳帧间隔（降低计算量）
        :param show_landmarks: 是否显示面部68关键点
        :return: 分析结果列表
        """
        # 初始化视频捕获对象
        cap = cv2.VideoCapture(video_path)
        # 检查视频是否成功打开
        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            return {
                "success": False,
                "error": "视频文件为空或不存在",
                "results": []
            }

        results = []  # 存储每帧的分析结果
        frame_idx = 0  # 当前帧索引
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数

        # 获取人脸检测器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # 初始化视频写入器
        actual_output_path = None
        video_writer = None
        if output_path:
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = max(fps, 1.0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            actual_output_path = os.path.splitext(output_path)[0] + '.avi'
            os.makedirs(os.path.dirname(actual_output_path), exist_ok=True)

            video_writer = cv2.VideoWriter(
                actual_output_path,
                fourcc,
                fps / (skip_frames + 1),
                (width, height)
            )

            if not video_writer.isOpened():
                logger.error(f"视频写入器初始化失败: {output_path}")
                return {
                    "success": False,
                    "error": "无法初始化视频写入器，请检查OpenCV是否支持MJPG编码器"
                }

        # 循环读取视频帧
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # 视频读取结束

            # 每隔skip_frames帧处理一次
            if frame_idx % (skip_frames + 1) != 0:
                frame_idx += 1
                continue

            # 提取当前帧特征
            features = self._extract_features(frame)
            # 初始化当前帧结果
            current_result = {
                "frame": frame_idx,
                "success": False,
                "emotions": {},
                "keywords": [],
                "confidence": 0.0,
                "face_count": 0,
                "face_details": [],
                "emotion_class": ""
            }

            output_frame = frame.copy()  # 用于绘制结果的副本
            face_details = []  # 存储当前帧的人脸信息

            # 如果成功提取特征
            if features is not None:
                try:
                    # 特征标准化处理
                    features_scaled = self.scaler.transform(features.reshape(1, -1))
                    # 选择重要特征
                    features_selected = features_scaled[:, self.selected_features]
                    # 模型预测
                    probs = self.model.predict(features_selected)[0]
                    # 获取预测标签
                    pred_label = self.EMOTION_LABELS[np.argmax(probs)]

                    # 构建情绪概率字典，使用英文键名
                    emotion_probs = {
                        'fear': probs[self.EMOTION_LABELS.index("恐惧")],
                        'anger': probs[self.EMOTION_LABELS.index("愤怒")],
                        'disgust': probs[self.EMOTION_LABELS.index("厌恶")],
                        'neutral': probs[self.EMOTION_LABELS.index("中性")],
                        'sadness': probs[self.EMOTION_LABELS.index("悲伤")],
                        'happy': probs[self.EMOTION_LABELS.index("高兴")],
                        'surprise': probs[self.EMOTION_LABELS.index("惊讶")]
                    }

                    # 转换为灰度图用于人脸检测
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # 检测人脸
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(40, 40))

                    # 如果检测到人脸
                    if len(faces) > 0:
                        # 对每个人脸进行处理
                        for (x, y, w, h) in faces:
                            # 仅添加边界框信息
                            face_details.append({"bounding_box": [int(x) for x in [x, y, w, h]]})

                            # 绘制边界框
                            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                            # 获取情绪标签（英文转中文）
                            emotion_map = {
                                'fear': '恐惧',
                                'anger': '愤怒',
                                'disgust': '厌恶',
                                'neutral': '中性',
                                'sadness': '悲伤',
                                'happy': '高兴',
                                'surprise': '惊讶'
                            }
                            emotion_label = emotion_map.get(
                                next(k for k, v in emotion_probs.items() if v == np.max(probs)), "未知")

                            # 绘制情绪标签
                            label_text = f"{emotion_label} ({np.max(probs):.2f})"
                            # cv2.putText(output_frame, label_text, (x, y - 10),
                            #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            # 替换为：使用支持中文的函数
                            output_frame = put_chinese_text(
                                output_frame,
                                label_text,
                                position=(x, y - 10),  # (x,y)是人脸框左上角坐标，-10让文字略高于人脸框
                                font_size=32,  # 设置字体大小为32
                                color=(0, 255, 0)  # 使用绿色绘制文本
                            )

                            # 如果需要显示关键点
                            if show_landmarks:
                                # 重新检测关键点用于绘制
                                face_rect = dlib.rectangle(x, y, x + w, y + h)
                                shape = self.predictor(gray, face_rect)
                                shape_np = face_utils.shape_to_np(shape)

                                # 绘制68个关键点
                                for (x_point, y_point) in shape_np:
                                    # 绘制面部关键点，红色实心圆
                                    # x_point, y_point: 当前关键点的坐标
                                    # 2: 圆的半径大小
                                    # (0, 0, 255): BGR颜色值（红色）
                                    # -1: 表示填充圆形
                                    cv2.circle(output_frame, (x_point, y_point), 2, (0, 0, 255), -1)

                    # 构建成功预测的结果
                    current_result.update({
                        "frame": frame_idx,
                        "success": True,
                        "error": "",
                        "emotions": {k: f"{v:.8f}" for k, v in emotion_probs.items()},
                        "keywords": [],
                        "confidence": float(np.max(probs)),
                        "face_count": len(faces),  # 检测到的人脸数量
                        "face_details": face_details,  # 包含所有人脸的详细信息
                        "emotion_class": next(k for k, v in emotion_probs.items() if v == np.max(probs))
                    })

                except Exception as e:
                    # 记录预测过程中的异常信息
                    current_result["error"] = str(e)

            # 添加当前帧结果到总结果中
            results.append(current_result)

            # 写入视频帧
            if video_writer and output_frame is not None:
                video_writer.write(output_frame)

            # 帧计数递增
            frame_idx += 1

        # 释放资源
        cap.release()  # 释放视频捕获对象
        if video_writer:
            video_writer.release()  # 释放视频写入器

        # 计算处理帧数
        processed_frames = len(results)

        # 返回最终分析结果
        return {
            "success": True,
            "error": "",
            "results": results,
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "output_path": actual_output_path
        }

# # ==========================================================测试==========================================================
# def main():
#     """
#     测试主函数：演示静态图片和视频的微表情分析用法
#     功能：
#     - 静态图片预测测试（支持保存结果）
#     - 动态视频处理测试（支持保存结果）
#     - 模型功能验证
#     """
#     import argparse
#     from datetime import datetime
#
#     # 创建命令行参数解析器
#     parser = argparse.ArgumentParser(description='微表情分析系统测试工具')
#     parser.add_argument('--image', type=str, help='测试图片路径')
#     parser.add_argument('--video', type=str, help='测试视频路径')
#     parser.add_argument('--image_out', type=str, help='处理后图片保存路径')
#     parser.add_argument('--video_out', type=str, help='处理后视频保存路径')
#     parser.add_argument('--show_landmarks', action='store_true', help='显示面部关键点')
#     parser.add_argument('--skip_frames', type=int, default=5, help='视频跳帧间隔')
#     args = parser.parse_args()
#     # 使用当前时间生成默认输出文件名
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#
#     # 示例1: 静态图片预测测试
#     if args.image or input("是否测试静态图片处理？(y/n): ").lower() == 'y':
#         try:
#             # 获取图片路径
#             image_path = args.image or input("请输入测试图片路径: ")
#             if not os.path.exists(image_path):
#                 print("图片文件不存在，请检查路径")
#                 return
#             # 获取保存路径
#             save_path = args.image_out or input("请输入保存路径(可选，直接回车跳过保存): ")
#             if save_path and not save_path.endswith(('.jpg', '.jpeg', '.png')):
#                 save_path += '.jpg'
#             # 初始化静态图片处理器
#             image_processor = StaticImageProcessor()
#             # 打印测试信息
#             print(f"[测试] 正在处理图片: {image_path}")
#             # 执行图片预测
#             image_result = image_processor.predict_image(
#                 image_path=image_path,
#                 save_path=save_path,
#                 show_landmarks=args.show_landmarks
#             )
#             # 处理预测结果
#             if image_result["success"]:
#                 # 成功情况下的结果展示
#                 print(f"检测结果:")
#                 print(f"  表情: {image_result['emotion_class']} (置信度: {image_result['confidence']:.2f})")
#                 print("  概率分布:")
#                 for label, prob in image_result["emotions"].items():
#                     print(f"    {label}: {float(prob):.4f}")
#                 # 显示保存信息
#                 if save_path:
#                     print(f"  处理后的图片已保存至: {save_path}")
#             else:
#                 # 失败情况下的错误提示
#                 print(f"图片处理失败: {image_result['error']}")
#         except Exception as e:
#             print(f"[错误] 图片处理发生异常: {str(e)}")
#
#     # 示例2: 动态视频预测测试
#     if args.video or input("是否测试视频处理？(y/n): ").lower() == 'y':
#         try:
#             # 获取视频路径
#             video_path = args.video or input("请输入测试视频路径: ")
#             if not os.path.exists(video_path):
#                 print("视频文件不存在，请检查路径")
#                 return
#             # 获取保存路径
#             save_path = args.video_out or input("请输入保存路径(可选，直接回车跳过保存): ")
#             if save_path and not save_path.endswith('.mp4'):
#                 save_path += '.mp4'
#             # 初始化视频处理器
#             video_processor = VideoProcessor()
#             # 打印测试信息
#             print(f"[测试] 正在处理视频: {video_path}")
#             # 导入时间模块用于计时
#             import time
#             # 记录开始时间
#             start_time = time.time()
#             # 执行视频处理
#             video_result = video_processor.process_video(
#                 video_path=video_path,
#                 output_path=save_path,
#                 skip_frames=args.skip_frames,
#                 show_landmarks=args.show_landmarks
#             )
#             # 记录结束时间
#             end_time = time.time()
#             # 计算并输出处理时间
#             processing_time = end_time - start_time
#             processed_frames = len(video_result.get('results', []))
#             if processed_frames > 0:
#                 print(f"\n性能测试结果:")
#                 print(f"  处理{processed_frames}帧所需时间: {processing_time:.4f} 秒")
#                 print(f"  平均每帧耗时: {(processing_time / processed_frames):.4f} 秒")
#             # 输出处理摘要信息
#             print("\n视频处理摘要:")
#             print(f"  总帧数: {video_result.get('total_frames', 0)}")
#             print(f"  处理帧数: {processed_frames}")
#             # 显示保存信息
#             if save_path and video_result.get('success', False):
#                 print(f"  处理后的视频已保存至: {save_path}")
#             # 显示结果示例
#             if video_result.get('results'):
#                 print(f"  结果示例: {video_result['results'][0] if video_result['results'] else '无结果'}")
#         except Exception as e:
#             print(f"[错误] 视频处理发生异常: {str(e)}")
#
# if __name__ == "__main__":
#     """
#     程序入口点
#     当脚本被直接运行时执行main()函数
#     """
#     main()