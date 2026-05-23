"""
微表情分析模型训练管道

功能概述：
- 使用LightGBM实现微表情识别
- 提取面部关键点几何特征
- 支持数据增强和特征选择
- 包含完整的数据预处理和模型评估流程

依赖库：
    - 标准库: os
    - 图像处理: cv2, dlib, imutils
    - 数值计算: numpy
    - 机器学习: lightgbm, sklearn
    - 数据可视化: matplotlib

编写日期：2025年07月
班级：物联一班
学号：202378040109
作者：李正标
"""
# 标准库导入
import os  # 提供与操作系统交互的功能，用于文件路径操作和删除无效文件
# 图像处理相关库
import cv2  # OpenCV库，用于图像读取、灰度转换及人脸检测
import dlib  # 用于人脸检测和68个关键点定位的深度学习库
from imutils import face_utils  # 辅助处理dlib的人脸关键点坐标格式
# 数值计算与数据处理
import numpy as np  # 提供高效的多维数组运算支持
import pandas as pd  # 虽未直接使用，但常用于结构化数据加载与处理
# 机器学习与模型构建
import lightgbm as lgb  # 高效梯度提升决策树（LightGBM）实现，适用于分类任务
# 数据增强与可视化
import albumentations as aa  # 强大的图像增强库，用于训练时扩充数据集，提高模型泛化能力
import matplotlib.pyplot as plt  # 绘图工具，用于可视化特征重要性等信息
# Scikit-learn 模块导入 - 用于模型评估、预处理、交叉验证和类别权重计算
from sklearn.preprocessing import StandardScaler  # 特征标准化工具，用于归一化处理
from sklearn.utils.class_weight import compute_class_weight  # 处理类别不平衡问题，为不同类别分配权重
from sklearn.model_selection import StratifiedKFold, train_test_split  # 分层交叉验证与数据集划分策略
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # 模型性能评估指标
import multiprocessing as mp

# 输出 LightGBM 版本，确保版本为 4.6.0（LightGBM 最新稳定版）
print(f"LightGBM版本: {lgb.__version__}")  # 确保输出版本为 4.6.0 或以上
# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体以支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

def get_augmentations():
    """
    创建一组图像增强策略。
    返回:
        albumentations.Compose: 包含多个增强变换的组合对象
    """
    return aa.Compose([
        # 50% 概率水平翻转
        aa.HorizontalFlip(p=0.5),
        # 50% 概率旋转 ±10 度
        aa.Rotate(limit=10, p=0.5),
        # 30% 概率应用高斯模糊
        aa.GaussianBlur(blur_limit=(3, 7), p=0.3),
        # 30% 概率调整亮度/对比度
        aa.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3)
    ])
def _extract_geometric_features(shape_np, face_width):
    """
    提取人脸关键点的几何特征。
    优化亮点：
    - 使用向量化矩阵运算替代双重循环，显著提升计算效率
    - 明确定义面部区域特征，增强特征提取的可解释性
    - 增加归一化处理，提升模型鲁棒性和泛化能力
    参数:
        shape_np (np.array): 面部关键点坐标数组 (68x2)
        face_width (float): 面部宽度（用于特征归一化）
    返回:
        list: 包含所有提取的几何特征值列表
    """
    features = []  # 初始化特征列表
    # 向量化计算所有点对之间的欧几里得距离
    diff_matrix = shape_np[:, np.newaxis, :] - shape_np[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff_matrix, axis=2) / face_width
    # 提取上三角矩阵非对角线元素作为全局特征
    triu_indices = np.triu_indices_from(dist_matrix, k=1)
    features.extend(dist_matrix[triu_indices])
    # 定义面部关键区域索引范围
    JAW = slice(0, 17)  # 下巴轮廓索引范围 [0, 16]
    NOSE = slice(27, 36)  # 鼻子区域索引范围 [27, 35]
    MOUTH_WIDTH = [48, 54]  # 嘴宽关键点索引
    MOUTH_HEIGHT = [51, 57]  # 嘴高关键点索引
    EYE_LEFT = slice(36, 42)  # 左眼区域索引范围 [36, 41]
    EYE_RIGHT = slice(42, 48)  # 右眼区域索引范围 [42, 47]
    BROW_LEFT = slice(17, 22)  # 左眉区域索引范围 [17, 21]
    BROW_RIGHT = slice(22, 27)  # 右眉区域索引范围 [22, 26]
    # 提取眼睛特征
    left_eye = shape_np[EYE_LEFT]  # 获取左眼关键点坐标
    right_eye = shape_np[EYE_RIGHT]  # 获取右眼关键点坐标
    eye_height_left = np.linalg.norm(left_eye[1] - left_eye[5])  # 左眼高度（垂直方向）
    eye_width_left = np.linalg.norm(left_eye[0] - left_eye[3])  # 左眼宽度（水平方向）
    eye_height_right = np.linalg.norm(right_eye[1] - right_eye[5])  # 右眼高度
    eye_width_right = np.linalg.norm(right_eye[0] - right_eye[3])  # 右眼宽度
    # 提取嘴巴特征
    mouth_width = np.linalg.norm(shape_np[MOUTH_WIDTH[0]] - shape_np[MOUTH_WIDTH[1]])  # 嘴宽
    mouth_height = np.linalg.norm(shape_np[MOUTH_HEIGHT[0]] - shape_np[MOUTH_HEIGHT[1]])  # 嘴高
    # 提取眉毛特征
    left_eyebrow = shape_np[BROW_LEFT]  # 左眉关键点
    right_eyebrow = shape_np[BROW_RIGHT]  # 右眉关键点
    eyebrow_height_left = np.mean([np.linalg.norm(p - left_eye[1]) for p in left_eyebrow])  # 左眉到左眼高度
    eyebrow_height_right = np.mean([np.linalg.norm(p - right_eye[1]) for p in right_eyebrow])  # 右眉到右眼高度
    # 新增鼻子和下巴特征
    nose_points = shape_np[NOSE]  # 鼻子关键点
    nose_width = np.linalg.norm(nose_points[0] - nose_points[4])  # 鼻翼宽度
    jaw_points = shape_np[JAW]  # 下巴轮廓点
    jaw_width = np.linalg.norm(jaw_points[0] - jaw_points[16])  # 下巴最宽处
    # 添加结构化特征并进行归一化处理
    features.extend([
        eye_height_left / eye_width_left,  # 左眼纵横比
        eye_height_right / eye_width_right,  # 右眼纵横比
        mouth_width / face_width,  # 嘴宽与面部宽度比例
        mouth_height / face_width,  # 嘴高与面部宽度比例
        eyebrow_height_left / face_width,  # 左眉高度比例
        eyebrow_height_right / face_width,  # 右眉高度比例
        nose_width / face_width,  # 鼻宽比例
        jaw_width / face_width  # 下巴宽度比例
    ])
    # 新增左右对称性特征
    left_features = [
        eye_height_left / eye_width_left,  # 左眼纵横比
        eyebrow_height_left / face_width  # 左眉高度比例
    ]
    right_features = [
        eye_height_right / eye_width_right,  # 右眼纵横比
        eyebrow_height_right / face_width  # 右眉高度比例
    ]
    symmetry_features = [abs(l - r) for l, r in zip(left_features, right_features)]  # 左右差异绝对值
    features.extend(symmetry_features)  # 添加对称性特征
    return features  # 返回最终特征列表


class MicroExpressionAnalyzer:
    """
    微表情分析器类，用于基于面部特征点进行微表情识别。
    包括以下功能：
        - 提取面部特征点
        - 加载数据集并提取特征
        - 训练 LightGBM 模型
        - 预测和评估模型性能
        - 保存/加载模型及其参数
        - 绘制特征重要性图
    """
    def __init__(self):
        """初始化模型组件及超参数"""
        self.model = None  # LightGBM 主模型
        self.ensemble_models = []  # 模型集成集合，用于交叉验证模型
        self.selected_features = None  # 被选中的特征索引
        self.feature_importance = None  # 特征重要性评分
        self.scaler = StandardScaler()  # 标准化器，用于特征缩放
        # LightGBM 模型超参数配置
        self.model_params = {
            'verbose': -1,  # 控制输出信息级别，-1 表示不输出日志信息，减少训练时的冗余输出
            'max_depth': 7,  # 设置树的最大深度，增加深度可以提升模型表达能力，但可能导致过拟合
            'num_class': 7,  # 输出类别数，对应七种微表情（愤怒、厌恶、恐惧、高兴、悲伤、惊讶、中性）
            'num_leaves': 63,  # 叶子节点最大数量，控制模型复杂度，值越大模型越复杂，但也更容易过拟合
            'lambda_l1': 0.2,  # L1 正则化系数，用于防止过拟合，提高模型泛化能力
            'lambda_l2': 0.2,  # L2 正则化系数，与 L1 一起使用可进一步增强正则化效果
            'bagging_freq': 5,  # bagging 频率，每 5 次迭代执行一次随机采样，有助于提升泛化能力并减少过拟合
            'random_state': 42,  # 固定随机种子，确保实验结果可重复
            'learning_rate': 0.03,  # 学习率，较小的学习率使模型训练更稳定，但可能需要更多迭代次数才能收敛
            'min_child_samples': 10,  # 子节点所需最小样本数，减少该值可让模型学习到更细粒度的模式，但也可能引入噪声
            'feature_fraction': 0.7,  # 每次迭代使用的特征比例，设置为 0.7 可以降低过拟合风险并提高模型泛化能力
            'bagging_fraction': 0.8,  # 每次迭代使用的数据比例，结合 bagging_freq 使用，能有效防止过拟合
            'boosting_type': 'gbdt',  # 提升类型选择梯度提升决策树，适用于大多数任务场景
            'objective': 'multiclass',  # 优化目标为多分类任务，与 num_class 配合定义具体类别数
            'metric': 'multi_logloss',  # 评估指标为多分类交叉熵损失函数，适合衡量分类模型的性能
        }
        # 初始化 dlib 面部关键点检测器
        try:
            # 构建模型文件路径
            model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'LightGBM')
            # 指定关键点检测模型文件路径
            predictor_model_path = os.path.join(model_dir, 'shape_predictor_68_face_landmarks.dat')
            # 检查模型目录是否存在
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"[错误] 模型目录不存在: {model_dir}")
            # 检查模型文件是否存在
            if not os.path.exists(predictor_model_path):
                raise FileNotFoundError(f"[错误] 关键点检测模型文件不存在: {predictor_model_path}")
            # 加载 dlib 的 shape_predictor 模型
            self.predictor = dlib.shape_predictor(predictor_model_path)
            print("[加载] dlib模型成功加载")
            self.predictor_model_path = predictor_model_path
        except Exception as e:
            # 捕获并打印初始化失败的错误信息，然后重新抛出异常
            print(f"[错误] 初始化dlib模型失败: {str(e)}")
            raise

    def extract_features(self, image_path, augment=False):
        """
        增强特征提取函数，支持数据增强
        参数:
            image_path (str): 图像文件路径
            augment (bool): 是否应用数据增强，默认为False
        返回:
            np.array: 提取的特征向量，若处理失败则返回None
        """
        # 使用OpenCV读取图像
        image = cv2.imread(image_path)
        # 检查图像是否成功加载
        if image is None:
            print(f"无法读取图像: {image_path}")
            # # 尝试删除无效文件
            try:
                os.remove(image_path)
                print(f"[删除] 图片已删除: {image_path}")
            except Exception as del_e:
                print(f"[错误] 删除图片失败 {image_path}: {str(del_e)}")
            return None
        # 应用数据增强（如果启用）
        if augment:
            # 获取预定义的增强策略
            transform = get_augmentations()
            # 对图像应用增强变换
            image = transform(image=image)['image']
        # 将图像转换为灰度图，用于后续人脸检测和关键点定位
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        try:
            # 使用OpenCV的Haar级联分类器进行人脸检测，调用OpenCV库预先训练好的Haar级联库检测面部
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            # 人脸检测参数优化：
            # scaleFactor: 图像缩放因子，值越小检测越细致但计算量增加（1.05表示每次缩小5%）
            # minNeighbors: 检测框保留阈值，值越大检测结果越稳定但可能漏检
            # minSize: 最小人脸尺寸，过滤过小的人脸区域
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,   # 缩放步长更精细
                minNeighbors=7,    # 增加检测准确性
                minSize=(40, 40)   # 过滤太小的人脸
            )
            if len(faces) == 0:
                # 如果默认分类器未检测到人脸，尝试使用替代分类器，调用OpenCV库预先训练好的Haar级联库检测人脸属性
                face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
                # 使用相同参数再次检测
                faces = face_cascade_alt.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=7,
                    minSize=(40, 40)
                )
                if len(faces) == 0:
                    print(f"[警告] 未检测到人脸: {image_path}")
                    # 删除无效图片以保持数据集整洁
                    os.remove(image_path)
                    print(f"[删除] 未检测到人脸的图片已删除: {image_path}")
                    return None
            # 选择面积最大的人脸区域作为目标
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            face_rect = dlib.rectangle(x, y, x + w, y + h)  # 转换为dlib矩形格式
            # 使用dlib的68点面部关键点检测器
            shape = self.predictor(gray, face_rect)
            if shape is None:
                print("[警告] 关键点检测失败")
                # 避免重复删除，仅当文件存在时才删除
                if os.path.exists(image_path):
                    os.remove(image_path)
                    print(f"[删除] 关键点检测失败的图片已删除: {image_path}")
                return None
            # 将关键点转换为numpy数组格式
            shape_np = face_utils.shape_to_np(shape)
            # 计算面部宽度（左下巴角到右下巴角的距离）
            face_width = np.linalg.norm(shape_np[16] - shape_np[0])
            # 提取几何特征
            features = _extract_geometric_features(shape_np, face_width)
            return np.array(features)
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            # 避免重复删除，仅当文件存在时才删除
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"[删除] 出错图片已删除: {image_path}")
            return None

    def load_data(self, data_dir, augment=False, augment_factor=2):
        """
        加载数据集并提取每张图像的特征向量和对应的标签。函数遍历指定目录结构为 data_dir/表情类别/图像文件 的数据集，对每张图像提取特征，并可选地进行数据增强以提升模型泛化能力。
        参数:
            data_dir (str): 数据集根目录路径，目录结构应为 data_dir/表情类别/图像文件
            augment (bool): 是否启用数据增强，默认为False
            augment_factor (int): 每张原始图像生成的增强样本数量，默认为2个
        返回:
            X (np.array): 提取的特征矩阵，形状为 (样本数, 特征维度)
            y (np.array): 对应的标签数组，形状为 (样本数,)
        """
        X = []  # 存储特征向量
        y = []  # 存储对应的表情类别标签

        # 遍历每个表情类别目录
        for emotion_idx, emotion in enumerate(os.listdir(data_dir)):
            emotion_dir = os.path.join(data_dir, emotion)  # 构建当前表情类别的路径
            if not os.path.isdir(emotion_dir):  # 跳过非目录的条目（如隐藏文件）
                continue

            print(f"加载表情: {emotion} ({emotion_idx})")  # 输出当前处理的表情类别信息

            # 遍历当前表情目录下的所有图像文件
            for image_name in os.listdir(emotion_dir):
                image_path = os.path.join(emotion_dir, image_name)  # 构建图像文件的完整路径

                if not os.path.exists(image_path):  # 如果文件不存在则跳过
                    print(f"图像文件 {image_path} 不存在")
                    continue

                # 提取原始图像的特征
                features = self.extract_features(image_path)  # 调用 extract_features 方法提取特征
                if features is not None:  # 如果特征提取成功
                    X.append(features)  # 将特征添加到特征列表
                    y.append(emotion_idx)  # 将对应的类别索引添加到标签列表

                    # 数据增强：如果启用增强，则生成多个增强样本
                    if augment:
                        for _ in range(augment_factor):  # 循环生成指定数量的增强样本
                            aug_features = self.extract_features(image_path, augment=True)  # 提取增强后的特征
                            if aug_features is not None:  # 如果增强特征提取成功
                                X.append(aug_features)  # 添加增强后的特征
                                y.append(emotion_idx)  # 标签与原图一致

        return np.array(X), np.array(y)  # 将列表转换为 numpy 数组返回

    def train(self, X, y, test_size=0.2, early_stopping_rounds=20, num_boost_round=500, n_splits=5):
        """增强训练函数，支持交叉验证和特征选择优化"""
        # 计算类别权重
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        sample_weights = np.array([class_weights[i] for i in y])
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        # 特征选择优化 - 使用所有特征进行初始训练
        self.selected_features = np.arange(X.shape[1])
        # 使用交叉验证训练多个模型
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_models = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
            print(f"\n训练折叠 {fold_idx + 1}/{n_splits}")
            # 划分训练集和验证集
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            # 根据类别权重获取当前fold的样本权重
            w_train = sample_weights[train_idx]
            # 创建LightGBM数据集
            train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            print("开始训练...")
            model = lgb.train(
                params=self.model_params,  # 使用初始化时配置的超参数
                train_set=train_data,      # 训练数据集
                num_boost_round=num_boost_round,  # 最大迭代次数
                valid_sets=[valid_data],         # 验证集
                callbacks=[
                    # 早停机制：如果验证集loss在指定轮数内无改进则停止训练
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                    # 日志记录：每50轮输出一次训练信息
                    lgb.log_evaluation(period=50)
                ],
            )
            # 保存当前fold的模型
            fold_models.append(model)
            print(f"折叠 {fold_idx + 1} 最佳迭代: {model.best_iteration}")
        # 保存所有模型用于集成
        self.ensemble_models = fold_models
        # 基于特征重要性选择特征
        feature_importance = np.zeros(X.shape[1])
        # 计算特征重要性（基于信息增益）
        for model in fold_models:
            # 获取当前模型特征重要性
            importance = model.feature_importance(importance_type='gain')
            # 累加特征重要性，考虑可能的维度差异
            feature_importance[:len(importance)] += importance
        self.feature_importance = feature_importance  # 保存全局特征重要性
        # 按重要性排序特征索引（从高到低）
        sorted_idx = np.argsort(feature_importance)[::-1]  
        # 选择前500个最重要的特征
        n_features = min(500, X.shape[1])
        self.selected_features = sorted_idx[:n_features]
        print(f"选择了前 {n_features} 个最重要特征")
        # 使用选定特征训练最终模型
        X_selected = X_scaled[:, self.selected_features]
        X_train, X_val, y_train, y_val = train_test_split(
            X_selected, y, test_size=test_size, stratify=y, random_state=42
        )
        # 重新计算训练集的样本权重
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        sample_weights_train = np.array([class_weights[i] for i in y_train])
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        self.model = lgb.train(
            params=self.model_params,
            train_set=train_data,
            num_boost_round=num_boost_round,
            valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=50)
            ],
        )
        print(f"最终模型最佳迭代次数: {self.model.best_iteration}")
        return self.model

    def predict(self, X, use_ensemble=False):
        """
        使用训练好的模型对输入数据进行预测。

        参数:
            X (np.array): 输入特征矩阵
            use_ensemble (bool): 是否使用模型集成进行预测

        返回:
            y_pred (np.array): 预测的类别标签
            y_pred_proba (np.array): 预测的概率分布
        """
        if self.model is None and not use_ensemble:
            raise Exception("模型未训练，请先调用train()方法")

        X_scaled = self.scaler.transform(X)
        X_selected = X_scaled[:, self.selected_features]

        if use_ensemble and len(self.ensemble_models) > 0:
            # 使用模型集成进行预测
            proba_sum = np.zeros((X_selected.shape[0], 7))

            for model in self.ensemble_models:
                # 确保每个模型使用相同的特征选择
                X_model = X_scaled[:, self.selected_features]
                proba = model.predict(X_model)
                proba_sum += proba

            y_pred_proba = proba_sum / len(self.ensemble_models)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            # 使用单一模型预测
            y_pred_proba = self.model.predict(X_selected)
            y_pred = np.argmax(y_pred_proba, axis=1)

        return y_pred, y_pred_proba

    def evaluate(self, X, y, class_names=None, use_ensemble=False):
        """
        评估模型性能，包括准确率、分类报告和混淆矩阵。

        参数:
            X (np.array): 测试数据特征矩阵
            y (np.array): 测试数据真实标签
            class_names (list): 类别名称列表
            use_ensemble (bool): 是否使用模型集成进行预测

        返回:
            accuracy (float): 准确率
        """
        y_pred, _ = self.predict(X, use_ensemble=use_ensemble)

        # 计算准确率
        accuracy = accuracy_score(y, y_pred)
        print(f"准确率: {accuracy:.4f}")

        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(y, y_pred, target_names=class_names))

        # 打印混淆矩阵
        cm = confusion_matrix(y, y_pred)
        print("\n混淆矩阵:")
        print(cm)

        return accuracy

    def save_model(self, model_path):
        """
        保存训练好的模型及相关参数到文件。

        参数:
            model_path (str): 模型保存路径
        """
        if self.model is not None:
            model_dir = os.path.dirname(model_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)  # 创建目标目录（如果不存在）

            self.model.save_model(model_path)  # 保存模型
            print(f"模型已保存到: {model_path}")

            # 保存标准化器参数
            np.save(os.path.join(model_dir, 'scaler_mean.npy'), self.scaler.mean_)
            np.save(os.path.join(model_dir, 'scaler_scale.npy'), self.scaler.scale_)
            print(f"标准化器参数已保存")

            # 保存特征选择索引
            if self.selected_features is not None:
                np.save(os.path.join(model_dir, 'selected_features.npy'), self.selected_features)
                print(f"特征选择索引已保存")

    def load_model(self, model_path):
        """
        从文件加载模型及相关的标准化器和特征选择索引。

        参数:
            model_path (str): 模型文件路径
        """
        if os.path.exists(model_path):
            self.model = lgb.Booster(model_file=model_path)  # 加载模型
            print(f"模型已从 {model_path} 加载")

            # 加载标准化器参数
            model_dir = os.path.dirname(model_path)
            mean_path = os.path.join(model_dir, 'scaler_mean.npy')
            scale_path = os.path.join(model_dir, 'scaler_scale.npy')
            features_path = os.path.join(model_dir, 'selected_features.npy')

            if os.path.exists(mean_path) and os.path.exists(scale_path):
                self.scaler.mean_ = np.load(mean_path)
                self.scaler.scale_ = np.load(scale_path)
                print(f"标准化器参数已加载")
            else:
                print("警告: 未找到标准化器参数，预测时可能需要重新训练标准化器")

            if os.path.exists(features_path):
                self.selected_features = np.load(features_path)
                print(f"特征选择索引已加载")
            else:
                print("警告: 未找到特征选择索引，预测时可能会出错")
        else:
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")

    def plot_feature_importance(self, top_n=20):
        """
        绘制 LightGBM 模型的特征重要性图。

        参数:
            top_n (int): 要显示的前 N 个重要特征
        """
        if self.model is None:
            raise Exception("模型未训练，请先调用train()方法")

        lgb.plot_importance(self.model, max_num_features=top_n)  # 绘制特征重要性图
        plt.xlabel('特征重要性')  # x轴标签
        plt.ylabel('特征')  # y轴标签
        plt.title('特征重要性图')  # 图表标题
        plt.tight_layout()  # 自动调整布局

        fig_dir = os.path.join(os.getcwd(), 'plots')  # 图片保存目录
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)  # 创建目录（如果不存在）

        fig_path = os.path.join(fig_dir, '特征重要性图.png')  # 图片文件路径
        plt.savefig(fig_path)  # 保存图片
        plt.show()  # 显示图表

    def load_preprocessed_features(self, npy_path):
        """加载离线保存的特征文件"""
        data = np.load(npy_path, allow_pickle=True)
        return data['X'], data['y']

def main():
    analyzer = MicroExpressionAnalyzer()
    data_dir = "data"                     # 原始数据目录（7个小写英文子文件夹）
    feature_file = "model\\features.npz"         # 离线特征保存文件

    # 如果特征文件已存在，直接加载
    if os.path.exists(feature_file):
        print("找到离线特征文件，直接加载...")
        X, y = analyzer.load_preprocessed_features(feature_file)
    else:
        print("未找到离线特征文件，开始提取特征（单线程，耗时较长）...")
        # 调用原有的 load_data（含数据增强），提取特征
        X, y = analyzer.load_data(data_dir, augment=True, augment_factor=2)
        if len(X) > 0:
            # 保存特征到文件
            np.savez_compressed(feature_file, X=X, y=y)
            print(f"特征已保存到 {feature_file}，样本数: {len(X)}")
        else:
            print("未提取到任何特征，请检查数据目录")
            return

    print(f"数据加载完成，样本数: {len(X)}, 特征数: {X.shape[1]}")

    # 训练模型（5折交叉验证）
    print("开始训练模型...")
    analyzer.train(X, y, n_splits=5)

    # 评估模型（注意：这里传入类别名称以便报告显示）
    class_names = ['angry', 'disgust', 'fear', 'happy', 'natural', 'sad', 'surprised']
    print("评估模型性能...")
    analyzer.evaluate(X, y, class_names=class_names, use_ensemble=False)

    # 保存模型
    analyzer.save_model("model\\micro_expression_model.txt")

    # 绘制特征重要性
    analyzer.plot_feature_importance()


if __name__ == "__main__":
    main()