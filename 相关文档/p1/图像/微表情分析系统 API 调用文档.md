# 微表情分析系统 API 调用文档


## 一、系统概述
微表情分析系统基于LightGBM框架和68点人脸关键点检测技术，提供静态图片和动态视频的微表情识别功能，支持7类微表情（愤怒、厌恶、恐惧、高兴、中性、悲伤、惊讶）的概率预测及可视化输出。

核心功能模块：
- 静态图片处理器（`StaticImageProcessor`）：单张图片微表情分析
- 动态视频处理器（`VideoProcessor`）：视频流实时微表情检测


## 二、环境准备
### 2.1 依赖库要求
| 依赖库         | 版本要求   | 说明                     |
|----------------|------------|--------------------------|
| `lightgbm`     | ≥4.6.0     | 核心分类模型框架         |
| `opencv-python`| ≥4.11.0    | 图像处理基础库           |
| `dlib`         | ≥20.0.0    | 人脸关键点检测           |
| `imutils`      | ≥0.5.4     | 关键点格式转换工具       |
| `numpy`        | ≥1.24.3    | 数值计算基础             |
| `scikit-learn` | ≥1.7.0     | 特征标准化工具           |

### 2.2 模型文件
系统需加载以下模型文件（默认路径为`model/`目录）：
- `micro_expression_model.txt`：LightGBM预训练模型
- `scaler_mean.npy`/`scaler_scale.npy`：特征标准化参数
- `selected_features.npy`：筛选后的特征索引
- `shape_predictor_68_face_landmarks.dat`：68点人脸关键点模型


## 三、API 接口详情

### 3.1 静态图片分析接口

#### 3.1.1 类初始化
```python
from image_live_model import StaticImageProcessor

# 初始化处理器（自动加载模型资源）
processor = StaticImageProcessor()
```

#### 3.1.2 核心方法：`predict_image`
**功能**：分析单张图片的微表情，返回预测结果及可选的可视化图片。

**参数说明**：
| 参数名          | 类型   | 默认值   | 说明                     |
|-----------------|--------|----------|--------------------------|
| `image_path`    | str    | 必传     | 输入图片路径（支持jpg/png格式） |
| `save_path`     | str    | None     | 结果图片保存路径，为None则不保存 |
| `show_landmarks`| bool   | False    | 是否在结果图中显示68个面部关键点 |

**返回结果**：
```python
{
    "success": bool,          # 处理是否成功
    "error": str,             # 错误信息（success为False时非空）
    "emotions": {             # 各情绪概率（英文键名）
        "fear": str,          # 恐惧概率（保留8位小数）
        "anger": str,         # 愤怒概率
        "disgust": str,       # 厌恶概率
        "neutral": str,       # 中性概率
        "sadness": str,       # 悲伤概率
        "happy": str,         # 高兴概率
        "surprise": str       # 惊讶概率
    },
    "confidence": float,      # 最高概率值
    "face_count": int,        # 检测到的人脸数量
    "face_details": [         # 人脸详细信息列表
        {
            "bounding_box": [x, y, w, h],  # 人脸 bounding box
            "emotion": str                 # 预测的情绪（中文）
        }
    ],
    "emotion_class": str      # 最高概率对应的情绪（英文）
}
```

**调用示例**：
```python
result = processor.predict_image(
    image_path="test_image.jpg",
    save_path="result_image.jpg",
    show_landmarks=True
)
print(f"预测结果：{result['emotion_class']}，置信度：{result['confidence']:.2f}")
```


### 3.2 视频流分析接口

#### 3.2.1 类初始化
```python
from image_live_model import VideoProcessor

# 初始化处理器
processor = VideoProcessor()
```

#### 3.2.2 核心方法：`process_video`
**功能**：批量处理视频帧，输出每帧的微表情预测结果，支持结果视频保存。

**参数说明**：
| 参数名          | 类型   | 默认值   | 说明                     |
|-----------------|--------|----------|--------------------------|
| `video_path`    | str    | 必传     | 输入视频路径（支持mp4/avi格式） |
| `output_path`   | str    | None     | 结果视频保存路径，为None则不保存 |
| `skip_frames`   | int    | 5        | 跳帧间隔（每n+1帧处理1帧，降低计算量） |
| `show_landmarks`| bool   | False    | 是否在结果视频中显示面部关键点 |

**返回结果**：
```python
{
    "success": bool,              # 处理是否成功
    "error": str,                 # 错误信息
    "results": [                  # 每帧处理结果列表
        {
            "frame": int,         # 帧索引
            "success": bool,      # 该帧处理是否成功
            "emotions": dict,     # 同图片接口的emotions格式
            "confidence": float,  # 最高概率值
            "face_count": int,    # 该帧人脸数量
            "face_details": list, # 同图片接口的face_details格式
            "emotion_class": str  # 该帧预测的情绪（英文）
        }
    ],
    "total_frames": int,          # 视频总帧数
    "processed_frames": int       # 实际处理的帧数
}
```

**调用示例**：
```python
result = processor.process_video(
    video_path="test_video.mp4",
    output_path="result_video.avi",
    skip_frames=3,
    show_landmarks=False
)
print(f"总帧数：{result['total_frames']}，处理帧数：{result['processed_frames']}")
```


## 四、错误处理
| 错误类型                | 可能原因                          | 解决方案                          |
|-------------------------|-----------------------------------|-----------------------------------|
| 图片无法读取            | 路径错误或文件损坏                | 检查图片路径，确认文件可正常打开  |
| 未检测到有效人脸        | 图片/视频帧中无人脸或人脸模糊     | 调整拍摄角度，确保人脸清晰可见    |
| 模型文件缺失            | 模型路径错误或文件未部署          | 检查`model/`目录下是否存在完整模型文件 |
| 视频写入失败            | 输出路径不可写或编码器不支持      | 更换输出路径，使用.avi格式保存    |


## 五、返回字段说明
| 一级字段         | 二级字段         | 说明                                  |
|------------------|------------------|---------------------------------------|
| `success`        | -                | 操作是否成功，`True`/`False`          |
| `error`          | -                | 错误信息，成功时为空字符串            |
| `emotions`       | 英文情绪键名     | 对应情绪的概率（字符串格式，保留8位小数） |
| `confidence`     | -                | 最高概率的数值（0~1范围）             |
| `face_count`     | -                | 检测到的人脸数量                      |
| `face_details`   | `bounding_box`   | 人脸位置坐标（x,y,w,h）               |
|                  | `emotion`        | 中文情绪标签（如“高兴”）              |
| `emotion_class`  | -                | 最高概率对应的英文情绪标签（如“happy”） |


## 六、使用注意事项
1. 输入图片/视频需保证光线充足，避免人脸遮挡，以提高关键点检测精度；
2. 视频处理时，`skip_frames`建议设置为3~5（平衡速度与精度）；
3. 显示关键点（`show_landmarks=True`）会增加计算耗时，建议仅用于调试；
4. 模型支持的7类情绪标签对应关系：
   - 英文：fear/anger/disgust/neutral/sadness/happy/surprise
   - 中文：恐惧/愤怒/厌恶/中性/悲伤/高兴/惊讶
## 七、其他
**撰写人**：李正标