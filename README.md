# EmotionRecognitionSystem
## 项目概述
EmotionRecognitionSystem是一个基于深度学习的多模态情绪识别平台，属于专业实训项目。该系统支持文本、图像、视频三种输入类型，可精准识别愤怒、厌恶、恐惧、喜悦、中性、悲伤、惊讶7类基础情绪，为用户提供多场景下的情绪分析服务，并具备用户交互、历史记录管理及数据可视化功能。

## 功能特点
1. **多模态情绪识别**
    - **文本情绪分析**：用户输入中文文本（最长100字），系统分析其中的情绪倾向，返回7类情绪标签及置信度、情绪概率分布。
    - **图片情绪识别**：用户上传含人脸的图片（支持常见图片格式），系统检测人脸并标注情绪标签，显示情绪置信度。
    - **实时视频识别**：用户开启摄像头，通过摄像头录制视频并上传进行情绪识别，显示当前情绪标签及情绪强度，处理后的视频中绘制人脸边界框及关键点。
2. **历史记录管理**：集中展示用户的文本、图像、视频情绪识别结果，支持按时间范围（全部/本周/本月）、情绪类型（7类标准情绪）、数据类型（文本/图像/视频）筛选记录，可分页浏览、查看详情、删除记录及下载报告。
3. **数据看板**：通过图表形式直观呈现用户的情绪分布规律和时间变化趋势，包括情绪分布统计柱状图和趋势分析散点图。
4. **用户中心**：用户可管理个人信息（修改用户名、邮箱）、进行安全设置（修改密码），查看情绪分析次数和最近活跃时间等。
5. **分享功能**：允许用户生成临时分享链接，将特定识别记录共享给他人，链接具有时效性（默认1小时）和权限控制。

## 使用说明
1. **系统启动**：运行app.py文件启动系统，命令如下：
   ```bash
   python app.py
   ```
2. **访问系统**：可在本地访问http://localhost:5000。
3. **用户操作**
    - **注册/登录**：新用户通过邮箱注册，输入用户名、密码、邮箱并获取验证码进行验证；注册成功后登录系统。
    - **情绪识别**：在情绪识别页面选择相应的识别类型（文本/图片/视频），按照提示进行操作，查看识别结果。
    - **历史记录管理**：在历史记录页面进行筛选、查看、删除、下载报告等操作。
    - **数据看板查看**：在数据看板页面查看情绪分布统计和趋势分析图表。
    - **个人信息管理**：在用户中心修改个人信息和密码。

## 项目结构
```
EmotionRecognitionSystem/
├── p0_MySQL/
│   ├── sql_base.py
│   ├── sql_model.py
│   └── __init__.py
├── p1_models_train/
│   ├── LightGBM/
│   │   ├── data/
│   │   │   ├── anger/
│   │   │   ├── disgust/
│   │   │   ├── fear/
│   │   │   ├── happy/
│   │   │   ├── neutral/
│   │   │   ├── sadness/
│   │   │   └── surprise/
│   │   ├── model/
│   │   │   ├── features.npz
│   │   │   ├── micro_expression_model.txt
│   │   │   ├── scaler_mean.npy
│   │   │   ├── scaler_scale.npy
│   │   │   ├── selected_features.npy
│   │   │   └── test_features.npz
│   │   ├── plots/
│   │   ├── image_live_model.py
│   │   └── LightGBM_MicroExpression_Analysis_Pipeline.py
│   └── TextModel/
│       ├── nlp_structbert_emotion_classification_chinese_base/
│       └── text_model.py
├── p2_web_frontend/
│   ├── static/
│   │   ├── css/
│   │   │   ├── font-awesome.css
│   │   │   └── style.css
│   │   ├── fonts/
│   │   │   ├── fontawesome-webfont.eot
│   │   │   ├── fontawesome-webfont.svg
│   │   │   ├── fontawesome-webfont.ttf
│   │   │   ├── fontawesome-webfont.woff
│   │   │   ├── fontawesome-webfont.woff2
│   │   │   ├── FontAwesome.otf
│   │   │   └── simsun.ttc
│   │   ├── img/
│   │   │   └── into.png
│   │   └── js/
│   │       ├── auth.js
│   │       ├── echarts.min.js
│   │       ├── jquery.min.js
│   │       └── text_emotion.js
│   └── templates/
│       ├── dashboard.html
│       ├── history.html
│       ├── login.html
│       ├── recognition.html
│       ├── register.html
│       ├── share.html
│       └── user.html
├── p3_resources/
│   └── {user_id}/
│       ├── images/
│       └── videos/
├── 相关文档/
│   ├── p0/
│   │   ├── emotion_recognition.md
│   │   ├── emotion_recognition.sql
│   │   └── 数据库操作说明.md
│   ├── p1/
│   │   ├── 图像/
│   │   │   ├── LigntGBM框架.txt
│   │   │   ├── 微表情分析系统 API 调用文档.md
│   │   │   └── 情绪映射与修正流程.drawio
│   │   └── 文本/
│   │       ├── 情绪映射与修正流程.drawio
│   │       ├── 文本情感分析 API 使用文档.md
│   │       └── 文本情绪分析模块文件结构.txt
│   ├── p2/
│   │   ├── 历史记录与分享功能说明文档.md
│   │   ├── 情绪识别功能说明文档.md
│   │   ├── 数据看板说明文档.md
│   │   ├── 注册功能说明文档.md
│   │   ├── 用户中心说明文档.md
│   │   ├── 登录功能说明文档.md
│   │   └── 页面模板.zip
│   ├── 技术框架说明文档.md
│   ├── 系统四层架构图.drawio
│   ├── 项目文件框架.txt
│   └── 项目系统依赖检测报告.md
├── app.py
├── README.md
├── requirements.txt
└── 部署流程.md
```

## 注意事项
1. 输入文本长度不超过100字，图片需包含清晰人脸且文件大小不超过5MB，视频识别依赖设备摄像头权限，建议在光线充足环境下使用。
2. 分享链接有效期为1小时，过期后无法访问，需重新生成。
3. 删除记录后，关联的图像/视频文件将被永久删除，无法恢复。
4. PDF报告生成依赖simsun.ttc字体，需确保该字体存在于p2_web_frontend/static/fonts目录。