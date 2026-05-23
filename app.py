"""
功能描述：基于Flask的Web应用主程序，实现情绪识别系统的核心功能
编写日期：2025年07月
班级：物联一班
学号：202378040109
作者：李正标
"""
# 标准库导入
import os
import json
import sys
import logging
import datetime
from dotenv import load_dotenv

load_dotenv()
# 第三方库导入
import numpy as np     # 用于数值计算，支持多维数组
import torch           # PyTorch深度学习框架，用于模型推理
import cv2             # OpenCV计算机视觉库，用于图像/视频处理
# Flask核心模块导入
from flask import (
    Flask,           # Flask核心类，创建Web应用实例
    render_template, # 渲染HTML模板
    request,         # 处理HTTP请求对象
    redirect,        # 执行HTTP重定向
    url_for,         # 生成URL
    flash,           # 显示一次性消息提示
    session,         # 管理会话状态（基于加密的Cookie）
    jsonify,         # 将Python对象转换为JSON响应
    send_file,
    send_from_directory
)
from PIL import Image as PILImage
import io
# Python标准库装饰器模块
from functools import wraps  # 用于编写装饰器函数，保留原函数元数据
# 自然语言处理模型导入
from transformers import BertTokenizer, BertForSequenceClassification  # BERT系列模型相关类
# 本地数据库模块导入
from p0_MySQL.sql_base import DatabaseManager, DatabaseError  # 数据库连接管理类和异常定义
from p0_MySQL.sql_model import EmotionResult, RecognitionResult, RecognitionResultFactory  # 情绪识别结果相关数据模型
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from tempfile import NamedTemporaryFile
from apscheduler.schedulers.background import BackgroundScheduler

# ------------------------------ 应用初始化 ------------------------------
# 1. 定义PROJECT_ROOT（解决"PROJECT_ROOT未定义"问题）
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))  # 确保在使用前定义
sys.path.append(PROJECT_ROOT)

# 2. 初始化Flask应用，指定模板文件夹路径
app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "p2_web_frontend", "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "p2_web_frontend", "static")
)
app.secret_key = os.environ.get('FLASK_SECRET_KEY')
if not app.secret_key:
    raise RuntimeError("未设置环境变量 FLASK_SECRET_KEY，请在 .env 文件中配置")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 初始化数据库管理器
db_manager = DatabaseManager()

# 3. 验证数据库连接（通过公共方法间接使用_db_cursor，解决访问受保护成员问题）
try:
    # 调用DatabaseManager的公共方法验证连接（而非直接访问_db_cursor）
    if db_manager.test_connection():  # 新增test_connection方法
        logger.info("数据库连接成功")
        
    # 初始化定时任务调度器
    scheduler = BackgroundScheduler()
    
    # 定义清理过期令牌的任务函数
    def clean_expired_tokens():
        try:
            deleted_count = db_manager.delete_expired_share_tokens()
            logger.info(f"清理过期分享令牌 {deleted_count} 条")
        except Exception as e:
            logger.error(f"清理过期令牌失败: {str(e)}")

    # 添加定时任务：每天凌晨2点执行
    scheduler.add_job(
        clean_expired_tokens,
        'cron',
        hour=2,
        minute=0
    )
    
    # 启动定时任务调度器
    if not scheduler.running:
        scheduler.start()
        logger.info("定时任务已启动，将每天凌晨2点清理过期分享令牌")

except DatabaseError as e:
    logger.error(f"数据库连接失败: {str(e)}")
    raise  # 连接失败时终止应用



# ------------------------------ 装饰器与工具函数 ------------------------------
def login_required(f):
    """装饰器：限制未登录用户访问"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('请先登录')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def current_user():
    """获取当前登录用户信息"""
    user_id = session.get('user_id')
    if user_id:
        return db_manager.get_user_by_id(user_id)
    return None


# ------------------------------ 应用生命周期管理 ------------------------------
@app.teardown_appcontext
def close_db_connection(exception):
    """应用上下文结束时关闭数据库连接"""
    db_manager.close()

# ------------------------------ 认证模块（登录/注册） ------------------------------
@app.route('/')
def index():
    """首页重定向到登录页"""
    return redirect(url_for('login'))


@app.route('/favicon.ico')
def favicon():
    return '', 204


@app.route('/login', methods=['GET', 'POST'])
def login():
    """用户登录处理"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash('用户名和密码不能为空')
            return render_template('login.html')

        try:
            if db_manager.verify_password(username, password):
                # 验证成功，初始化会话
                user = db_manager.get_user_by_username(username)
                session['user_id'] = user['id']
                session['user_info'] = user
                flash('登录成功')
                return redirect(url_for('recognition'))
            else:
                flash('用户名或密码错误')
        except DatabaseError as e:
            logger.error(f"登录失败: {str(e)}")
            flash(f'登录异常: {str(e)}')

    return render_template('login.html')


# 恢复验证码发送路由
@app.route('/send-verification-code', methods=['POST'])
def send_verification_code():
    data = request.json
    email = data.get('email')  # 只接受邮箱
    if not email or '@' not in email:
        return jsonify({'code': 400, 'message': '请提供正确的邮箱地址'}), 400
    try:
        code = db_manager.generate_email_verification_code()
        success = db_manager.send_email_verification_code(email)  # 仅调用邮箱发送方法
        if success:
            return jsonify({'code': 200, 'message': '验证码发送成功'}), 200
        else:
            return jsonify({'code': 500, 'message': '验证码发送失败，请检查邮箱'}), 500
    except DatabaseError as e:
        return jsonify({'code': 500, 'message': f'发送失败: {str(e)}'}), 500


# 修正注册路由的验证码校验逻辑
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        email = request.form.get('email', '').strip()
        # phone = request.form.get('phone', '').strip()  # 注释掉手机注册字段
        verification_code = request.form.get('verification_code', '').strip()

        # 基础校验
        if not username or not password:
            flash('用户名和密码不能为空')
            return render_template('register.html', username=username), 400
        if password != confirm_password:
            flash('两次密码不一致')
            return render_template('register.html', username=username), 400

        # 强制使用邮箱注册
        register_type = 'email'
        contact = email = request.form.get('email', '').strip()  # 只保留邮箱字段
        if not email:
            flash('请提供邮箱地址')
            return render_template('register.html', username=username), 400

        # 验证码校验 - 只保留邮箱验证逻辑
        try:
            db_manager.verify_code(contact, verification_code)  # 使用邮箱验证码验证

            # 创建用户 - 移除手机相关参数
            user_id = db_manager.create_user(
                username=username,
                password=password,
                email=email  # 仅保留邮箱参数
            )
            # 清理验证码缓存 - 仅处理邮箱缓存
            if contact in db_manager.email_verification_cache:
                del db_manager.email_verification_cache[contact]

            flash('注册成功，请登录')
            return redirect(url_for('login'))
        except DatabaseError as e:
            flash(f'注册失败: {str(e)}')
            return render_template('register.html', username=username), 500
    return render_template('register.html')

# ------------------------------ 情绪识别模块 ------------------------------
@app.route('/recognition')
@login_required
def recognition():
    """情绪识别主页面"""
    return render_template('recognition.html')

@app.route('/api/text/analyze', methods=['POST'])
@login_required
def text_emotion():
    """文本情绪分析API"""
    data = request.json
    text = data.get('text', '').strip()
    if not text:
        return jsonify({"code": 400, "message": "文本不能为空"}), 400
    user = current_user()
    try:
        env_text_model = os.environ.get("TEXT_MODEL_PATH")
        if env_text_model:
            model_path = env_text_model if os.path.isabs(env_text_model) else os.path.abspath(os.path.join(PROJECT_ROOT, env_text_model))
        else:
            model_path = os.path.join(
                PROJECT_ROOT, "p1_models_train", "TextModel",
                "nlp_structbert_emotion_classification_chinese_base"
            )
        
        # 调用封装好的文本情感分析函数
        from p1_models_train.TextModel.text_model import analyze_text_emotion
        result = analyze_text_emotion(text, model_path)
        
        # 构建结果并保存历史
        emotion_result = EmotionResult(
            emotion_class=result["emotion_class"],
            confidence=result["confidence"],
            emotions=result["emotions"],
            keywords=result["keywords"],
            face_count=result["face_count"],
            face_details=result["face_details"]
        )
        recognition_result = RecognitionResultFactory.create_text_result(
            user_id=user['id'],
            text_content=text,
            emotion_class=result["emotion_class"],
            confidence=result["confidence"],
            emotions=result["emotions"],
            keywords=result["keywords"]
        )
        db_manager.save_recognition_history(recognition_result)
        
        # 返回标准化响应格式
        return jsonify({
            "code": 200,
            "data": {
                "label": result["emotion_class"],
                "confidence": float(result["confidence"]),
                "keywords": result["keywords"]
            }
        })
    except Exception as e:
        logger.error(f"文本分析失败: {str(e)}")
        return jsonify({"code": 500, "message": f"分析失败: {str(e)}"}), 500

@app.route('/api/resources/<path:filename>')
@login_required  # 仅登录用户可访问
def serve_resource(filename):
    """加载资源文件（图片/视频帧）"""
    if not filename:
        abort(404)
    filename = filename.replace('\\', '/')
    resource_dir = os.path.join(PROJECT_ROOT, "p3_resources")
    file_path = os.path.normpath(os.path.join(resource_dir, filename))
    if not file_path.startswith(os.path.normpath(resource_dir)):
        abort(403)
    if not os.path.exists(file_path) or os.path.isdir(file_path):
        abort(404)
    return send_file(file_path)



def process_media_emotion(file, file_type):
    """
    处理媒体文件的情绪分析核心函数
    支持图像和视频两种类型，实现文件存储、情绪分析及结果记录功能
    参数:
        file: 上传的媒体文件对象
        file_type: 文件类型('image'或'video')
    返回:
        JSON格式的响应结果
    """
    # 获取当前登录用户信息
    user = current_user()
    if not user:
        return jsonify({"code": 401, "message": "未登录"}), 401

    try:
        # 动态构建模块路径并导入
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, "p1_models_train", "LightGBM")
        sys.path.append(model_path)
        
        if file_type == "image":
            from p1_models_train.LightGBM.image_live_model import StaticImageProcessor
        elif file_type == "video":
            from p1_models_train.LightGBM.image_live_model import VideoProcessor

        # 构建用户专属的文件保存路径
        save_dir = os.path.join(
            PROJECT_ROOT, 
            "p3_resources", 
            user.get('username'),  
            f"{file_type}s"
        )
        
        # 创建目录（如果不存在）
        os.makedirs(save_dir, exist_ok=True)
        
        # 定义文件路径
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        file_extension = os.path.splitext(file.filename)[1]
        
        # 原始文件名：用户id+时间戳+原始文件名（仅视频保留原始文件名）
        original_filename = f"{user['id']}_{timestamp}"
        if file_type == "video":
            original_filename += f"_{file.filename}"
        original_filename += file_extension
        
        file_path = os.path.join(save_dir, original_filename)
        
        # 处理后文件名：用户id+'r'+时间戳
        processed_filename = f"{user['id']}_r{timestamp}"
        if file_type == "video":
            processed_filename += f"_{file.filename}"
        processed_filename += file_extension
        
        file_result_path = os.path.join(save_dir, processed_filename)
        
        # 保存上传文件
        file.save(file_path)
        
        # 记录文件保存日志
        file_size = os.path.getsize(file_path)
        logger.info(f"用户{user['id']}的{file_type}已保存: {original_filename}({timestamp})，大小: {file_size} bytes")
        
        # 初始化处理器并执行预测
        if file_type == "image":
            static_image_processor = StaticImageProcessor()
            result = static_image_processor.predict_image(
                file_path,
                file_result_path, 
                show_landmarks=True
            )
        elif file_type == "video":
            video_processor = VideoProcessor()
            result = video_processor.process_video(
                video_path=file_path,
                output_path=file_result_path,
                skip_frames=10,
                show_landmarks=True
            )
        if result.get('success'):
            if file_type == "video" and result.get('output_path'):
                file_result_path = result['output_path']
            logger.info(f"处理后{file_type}保存路径: {file_result_path}")
        else:
            if os.path.exists(file_path):
                os.remove(file_path)
            logger.warning(f"{file_type}处理失败: {result.get('error', '未知原因')}")
            return jsonify({
                "code": 500,
                "message": f"{file_type}处理失败: {result.get('error', '未知错误')}"
            }), 500
        # 修改为使用相对路径存储到数据库
        resource_dir = os.path.join(PROJECT_ROOT, "p3_resources")
        relative_original_path = os.path.relpath(file_path, resource_dir)
        relative_processed_path = os.path.relpath(file_result_path, resource_dir)
        
        # 提取结果数据并处理不同类型的返回结构
        if 'results' in result:  # 视频处理结果
            # 取第一帧的检测结果作为整体情绪判断
            first_frame = result['results'][0] if result['results'] else {}
            emotion_class = first_frame.get('emotion_class') or "unknown"
            confidence = first_frame.get('confidence') or 0.0
            emotions = first_frame.get('emotions') or {}
            keywords = first_frame.get('keywords') or []
            face_count = sum(1 for r in result['results'] if r.get('face_count', 0) > 0)
            face_details = [item for r in result['results'] 
                          for item in r.get('face_details', [])]
        else:  # 图片处理结果
            emotion_class = result.get('emotion_class') or "unknown"
            confidence = result.get('confidence') or 0.0
            emotions = result.get('emotions') or {}
            keywords = result.get('keywords') or []
            face_count = result.get('face_count', 0)
            face_details = result.get('face_details', [])
        
        # 新增：检查情绪类别是否为空，设置默认值
        if not emotion_class:
            emotion_class = "unknown"
            logger.warning("模型未返回有效情绪类别，使用默认值'unknown'")
            
        # 创建情绪分析结果对象
        emotion_result = EmotionResult(
            emotion_class=emotion_class,
            confidence=confidence,
            emotions=emotions,
            keywords=keywords,
            face_count=face_count,
            face_details=face_details,
        )

        # 根据文件类型创建对应的识别结果对象
        if file_type == "image":
            recognition_result = RecognitionResultFactory.create_image_result(
                user_id=user['id'],
                image_path=relative_original_path,
                emotion_class=emotion_result.emotion_class,
                confidence=emotion_result.confidence,
                emotions=emotion_result.emotions,
                keywords=emotion_result.keywords,
                face_count=emotion_result.face_count,
                face_details=emotion_result.face_details,
            )
        elif file_type == "video":
            recognition_result = RecognitionResultFactory.create_video_result(
                user_id=user['id'],
                video_path=relative_original_path,
                emotion_class=emotion_result.emotion_class,
                confidence=emotion_result.confidence,
                emotions=emotion_result.emotions,
                keywords=emotion_result.keywords,
                face_count=emotion_result.face_count,
                face_details=emotion_result.face_details,
            )

        # 将识别结果保存到数据库
        db_manager.save_recognition_history(recognition_result)

        # 返回标准化响应
        return jsonify({
            "code": 200,
            "data": {
                "label": emotion_class,
                "confidence": float(confidence) if confidence is not None else 0.0,
                "face_count": face_count,
                "face_details": face_details
            }
        })

    except Exception as e:
        # 捕获并记录异常
        logger.error(f"{file_type}分析失败: {str(e)}")
        return jsonify({"code": 500, "message": f"分析失败: {str(e)}"}), 500

@app.route('/api/video/upload', methods=['POST'])
@login_required
def upload_video():
    """接收前端录制的视频，保存并调用处理逻辑"""
    if 'video' not in request.files:
        return jsonify({"code": 400, "message": "未上传视频"}), 400
    return process_media_emotion(request.files['video'], 'video')

@app.route('/api/image/analyze', methods=['POST'])
@login_required
def image_emotion():
    """图像情绪分析API"""
    if 'image' not in request.files:
        return jsonify({"code": 400, "message": "未上传图像"}), 400
    return process_media_emotion(request.files['image'], 'image')

# ------------------------------ 历史记录与数据看板 ------------------------------
@app.route('/history')
@login_required
def history():
    """历史记录页面"""
    user_id = session['user_id']
    try:
        records = db_manager.get_user_recognition_history(user_id)
        # 解析情绪结果
        for record in records:
            try:
                record['result'] = json.loads(record.get('result', '{}'))
                record['emotion_class'] = record['result'].get('emotion_class', '未知')
            except json.JSONDecodeError:
                record['emotion_class'] = '解析失败'
        return render_template('history.html', recognition_history=records)
    except DatabaseError as e:
        logger.error(f"获取历史记录失败: {str(e)}")
        flash(f'获取记录失败: {str(e)}')
        return render_template('history.html', recognition_history=[])


@app.route('/api/history/delete/<int:record_id>', methods=['DELETE'])
@login_required
def delete_history(record_id):
    """
    删除指定的历史记录及其关联的媒体文件
    参数:
        record_id (int): 要删除的记录ID
    返回:
        JSON格式响应，包含以下字段：
        - code: 状态码(200成功, 403无权操作, 500服务器错误)
        - message: 操作结果描述信息
    """
    # 获取当前登录用户信息
    user = current_user()
    
    try:
        # 1. 获取记录详情（包含type_name）
        record = db_manager.get_recognition_by_id(record_id)
        if not record or record['user_id'] != user['id']:
            return jsonify({
                "code": 403,
                "message": "无权删除该记录"
            }), 403

        # 2. 执行数据库删除
        db_manager.delete_recognition_record(record_id, user['id'])
        
        # 3. 删除关联媒体文件（图片/视频）
        try:
            # 打印调试日志，确认类型和路径
            logger.info(f"删除记录类型: {record.get('type_name')}, 图片路径: {record.get('image_path')}, 视频路径: {record.get('video_path')}")
            
            if record['type_name'] == 'image' and record.get('image_path'):
                # 处理原始图片路径
                resource_dir = os.path.join(PROJECT_ROOT, "p3_resources")
                normalized_original = os.path.normpath(record['image_path'])
                full_original = os.path.join(resource_dir, normalized_original) if not os.path.isabs(normalized_original) else normalized_original

                # 生成处理后图片路径（根据命名规则逆推）
                dir_name = os.path.dirname(full_original)
                file_name = os.path.basename(full_original)
                processed_file_name = file_name.replace(f"{user['id']}_", f"{user['id']}_r", 1)
                full_processed = os.path.join(dir_name, processed_file_name)
                
                # 删除原始图片
                if os.path.exists(full_original):
                    os.remove(full_original)
                    logger.info(f"原始图片已删除: {full_original}")
                
                # 删除处理后图片
                if os.path.exists(full_processed):
                    os.remove(full_processed)
                    logger.info(f"处理后图片已删除: {full_processed}")
                else:
                    logger.warning(f"处理后图片不存在: {full_processed}")
            
            elif record['type_name'] == 'video' and record.get('video_path'):
                resource_dir = os.path.join(PROJECT_ROOT, "p3_resources")
                normalized_original = os.path.normpath(record['video_path'])
                full_original = os.path.join(resource_dir, normalized_original) if not os.path.isabs(normalized_original) else normalized_original
                dir_name = os.path.dirname(full_original)
                file_name = os.path.basename(full_original)
                processed_file_name = file_name.replace(f"{user['id']}_", f"{user['id']}_r", 1)
                full_processed = os.path.join(dir_name, processed_file_name)
                
                if os.path.exists(full_original):
                    os.remove(full_original)
                    logger.info(f"原始视频已删除: {full_original}")
                
                if os.path.exists(full_processed):
                    os.remove(full_processed)
                    logger.info(f"处理后视频已删除: {full_processed}")
                else:
                    avi_path = os.path.splitext(full_processed)[0] + '.avi'
                    if os.path.exists(avi_path):
                        os.remove(avi_path)
                        logger.info(f"处理后视频已删除: {avi_path}")
                    else:
                        logger.warning(f"处理后视频不存在: {full_processed}")
        
        except Exception as e:
            logger.warning(f"删除媒体文件失败: {str(e)}")  # 仅警告，不影响记录删除

        # 返回成功响应
        return jsonify({
            "code": 200,
            "message": "记录及关联文件删除成功"
        })
        
    except DatabaseError as e:
        # 记录数据库相关错误的日志
        logger.error(f"删除记录失败: {str(e)}")
        
        # 返回通用的服务器内部错误响应
        return jsonify({
            "code": 500,
            "message": f"删除失败: {str(e)}"
        }), 500

@app.route('/api/history_filter', methods=['GET'])
@login_required
def history_filter():
    try:
        params = {
            'user_id': session['user_id'],
            'time_range': request.args.get('timeRange', 'all'),
            'emotion_type': request.args.get('emotionType', 'all'),
            'data_type': request.args.get('dataType', 'all'),
            'limit': int(request.args.get('limit', 10)),
            'offset': int(request.args.get('offset', 0)),
        }
        # 筛选记录（带分页）
        filtered = db_manager.filter_recognition_history(**params)
        
        # 新增：解析每条记录的result字段，提取emotion_class
        for record in filtered:
            try:
                # 将result字段从JSON字符串转为字典
                record_result = json.loads(record.get('result', '{}'))
                # 提取emotion_class字段，默认设为'未知'
                record['emotion_class'] = record_result.get('emotion_class', '未知')
            except json.JSONDecodeError:
                # JSON解析失败时设置为固定值
                record['emotion_class'] = '解析失败'

        # 统计总数时移除limit和offset参数（这两个参数仅用于分页，不影响总数统计）
        count_params = params.copy()
        count_params.pop('limit', None)
        count_params.pop('offset', None)
        total = db_manager.count_recognition_history(**count_params)
        
        return jsonify({
            "code": 200,
            "data": filtered,
            "total": total
        })
    except Exception as e:
        logger.error(f"筛选记录失败: {str(e)}")
        # 明确返回错误码和错误信息，便于前端处理
        return jsonify({"code": 500, "message": f"筛选失败: {str(e)}"}), 500

# 下载报告接口
@app.route('/api/download_report/<int:record_id>', methods=['GET'])
@login_required
def download_report(record_id):
    user = current_user()
    try:
        # 1. 获取记录详情
        record = db_manager.get_recognition_by_id(record_id)
        if not record or record['user_id'] != user['id']:
            return jsonify({"code": 403, "message": "无权访问"}), 403
        
        # 2. 解析记录数据
        result = json.loads(record.get('result', '{}'))  # 情绪分析结果
        emotion_class = result.get('emotion_class', '未知')  # 主要情绪
        emotions = result.get('emotions', {})  # 所有情绪占比
        created_at = record.get('created_at', datetime.datetime.now()).strftime('%Y-%m-%d %H:%M:%S')  # 时间
        analysis_type = record.get('type_name', '未知')  # 分析类型（文本/图片/视频）
        username = user.get('username', '未知用户')  # 用户名
        
        # 从记录中获取原始文本（新增字段）
        original_text = record.get('text_content', '（无原始文本记录）')  # 原始输入文本

        # ========== 新增：注册中文字体 ==========
        # 修改字体路径为静态资源目录下的fonts文件夹
        font_path = os.path.join(
            PROJECT_ROOT, 
            "p2_web_frontend", 
            "static", 
            "fonts", 
            "simsun.ttc"
        )
        if not os.path.exists(font_path):
            raise FileNotFoundError(
                "未找到中文字体文件！请将 SimSun 字体（simsun.ttc）放入项目的 p2_web_frontend/static/fonts 目录。"
            )
        pdfmetrics.registerFont(TTFont('SimSun', font_path))  # 注册字体，命名为 SimSun

        # 3. 准备图片路径（源文件和处理后文件）
        resource_dir = os.path.join(PROJECT_ROOT, "p3_resources")
        image_paths = []
        if analysis_type == 'image':
            # 图片源文件和处理后文件
            orig_img_path = os.path.join(resource_dir, record.get('image_path', ''))
            orig_img_path = os.path.normpath(orig_img_path)  # 规范化
            
            proc_img_path = os.path.join(
                os.path.dirname(orig_img_path),
                os.path.basename(orig_img_path).replace(f"{user['id']}_", f"{user['id']}_r", 1)
            )
            proc_img_path = os.path.normpath(proc_img_path)  # 规范化
            if os.path.exists(orig_img_path):
                image_paths.append(("原始图片", orig_img_path))
            if os.path.exists(proc_img_path):
                image_paths.append(("处理后图片", proc_img_path))
        elif analysis_type == 'video':
            # 视频取首帧预览（假设结果中包含帧路径）
            frame_path = result.get('face_details', [{}])[0].get('frame_path', '')
            if frame_path:
                video_frame_path = os.path.join(resource_dir, frame_path)
                if os.path.exists(video_frame_path):
                    image_paths.append(("视频关键帧", video_frame_path))
        
        # ========== 替换原PDF生成逻辑 ==========
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
        from reportlab.lib.units import cm
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        
        with NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as temp_pdf:
            # 创建文档并设置样式
            doc = SimpleDocTemplate(temp_pdf.name, pagesize=A4)
            
            # ========== 修改样式：全局使用中文字体 ==========
            styles = getSampleStyleSheet()
            # 遍历所有样式，强制设置字体为 SimSun，并左对齐（可选）
            for style_name in styles.byName:
                style = styles.byName[style_name]
                style.fontName = 'SimSun'       # 替换为注册的中文字体
                style.alignment = TA_LEFT       # 文字左对齐（根据需求调整）
                style.leading = 18              # 行间距（可选优化）

            # 自定义标题样式 - 居中显示
            styles.add(ParagraphStyle(
                name='ReportTitle',
                fontName='SimSun',
                fontSize=24,
                alignment=TA_CENTER,  # 标题居中
                spaceAfter=24         # 标题与下文间距
            ))
            
            elements = []

            # 标题（系统名称）
            title = Paragraph("七情六欲情绪识别系统", styles['ReportTitle'])
            elements.append(title)

            # 基本信息
            elements.append(Paragraph(f"用户：{username}", styles['Normal']))
            elements.append(Paragraph(f"分析时间：{created_at}", styles['Normal']))
            elements.append(Paragraph(f"分析类型：{analysis_type}", styles['Normal']))
            elements.append(Paragraph(f"主要情绪：{emotion_class}", styles['Normal']))
            elements.append(Spacer(1, 0.5*cm))
            
            # 新增：原始文本（仅文本分析时显示）
            if analysis_type == 'text':
                elements.append(Paragraph("原始输入文本：", styles['Heading3']))
                elements.append(Paragraph(original_text, styles['Normal']))
                elements.append(Spacer(1, 0.5*cm))
            
            # 情绪占比
            elements.append(Paragraph("情绪占比：", styles['Heading2']))
            for emotion, ratio in emotions.items():
                elements.append(Paragraph(f"- {emotion}：{ratio*100:.2f}%", styles['Normal']))
            elements.append(Spacer(1, 1*cm))
            
            # 插入图片（源文件和处理后文件）
            for img_title, img_path in image_paths:
                elements.append(Paragraph(img_title, styles['Heading3']))
                try:
                    with PILImage.open(img_path) as pil_img:
                        if pil_img.mode in ('RGBA', 'LA', 'P'):
                            pil_img = pil_img.convert('RGB')
                        img_byte_arr = io.BytesIO()
                        pil_img.save(img_byte_arr, format='JPEG')
                        img_byte_arr.seek(0)
                        
                        # 2. 使用处理后的图片数据创建ReportLab Image对象
                        img = Image(img_byte_arr)
                        
                        # 3. 计算缩放尺寸
                        img.drawWidth = 15*cm
                        img.drawHeight = img.drawWidth * pil_img.height / pil_img.width  # 使用Pillow获取的尺寸
                        elements.append(img)
                        elements.append(Spacer(1, 0.5*cm))
                except Exception as e:
                    logger.warning(f"插入图片失败：{img_path}，错误：{str(e)}")
                    elements.append(Paragraph(f"[图片无法显示：{img_title}（{str(e)}）]", styles['Normal']))

            # 生成PDF到临时文件
            doc.build(elements)
            
            # 重置文件指针，确保可读取
            temp_pdf.seek(0)
            
            # 返回文件下载
            return send_file(
                temp_pdf.name,
                as_attachment=True,
                download_name=f"情绪识别报告_{record_id}.pdf",
                mimetype='application/pdf'
            )
        
    except Exception as e:
        logger.error(f"生成报告失败: {str(e)}")
        return jsonify({"code": 500, "message": f"下载失败: {str(e)}"}), 500
    finally:
        # 确保临时文件被删除（防御性处理）
        try:
            if 'temp_pdf' in locals() and os.path.exists(temp_pdf.name):
                os.remove(temp_pdf.name)
        except:
            pass

# 在分享记录相关路由处添加
@app.route('/api/generate_share_token/<int:record_id>', methods=['GET'])
@login_required
def generate_share_token(record_id):
    """生成分享记录的临时Token"""
    try:
        user = current_user()
        # 验证记录归属权
        record = db_manager.get_recognition_by_id(record_id)
        if not record or record['user_id'] != user['id']:
            return jsonify({"code": 403, "message": "无权生成分享链接"}), 403
        # 调用数据库管理器生成Token（有效期1小时）
        token = db_manager.generate_share_token(record_id, expires_in=3600)
        return jsonify({"code": 200, "token": token})
    except Exception as e:
        logger.error(f"生成分享Token失败: {str(e)}")
        return jsonify({"code": 500, "message": f"生成失败: {str(e)}"}), 500

@app.route('/dashboard')
@login_required
def dashboard():
    """数据看板页面"""
    user_id = session['user_id']
    try:
        records = db_manager.get_user_recognition_history(user_id)
        emotion_dist = {}  # 情绪分布统计
        trend_data = []    # 时间趋势数据

        for record in records:
            try:
                result = json.loads(record.get('result', '{}'))
                emotion = result.get('emotion_class')
                created_at = record.get('created_at')
                if not emotion or not created_at:
                    continue

                # 情绪分布
                emotion_dist[emotion] = emotion_dist.get(emotion, 0) + 1

                # 趋势数据（格式化时间）
                if isinstance(created_at, datetime.datetime):
                    time_str = created_at.strftime('%Y-%m-%d %H:%M')
                else:
                    time_str = str(created_at)[:16]
                trend_data.append({"time": time_str, "emotionValue": emotion})

            except json.JSONDecodeError as e:
                logger.warning(f"解析记录失败: {str(e)}")
                continue

        return render_template(
            'dashboard.html',
            emotion_distribution=emotion_dist,
            trend_data=trend_data
        )
    except Exception as e:
        logger.error(f"加载数据看板失败: {str(e)}")
        return render_template('dashboard.html', emotion_distribution={}, trend_data=[])

@app.route('/api/recognition_detail/<int:record_id>', methods=['GET'])
@login_required
def recognition_detail(record_id):
    """
    获取情绪识别记录详情的API接口
    URL参数:
        record_id (int): 需要查询的记录ID
    返回:
        JSON格式响应，包含以下字段：
        - code: 状态码(200成功, 404未找到, 500服务器错误)
        - message: 错误信息（仅当出错时出现）
        - data: 包含记录数据的对象
    """
    try:
        # 调用数据库管理器获取指定ID的记录
        user = current_user()
        record = db_manager.get_recognition_by_id(record_id)
        if not record or record['user_id'] != user['id']:
            return jsonify({"code": 403, "message": "无权访问该记录"}), 403

        if record:
            try:
                # 尝试将记录中的JSON字符串解析为Python对象
                # 如果result字段不存在或为空，则使用空字典作为默认值
                record['result'] = json.loads(record.get('result', '{}'))
                
                # 强制提取emotion_class，增加默认值
                record['emotion_class'] = record['result'].get('emotion_class', 'unknown')
            except json.JSONDecodeError:
                # JSON解析失败时设置emotion_class为解析失败
                record['emotion_class'] = '解析失败'
            
            # 返回成功响应，包含解析后的记录数据
            return jsonify({
                "code": 200,
                "data": record  # 返回完整的记录数据
            })
        
        else:
            # 如果未找到记录，返回404响应
            return jsonify({
                "code": 404,
                "message": "记录未找到"  # 提供清晰的错误信息
            }), 404
            
    except DatabaseError as e:
        # 捕获并记录数据库相关错误
        logger.error(f"获取记录详情失败: {str(e)}")
        
        # 返回500内部服务器错误响应
        return jsonify({
            "code": 500,
            "message": f"获取记录详情失败: {str(e)}"  # 包含具体的错误信息
        }), 500

# ------------------------------ 用户中心 ------------------------------
@app.route('/user')
@login_required
def user_center():
    """用户中心页面"""
    user = current_user()
    records = db_manager.get_user_recognition_history(user['id'])
    return render_template(
        'user.html',
        user_info=user,
        analysis_count=len(records),
        last_active_time=records[0]['created_at'] if records else '无'
    )


@app.route('/update-password', methods=['POST'])
@login_required
def update_password():
    """更新密码API"""
    data = request.json
    try:
        success = db_manager.update_password(
            user_id=session['user_id'],
            old_password=data.get('old_password'),
            new_password=data.get('new_password')
        )
        if success:
            return jsonify({"code": 200, "message": "密码更新成功"})
        return jsonify({"code": 400, "message": "旧密码错误或新密码不符合要求"})
    except DatabaseError as e:
        return jsonify({"code": 400, "message": str(e)}), 400


@app.route('/update-user-info', methods=['POST'])
@login_required
def update_user_info():
    try:
        if 'user_info' not in session:
            return jsonify({'code': 401, 'message': '请先登录'}), 401

        data = request.json
        user_id = session['user_info']['id']
        username = data.get('username')
        email = data.get('email')
        verification_code = data.get('verification_code')

        current_username = session['user_info']['username']
        current_email = session['user_info'].get('email')

        # 只有在用户名发生更改时才进行验证
        if username and username != current_username:
            if db_manager.check_username_exists(username, exclude_id=user_id):
                return jsonify({'code': 400, 'message': '用户名已存在'}), 400

        # 只有在邮箱发生更改时才进行验证
        if email and email != current_email:
            # 简单的邮箱格式验证（可以根据需要添加更复杂的验证）
            if '@' not in email or '.' not in email.split('@')[-1]:
                return jsonify({'code': 400, 'message': '邮箱格式不正确'}), 400

            # 验证验证码
            if verification_code:
                try:
                    db_manager.verify_code(email, verification_code)
                except DatabaseError as e:
                    return jsonify({'code': 400, 'message': str(e)}), 400
            else:
                return jsonify({'code': 400, 'message': '请输入验证码'}), 400

        # 更新用户信息
        updated = False
        if username and username != current_username:
            if db_manager.update_username(user_id, username):
                session['user_info']['username'] = username
                updated = True
                
        if email and email != current_email and verification_code:
            if db_manager.update_email(user_id, email, verification_code):
                session['user_info']['email'] = email
                updated = True
                # 从缓存中移除已使用的验证码
                if email in db_manager.email_verification_cache:
                    del db_manager.email_verification_cache[email]

        if updated:
            # 刷新用户信息
            session['user_info'] = db_manager.get_user_by_id(user_id)
            return jsonify({'code': 200, 'message': '更新成功'})
        else:
            return jsonify({'code': 400, 'message': '未检测到有效更新内容'}), 400

    except Exception as e:
        # 记录错误日志
        app.logger.error(f"更新用户信息失败: {str(e)}", exc_info=True)
        return jsonify({'code': 500, 'message': '服务器内部错误，请稍后重试'}), 500


@app.route('/share/<token>')
def share_record(token):
    """处理分享链接访问，验证令牌并展示记录详情"""
    try:
        # 1. 查询令牌信息（从share_tokens表）
        share_token = db_manager.get_share_token(token)
        if not share_token:
            return render_template('error.html', message="无效的分享链接"), 404

        # 2. 检查令牌是否过期
        current_time = datetime.datetime.now()
        if share_token['expires_at'] < current_time:
            return render_template('error.html', message="分享链接已过期"), 404

        # 3. 根据record_id查询识别记录
        record_id = share_token['record_id']
        record = db_manager.get_recognition_by_id(record_id)
        if not record:
            return render_template('error.html', message="分享的记录不存在"), 404

        # 4. 解析记录中的情绪结果（JSON字符串转字典）
        try:
            record['result'] = json.loads(record.get('result', '{}'))
            record['emotion_class'] = record['result'].get('emotion_class', '未知')
        except json.JSONDecodeError:
            record['emotion_class'] = '解析失败'

        # 新增：路径标准化处理
        # 处理图片路径中的反斜杠问题
        if record.get('image_path'):
            record['image_path'] = record['image_path'].replace('\\', '/')
        
        # 处理视频关键帧路径（假设result是JSON格式，含frame_path）
        if record.get('result') and isinstance(record['result'], dict):
            for detail in record['result'].get('face_details', []):
                if detail.get('frame_path'):
                    detail['frame_path'] = detail['frame_path'].replace('\\', '/')

        # 5. 渲染分享页面（需创建share.html模板）
        return render_template('share.html', record=record)

    except DatabaseError as e:
        logger.error(f"分享链接处理失败: {str(e)}")
        return render_template('error.html', message="服务器错误"), 500
# ------------------------------ 应用启动 ------------------------------
if __name__ == '__main__':
    app.run(
        host=os.environ.get("APP_HOST", "127.0.0.1"),
        port=int(os.environ.get("APP_PORT", "5000")),
        debug=os.environ.get("DEBUG", "false").lower() == "true"
    )
import atexit

@atexit.register
def shutdown_scheduler():
    try:
        scheduler.shutdown()
        logger.info("定时任务已停止")
    except NameError:
        pass
