"""
数据库操作管理模块

功能说明：
- 提供用户管理（创建、查询、更新）
- 管理识别历史记录（保存、查询、筛选、删除）
- 实现验证码发送与验证（邮箱）
- 数据库连接管理（自动重连、事务支持）

安全特性：
- 密码BCrypt加密存储
- 验证码有效期控制
- 参数化SQL查询防止注入

依赖组件：
- pymysql：MySQL数据库驱动
- bcrypt：密码加密
- smtplib/email：邮件发送支持
- contextlib：上下文管理器支持

编写日期：2025年07月
班级：物联一班
学号：202378040109
作者：李正标
"""

import bcrypt
import random
import string
import os
import pymysql
import smtplib
import logging
import uuid
from email.mime.text import MIMEText
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dotenv import load_dotenv
from p0_MySQL.sql_model import RecognitionResult

load_dotenv()

class DatabaseError(Exception):
    """
    数据库操作异常类
    """
    pass
class DatabaseManager:
    """
    数据库操作管理类
    """
    def __init__(self):
        """
        初始化数据库连接
        采用项目指定的连接参数，使用DictCursor返回字典格式结果，关闭自动提交以支持事务
        """

        # 初始化日志记录器，设置日志级别为ERROR
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.ERROR)

        # 创建日志格式器和控制台处理器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)  # 将处理器添加到logger

        # 创建临时存储邮箱验证码的缓存字典
        # 存储结构：{邮箱: (验证码, 发送时间)}
        self.email_verification_cache = {}
        # 创建临时存储短信验证码的缓存字典
        # 存储结构：{手机号: (验证码, 发送时间)}
        self.phone_verification_cache = {}

        _db_host = os.environ.get("DB_HOST")
        _db_user = os.environ.get("DB_USER")
        _db_password = os.environ.get("DB_PASSWORD")
        _db_name = os.environ.get("DB_NAME")
        if not all([_db_host, _db_user, _db_password, _db_name]):
            missing = [k for k in ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"] if not os.environ.get(k)]
            raise DatabaseError(f"缺少必要的数据库环境变量: {', '.join(missing)}，请在 .env 文件中配置")

        try:
            self.conn = pymysql.connect(
                host=_db_host,
                port=int(os.environ.get("DB_PORT", "3306")),
                user=_db_user,
                password=_db_password,
                db=_db_name,
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=False
            )
        except pymysql.MySQLError as e:
            # 捕获数据库连接异常，记录错误日志并抛出自定义异常
            self.logger.error(f"数据库连接失败: {str(e)}")  # 打印具体错误信息
            raise DatabaseError(f"连接数据库失败: {str(e)}") from e  # 抛出自定义数据库异常

    def test_connection(self):
        """公共方法：测试数据库连接，避免外部直接访问_db_cursor"""
        try:
            with self._db_cursor():  # 内部使用受保护成员
                return True
        except DatabaseError:
            return False
    @contextmanager
    def _db_cursor(self):
        """数据库游标上下文管理器"""
        # 检查数据库连接是否存在或是否已关闭
        if not self.conn or not self.conn.open:
            # 尝试重新建立数据库连接
            try:
                self.conn = pymysql.connect(
                    host=os.environ.get("DB_HOST"),
                    port=int(os.environ.get("DB_PORT", "3306")),
                    user=os.environ.get("DB_USER"),
                    password=os.environ.get("DB_PASSWORD"),
                    db=os.environ.get("DB_NAME"),
                    cursorclass=pymysql.cursors.DictCursor,
                    autocommit=True
                )
            except pymysql.MySQLError as e:     # 捕获连接数据库时发生的错误
                # 抛出自定义数据库异常，并记录原始异常信息
                raise DatabaseError(f"数据库重连失败: {str(e)}") from e

        # 创建数据库游标对象
        cursor = self.conn.cursor()

        try:
            # 将游标对象提供给with语句内的代码使用
            yield cursor

            # 如果操作成功，提交事务
            self.conn.commit()

        except pymysql.MySQLError as e:         # 捕获数据库操作中的错误
            # 如果连接仍然打开，则执行回滚
            if self.conn.open:
                self.conn.rollback()

            # 抛出自定义数据库异常，并记录原始异常信息
            raise DatabaseError(f"数据库操作失败: {str(e)}") from e

        finally:
            # 不管操作是否成功，最后都要关闭游标
            cursor.close()

    # ========================== 用户管理操作 ==========================
    def create_user(self, username: str, password: str, phone_num: str = "", email: str = "") -> int:
        """创建新用户
        验证用户名/手机号/邮箱唯一性，密码BCrypt加密存储，返回新用户ID

        :param username: 用户名（唯一）
        :param password: 原始密码（需满足强度要求）
        :param phone_num: 手机号（可选，唯一）
        :param email: 邮箱（可选，唯一）
        :return: 新用户ID
        :raises DatabaseError: 当用户名/手机号/邮箱已存在或密码不符合要求时
        """
        # 唯一性验证
        if self.get_user_by_username(username):
            raise DatabaseError("用户名已存在")
        if phone_num and self.get_user_by_phone(phone_num):
            raise DatabaseError(f"手机号 {phone_num} 已被注册")
        if email and self.get_user_by_email(email):
            raise DatabaseError(f"邮箱 {email} 已被注册")

        # 密码强度验证（至少8位，包含字母和数字）
        if len(password) < 8 or not any(c.isalpha() for c in password) or not any(c.isdigit() for c in password):
            raise DatabaseError("密码强度不足：需至少8位且包含字母和数字")

        # 密码加密
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # 处理空值（数据库字段允许NULL）
        phone_num = phone_num if phone_num else None
        email = email if email else None

        # 执行插入
        with self._db_cursor() as cursor:
            cursor.execute("""
                INSERT INTO USERS (username, password_hash, phone_num, email)
                VALUES (%s, %s, %s, %s)
            """, (username, password_hash, phone_num, email))
            return cursor.lastrowid

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """通过用户ID查询用户信息

        :param user_id: 用户ID
        :return: 用户信息字典（含id, username等字段），不存在则返回None
        """
        with self._db_cursor() as cursor:
            cursor.execute("SELECT * FROM USERS WHERE id = %s", (user_id,))
            return cursor.fetchone()

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """通过用户名查询用户信息

        :param username: 用户名
        :return: 用户信息字典，不存在则返回None
        """
        with self._db_cursor() as cursor:
            cursor.execute("SELECT * FROM USERS WHERE username = %s", (username,))
            return cursor.fetchone()

    def get_user_by_phone(self, phone_num: str) -> Optional[Dict]:
        """通过手机号查询用户信息

        :param phone_num: 手机号
        :return: 用户信息字典，不存在则返回None
        """
        with self._db_cursor() as cursor:
            cursor.execute("SELECT * FROM USERS WHERE phone_num = %s", (phone_num,))
            return cursor.fetchone()

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """通过邮箱查询用户信息

        :param email: 邮箱地址
        :return: 用户信息字典，不存在则返回None
        """
        with self._db_cursor() as cursor:
            cursor.execute("SELECT * FROM USERS WHERE email = %s", (email,))
            return cursor.fetchone()
    
    # ========================== 用户信息更新 ==========================
    def update_username(self, user_id: int, new_username: str) -> bool:
        """更新用户名（需确保新用户名唯一）

        :param user_id: 用户ID
        :param new_username: 新用户名
        :return: 更新成功返回True，否则False
        :raises DatabaseError: 新用户名已存在时
        """
        if self.get_user_by_username(new_username):
            raise DatabaseError("新用户名已存在")

        with self._db_cursor() as cursor:
            cursor.execute(
                "UPDATE USERS SET username = %s WHERE id = %s",
                (new_username, user_id)
            )
            return cursor.rowcount > 0

    def update_password(self, user_id: int, old_password: str, new_password: str) -> bool:
        """更新密码（需验证旧密码）

        :param user_id: 用户ID
        :param old_password: 原始密码
        :param new_password: 新密码（需满足强度要求）
        :return: 更新成功返回True
        :raises DatabaseError: 旧密码验证失败或新密码不符合要求
        """
        # 验证旧密码
        user = self.get_user_by_id(user_id)
        if not user or not self.verify_password(user["username"], old_password):
            raise DatabaseError("旧密码验证失败")

        # 验证新密码强度
        if len(new_password) < 8 or not any(c.isalpha() for c in new_password) or not any(c.isdigit() for c in new_password):
            raise DatabaseError("新密码强度不足")

        # 加密新密码并更新
        new_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        with self._db_cursor() as cursor:
            cursor.execute(
                "UPDATE USERS SET password_hash = %s WHERE id = %s",
                (new_hash, user_id)
            )
            return cursor.rowcount > 0

    def update_email(self, user_id: int, new_email: str, verification_code: str) -> bool:
        """更新邮箱（需验证验证码，有效期60秒）

        :param user_id: 用户ID
        :param new_email: 新邮箱地址
        :param verification_code: 验证码
        :return: 更新成功返回True
        :raises DatabaseError: 验证码无效、过期或邮箱已被注册
        """
        # 验证新邮箱唯一性
        if self.get_user_by_email(new_email):
            raise DatabaseError("新邮箱已被注册")
            
        # 验证验证码
        self.verify_code(new_email, verification_code)
        
        # 执行更新
        with self._db_cursor() as cursor:
            cursor.execute(
                "UPDATE USERS SET email = %s WHERE id = %s",
                (new_email, user_id)
            )
            del self.email_verification_cache[new_email]  # 验证成功后清理缓存
            return cursor.rowcount > 0

    # ========================== 识别历史管理 ==========================
    def get_recognition_by_id(self, record_id):
        """
        根据记录ID查询单条识别记录，关联recognition_types表获取type_name
        参数：
            record_id (int): 要查询的记录ID
        返回:
            dict: 包含识别记录及其类型名称(type_name)的字典对象，
                  如果未找到记录则返回None
        异常:
            DatabaseError: 当数据库查询过程中发生错误时抛出
        """
        try:
            # 使用数据库游标上下文管理器自动处理资源释放
            with self._db_cursor() as cursor:
                # 执行SQL查询，关联recognition_history和recognition_types表
                # 查询字段包含recognition_history表的所有字段以及recognition_types.type_name
                cursor.execute("""
                    SELECT rh.*, rt.type_name 
                    FROM recognition_history rh
                    JOIN recognition_types rt ON rh.type_id = rt.id
                    WHERE rh.id = %s  -- 按照传入的record_id进行过滤
                """, (record_id,))  # 参数化查询防止SQL注入
                # 获取单条查询结果
                record = cursor.fetchone()
                # 返回查询结果（如果没有找到记录则返回None）
                return record
        except Exception as e:
            # 捕获所有异常并包装为自定义DatabaseError，提供更清晰的错误信息
            raise DatabaseError(f"查询记录失败：{str(e)}") from e  # 保留原始异常上下文
    def get_latest_recognition_history(self, user_id, recognition_type_name):
        """
        获取用户最近的识别历史记录
        :param user_id: 用户ID
        :param recognition_type_name: 识别类型名称(如"video", "image")
        :return: 最近的识别记录，如果找不到则为None
        """
        try:
            with self._db_cursor() as cursor:
                cursor.execute(
                    """
                    SELECT rh.* 
                    FROM recognition_history rh
                    JOIN recognition_types rt ON rh.type_id = rt.id
                    WHERE rh.user_id = %s AND rt.name = %s
                    ORDER BY rh.created_at DESC
                    LIMIT 1
                    """,
                    (user_id, recognition_type_name)
                )
                record = cursor.fetchone()
                if record:
                    try:
                        import json
                        record['result'] = json.loads(record.get('result', '{}'))
                    except json.JSONDecodeError:
                        record['result'] = {}
                return record
        except Exception as e:
            raise DatabaseError(f"查询最近{recognition_type_name}识别记录失败：{str(e)}")

    def count_recognition_history(
            self, user_id, time_range, emotion_type, data_type
    ):
        """
        统计符合筛选条件的记录总数
        与 filter_recognition_history 共享相同的筛选逻辑
        """
        try:
            # 构建与筛选逻辑一致的查询，但只返回计数
            query = """
                SELECT COUNT(*) as total 
                FROM recognition_history rh
                JOIN recognition_types rt ON rh.type_id = rt.id
                WHERE rh.user_id = %s
            """
            params = [user_id]

            # 时间范围筛选（复用 filter_recognition_history 的逻辑）
            if time_range == 'week':
                query += " AND rh.created_at >= NOW() - INTERVAL 1 WEEK"
            elif time_range == 'month':
                query += " AND rh.created_at >= NOW() - INTERVAL 1 MONTH"

            # 情绪类型筛选
            if emotion_type != 'all':
                query += " AND JSON_EXTRACT(rh.result, '$.emotion_class') = %s"
                params.append(emotion_type)

            # 数据类型筛选
            if data_type != 'all':
                query += " AND rt.type_name = %s"
                params.append(data_type)

            with self._db_cursor() as cursor:
                cursor.execute(query, params)
                result = cursor.fetchone()
                return result['total'] if result else 0  # 返回总数，默认为0
        except Exception as e:
            raise DatabaseError(f"统计记录总数失败：{str(e)}")
    # 修改方法参数，通过参数接收时间范围
    def filter_recognition_history(
            self, user_id, time_range, emotion_type, data_type, limit=None, offset=0
    ):
        try:
            query = """
                SELECT rh.*, rt.type_name 
                FROM recognition_history rh
                JOIN recognition_types rt ON rh.type_id = rt.id
                WHERE rh.user_id = %s
            """
            params = [user_id]
            # 时间范围筛选（使用参数传递的start_time和end_time）
            if time_range == 'week':
                query += " AND rh.created_at >= NOW() - INTERVAL 1 WEEK"
            elif time_range == 'month':
                query += " AND rh.created_at >= NOW() - INTERVAL 1 MONTH"

            # 情绪类型筛选
            if emotion_type != 'all':
                query += " AND JSON_EXTRACT(rh.result, '$.emotion_class') = %s"
                params.append(emotion_type)

            # 数据类型筛选（通过关联表的type_name）
            if data_type != 'all':
                query += " AND rt.type_name = %s"  # 关联筛选type_name
                params.append(data_type)

            # 分页与排序
            query += " ORDER BY rh.created_at DESC"
            if limit is not None:
                query += " LIMIT %s OFFSET %s"
                params.extend([limit, offset])

            with self._db_cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
        except Exception as e:
            raise DatabaseError(f"筛选记录失败：{str(e)}")

    def get_user_recognition_history(self, user_id, limit=20, record_type="all"):
        return self.filter_recognition_history(
            user_id=user_id,
            time_range=None,  # 不筛选时间
            emotion_type="all",  # 不筛选情绪
            data_type=record_type,  # 复用类型筛选
            limit=limit  # 传递limit参数（需确保filter_recognition支持）
        )
    def save_recognition_history(self, history: RecognitionResult) -> int:
        """保存识别记录到数据库
        :param history: 识别结果对象（RecognitionResult实例）
        :return: 新记录ID
        """
        # 转换结果为JSON字符串
        result_json = str(history.emotion_result.to_json()).replace("'", '"')

        # 按类型区分媒体路径
        video_path = history.media_path if history.recognition_type == "video" else None
        image_path = history.media_path if history.recognition_type == "image" else None
        text_content = history.raw_content if history.recognition_type == "text" else None

        with self._db_cursor() as cursor:
            cursor.execute("""
                INSERT INTO RECOGNITION_HISTORY 
                (user_id, type_id, result, confidence, video_path, image_path, text_content)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                history.user_id,
                history.get_recognition_type_id(history.recognition_type),
                result_json,
                history.emotion_result.confidence,
                video_path,
                image_path,
                text_content
            ))
            return cursor.lastrowid

    def delete_recognition_record(self, record_id: int, user_id: int) -> bool:
        """删除用户的识别记录（验证权限）
        参数:
            record_id: 要删除的记录ID (int)
            user_id: 当前用户ID (int)
        返回:
            成功删除返回True (bool)
        抛出:
            DatabaseError: 如果无权限删除或记录不存在
        """
        # 1. 先删除关联的分享令牌（无论是否有外键，强制清理）
        with self._db_cursor() as cursor:
            cursor.execute(
                "DELETE FROM share_tokens WHERE record_id = %s",
                (record_id,)
            )

        # 2. 验证记录所有权并删除主表记录
        detail = self.get_recognition_by_id(record_id)
        if not detail or detail["user_id"] != user_id:
            raise DatabaseError("无权限删除此记录")

        with self._db_cursor() as cursor:
            cursor.execute(
                "DELETE FROM recognition_history WHERE id = %s",
                (record_id,)
            )
            return cursor.rowcount > 0

    # ========================== 辅助方法 ==========================
    def verify_password(self, username: str, password: str) -> bool:
        """验证密码（BCrypt校验）
        :param username: 用户名
        :param password: 原始密码
        :return: 验证通过返回True
        """
        user = self.get_user_by_username(username)
        if not user:
            return False
        return bcrypt.checkpw(
            password.encode('utf-8'),
            user["password_hash"].encode('utf-8')
        )

    def generate_share_token(self, record_id: int, expires_in: int = 3600) -> str:
        """生成带过期时间的分享token"""
        token = uuid.uuid4().hex
        expires_at = datetime.now() + timedelta(seconds=expires_in)  # 直接使用类
        with self._db_cursor() as cursor:
            cursor.execute(
                "INSERT INTO share_tokens (token, record_id, expires_at) VALUES (%s, %s, %s)",
                (token, record_id, expires_at)
            )
        return token

    def verify_share_token(self, token: str) -> Optional[int]:
        """验证token并返回记录ID（过期则失效）"""
        with self._db_cursor() as cursor:
            cursor.execute(
                "SELECT record_id FROM share_tokens WHERE token = %s AND expires_at > NOW()",
                (token,)
            )
            result = cursor.fetchone()
            return result['record_id'] if result else None

    def get_share_token(self, token):
        """根据token查询分享令牌信息"""
        try:
            with self._db_cursor() as cursor:
                cursor.execute(
                    "SELECT id, token, record_id, expires_at FROM share_tokens WHERE token = %s",
                    (token,)
                )
                result = cursor.fetchone()
                return result if result else None
        except Exception as e:
            raise DatabaseError(f"查询分享令牌失败: {str(e)}")

    # 在 sql_base.py 的 DatabaseManager 类中添加
    def delete_expired_share_tokens(self) -> int:
        """删除所有已过期的分享令牌，返回删除的记录数"""
        with self._db_cursor() as cursor:
            cursor.execute("""
                DELETE FROM share_tokens 
                WHERE expires_at < NOW()  -- 删除过期记录
            """)
            return cursor.rowcount  # 返回删除的行数
    @staticmethod
    def generate_email_verification_code() -> str:
        """生成6位数字邮箱验证码
        :return: 6位数字字符串
        """
        return ''.join(random.choices(string.digits, k=6))

    def check_username_exists(self, username: str, exclude_id: Optional[int] = None) -> bool:
        """
        检查用户名是否存在（支持排除特定用户ID）
        
        用途：
        - 用户注册时验证用户名唯一性
        - 用户更新信息时检查是否更改了用户名
        
        参数说明：
        :param username: 要检查的用户名
        :param exclude_id: 可选参数，用于排除指定用户ID（如当前登录用户）
        :return: 如果用户名存在返回True，否则返回False
        
        特性：
        - 使用参数化查询防止SQL注入
        - 自动处理数据库异常并记录日志
        - 支持类型提示增强代码可读性
        """
        try:
            # 使用上下文管理器自动处理游标生命周期
            with self.conn.cursor() as cursor:
                # 初始化基础查询语句和参数列表
                query = "SELECT id FROM users WHERE username = %s"
                params = [username]
                
                # 如果提供了exclude_id，则添加排除条件
                if exclude_id:
                    query += " AND id != %s"  # 添加排除特定用户的条件
                    params.append(exclude_id)  # 将排除的ID加入参数列表
                
                # 执行参数化查询
                cursor.execute(query, params)
                
                # 返回查询结果是否存在（即用户名是否已被占用）
                return cursor.fetchone() is not None
                
        except Exception as e:
            # 记录详细的错误日志
            self.logger.error(f"检查用户名 [{username}] 是否存在时发生异常: {str(e)}", exc_info=True)
            
            # 发生异常时保守返回False，避免阻止合法用户操作
            # 这有助于在数据库暂时不可用时仍允许用户继续操作
            return False
    def send_email_verification_code(self, email: str) -> bool | None:
        """发送验证码邮件（SMTP协议）

        :param email: 目标邮箱
        :return: 发送成功返回True
        """
        sender = os.environ.get("SMTP_USER", "")
        auth_code = os.environ.get("SMTP_PASSWORD", "")
        smtp_host = os.environ.get("SMTP_HOST", "smtp.qq.com")
        smtp_port = int(os.environ.get("SMTP_PORT", "465"))
        receiver = email
        # 检查是否已存在未过期的验证码
        if receiver in self.email_verification_cache:
            stored_code, send_time = self.email_verification_cache[receiver]
            if datetime.now() - send_time < timedelta(seconds=60):
                # 如果验证码已经存在且未过期，不发送新验证码
                raise DatabaseError("验证码已发送，请勿重复请求")
        # 标记邮件是否发送成功
        is_sent = False
        server = None
        stored_code = self.generate_email_verification_code()
        app_url = os.environ.get("APP_URL", "http://localhost:5000")
        msg = MIMEText(f"你的验证码是：{stored_code}，有效期60秒。\n注册请进入：{app_url}/register\n更改邮箱号请进入：{app_url}/user\n", "plain", "utf-8")
        msg["Subject"] = "情感识别系统 - 邮箱验证"
        msg["From"] = sender
        msg["To"] = receiver

        try:
            with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
                server.login(sender, auth_code)
                server.sendmail(sender, [receiver], msg.as_string())
                is_sent = True  # 发送成功
            # 缓存验证码及发送时间
            self.email_verification_cache[receiver] = (stored_code, datetime.now())
        except smtplib.SMTPException as e:
            self.logger.error(f"邮件发送过程出错: {str(e)}")
        finally:
            # 单独处理关闭连接的逻辑，避免影响发送结果的判断
            if server:
                try:
                    server.quit()  # 尝试正常关闭
                except smtplib.SMTPException as e:
                    # 关闭失败仅警告，不影响“发送成功”的判定
                    self.logger.warning(f"关闭SMTP连接时出错: {str(e)}")
        if is_sent:
            # 发送成功后缓存验证码
            self.email_verification_cache[receiver] = (stored_code, datetime.now())
            return True
        else:
            return False

    def verify_code(self, email: str, verification_code: str):
        """验证邮箱验证码（有效期60秒）

        :param email: 邮箱地址
        :param verification_code: 待验证的验证码
        :raises DatabaseError: 当验证码无效或过期时
        """
        cache_key = email
        if cache_key not in self.email_verification_cache:
            raise DatabaseError("未获取验证码，请重新发送")
            
        stored_code, send_time = self.email_verification_cache[cache_key]
        if datetime.now() - send_time > timedelta(seconds=60):
            del self.email_verification_cache[cache_key]  # 清理过期缓存
            raise DatabaseError("验证码已过期（有效期60秒）")
            
        if stored_code != verification_code:
            raise DatabaseError("验证码错误")

    def close(self):
        """关闭数据库连接"""
        if self.conn and self.conn.open:
            try:
                self.conn.close()
            except Exception as e:
                self.logger.warning(f"关闭数据库连接时出错: {str(e)}")

    def __del__(self):
        """对象销毁时自动关闭连接"""
        self.close()