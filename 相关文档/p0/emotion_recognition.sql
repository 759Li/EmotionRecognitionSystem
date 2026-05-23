-- 创建用户表（USERS）
CREATE TABLE `USERS` (
    `id` INTEGER AUTO_INCREMENT COMMENT '用户ID',
    `username` VARCHAR(50) NOT NULL UNIQUE COMMENT '用户登录名（唯一）',
    `password_hash` VARCHAR(255) NOT NULL COMMENT 'BCrypt加密后的密码',
    `phone_num` VARCHAR(100) UNIQUE COMMENT '手机号（唯一）',
    `email` VARCHAR(100) UNIQUE COMMENT '邮箱地址（唯一）',
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '账户创建时间',
    `status` TINYINT DEFAULT 1 COMMENT '用户状态：1 - 正常，0 - 禁用',
    PRIMARY KEY (`id`)
) COMMENT = '用户表';

-- 创建用户表的索引
CREATE INDEX `idx_users_username` ON `USERS` (`username`);
CREATE INDEX `idx_users_phone` ON `USERS` (`phone_num`);
CREATE INDEX `idx_users_email` ON `USERS` (`email`);

-- 创建识别类型表（RECOGNITION_TYPES）
CREATE TABLE `RECOGNITION_TYPES` (
    `id` INTEGER AUTO_INCREMENT COMMENT '类型ID',
    `type_name` VARCHAR(50) NOT NULL UNIQUE COMMENT '识别类型名称',
    `description` TEXT(65535) DEFAULT NULL COMMENT '类型描述',
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `is_active` TINYINT DEFAULT 1 COMMENT '是否启用：1 - 是，0 - 否',
    PRIMARY KEY (`id`)
) COMMENT = '识别类型表';

-- 插入识别类型数据
INSERT INTO `RECOGNITION_TYPES` (`type_name`, `description`) VALUES
('video', '通过视频分析面部特征来识别不同的情绪'),
('image', '通过图片分析面部特征来识别不同的情绪'),
('text', '对文本内容进行情感极性判断');

-- 创建识别历史表（RECOGNITION_HISTORY）
CREATE TABLE `RECOGNITION_HISTORY` (
    `id` INTEGER AUTO_INCREMENT COMMENT '记录ID',
    `user_id` INTEGER NOT NULL COMMENT '用户ID（外键）',
    `type_id` INTEGER NOT NULL COMMENT '识别类型ID（外键）',
    `result` JSON NOT NULL COMMENT '识别结果JSON对象',
    `confidence` FLOAT(4, 2) NOT NULL COMMENT '置信度（0 - 1）',
    `video_path` VARCHAR(255) DEFAULT NULL COMMENT '视频存储路径',
    `image_path` VARCHAR(255) DEFAULT NULL COMMENT '图像存储路径',
    `text_content` TEXT(65535) DEFAULT NULL COMMENT '识别文本内容',
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '识别时间',
    PRIMARY KEY (`id`)
) COMMENT = '识别历史表';

-- 创建识别历史表的索引
CREATE INDEX `idx_history_user_time` ON `RECOGNITION_HISTORY` (`user_id`, `created_at`);
CREATE INDEX `idx_history_type` ON `RECOGNITION_HISTORY` (`type_id`);
CREATE INDEX `idx_history_time` ON `RECOGNITION_HISTORY` (`created_at`);

-- 为识别历史表添加外键约束
ALTER TABLE `RECOGNITION_HISTORY`
ADD FOREIGN KEY (`user_id`) REFERENCES `USERS` (`id`)
ON UPDATE NO ACTION ON DELETE NO ACTION;

ALTER TABLE `RECOGNITION_HISTORY`
ADD FOREIGN KEY (`type_id`) REFERENCES `RECOGNITION_TYPES` (`id`)
ON UPDATE NO ACTION ON DELETE NO ACTION;

-- 创建分享令牌表（SHARE_TOKENS）
CREATE TABLE `SHARE_TOKENS` (
    `id` INTEGER AUTO_INCREMENT COMMENT '令牌ID',
    `token` VARCHAR(64) NOT NULL UNIQUE COMMENT '分享令牌（UUID生成的唯一字符串）',
    `record_id` INTEGER NOT NULL COMMENT '关联的识别记录ID',
    `expires_at` DATETIME NOT NULL COMMENT '令牌过期时间',
    PRIMARY KEY (`id`)
) COMMENT = '分享令牌表';

-- 创建分享令牌表的索引
CREATE INDEX `idx_token` ON `SHARE_TOKENS` (`token`);
CREATE INDEX `idx_expires_at` ON `SHARE_TOKENS` (`expires_at`);
CREATE INDEX `idx_record_id` ON `SHARE_TOKENS` (`record_id`);

-- 为分享令牌表添加外键约束
ALTER TABLE `SHARE_TOKENS`
ADD FOREIGN KEY (`record_id`) REFERENCES `RECOGNITION_HISTORY` (`id`)
ON UPDATE NO ACTION ON DELETE NO ACTION;
