"""
日誌配置模組

提供統一的日誌配置和管理功能。
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional
import json


class JSONFormatter(logging.Formatter):
    """JSON 格式的日誌格式化器"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        # 添加額外的上下文資訊
        if hasattr(record, 'error_code'):
            log_entry['error_code'] = record.error_code
        if hasattr(record, 'task_id'):
            log_entry['task_id'] = record.task_id
        if hasattr(record, 'audio_file'):
            log_entry['audio_file'] = record.audio_file
        if hasattr(record, 'file_size'):
            log_entry['file_size'] = record.file_size
        if hasattr(record, 'text_length'):
            log_entry['text_length'] = record.text_length
        if hasattr(record, 'model'):
            log_entry['model'] = record.model
        if hasattr(record, 'url'):
            log_entry['url'] = record.url
        if hasattr(record, 'endpoint'):
            log_entry['endpoint'] = record.endpoint
        
        # 添加異常資訊
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """彩色控制台日誌格式化器"""
    
    # ANSI 顏色代碼
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 綠色
        'WARNING': '\033[33m',  # 黃色
        'ERROR': '\033[31m',    # 紅色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重置
    }
    
    def format(self, record):
        # 添加顏色
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # 格式化訊息
        formatted = super().format(record)
        return f"{color}{formatted}{reset}"


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    app_name: str = "whisper",
    enable_json_logs: bool = True,
    enable_console_colors: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    設置統一的日誌系統
    
    Args:
        log_level: 日誌級別
        log_dir: 日誌目錄
        app_name: 應用程式名稱
        enable_json_logs: 是否啟用 JSON 格式日誌
        enable_console_colors: 是否啟用控制台顏色
        max_file_size: 日誌檔案最大大小
        backup_count: 備份檔案數量
        
    Returns:
        配置好的 logger
    """
    # 創建日誌目錄
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 獲取根 logger
    logger = logging.getLogger(app_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除現有的 handlers
    logger.handlers.clear()
    
    # 檔案處理器 - 一般日誌
    log_file = os.path.join(log_dir, f"{app_name}.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    
    # 檔案處理器 - 錯誤日誌
    error_log_file = os.path.join(log_dir, f"{app_name}-error.log")
    error_file_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_file_handler.setLevel(logging.ERROR)
    
    # JSON 日誌檔案處理器
    if enable_json_logs:
        json_log_file = os.path.join(log_dir, f"{app_name}-json.log")
        json_file_handler = logging.handlers.RotatingFileHandler(
            json_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        json_file_handler.setLevel(logging.DEBUG)
        json_file_handler.setFormatter(JSONFormatter())
        logger.addHandler(json_file_handler)
    
    # 控制台處理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 設置格式化器
    if enable_console_colors and sys.stdout.isatty():
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # 應用格式化器
    file_handler.setFormatter(file_formatter)
    error_file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # 添加處理器
    logger.addHandler(file_handler)
    logger.addHandler(error_file_handler)
    logger.addHandler(console_handler)
    
    # 防止日誌傳播到根 logger
    logger.propagate = False
    
    logger.info(f"日誌系統初始化完成 - 級別: {log_level}, 目錄: {log_dir}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    獲取指定名稱的 logger
    
    Args:
        name: logger 名稱
        
    Returns:
        logger 實例
    """
    return logging.getLogger(f"whisper.{name}")


# 預設 logger 實例
default_logger = setup_logging()


class LogContext:
    """日誌上下文管理器，用於添加結構化日誌資訊"""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_factory = logging.getLogRecordFactory()
    
    def __enter__(self):
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


def log_performance(func):
    """效能監控裝飾器"""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger("performance")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(
                f"函數 {func.__name__} 執行完成",
                extra={
                    "function": func.__name__,
                    "duration": duration,
                    "status": "success"
                }
            )
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"函數 {func.__name__} 執行失敗: {str(e)}",
                extra={
                    "function": func.__name__,
                    "duration": duration,
                    "status": "error",
                    "error": str(e)
                }
            )
            raise
    
    return wrapper
