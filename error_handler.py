"""
統一錯誤處理模組
提供一致的錯誤處理和日誌記錄
"""

import logging
import traceback
from typing import Optional, Callable, Any
from functools import wraps

# 設定日誌
logger = logging.getLogger(__name__)

# 匯入 HTTPClient 作為向後相容導出
try:
    from http_client import HTTPClient
except ImportError:
    HTTPClient = None


class WhisperError(Exception):
    """Whisper 相關錯誤的基礎異常類別"""
    pass


class FileProcessingError(WhisperError):
    """檔案處理錯誤"""
    pass


class TranslationError(WhisperError):
    """翻譯錯誤"""
    pass


class ResourceError(WhisperError):
    """資源管理錯誤"""
    pass


class ValidationError(WhisperError):
    """驗證錯誤"""
    pass


def handle_errors(fallback_message: str = "處理過程中發生錯誤", log_error: bool = True):
    """
    錯誤處理裝飾器

    Args:
        fallback_message: 當發生錯誤時的預設訊息
        log_error: 是否記錄錯誤日誌
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except WhisperError:
                # 重新拋出自定義異常，不做額外處理
                raise
            except Exception as e:
                if log_error:
                    logger.error(f"函式 {func.__name__} 執行時發生錯誤: {str(e)}")
                    logger.error(f"錯誤詳情: {traceback.format_exc()}")

                # 將一般異常轉換為 WhisperError
                raise WhisperError(f"{fallback_message}: {str(e)}") from e
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, fallback_value: Any = None, **kwargs) -> Any:
    """
    安全執行函式，如果發生錯誤返回預設值

    Args:
        func: 要執行的函式
        *args: 函式參數
        fallback_value: 發生錯誤時的預設返回值
        **kwargs: 函式關鍵字參數

    Returns:
        函式執行結果或預設值
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"安全執行 {func.__name__} 失敗，使用預設值: {str(e)}")
        return fallback_value


class ErrorHandler:
    """統一錯誤處理器"""

    @staticmethod
    def log_and_raise(error: Exception, message: Optional[str] = None):
        """記錄錯誤並重新拋出"""
        if message:
            logger.error(message)
        logger.error(f"錯誤詳情: {str(error)}")
        logger.error(f"堆疊追蹤: {traceback.format_exc()}")
        raise error

    @staticmethod
    def handle_file_error(file_path: str, error: Exception) -> FileProcessingError:
        """處理檔案相關錯誤"""
        message = f"處理檔案 '{file_path}' 時發生錯誤: {str(error)}"
        logger.error(message)
        return FileProcessingError(message)

    @staticmethod
    def handle_translation_error(text: str, error: Exception) -> TranslationError:
        """處理翻譯相關錯誤"""
        message = f"翻譯文字時發生錯誤: {str(error)}"
        logger.warning(message)
        logger.debug(f"原文: {text[:100]}...")
        return TranslationError(message)

    @staticmethod
    def handle_resource_error(resource_name: str, error: Exception) -> ResourceError:
        """處理資源管理錯誤"""
        message = f"資源 '{resource_name}' 處理錯誤: {str(error)}"
        logger.error(message)
        return ResourceError(message)

    @staticmethod
    def validate_condition(condition: bool, error_message: str) -> None:
        """驗證條件，如果不滿足則拋出異常"""
        if not condition:
            logger.error(f"驗證失敗: {error_message}")
            raise ValidationError(error_message)


# 便捷函式
def log_error(message: str, error: Optional[Exception] = None):
    """記錄錯誤訊息"""
    logger.error(message)
    if error:
        logger.error(f"錯誤詳情: {str(error)}")


def log_warning(message: str):
    """記錄警告訊息"""
    logger.warning(message)


def log_info(message: str):
    """記錄資訊訊息"""
    logger.info(message)


# 上下文管理器用於錯誤追蹤
from contextlib import contextmanager


@contextmanager
def error_context(operation: str, **context_data):
    """
    錯誤上下文管理器，用於追蹤操作和相關資料

    Args:
        operation: 操作名稱
        **context_data: 相關的上下文資料
    """
    try:
        log_info(f"開始操作: {operation}")
        if context_data:
            logger.debug(f"操作上下文: {context_data}")
        yield
        log_info(f"操作完成: {operation}")
    except Exception as e:
        log_error(f"操作失敗: {operation} - {str(e)}")
        if context_data:
            logger.debug(f"失敗時的上下文: {context_data}")
        raise


def retry_on_exception(max_retries: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """
    重試裝飾器

    Args:
        max_retries: 最大重試次數
        delay: 重試間隔（秒）
        exceptions: 需要重試的異常類型
    """
    import time

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"函式 {func.__name__} 執行失敗 (嘗試 {attempt + 1}/{max_retries + 1})，{delay} 秒後重試: {str(e)}")
                        time.sleep(delay)
                    else:
                        logger.error(f"函式 {func.__name__} 在 {max_retries + 1} 次嘗試後仍然失敗")
                        raise

            # 這行程式碼理論上不會執行到，但為了完整性還是加上
            raise last_exception or Exception("未知錯誤")
        return wrapper
    return decorator


def handle_file_operations(func: Callable) -> Callable:
    """
    檔案操作裝飾器，提供統一的檔案操作錯誤處理

    Args:
        func: 要裝飾的函式

    Returns:
        裝飾後的函式
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            error_msg = f"檔案未找到: {str(e)}"
            logger.error(error_msg)
            raise FileProcessingError(error_msg) from e
        except PermissionError as e:
            error_msg = f"檔案權限錯誤: {str(e)}"
            logger.error(error_msg)
            raise FileProcessingError(error_msg) from e
        except OSError as e:
            error_msg = f"檔案操作錯誤: {str(e)}"
            logger.error(error_msg)
            raise FileProcessingError(error_msg) from e
        except Exception as e:
            error_msg = f"檔案操作過程中發生未預期的錯誤: {str(e)}"
            logger.error(error_msg)
            raise FileProcessingError(error_msg) from e
    return wrapper
