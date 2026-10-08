"""
自定義異常類別模組

提供專案特定的異常類別，用於更精確的錯誤處理和分類。
"""

from typing import Optional, Dict, Any
from enum import Enum


class ErrorCode(Enum):
    """錯誤代碼枚舉"""
    # 系統錯誤 (1000-1999)
    SYSTEM_ERROR = 1000
    MEMORY_ERROR = 1001
    DEVICE_ERROR = 1002
    INITIALIZATION_ERROR = 1003
    
    # 檔案錯誤 (2000-2999)
    FILE_NOT_FOUND = 2000
    FILE_READ_ERROR = 2001
    FILE_WRITE_ERROR = 2002
    FILE_FORMAT_ERROR = 2003
    FILE_SIZE_ERROR = 2004
    
    # 模型錯誤 (3000-3999)
    MODEL_LOAD_ERROR = 3000
    MODEL_INFERENCE_ERROR = 3001
    MODEL_NOT_INITIALIZED = 3002
    
    # API 錯誤 (4000-4999)
    API_CONNECTION_ERROR = 4000
    API_TIMEOUT_ERROR = 4001
    API_AUTHENTICATION_ERROR = 4002
    API_RATE_LIMIT_ERROR = 4003
    API_RESPONSE_ERROR = 4004
    API_INVALID_REQUEST = 4005
    
    # 轉錄錯誤 (5000-5999)
    TRANSCRIPTION_ERROR = 5000
    AUDIO_FORMAT_ERROR = 5001
    AUDIO_DURATION_ERROR = 5002
    
    # 翻譯錯誤 (6000-6999)
    TRANSLATION_ERROR = 6000
    LANGUAGE_NOT_SUPPORTED = 6001
    
    # 摘要錯誤 (7000-7999)
    SUMMARY_ERROR = 7000
    TEXT_TOO_SHORT = 7001
    TEXT_TOO_LONG = 7002
    
    # 任務錯誤 (8000-8999)
    TASK_NOT_FOUND = 8000
    TASK_TIMEOUT = 8001
    TASK_CANCELLED = 8002
    QUEUE_FULL = 8003


class WhisperBaseException(Exception):
    """Whisper 專案基礎異常類別"""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.original_exception = original_exception
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式，用於 API 回應"""
        return {
            "error": {
                "code": self.error_code.value,
                "message": self.message,
                "details": self.details,
                "type": self.__class__.__name__
            }
        }
    
    def __str__(self) -> str:
        return f"[{self.error_code.name}] {self.message}"


class SystemError(WhisperBaseException):
    """系統相關錯誤"""
    pass


class FileError(WhisperBaseException):
    """檔案相關錯誤"""
    pass


class ModelError(WhisperBaseException):
    """模型相關錯誤"""
    pass


class APIError(WhisperBaseException):
    """API 相關錯誤"""
    pass


class TranscriptionError(WhisperBaseException):
    """轉錄相關錯誤"""
    pass


class TranslationError(WhisperBaseException):
    """翻譯相關錯誤"""
    pass


class SummaryError(WhisperBaseException):
    """摘要相關錯誤"""
    pass


class TaskError(WhisperBaseException):
    """任務相關錯誤"""
    pass


# 便利函數用於快速創建常見異常
def create_file_not_found_error(file_path: str) -> FileError:
    """創建檔案未找到錯誤"""
    return FileError(
        message=f"檔案未找到: {file_path}",
        error_code=ErrorCode.FILE_NOT_FOUND,
        details={"file_path": file_path}
    )


def create_model_load_error(model_name: str, reason: str) -> ModelError:
    """創建模型載入錯誤"""
    return ModelError(
        message=f"模型載入失敗: {model_name} - {reason}",
        error_code=ErrorCode.MODEL_LOAD_ERROR,
        details={"model_name": model_name, "reason": reason}
    )


def create_api_connection_error(url: str, reason: str) -> APIError:
    """創建 API 連線錯誤"""
    return APIError(
        message=f"API 連線失敗: {url} - {reason}",
        error_code=ErrorCode.API_CONNECTION_ERROR,
        details={"url": url, "reason": reason}
    )


def create_transcription_error(audio_file: str, reason: str) -> TranscriptionError:
    """創建轉錄錯誤"""
    return TranscriptionError(
        message=f"轉錄失敗: {audio_file} - {reason}",
        error_code=ErrorCode.TRANSCRIPTION_ERROR,
        details={"audio_file": audio_file, "reason": reason}
    )


def create_task_not_found_error(task_id: str) -> TaskError:
    """創建任務未找到錯誤"""
    return TaskError(
        message=f"任務未找到: {task_id}",
        error_code=ErrorCode.TASK_NOT_FOUND,
        details={"task_id": task_id}
    )
