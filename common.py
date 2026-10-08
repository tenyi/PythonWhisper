"""
共用模組 - 解決跨檔案重複問題
包含共用的類別、函式和資源管理
"""

import os
import logging
from typing import Optional, List
from contextlib import contextmanager
from opencc import OpenCC
from pydantic import BaseModel
import threading

# 設定日誌
logger = logging.getLogger(__name__)

# 匯入錯誤處理模組
from error_handler import ErrorHandler, safe_execute, handle_errors, log_error, log_info


# 語音片段結構體
class Segment(BaseModel):
    start: float
    end: float
    text: str

    def __init__(self, start: float, end: float, text: str, **data):
        super().__init__(start=start, end=end, text=text, **data)


# 轉錄資訊結構體
class TranscriptionInfo(BaseModel):
    language: str
    language_probability: float
    duration: float

    def __init__(
        self, language: str, language_probability: float, duration: float, **data
    ):
        super().__init__(
            language=language,
            language_probability=language_probability,
            duration=duration,
            **data,
        )


# OpenCC 資源管理器 - 解決重複建立問題
class OpenCCManager:
    _instance: Optional['OpenCCManager'] = None
    _lock = threading.Lock()

    def __init__(self):
        self._cc_instances = {}

    @classmethod
    def get_instance(cls) -> 'OpenCCManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_cc(self, config: str = "s2twp.json") -> OpenCC:
        """獲取或建立 OpenCC 實例"""
        if config not in self._cc_instances:
            try:
                self._cc_instances[config] = OpenCC(config)
                logger.info(f"已建立 OpenCC 實例: {config}")
            except Exception as e:
                logger.error(f"建立 OpenCC 實例失敗 {config}: {e}")
                raise
        return self._cc_instances[config]

    def cleanup(self):
        """清理所有 OpenCC 實例"""
        self._cc_instances.clear()
        logger.info("已清理所有 OpenCC 實例")


# 檔案處理工具
class FileHandler:
    @staticmethod
    def safe_path_split(file_path: str, separator: str, maxsplit: int = 1) -> List[str]:
        """安全地分割檔案路徑，避免索引錯誤"""
        parts = file_path.rsplit(separator, maxsplit)
        if len(parts) <= maxsplit:
            # 如果分割後的片段數量不足，返回原路徑作為第一個元素
            return [file_path]
        return parts

    @staticmethod
    def validate_audio_file(file_path: str, max_size_mb: int = 100) -> bool:
        """驗證音訊檔案"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"檔案不存在: {file_path}")

        # 檢查檔案大小
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise ValueError(f"檔案過大: {file_size_mb:.1f}MB (最大限制: {max_size_mb}MB)")

        # 檢查檔案副檔名
        allowed_extensions = {'.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.ogg', '.wav', '.webm'}
        _, ext = os.path.splitext(file_path.lower())
        if ext not in allowed_extensions:
            raise ValueError(f"不支援的檔案格式: {ext}")

        return True


# 時間格式化工具
class TimeFormatter:
    @staticmethod
    def format_time_vtt(seconds_val: float) -> str:
        """格式化為 VTT 時間格式"""
        hours = int(seconds_val // 3600)
        minutes = int((seconds_val % 3600) // 60)
        remaining_seconds = seconds_val % 60
        milliseconds = int((remaining_seconds - int(remaining_seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(remaining_seconds):02d}.{milliseconds:03d}"

    @staticmethod
    def format_time_srt(seconds_val: float) -> str:
        """格式化為 SRT 時間格式"""
        hours = int(seconds_val // 3600)
        minutes = int((seconds_val % 3600) // 60)
        remaining_seconds = seconds_val % 60
        milliseconds = int((remaining_seconds - int(remaining_seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(remaining_seconds):02d},{milliseconds:03d}"


# 語言偵測工具
class LanguageDetector:
    @staticmethod
    def detect_language_from_text(text: str) -> str:
        """從文字內容偵測語言"""
        import re
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.replace(' ', ''))

        if total_chars > 0 and chinese_chars / total_chars > 0.3:
            return "zh"
        else:
            return "en"


# 資源管理器
class ResourceManager:
    def __init__(self):
        self._resources = []
        self._lock = threading.Lock()

    @contextmanager
    def manage_file(self, file_path: str, mode: str = 'r', encoding: str = 'utf-8'):
        """檔案資源管理器"""
        file_obj = None
        try:
            file_obj = open(file_path, mode, encoding=encoding)
            yield file_obj
        finally:
            if file_obj:
                file_obj.close()

    def add_resource(self, resource):
        """加入資源到管理列表"""
        with self._lock:
            self._resources.append(resource)

    def cleanup(self):
        """清理所有資源"""
        with self._lock:
            for resource in self._resources:
                try:
                    if hasattr(resource, 'close'):
                        resource.close()
                    elif hasattr(resource, 'cleanup'):
                        resource.cleanup()
                except Exception as e:
                    logger.error(f"清理資源時發生錯誤: {e}")
            self._resources.clear()


# 全域資源管理器實例
resource_manager = ResourceManager()
opencc_manager = OpenCCManager.get_instance()