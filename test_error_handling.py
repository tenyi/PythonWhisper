#!/usr/bin/env python3
"""
錯誤處理機制單元測試

測試自定義異常類別、錯誤處理工具和重試機制。
"""

import pytest
import requests
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import json

# 匯入要測試的模組
from exceptions import (
    WhisperBaseException, SystemError, FileError, ModelError, APIError,
    TranscriptionError, TranslationError, SummaryError, TaskError,
    ErrorCode, create_file_not_found_error, create_model_load_error,
    create_api_connection_error, create_transcription_error, create_task_not_found_error
)
from error_handler import (
    retry_on_exception, error_context, HTTPClient, handle_file_operations
)


class TestWhisperBaseException:
    """測試基礎異常類別"""
    
    def test_basic_exception_creation(self):
        """測試基本異常創建"""
        error = WhisperBaseException(
            message="測試錯誤",
            error_code=ErrorCode.SYSTEM_ERROR,
            details={"key": "value"}
        )
        
        assert str(error) == "[SYSTEM_ERROR] 測試錯誤"
        assert error.message == "測試錯誤"
        assert error.error_code == ErrorCode.SYSTEM_ERROR
        assert error.details == {"key": "value"}
    
    def test_to_dict_method(self):
        """測試轉換為字典格式"""
        error = WhisperBaseException(
            message="測試錯誤",
            error_code=ErrorCode.API_CONNECTION_ERROR,
            details={"url": "http://test.com"}
        )
        
        result = error.to_dict()
        expected = {
            "error": {
                "code": ErrorCode.API_CONNECTION_ERROR.value,
                "message": "測試錯誤",
                "details": {"url": "http://test.com"},
                "type": "WhisperBaseException"
            }
        }
        
        assert result == expected
    
    def test_with_original_exception(self):
        """測試包含原始異常"""
        original = ValueError("原始錯誤")
        error = WhisperBaseException(
            message="包裝錯誤",
            error_code=ErrorCode.SYSTEM_ERROR,
            original_exception=original
        )
        
        assert error.original_exception == original


class TestErrorCodeEnum:
    """測試錯誤代碼枚舉"""
    
    def test_error_code_values(self):
        """測試錯誤代碼值"""
        assert ErrorCode.SYSTEM_ERROR.value == 1000
        assert ErrorCode.FILE_NOT_FOUND.value == 2000
        assert ErrorCode.MODEL_LOAD_ERROR.value == 3000
        assert ErrorCode.API_CONNECTION_ERROR.value == 4000
        assert ErrorCode.TRANSCRIPTION_ERROR.value == 5000
        assert ErrorCode.TRANSLATION_ERROR.value == 6000
        assert ErrorCode.SUMMARY_ERROR.value == 7000
        assert ErrorCode.TASK_NOT_FOUND.value == 8000


class TestConvenienceFunctions:
    """測試便利函數"""
    
    def test_create_file_not_found_error(self):
        """測試創建檔案未找到錯誤"""
        error = create_file_not_found_error("/path/to/file.txt")
        
        assert isinstance(error, FileError)
        assert error.error_code == ErrorCode.FILE_NOT_FOUND
        assert "檔案未找到" in error.message
        assert error.details["file_path"] == "/path/to/file.txt"
    
    def test_create_model_load_error(self):
        """測試創建模型載入錯誤"""
        error = create_model_load_error("test_model", "記憶體不足")
        
        assert isinstance(error, ModelError)
        assert error.error_code == ErrorCode.MODEL_LOAD_ERROR
        assert "模型載入失敗" in error.message
        assert error.details["model_name"] == "test_model"
        assert error.details["reason"] == "記憶體不足"
    
    def test_create_api_connection_error(self):
        """測試創建 API 連線錯誤"""
        error = create_api_connection_error("http://test.com", "連線超時")
        
        assert isinstance(error, APIError)
        assert error.error_code == ErrorCode.API_CONNECTION_ERROR
        assert "API 連線失敗" in error.message
        assert error.details["url"] == "http://test.com"
        assert error.details["reason"] == "連線超時"


class TestRetryDecorator:
    """測試重試裝飾器"""
    
    def test_successful_execution(self):
        """測試成功執行（無需重試）"""
        @retry_on_exception(max_retries=3)
        def successful_function():
            return "success"
        
        result = successful_function()
        assert result == "success"
    
    def test_retry_on_exception(self):
        """測試異常重試"""
        call_count = 0
        
        @retry_on_exception(max_retries=3, delay=0.01, exceptions=(ValueError,))
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("測試錯誤")
            return "success"
        
        result = failing_function()
        assert result == "success"
        assert call_count == 3
    
    def test_max_retries_exceeded(self):
        """測試超過最大重試次數"""
        @retry_on_exception(max_retries=2, delay=0.01, exceptions=(ValueError,))
        def always_failing_function():
            raise ValueError("總是失敗")
        
        with pytest.raises(ValueError, match="總是失敗"):
            always_failing_function()
    
    def test_non_retryable_exception(self):
        """測試不可重試的異常"""
        @retry_on_exception(max_retries=3, exceptions=(ValueError,))
        def function_with_non_retryable_error():
            raise TypeError("不可重試的錯誤")
        
        with pytest.raises(TypeError, match="不可重試的錯誤"):
            function_with_non_retryable_error()


class TestErrorContext:
    """測試錯誤上下文管理器"""
    
    def test_successful_context(self):
        """測試成功的上下文"""
        with error_context("test_operation", param1="value1"):
            result = "success"
        
        assert result == "success"
    
    def test_context_with_whisper_exception(self):
        """測試上下文中的 Whisper 異常"""
        with pytest.raises(SystemError):
            with error_context("test_operation"):
                raise SystemError(
                    message="測試錯誤",
                    error_code=ErrorCode.SYSTEM_ERROR
                )
    
    def test_context_with_generic_exception(self):
        """測試上下文中的一般異常"""
        with pytest.raises(ValueError):
            with error_context("test_operation"):
                raise ValueError("一般錯誤")


class TestHTTPClient:
    """測試 HTTP 客戶端"""
    
    def test_successful_post_request(self):
        """測試成功的 POST 請求"""
        with patch('requests.Session.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"result": "success"}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            client = HTTPClient("http://test.com")
            result = client.post("/api/test", {"data": "test"})
            
            assert result == {"result": "success"}
            mock_post.assert_called_once()
    
    def test_connection_error(self):
        """測試連線錯誤"""
        with patch('requests.Session.post') as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError("連線失敗")
            
            client = HTTPClient("http://test.com")
            
            with pytest.raises(APIError) as exc_info:
                client.post("/api/test", {"data": "test"})
            
            assert exc_info.value.error_code == ErrorCode.API_CONNECTION_ERROR
    
    def test_timeout_error(self):
        """測試超時錯誤"""
        with patch('requests.Session.post') as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout("請求超時")
            
            client = HTTPClient("http://test.com", timeout=5)
            
            with pytest.raises(APIError) as exc_info:
                client.post("/api/test", {"data": "test"})
            
            assert exc_info.value.error_code == ErrorCode.API_TIMEOUT_ERROR
    
    def test_http_error(self):
        """測試 HTTP 錯誤"""
        with patch('requests.Session.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
            mock_post.return_value = mock_response
            
            client = HTTPClient("http://test.com")
            
            with pytest.raises(APIError) as exc_info:
                client.post("/api/test", {"data": "test"})
            
            assert exc_info.value.error_code == ErrorCode.API_RESPONSE_ERROR
    
    def test_json_parse_error(self):
        """測試 JSON 解析錯誤"""
        with patch('requests.Session.post') as mock_post:
            mock_response = Mock()
            mock_response.json.side_effect = ValueError("JSON 解析失敗")
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            client = HTTPClient("http://test.com")
            
            with pytest.raises(APIError) as exc_info:
                client.post("/api/test", {"data": "test"})
            
            assert exc_info.value.error_code == ErrorCode.API_RESPONSE_ERROR


class TestFileOperationsDecorator:
    """測試檔案操作裝飾器"""
    
    def test_successful_file_operation(self):
        """測試成功的檔案操作"""
        @handle_file_operations
        def read_file(filename):
            return f"content of {filename}"
        
        result = read_file("test.txt")
        assert result == "content of test.txt"
    
    def test_file_not_found_error(self):
        """測試檔案未找到錯誤"""
        @handle_file_operations
        def read_nonexistent_file():
            raise FileNotFoundError("檔案不存在")
        
        with pytest.raises(FileError) as exc_info:
            read_nonexistent_file()
        
        assert exc_info.value.error_code == ErrorCode.FILE_NOT_FOUND
    
    def test_permission_error(self):
        """測試權限錯誤"""
        @handle_file_operations
        def access_restricted_file():
            raise PermissionError("權限不足")
        
        with pytest.raises(FileError) as exc_info:
            access_restricted_file()
        
        assert exc_info.value.error_code == ErrorCode.FILE_READ_ERROR
    
    def test_io_error(self):
        """測試 I/O 錯誤"""
        @handle_file_operations
        def file_io_operation():
            raise IOError("I/O 錯誤")
        
        with pytest.raises(FileError) as exc_info:
            file_io_operation()
        
        assert exc_info.value.error_code == ErrorCode.FILE_READ_ERROR
    
    def test_unicode_decode_error(self):
        """測試編碼錯誤"""
        @handle_file_operations
        def decode_file():
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "編碼錯誤")
        
        with pytest.raises(FileError) as exc_info:
            decode_file()
        
        assert exc_info.value.error_code == ErrorCode.FILE_FORMAT_ERROR


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
