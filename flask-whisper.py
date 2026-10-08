from flask import Flask, request, jsonify
import os
import logging
import tempfile
import time
import threading
import queue
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional
from fast import transcribe_full, transcribe_openai_format  # , release_model
from translator_ollama import translate_text_ollama, summary_text_ollama

# 匯入自定義異常和錯誤處理
from exceptions import (
    WhisperBaseException, FileError, TaskError, APIError, TranslationError,
    SummaryError, ErrorCode, create_task_not_found_error
)
from error_handler import error_context
from logging_config import get_logger

# 使用統一的日誌系統
flask_logger = get_logger("flask")


def handle_api_error(e: Exception) -> tuple:
    """
    統一的 API 錯誤處理函數

    Args:
        e: 異常物件

    Returns:
        (response_data, status_code) 元組
    """
    if isinstance(e, WhisperBaseException):
        flask_logger.error(f"業務邏輯錯誤: {str(e)}", extra={"error_code": e.error_code.value})
        return jsonify(e.to_dict()), 400
    elif isinstance(e, ValueError):
        flask_logger.error(f"參數錯誤: {str(e)}")
        return jsonify({
            "error": {
                "code": ErrorCode.API_INVALID_REQUEST.value,
                "message": f"參數錯誤: {str(e)}",
                "type": "ValueError"
            }
        }), 400
    else:
        flask_logger.error(f"未預期錯誤: {str(e)}", exc_info=True)
        return jsonify({
            "error": {
                "code": ErrorCode.SYSTEM_ERROR.value,
                "message": "內部伺服器錯誤",
                "type": "InternalServerError"
            }
        }), 500


def validate_file_upload(request) -> tuple:
    """
    驗證檔案上傳

    Args:
        request: Flask request 物件

    Returns:
        (audio_file, error_response) 元組，如果驗證失敗則 audio_file 為 None
    """
    if "file" not in request.files:
        error_response = jsonify({
            "error": {
                "code": ErrorCode.API_INVALID_REQUEST.value,
                "message": "沒有上傳檔案",
                "type": "FileUploadError"
            }
        }), 400
        return None, error_response

    if len(request.files) == 0:
        flask_logger.warning("沒有找到上傳的檔案")
        error_response = jsonify({
            "error": {
                "code": ErrorCode.API_INVALID_REQUEST.value,
                "message": "需要上傳檔案",
                "type": "FileUploadError"
            }
        }), 400
        return None, error_response

    audio_file = request.files["file"]

    # 檢查是否有選擇檔案
    if audio_file.filename == "":
        error_response = jsonify({
            "error": {
                "code": ErrorCode.API_INVALID_REQUEST.value,
                "message": "沒有選擇檔案",
                "type": "FileUploadError"
            }
        }), 400
        return None, error_response

    # 檢查檔案大小（假設最大 100MB）
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    if hasattr(audio_file, 'content_length') and audio_file.content_length > MAX_FILE_SIZE:
        error_response = jsonify({
            "error": {
                "code": ErrorCode.FILE_SIZE_ERROR.value,
                "message": f"檔案過大，最大允許 {MAX_FILE_SIZE // (1024*1024)}MB",
                "type": "FileSizeError"
            }
        }), 400
        return None, error_response

    return audio_file, None


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TranscribeTask:
    task_id: str
    status: TaskStatus
    file_path: str
    filename: str
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class TranscribeQueue:
    """管理 transcribe 任務的隊列系統"""

    def __init__(self, max_workers: int = 1):
        self.task_queue = queue.Queue()
        self.tasks: Dict[str, TranscribeTask] = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.is_processing = False
        self.current_task_id: Optional[str] = None

        # 啟動工作線程
        self.executor.submit(self._worker)

    def add_task(self, file_path: str, filename: str, task_type: str = "openai") -> str:
        """添加新的 transcribe 任務"""
        task_id = str(uuid.uuid4())
        task = TranscribeTask(
            task_id=task_id,
            status=TaskStatus.PENDING,
            file_path=file_path,
            filename=filename,
            created_at=time.time(),
        )

        with self.lock:
            self.tasks[task_id] = task
            self.task_queue.put((task_id, task_type))

        flask_logger.info(f"任務 {task_id} 已加入隊列，檔案: {filename}")
        return task_id

    def get_task_status(self, task_id: str) -> Optional[TranscribeTask]:
        """獲取任務狀態"""
        with self.lock:
            return self.tasks.get(task_id)

    def _worker(self):
        """工作線程，處理隊列中的任務"""
        while True:
            try:
                # 從隊列中獲取任務
                task_id, task_type = self.task_queue.get(timeout=1)

                with self.lock:
                    if task_id not in self.tasks:
                        continue

                    task: TranscribeTask = self.tasks[task_id]
                    task.status = TaskStatus.PROCESSING
                    task.started_at = time.time()
                    self.is_processing = True
                    self.current_task_id = task_id

                flask_logger.info(f"開始處理任務 {task_id}")

                try:
                    with error_context("task_processing", task_id=task_id, task_type=task_type):
                        # 執行 transcribe
                        if task_type == "full":
                            result = transcribe_full(task.file_path)
                        else:
                            result = transcribe_openai_format(task.file_path)

                        # 更新任務狀態
                        with self.lock:
                            task.status = TaskStatus.COMPLETED
                            task.result = result
                            task.completed_at = time.time()

                        flask_logger.info(f"任務 {task_id} 處理完成")

                except WhisperBaseException as e:
                    flask_logger.error(
                        f"處理任務 {task_id} 時發生業務邏輯錯誤: {str(e)}",
                        extra={"error_code": e.error_code.value, "task_id": task_id}
                    )

                    with self.lock:
                        task.status = TaskStatus.FAILED
                        task.error = e.to_dict()
                        task.completed_at = time.time()

                except Exception as e:
                    flask_logger.error(
                        f"處理任務 {task_id} 時發生未預期錯誤: {str(e)}",
                        exc_info=True,
                        extra={"task_id": task_id}
                    )

                    with self.lock:
                        task.status = TaskStatus.FAILED
                        task.error = {
                            "error": {
                                "code": ErrorCode.SYSTEM_ERROR.value,
                                "message": str(e),
                                "type": type(e).__name__
                            }
                        }
                        task.completed_at = time.time()

                finally:
                    # 清理臨時文件
                    try:
                        if os.path.exists(task.file_path):
                            os.remove(task.file_path)
                            flask_logger.info(f"已清理臨時文件: {task.file_path}")
                    except Exception as e:
                        flask_logger.error(f"清理臨時文件失敗: {str(e)}")

                    with self.lock:
                        self.is_processing = False
                        self.current_task_id = None

                    self.task_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                flask_logger.error(f"工作線程發生未預期錯誤: {str(e)}", exc_info=True)

    def get_queue_status(self) -> Dict:
        """獲取隊列狀態"""
        with self.lock:
            pending_count = self.task_queue.qsize()
            return {
                "is_processing": self.is_processing,
                "current_task_id": self.current_task_id,
                "pending_tasks": pending_count,
                "total_tasks": len(self.tasks),
            }


# 全域隊列實例
transcribe_queue = TranscribeQueue(max_workers=1)


# app = Flask(__name__)
def create_app() -> Flask:
    app = Flask(__name__)

    # @app.route("/release_model", methods=["POST"])
    # def release_model_endpoint():
    #     """
    #     API endpoint to release the Whisper model and free up GPU memory.
    #     """
    #     release_model()
    #     return jsonify({"message": "Model released successfully."}), 200

    @app.route("/translate", methods=["POST"])
    def translate():
        """翻譯文字 API"""
        try:
            with error_context("translate_api"):
                if not request.json:
                    raise ValueError("請求必須包含 JSON 資料")

                text = request.json.get("text")
                language = request.json.get("language", "zh-TW")

                if not text:
                    raise ValueError("text 參數不能為空")

                result = translate_text_ollama(text, language)
                return jsonify({"result": result})

        except Exception as e:
            return handle_api_error(e)

    @app.route("/summary", methods=["POST"])
    def summary():
        """文字摘要 API"""
        try:
            with error_context("summary_api"):
                if not request.json:
                    raise ValueError("請求必須包含 JSON 資料")

                text = request.json.get("text")

                if not text:
                    raise ValueError("text 參數不能為空")

                result = summary_text_ollama(text)
                return jsonify({"result": result})

        except Exception as e:
            return handle_api_error(e)

    @app.route("/transcribe", methods=["POST"])
    def transcribe():
        """提交 transcribe 任務到隊列"""
        temp_file_path = None
        audio_file = None

        try:
            with error_context("transcribe_api"):
                # 驗證檔案上傳
                audio_file, error_response = validate_file_upload(request)
                if error_response:
                    return error_response

                # 創建臨時文件保存上傳的音訊
                temp_file_fd, temp_file_path = tempfile.mkstemp(
                    suffix=os.path.splitext(audio_file.filename)[1]
                )
                os.close(temp_file_fd)
                audio_file.save(temp_file_path)
                flask_logger.info(f"已保存音訊檔案至臨時位置: {temp_file_path}")

                # 將任務加入隊列
                task_id = transcribe_queue.add_task(
                    temp_file_path, audio_file.filename, "openai"
                )

                return jsonify({
                    "task_id": task_id,
                    "status": "pending",
                    "message": "任務已加入隊列，請使用 task_id 查詢結果",
                }), 202

        except Exception as e:
            flask_logger.error(f"處理音訊時發生錯誤: {str(e)}", exc_info=True)

            # 清理臨時文件
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    flask_logger.info(f"已清理臨時文件: {temp_file_path}")
                except Exception as cleanup_error:
                    flask_logger.error(f"清理臨時文件失敗: {str(cleanup_error)}")

            return handle_api_error(e)

        finally:
            # 清理資源
            if audio_file:
                try:
                    audio_file.close()
                except Exception as e:
                    flask_logger.warning(f"關閉檔案時發生錯誤: {str(e)}")

    @app.route("/transcribe_full", methods=["POST"])
    def transcribefull():
        """提交完整 transcribe 任務到隊列"""
        # 檢查是否有檔案上傳
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        if len(request.files) == 0:
            flask_logger.warning("沒有找到上傳的檔案")
            return jsonify({"error": "需要上傳檔案"}), 400

        audio_file = request.files["file"]

        # 檢查是否有選擇檔案
        if audio_file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        try:
            # 創建臨時文件保存上傳的音訊
            temp_file_fd, temp_file_path = tempfile.mkstemp(
                suffix=os.path.splitext(audio_file.filename)[1]
            )
            os.close(temp_file_fd)
            audio_file.save(temp_file_path)
            flask_logger.info(f"已保存音訊檔案至臨時位置: {temp_file_path}")

            # 將任務加入隊列
            task_id = transcribe_queue.add_task(
                temp_file_path, audio_file.filename, "full"
            )

            return jsonify(
                {
                    "task_id": task_id,
                    "status": "pending",
                    "message": "任務已加入隊列，請使用 task_id 查詢結果",
                }
            ), 202

        except Exception as e:
            flask_logger.error(f"處理音訊時發生錯誤: {str(e)}", exc_info=True)
            # 清理臨時文件
            try:
                if "temp_file_path" in locals() and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
            except:
                pass
            return jsonify({"error": f"處理失敗: {str(e)}"}), 500

        finally:
            # 清理資源
            if audio_file:
                audio_file.close()

    @app.route("/task/<task_id>", methods=["GET"])
    def get_task_status(task_id: str):
        """查詢任務狀態"""
        try:
            with error_context("get_task_status", task_id=task_id):
                if not task_id or not task_id.strip():
                    raise ValueError("task_id 不能為空")

                task = transcribe_queue.get_task_status(task_id)

                if not task:
                    raise create_task_not_found_error(task_id)

                response_data = {
                    "task_id": task.task_id,
                    "status": task.status.value,
                    "filename": task.filename,
                    "created_at": task.created_at,
                    "started_at": task.started_at,
                    "completed_at": task.completed_at,
                }

                if task.status == TaskStatus.COMPLETED and task.result:
                    response_data["result"] = task.result
                    # 計算處理時間
                    if task.started_at and task.completed_at:
                        response_data["process_time"] = task.completed_at - task.started_at
                elif task.status == TaskStatus.FAILED and task.error:
                    response_data["error"] = task.error

                return jsonify(response_data)

        except Exception as e:
            return handle_api_error(e)

    @app.route("/queue/status", methods=["GET"])
    def get_queue_status():
        """查詢隊列狀態"""
        try:
            with error_context("get_queue_status"):
                status = transcribe_queue.get_queue_status()
                return jsonify(status)
        except Exception as e:
            return handle_api_error(e)

    @app.route("/health", methods=["GET"])
    def health_check():
        """健康檢查端點"""
        try:
            with error_context("health_check"):
                queue_status = transcribe_queue.get_queue_status()
                return jsonify({"status": "healthy", "queue_info": queue_status})
        except Exception as e:
            flask_logger.error(f"健康檢查失敗: {str(e)}")
            return jsonify({
                "status": "unhealthy",
                "error": str(e)
            }), 503

    # 全域錯誤處理器
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "error": {
                "code": 404,
                "message": "API 端點不存在",
                "type": "NotFound"
            }
        }), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            "error": {
                "code": 405,
                "message": "HTTP 方法不被允許",
                "type": "MethodNotAllowed"
            }
        }), 405

    @app.errorhandler(500)
    def internal_error(error):
        flask_logger.error(f"內部伺服器錯誤: {str(error)}")
        return jsonify({
            "error": {
                "code": 500,
                "message": "內部伺服器錯誤",
                "type": "InternalServerError"
            }
        }), 500

    return app


if __name__ == "__main__":
    # 使用 flask 啟動伺服器
    #app.run(host="0.0.0.0", port=22434, debug=True)
    print("Starting whisper server...\n")
    # 使用 waitress 啟動伺服器
    from waitress import serve
    serve(create_app(), host="0.0.0.0", threads=4, port=22434)


# 使用 waitress 啟動伺服器
# waitress-serve --threads=4 --port=22434 flask-whisper:create_app

# 使用 gunicorn 啟動伺服器
# gunicorn --workers=1 --worker-class=gthread --threads=4 --bind=0.0.0.0:22434 'flask-whisper:create_app()'
# 注意：建議使用 --workers 1 避免多進程間的模型衝突
