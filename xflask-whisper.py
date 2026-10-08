from flask import Flask, request, jsonify
import os
import logging
import tempfile
import time
from xfast import transcribe_full, transcribe_openai_format
from translator_ollama import translate_text_ollama, summary_text_ollama
from common import FileHandler, ErrorHandler, resource_manager, safe_execute
from error_handler import FileProcessingError, ValidationError

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("whisper-server.log"), logging.StreamHandler()],
)

logger = logging.getLogger("whisper-fastcgi")


def create_app() -> Flask:
    app = Flask(__name__)

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"內部伺服器錯誤: {str(error)}")
        return jsonify({"error": "內部伺服器錯誤"}), 500

    @app.errorhandler(400)
    def bad_request(error):
        logger.warning(f"錯誤的請求: {str(error)}")
        return jsonify({"error": "錯誤的請求"}), 400

    # @app.route("/release_model", methods=["POST"])
    # def release_model_endpoint():
    #     """
    #     API endpoint to release the Whisper model and free up GPU memory.
    #     """
    #     release_model()
    #     return jsonify({"message": "Model released successfully."}), 200

    @app.route("/translate", methods=["POST"])
    def translate():
        if not request.json:
            return jsonify({"error": "無效的 JSON 資料"}), 400
        text = request.json.get("text")
        language = request.json.get("language")
        return translate_text_ollama(text, language)

    @app.route("/summary", methods=["POST"])
    def summary():
        if not request.json:
            return jsonify({"error": "無效的 JSON 資料"}), 400
        text = request.json.get("text")
        return summary_text_ollama(text)

    @app.route("/transcribe", methods=["POST"])
    def transcribe():
        start_time = time.time()
        temp_file_path = None
        audio_file = None

        try:
            # 統一的檔案檢查邏輯
            if "file" not in request.files or len(request.files) == 0:
                logger.error("沒有找到上傳的檔案")
                return jsonify({"error": "需要上傳檔案"}), 400

            audio_file = request.files["file"]

            # 檢查檔案名稱是否有效
            if not audio_file.filename:
                logger.error("檔案名稱無效或為空")
                return jsonify({"error": "檔案名稱無效或為空"}), 400

            # 創建臨時文件保存上傳的音訊
            temp_file_fd, temp_file_path = tempfile.mkstemp(
                suffix=os.path.splitext(audio_file.filename)[1]
            )
            os.close(temp_file_fd)

            # 將上傳的檔案保存到臨時位置
            audio_file.save(temp_file_path)
            resource_manager.add_resource(temp_file_path)
            logger.info(f"已保存音訊檔案至臨時位置: {temp_file_path}")

            # 驗證檔案類型和大小
            try:
                FileHandler.validate_audio_file(temp_file_path)
            except ValidationError as e:
                logger.error(f"檔案驗證失敗: {str(e)}")
                return jsonify({"error": str(e)}), 400

            # 進行轉錄
            logger.info(f"開始處理音訊檔案: {audio_file.filename}")
            json_content = safe_execute(
                transcribe_openai_format,
                temp_file_path,
                fallback_value='{"error": "轉錄失敗"}',
            )

            # 計算處理時間
            process_time = time.time() - start_time
            logger.info(f"處理完成，耗時: {process_time:.2f} 秒")

            return (
                json_content,
                200,
                {
                    "Content-Type": "application/json; charset=utf-8",
                    "Process-Time": str(process_time),
                },
            )

        except FileProcessingError as e:
            logger.error(f"檔案處理錯誤: {str(e)}")
            return jsonify({"error": str(e)}), 500
        except Exception as e:
            logger.error(f"處理音訊時發生未預期的錯誤: {str(e)}", exc_info=True)
            return jsonify({"error": "處理音訊時發生錯誤"}), 500

        finally:
            # 確保資源被正確清理
            if audio_file:
                audio_file.close()

            # 清理臨時檔案
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.debug(f"已清理臨時檔案: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"清理臨時檔案失敗: {str(e)}")

    @app.route("/transcribe_full", methods=["POST"])
    def transcribefull():
        start_time = time.time()
        temp_file_path = None
        audio_file = None

        try:
            # 統一的檔案檢查邏輯
            if "file" not in request.files or len(request.files) == 0:
                logger.error("沒有找到上傳的檔案")
                return jsonify({"error": "需要上傳檔案"}), 400

            audio_file = request.files["file"]

            # 檢查檔案名稱是否有效
            if not audio_file.filename:
                logger.error("檔案名稱無效或為空")
                return jsonify({"error": "檔案名稱無效或為空"}), 400

            # 驗證檔案類型和大小
            try:
                FileHandler.validate_audio_file(audio_file.filename)
            except ValidationError as e:
                logger.error(f"檔案驗證失敗: {str(e)}")
                return jsonify({"error": str(e)}), 400

            # 創建臨時文件保存上傳的音訊
            temp_file_fd, temp_file_path = tempfile.mkstemp(
                suffix=os.path.splitext(audio_file.filename)[1]
            )
            os.close(temp_file_fd)

            # 將上傳的檔案保存到臨時位置
            audio_file.save(temp_file_path)
            resource_manager.add_resource(temp_file_path)  # 加入資源管理
            logger.info(f"已保存音訊檔案至臨時位置: {temp_file_path}")

            # 進行完整轉錄
            logger.info(f"開始處理音訊檔案: {audio_file.filename}")
            json_content = safe_execute(
                transcribe_full,
                temp_file_path,
                fallback_value='{"error": "完整轉錄失敗"}',
            )

            # 計算處理時間
            process_time = time.time() - start_time
            logger.info(f"處理完成，耗時: {process_time:.2f} 秒")

            return (
                json_content,
                200,
                {
                    "Content-Type": "application/json; charset=utf-8",
                    "Process-Time": str(process_time),
                },
            )

        except FileProcessingError as e:
            logger.error(f"檔案處理錯誤: {str(e)}")
            return jsonify({"error": str(e)}), 500
        except Exception as e:
            logger.error(f"處理音訊時發生未預期的錯誤: {str(e)}", exc_info=True)
            return jsonify({"error": "處理音訊時發生錯誤"}), 500

        finally:
            # 確保資源被正確清理
            if audio_file:
                audio_file.close()

            # 清理臨時檔案
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.debug(f"已清理臨時檔案: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"清理臨時檔案失敗: {str(e)}")

    @app.route("/health", methods=["GET"])
    def health_check():
        return jsonify({"status": "healthy", "service": "whisper-api"}), 200

    return app


if __name__ == "__main__":
    # 使用 flask 啟動伺服器
    # app.run(host="0.0.0.0", port=22434, debug=True)
    print("Starting whisper server...\n")
    # 使用 waitress 啟動伺服器
    from waitress import serve

    serve(create_app(), host="0.0.0.0", threads=4, port=22434)
