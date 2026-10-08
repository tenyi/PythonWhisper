# import json
import torch
from faster_whisper import WhisperModel  # 匯入 faster_whisper 模型庫
from faster_whisper.transcribe import Segment
from faster_whisper.transcribe import TranscriptionInfo
from opencc import OpenCC  # 匯入 opencc 用於簡繁轉換
from pydantic import BaseModel
import sys  # 匯入 sys 模組，用於處理命令列參數
from typing import List, Any  # 匯入類型提示
from translator_ollama import translate_text_ollama, summary_text_ollama, fix_s2twp
import uuid
import gc
import threading
import os
from pathlib import Path

# 匯入自定義異常和錯誤處理
from exceptions import (
    SystemError, ModelError, FileError, TranscriptionError,
    ErrorCode, create_model_load_error, create_file_not_found_error,
    create_transcription_error
)
from error_handler import error_context, handle_file_operations
from logging_config import get_logger

logger = get_logger("fast")

# 全域模型變數和鎖
model: WhisperModel | None = None
model_lock = threading.Lock()


def get_best_device() -> torch.device:
    """
    檢查系統支援的最佳運算裝置。

    優先順序：CUDA GPU > MPS (Apple Silicon) > ROCm (AMD GPU) > CPU

    回傳:
        torch.device: 最佳可用的運算裝置。

    Raises:
        SystemError: 當裝置檢查失敗時
    """
    try:
        with error_context("device_detection"):
            # 檢查 CUDA (NVIDIA GPU)
            if torch.cuda.is_available():
                device = torch.device("cuda")
                device_name = torch.cuda.get_device_name()
                logger.info(f"🚀 使用 CUDA GPU: {device_name}")
                return device

            # 檢查 MPS (Apple Silicon GPU)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("🚀 使用 Apple Silicon MPS GPU")
                return device

            # 檢查 ROCm (AMD GPU) - 透過檢查版本字串中是否包含 ROCm 相關資訊
            try:
                if "rocm" in torch.__version__.lower() or "hip" in torch.__version__.lower():
                    device = torch.device("cuda")  # ROCm 使用 cuda 介面
                    logger.info("🚀 使用 AMD ROCm GPU")
                    return device
            except Exception as e:
                logger.warning(f"檢查 ROCm 時發生例外: {e}")

            # 檢查其他可能的 GPU 後端
            if torch.cuda.is_available():  # 再次檢查以防萬一
                device = torch.device("cuda")
                device_name = torch.cuda.get_device_name()
                logger.info(f"🚀 使用 GPU: {device_name}")
                return device

            # 預設使用 CPU
            device = torch.device("cpu")
            logger.info("💻 使用 CPU (未偵測到 GPU 支援)")
            return device

    except Exception as e:
        raise SystemError(
            message=f"裝置檢查失敗: {str(e)}",
            error_code=ErrorCode.DEVICE_ERROR,
            original_exception=e
        )


def initialize_model():
    """
    初始化並載入 Whisper 模型（線程安全）。

    Raises:
        ModelError: 當模型初始化失敗時
        SystemError: 當裝置檢查失敗時
    """
    global model
    with model_lock:
        if model is None:
            try:
                with error_context("model_initialization"):
                    logger.info("初始化 Whisper 模型...")

                    # 設定 Whisper 模型的大小
                    #model_name = "breeze_asr_ct2"
                    model_name="large-v3"
                    device_type: str = get_best_device().type

                    compute_type = "default"
                    if device_type == "cuda":
                        compute_type = "float16"  # or "int8_float16"
                    elif device_type == "cpu":
                        compute_type = "int8"

                    # 載入模型
                    model = WhisperModel(
                        model_name, device=device_type, compute_type=compute_type
                    )
                    logger.info("模型初始化完成。")

            except Exception as e:
                raise create_model_load_error(model_name, str(e))
        else:
            logger.info("模型已經初始化，跳過重複初始化。")


def release_model():
    """
    釋放 Whisper 模型並清除 GPU 記憶體（線程安全）。

    Raises:
        SystemError: 當記憶體清理失敗時
    """
    global model
    with model_lock:
        if model is not None:
            try:
                with error_context("model_release"):
                    logger.info("正在釋放 Whisper 模型...")
                    # The underlying CTranslate2 model is deleted when the WhisperModel object is deleted.
                    del model
                    model = None

                    if torch.cuda.is_available():
                        logger.info("正在清除 CUDA 快取...")
                        torch.cuda.empty_cache()

                    gc.collect()
                    logger.info("模型已成功釋放，GPU 記憶體已清除。")

            except Exception as e:
                raise SystemError(
                    message=f"模型釋放失敗: {str(e)}",
                    error_code=ErrorCode.MEMORY_ERROR,
                    original_exception=e
                )
        else:
            logger.info("模型尚未載入，無需釋放。")


class OpenAI_Transcribe(BaseModel):
    text: str = ""
    language: str = ""
    duration: float = 0.0
    segments: List[Segment] = []


class Transcribe(BaseModel):
    """語音轉文字結果（支援自動型別檢查）"""

    transcript: str = ""  # 逐字稿
    file_name: str = ""  # 檔案名稱
    summary: str = ""  # 摘要
    srt: str = ""  # SRT 字幕
    vtt: str = ""  # WEBVTT 字幕
    translation: str = ""  # 翻譯
    guid: str = str(uuid.uuid4())  # 唯一識別碼
    duration: float = 0.0  # 音檔總時間長度（秒）
    whisper_api_cost: float = 0.0  # Whisper API 使用成本 (美元)


# 定義將秒數格式化為 VTT 時間格式的函式
def format_time_vtt(seconds_val: float) -> str:
    hours: int = int(seconds_val // 3600)  # 計算小時
    minutes: int = int((seconds_val % 3600) // 60)  # 計算分鐘
    remaining_seconds: float = seconds_val % 60  # 計算剩餘秒數
    milliseconds: int = int(
        (remaining_seconds - int(remaining_seconds)) * 1000
    )  # 計算毫秒
    return f"{hours:02d}:{minutes:02d}:{int(remaining_seconds):02d}.{milliseconds:03d}"  # 回傳格式化後的時間字串


# 定義將秒數格式化為 SRT 時間格式的函式
def format_time_srt(seconds_val: float) -> str:
    hours: int = int(seconds_val // 3600)  # 計算小時
    minutes: int = int((seconds_val % 3600) // 60)  # 計算分鐘
    remaining_seconds: float = seconds_val % 60  # 計算剩餘秒數
    milliseconds: int = int(
        (remaining_seconds - int(remaining_seconds)) * 1000
    )  # 計算毫秒
    return f"{hours:02d}:{minutes:02d}:{int(remaining_seconds):02d},{milliseconds:03d}"  # 回傳格式化後的時間字串


# 定義寫入逐字稿檔案的函式
@handle_file_operations
def write_transcript(
    segments_list: List[Segment], info: TranscriptionInfo, filename: str
) -> str:
    """
    將語音片段寫入逐字稿檔案

    Args:
        segments_list: 語音片段列表
        info: 轉錄資訊
        filename: 輸出檔案名稱

    Returns:
        逐字稿內容

    Raises:
        FileError: 當檔案操作失敗時
    """
    try:
        with error_context("write_transcript", filename=filename):
            with open(filename, "w", encoding="utf-8") as f:
                cc = None
                if info.language == "zh":  # 判斷偵測到的語言是否為中文
                    # 如果是中文，則初始化 OpenCC 物件，用於簡轉繁（台灣正體，包含詞彙轉換）
                    cc = OpenCC("s2twp.json")

                for segment in segments_list:  # 迭代處理每個語音片段
                    segment_text: str = segment.text
                    if cc is not None:  # 如果需要轉換繁體中文
                        segment_text = fix_s2twp(cc.convert(segment.text))  # 修正 s2twp 誤轉（如「只要」→「隻要」）
                    # 將格式化後的時間戳記和文字寫入檔案
                    f.write(
                        "[%s --> %s]\n %s\n\n"  # 時間戳記格式，並空一行
                        % (
                            format_time_vtt(segment.start),  # 格式化開始時間
                            format_time_vtt(segment.end),  # 格式化結束時間
                            segment_text,  # 語音片段文字
                        )
                    )

            # 讀取整個檔案內容並回傳
            with open(filename, "r", encoding="utf-8") as file:
                transcript = file.read()
            return transcript  # 回傳 逐字稿 內容

    except Exception as e:
        # handle_file_operations 裝飾器會處理檔案相關異常
        # 這裡處理其他可能的異常
        raise FileError(
            message=f"寫入逐字稿失敗: {str(e)}",
            error_code=ErrorCode.FILE_WRITE_ERROR,
            details={"filename": filename},
            original_exception=e
        )


# 定義翻譯逐字稿檔案的函式
@handle_file_operations
def write_translation(
    segments_list: List[Segment], info: TranscriptionInfo, filename: str
) -> str:
    """
    翻譯逐字稿並寫入檔案
    
    Args:
        segments_list: 語音片段列表
        info: 轉錄資訊
        filename: 輸出檔案名稱
        
    Returns:
        翻譯後的逐字稿內容
        
    Raises:
        FileError: 當檔案操作失敗時
        TranslationError: 當翻譯失敗時
    """
    try:
        with error_context("write_translation", filename=filename):
            with open(filename, "w", encoding="utf-8") as f:
                cc = None
                if info.language == "zh":  # 判斷偵測到的語言是否為中文
                    # 如果是中文，則初始化 OpenCC 物件，用於簡轉繁（台灣正體，包含詞彙轉換）
                    cc = OpenCC("s2twp.json")

                for segment in segments_list:  # 迭代處理每個語音片段
                    segment_text: str = segment.text
                    if not segment_text.strip():
                        continue
                    
                    # 翻譯文字
                    translated_text = translate_text_ollama(segment_text)
                    
                    # 如果需要轉換繁體中文
                    if cc is not None:
                        translated_text = fix_s2twp(cc.convert(translated_text))  # 修正 s2twp 誤轉（如「只要」→「隻要」）
                    
                    # 將格式化後的時間戳記和文字寫入檔案
                    f.write(
                        "[%s --> %s]\n %s\n\n"  # 時間戳記格式，並空一行
                        % (
                            format_time_vtt(segment.start),  # 格式化開始時間
                            format_time_vtt(segment.end),  # 格式化結束時間
                            translated_text,  # 翻譯後的文字
                        )
                    )

            # 讀取整個檔案內容並回傳
            with open(filename, "r", encoding="utf-8") as file:
                transcript = file.read()
            logger.info(f"翻譯檔案已寫入: {filename}")
            return transcript
    except Exception as e:
        logger.error(f"翻譯逐字稿失敗: {filename} - {str(e)}")
        raise


# 回傳 OpenAI Whisper API 格式 JSON 字串
def get_openai_json_text(segments: List[Segment], info: TranscriptionInfo) -> str:
    full_text = ""
    for segment in segments:
        full_text += segment.text + " "

    transcribe_json = OpenAI_Transcribe(
        text=full_text,
        language=info.language,
        duration=info.duration,
        segments=segments,
    )

    return transcribe_json.model_dump_json()


# 定義寫入 JSON 檔案的函式，回傳 OpenAI Whisper API 格式 JSON 字串
@handle_file_operations
def write_json(segments: List[Segment], info: TranscriptionInfo, filename: str) -> str:
    """
    寫入 JSON 檔案，回傳 OpenAI Whisper API 格式
    
    Args:
        segments: 語音片段列表
        info: 轉錄資訊
        filename: 輸出檔案名稱
        
    Returns:
        JSON 格式的字串
        
    Raises:
        FileError: 當檔案操作失敗時
    """
    try:
        with error_context("write_json", filename=filename):
            json_text = get_openai_json_text(segments, info)

            with open(filename, "w", encoding="utf-8") as f:
                f.write(json_text)

            logger.info(f"JSON 檔案已寫入: {filename}")
            return json_text
    except Exception as e:
        logger.error(f"寫入 JSON 檔案失敗: {filename} - {str(e)}")
        raise


# 定義寫入 SRT 字幕檔案的函式
def write_srt(segments_list: List[Segment], filename: str) -> str:
    with open(
        filename, "w", encoding="utf-8"
    ) as f:  # 開啟檔案以寫入模式，使用 UTF-8 編碼
        for i, segment in enumerate(segments_list):  # 迭代處理每個語音片段，並取得索引
            # 寫入 SRT 格式的字幕內容
            f.write(f"{i + 1}\n")  # 字幕序號
            f.write(
                "%s --> %s\n"  # 時間戳記格式
                % (
                    format_time_srt(segment.start),  # 格式化開始時間
                    format_time_srt(segment.end),  # 格式化結束時間
                )
            )
            f.write(f"{segment.text}\n\n")  # 字幕文字，並空一行
    # 讀取整個檔案內容並回傳
    # 這樣做是為了確保檔案已經寫入完成
    with open(filename, "r", encoding="utf-8") as file:
        srt_text = file.read()
    return srt_text  # 回傳 SRT 字幕內容


# 定義寫入 VTT 字幕檔案的函式
def write_vtt(segments_list: List[Segment], filename: str) -> str:
    with open(
        filename, "w", encoding="utf-8"
    ) as f:  # 開啟檔案以寫入模式，使用 UTF-8 編碼
        f.write("WEBVTT\n\n")  # VTT 檔案標頭
        for segment in segments_list:  # 迭代處理每個語音片段
            # 寫入 VTT 格式的字幕內容
            f.write(
                "%s --> %s\n"  # 時間戳記格式
                % (
                    format_time_vtt(segment.start),  # 格式化開始時間
                    format_time_vtt(segment.end),  # 格式化結束時間
                )
            )
            f.write(f"{segment.text}\n\n")  # 字幕文字，並空一行
    with open(filename, "r", encoding="utf-8") as file:
        vtt_text = file.read()
    return vtt_text  # 回傳 VTT 字幕內容


# 定義總結會議摘要的函式
def summary_text(segments: List[Segment]) -> str:
    summaries = []
    # 先找出各段落分開的總結
    for segment in segments:
        part_summary = summary_text_ollama(segment.text)
        summaries.append(part_summary)
    # 將各段落分開的總結合併成一個總結
    full_text = " ".join(summaries)
    # 總結會議摘要
    summary = summary_text_ollama(full_text)
    return summary


def transcribe_file(audio_file: str) -> tuple[list[Segment], TranscriptionInfo]:
    """
    轉錄音訊檔案

    Args:
        audio_file: 音訊檔案路徑

    Returns:
        語音片段列表和轉錄資訊

    Raises:
        FileError: 當音訊檔案不存在或無法讀取時
        TranscriptionError: 當轉錄失敗時
        ModelError: 當模型未初始化時
    """
    # 檢查檔案是否存在
    if not os.path.exists(audio_file):
        raise create_file_not_found_error(audio_file)

    # 檢查檔案大小
    file_size = os.path.getsize(audio_file)
    if file_size == 0:
        raise FileError(
            message=f"音訊檔案為空: {audio_file}",
            error_code=ErrorCode.FILE_SIZE_ERROR,
            details={"file_path": audio_file, "file_size": file_size}
        )

    try:
        with error_context("transcribe_file", audio_file=audio_file, file_size=file_size):
            initialize_model()  # 確保模型已載入

            if model is None:
                raise ModelError(
                    message="模型未正確初始化",
                    error_code=ErrorCode.MODEL_NOT_INITIALIZED
                )

            # 使用 Whisper 模型進行語音轉錄，beam_size 參數影響搜尋寬度
            segments, info_obj = model.transcribe(audio_file, beam_size=5)
            # 將 segments 生成器轉換為列表，以便可以多次迭代使用
            segments_list: List[Any] = list(segments)

            logger.info(f"轉錄完成，共 {len(segments_list)} 個片段")
            return segments_list, info_obj

    except Exception as e:
        if isinstance(e, (FileError, TranscriptionError, ModelError)):
            raise
        else:
            raise create_transcription_error(audio_file, str(e))


# 定義 transcribe 函式，回傳 OpenAI Whisper API 格式 JSON 字串
def transcribe_openai_format(audio_file: str) -> str:
    """
    轉錄音訊檔案並回傳 OpenAI Whisper API 格式的 JSON 字串

    Args:
        audio_file: 音訊檔案路徑

    Returns:
        OpenAI Whisper API 格式的 JSON 字串

    Raises:
        FileError: 當音訊檔案不存在或無法讀取時
        TranscriptionError: 當轉錄失敗時
        ModelError: 當模型未初始化時
    """
    try:
        with error_context("transcribe_openai_format", audio_file=audio_file):
            # 使用 Whisper 模型進行語音轉錄，回傳OpenAI Whisper API格式
            segments_list: list[Segment]
            info_obj: TranscriptionInfo
            segments_list, info_obj = transcribe_file(audio_file)

            # 記錄偵測到的語言及其機率
            logger.info(
                f"偵測到語言 '{info_obj.language}' 機率: {info_obj.language_probability:.4f}"
            )

            # 產生 JSON 檔案
            json_text = get_openai_json_text(segments_list, info_obj)
            return json_text

    except Exception as e:
        if isinstance(e, (FileError, TranscriptionError, ModelError)):
            raise
        else:
            raise create_transcription_error(audio_file, str(e))


def transcribe_full(audio_file: str) -> str:
    # 使用 Whisper 模型進行語音轉錄
    segments_list: list[Segment]
    info_obj: TranscriptionInfo
    segments_list, info_obj = transcribe_file(audio_file)
    # 印出偵測到的語言及其機率
    print(
        "Detected language '%s' with probability %f"
        % (info_obj.language, info_obj.language_probability)
    )

    # 根據輸入的音訊檔名產生輸出的檔名
    base_filename: str = audio_file.rsplit(".", 1)[0]  # 取得不含副檔名的基本檔名
    transcript_filename: str = f"{base_filename}_transcript.txt"  # 逐字稿檔名
    srt_filename: str = f"{base_filename}.srt"  # SRT 字幕檔名
    vtt_filename: str = f"{base_filename}.vtt"  # VTT 字幕檔名
    json_filename: str = f"{base_filename}.json"  # JSON 檔名
    translation_filename: str = f"{base_filename}_translation.txt"  # 翻譯檔名
    # 產生 JSON 檔案
    write_json(segments_list, info_obj, json_filename)
    # 產生逐字稿
    transcript = write_transcript(segments_list, info_obj, transcript_filename)
    # 產生 SRT 檔案
    srt_text = write_srt(segments_list, srt_filename)
    # 產生 VTT 檔案
    vtt_text = write_vtt(segments_list, vtt_filename)
    # 產生翻譯
    translation = write_translation(segments_list, info_obj, translation_filename)
    # summary = summary_text(segments_list)
    # if summary == "無法得出會議摘要":
    summary = summary_text_ollama(transcript)

    transcribe = Transcribe(
        transcript=transcript,
        file_name=audio_file,
        summary=summary,
        srt=srt_text,
        vtt=vtt_text,
        translation=translation,
        duration=info_obj.duration,
    )
    return transcribe.model_dump_json()


# 定義主函式，處理音訊檔案並產生輸出
def main(audio_file: str) -> None:
    """
    主函式，處理音訊檔案並產生輸出

    Args:
        audio_file: 音訊檔案路徑

    Raises:
        FileError: 當音訊檔案不存在或無法讀取時
        TranscriptionError: 當轉錄失敗時
        ModelError: 當模型操作失敗時
        SystemError: 當系統操作失敗時
    """
    try:
        with error_context("main_process", audio_file=audio_file):
            result = transcribe_full(audio_file)
            logger.info("轉錄處理完成")
            return result
    except Exception as e:
        logger.error(f"主程式執行失敗: {str(e)}")
        raise
    finally:
        try:
            release_model()  # 釋放模型資源
        except Exception as e:
            logger.error(f"模型釋放失敗: {str(e)}")


# 定義顯示使用說明的函式
def usage(script_name: str) -> None:
    print(f"Usage: python {script_name} <audio_file>")  # 印出使用語法
    print(f"Example: python {script_name} audio.mp3")  # 印出使用範例


# 主程式進入點
if __name__ == "__main__":
    if len(sys.argv) > 1:  # 檢查是否有提供命令列參數 (音訊檔案路徑)
        main(sys.argv[1])  # 如果有，則呼叫 main 函式並傳入音訊檔案路徑
    else:
        usage(sys.argv[0])  # 如果沒有提供參數，則顯示使用說明
