# import json
import torch
import whisper  # 匯入 openai-whisper 模型庫
from pydantic import BaseModel
import sys  # 匯入 sys 模組，用於處理命令列參數
from typing import List, Optional  # 匯入類型提示
from translator_ollama import translate_text_ollama, summary_text_ollama
import uuid
import gc
import threading
import logging

# 匯入共用模組
from common import (
    Segment,
    TranscriptionInfo,
    FileHandler,
    opencc_manager,
    safe_execute,
    handle_errors,
    log_error,
    log_info,
)
from error_handler import FileProcessingError, ValidationError

# 設定日誌
logger = logging.getLogger(__name__)

# 全域模型變數和鎖 - 使用更好的類型提示
model: Optional[whisper.Whisper] = None
model_lock = threading.Lock()


# 類別定義已從 common 模組匯入


def get_best_device() -> torch.device:
    """
    檢查系統支援的最佳運算裝置。

    優先順序：CUDA GPU > MPS (Apple Silicon) > ROCm (AMD GPU) > CPU

    回傳:
        torch.device: 最佳可用的運算裝置。
    """
    # 檢查 CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 使用 CUDA GPU: {torch.cuda.get_device_name()}")
        return device

    # 檢查 MPS (Apple Silicon GPU)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 使用 Apple Silicon MPS GPU")
        return device

    # 檢查 ROCm (AMD GPU) - 透過檢查版本字串中是否包含 ROCm 相關資訊
    try:
        if "rocm" in torch.__version__.lower() or "hip" in torch.__version__.lower():
            device = torch.device("cuda")  # ROCm 使用 cuda 介面
            print("🚀 使用 AMD ROCm GPU")
            return device
    except Exception as e:
        print(f"檢查 ROCm 時發生例外: {e}")

    # 檢查其他可能的 GPU 後端
    if torch.cuda.is_available():  # 再次檢查以防萬一
        device = torch.device("cuda")
        print(f"🚀 使用 GPU: {torch.cuda.get_device_name()}")
        return device

    # 預設使用 CPU
    device = torch.device("cpu")
    print("💻 使用 CPU (未偵測到 GPU 支援)")
    return device


@handle_errors("初始化 Whisper 模型時發生錯誤")
def initialize_model():
    """
    初始化並載入 Whisper 模型（線程安全）。
    """
    global model

    # 雙重檢查鎖定模式，避免競態條件
    if model is not None:
        log_info("模型已經初始化，跳過重複初始化。")
        return

    with model_lock:
        # 再次檢查以防在等待鎖時被其他線程初始化
        if model is not None:
            log_info("模型已經初始化，跳過重複初始化。")
            return

        try:
            log_info("初始化 Whisper 模型...")
            # 使用 openai-whisper，支援的模型大小：tiny, base, small, medium, large, turbo
            model_size = "turbo"  # 可以根據需要調整模型大小

            # 檢查是否有 GPU 可用
            device = get_best_device()

            # 載入模型
            model = whisper.load_model(model_size, device=device.type)
            log_info(f"模型初始化完成，使用模型：{model_size}，裝置：{device.type}")

        except Exception as e:
            log_error(f"模型初始化失敗: {str(e)}")
            model = None  # 重置模型變數
            raise


@handle_errors("釋放 Whisper 模型時發生錯誤")
def release_model():
    """
    釋放 Whisper 模型並清除 GPU 記憶體（線程安全）。
    """
    global model

    # 先檢查模型是否存在，避免不必要的鎖競爭
    if model is None:
        log_info("模型尚未載入，無需釋放。")
        return

    with model_lock:
        # 再次檢查以防在等待鎖時被其他線程釋放
        if model is None:
            log_info("模型已經被釋放。")
            return

        model_to_delete = None
        try:
            log_info("正在釋放 Whisper 模型...")
            # 儲存模型引用以便安全刪除
            model_to_delete = model
            model = None  # 立即設為 None，避免其他線程使用

            # 刪除模型
            del model_to_delete
            model_to_delete = None  # 清除引用

            # 清理 GPU 記憶體
            if torch.cuda.is_available():
                log_info("正在清除 CUDA 快取...")
                torch.cuda.empty_cache()

            # 強制垃圾回收
            gc.collect()
            log_info("模型已成功釋放，GPU 記憶體已清除。")

        except Exception as e:
            log_error(f"釋放模型時發生錯誤: {str(e)}")
            # 如果釋放失敗，嘗試恢復模型引用
            if model_to_delete is not None:
                model = model_to_delete
            raise


class OpenAI_Transcribe(BaseModel):
    text: str = ""
    language: str = ""
    duration: float = 0.0
    segments: List[Segment] = []


class Transcribe(BaseModel):
    """語音轉文字結果（支援自動型別檢查）"""

    model_config = {"arbitrary_types_allowed": True}

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
def write_transcript(
    segments_list: List[Segment], info: TranscriptionInfo, filename: str
) -> str:
    with open(
        filename, "w", encoding="utf-8"
    ) as f:  # 開啟檔案以寫入模式，使用 UTF-8 編碼
        # 如果是中文，則初始化 OpenCC 物件，用於簡轉繁（台灣正體，包含詞彙轉換）
        cc = None
        if info.language == "zh":
            cc = opencc_manager.get_cc("s2twp.json")
        for segment in segments_list:  # 迭代處理每個語音片段
            segment_text: str = segment.text
            if info.language == "zh" and cc:  # 再次判斷語言是否為中文
                # 如果是中文，則將片段文字轉換為繁體中文
                segment_text = cc.convert(segment.text)
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
    # 這樣做是為了確保檔案已經寫入完成
    with open(filename, "r", encoding="utf-8") as file:
        transcript = file.read()
    return transcript  # 回傳 逐字稿 內容


# 定義翻譯逐字稿檔案的函式
def write_translation(
    segments_list: List[Segment], info: TranscriptionInfo, filename: str
) -> str:
    with open(
        filename, "w", encoding="utf-8"
    ) as f:  # 開啟檔案以寫入模式，使用 UTF-8 編碼
        for segment in segments_list:  # 迭代處理每個語音片段
            segment_text: str = segment.text
            segment_text = translate_text_ollama(segment.text)
            # 將格式化後的時間戳記和文字寫入檔案
            f.write(
                "[%s --> %s]\n %s\n\n"  # 時間戳記格式，並空一行
                % (
                    format_time_vtt(segment.start),  # 格式化開始時間
                    format_time_vtt(segment.end),  # 格式化結束時間
                    segment_text,  # 語音片段文字
                )
            )

    # 這樣做是為了確保檔案已經寫入完成
    with open(filename, "r", encoding="utf-8") as file:
        transcript = file.read()
    return transcript  # 回傳 逐字稿 內容


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
def write_json(segments: List[Segment], info: TranscriptionInfo, filename: str) -> str:
    json_text = get_openai_json_text(segments, info)

    # 開啟檔案以寫入模式，使用 UTF-8 編碼
    with open(filename, "w", encoding="utf-8") as f:
        # 將語音片段和資訊寫入 JSON 檔案
        f.write(json_text)

    return json_text


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
@handle_errors("總結會議摘要時發生錯誤")
def summary_text(segments: List[Segment]) -> str:
    if not segments:
        return "沒有可用的語音片段來生成摘要"

    summaries = []
    # 先找出各段落分開的總結，使用安全執行避免單一失敗影響整體
    for segment in segments:
        part_summary = safe_execute(
            summary_text_ollama,
            segment.text,
            fallback_value=f"摘要失敗: {segment.text[:50]}...",
        )
        summaries.append(part_summary)

    # 將各段落分開的總結合併成一個總結
    full_text = " ".join(summaries)
    # 總結會議摘要
    final_summary = safe_execute(
        summary_text_ollama,
        full_text,
        fallback_value="摘要功能暫時無法使用，請確認 Ollama 服務正在運行。",
    )

    return final_summary


@handle_errors("轉錄音訊檔案時發生錯誤")
def transcribe_file(audio_file: str) -> tuple[list[Segment], TranscriptionInfo]:
    # 驗證檔案
    try:
        FileHandler.validate_audio_file(audio_file)
    except ValidationError as e:
        raise FileProcessingError(f"檔案驗證失敗: {str(e)}")

    # 確保模型已載入
    initialize_model()

    # 檢查模型是否成功載入
    if model is None:
        raise FileProcessingError("Whisper 模型載入失敗")

    try:
        # 使用 openai-whisper 進行語音轉錄，啟用詳細輸出以獲取分段資訊
        result = model.transcribe(audio_file, verbose=True, word_timestamps=False)  # type: ignore
    except Exception as e:
        raise FileProcessingError(f"Whisper 轉錄失敗: {str(e)}")

    # 轉換 openai-whisper 的結果格式為與 faster-whisper 相容的格式
    segments_list = []
    if "segments" in result and result["segments"]:
        for segment_data in result["segments"]:
            # 明確轉換型別以避免型別錯誤
            start_time = 0.0
            end_time = 0.0
            text_content = ""

            if isinstance(segment_data, dict):
                start_time = float(segment_data.get("start", 0.0))
                end_time = float(segment_data.get("end", 0.0))
                text_content = str(segment_data.get("text", ""))

            segment = Segment(start=start_time, end=end_time, text=text_content)
            segments_list.append(segment)
    else:
        # 如果沒有 segments 資訊，建立單一片段
        # 使用 whisper 內建的音檔載入功能獲取時長
        try:
            audio_data = whisper.load_audio(audio_file)
            duration = len(audio_data) / whisper.audio.SAMPLE_RATE
        except Exception:
            duration = 0.0

        segment = Segment(start=0.0, end=duration, text=str(result.get("text", "")))
        segments_list.append(segment)

    # 建立 TranscriptionInfo 物件
    duration = segments_list[-1].end if segments_list else 0.0
    info_obj = TranscriptionInfo(
        language=str(result.get("language", "unknown")),
        language_probability=1.0,  # openai-whisper 不提供此資訊，設為 1.0
        duration=duration,
    )

    return segments_list, info_obj


# 定義 transcribe 函式，回傳 OpenAI Whisper API 格式 JSON 字串
def transcribe_openai_format(audio_file: str) -> str:
    # 使用 Whisper 模型進行語音轉錄，回傳OpenAI Whisper API格式
    segments_list: list[Segment]
    info_obj: TranscriptionInfo
    segments_list, info_obj = transcribe_file(audio_file)
    # 印出偵測到的語言及其機率
    print(
        "Detected language '%s' with probability %f"
        % (info_obj.language, info_obj.language_probability)
    )

    # 產生 JSON 檔案
    json_text = get_openai_json_text(segments_list, info_obj)

    return json_text


def transcribe_full(audio_file: str) -> str:
    # 使用 Whisper 模型進行語音轉錄
    segments_list: list[Segment]
    info_obj: TranscriptionInfo
    segments_list, info_obj = transcribe_file(audio_file)
    # 印出偵測到的語言及其機率
    print(
        "Detected language '%s' with 概率 %f"
        % (info_obj.language, info_obj.language_probability)
    )

    # 根據輸入的音訊檔名產生輸出的檔名
    base_filename: str = FileHandler.safe_path_split(audio_file, ".", 1)[
        0
    ]  # 取得不含副檔名的基本檔名
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

    # 生成摘要，加入錯誤處理
    summary = safe_execute(
        summary_text_ollama,
        transcript,
        fallback_value="摘要功能暫時無法使用，請確認 Ollama 服務正在運行。",
    )

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
def main(audio_file: str) -> str:
    result = transcribe_full(audio_file)
    release_model()  # 釋放模型資源
    return result


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
