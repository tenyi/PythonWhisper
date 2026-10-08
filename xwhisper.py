# import json
import subprocess
import os
import re
import logging
from pydantic import BaseModel
import sys  # 匯入 sys 模組，用於處理命令列參數
from typing import List, Optional  # 匯入類型提示
from translator_ollama import translate_text_ollama, summary_text_ollama, fix_s2twp
import uuid

# 匯入共用模組
from common import (
    Segment, TranscriptionInfo, FileHandler,
    TimeFormatter, LanguageDetector, resource_manager,
    opencc_manager, safe_execute, handle_errors, log_error, log_info
)
from error_handler import TranslationError, FileProcessingError

# 設定日誌
logger = logging.getLogger(__name__)

# 全域變數
# 移除模型相關的全域變數，改用命令行工具


# 類別定義已從 common 模組匯入


# 解析 SRT 時間格式的函式
def parse_srt_time(time_str: str) -> float:
    """將 SRT 時間格式 (HH:MM:SS,mmm) 轉換為秒數"""
    time_parts = time_str.replace(',', '.').split(':')
    hours = int(time_parts[0])
    minutes = int(time_parts[1])
    seconds = float(time_parts[2])
    return hours * 3600 + minutes * 60 + seconds


# 解析 SRT 檔案的函式
@handle_errors("解析 SRT 檔案時發生錯誤")
def parse_srt_file(srt_path: str) -> tuple[List[Segment], str]:
    """解析 SRT 檔案並回傳 segments 和偵測到的語言"""
    segments = []

    # 驗證檔案存在
    if not os.path.exists(srt_path):
        log_error(f"SRT 檔案不存在: {srt_path}")
        return segments, "unknown"

    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception as e:
        raise FileProcessingError(f"無法讀取 SRT 檔案 {srt_path}: {str(e)}")

    # 使用正規表達式解析 SRT 格式
    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\d+\n|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)

    # 使用列表來收集文字，避免記憶體累積
    text_parts = []

    for match in matches:
        index, start_time, end_time, text = match
        start_seconds = parse_srt_time(start_time.strip())
        end_seconds = parse_srt_time(end_time.strip())
        text_content = text.strip()

        segment = Segment(start=start_seconds, end=end_seconds, text=text_content)
        segments.append(segment)
        text_parts.append(text_content)

    # 使用 LanguageDetector 進行語言偵測
    full_text = " ".join(text_parts)
    detected_language = LanguageDetector.detect_language_from_text(full_text)

    log_info(f"成功解析 SRT 檔案: {srt_path}, 偵測到 {len(segments)} 個片段, 語言: {detected_language}")
    return segments, detected_language


# 使用 whisper 命令行工具進行轉錄
def run_whisper_cli(audio_file: str, output_dir: Optional[str] = None) -> str:
    """使用 whisper 命令行工具產生 SRT 檔案"""
    if output_dir is None:
        output_dir = os.path.dirname(audio_file) or "."
    
    # 構建 whisper 命令
    cmd = [
        #sys.executable, "-m", 
        "whisper", 
        audio_file,
        "--output_format", "srt",
        "--output_dir", output_dir,
        "--model", "large-v3"
    ]
    
    try:
        print(f"執行 whisper 命令: {' '.join(cmd)}")
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Whisper 執行完成")
        
        # 找出產生的 SRT 檔案
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        srt_file = os.path.join(output_dir, f"{base_name}.srt")
        
        if os.path.exists(srt_file):
            return srt_file
        else:
            raise FileNotFoundError(f"SRT 檔案未找到: {srt_file}")
            
    except subprocess.CalledProcessError as e:
        print(f"Whisper 執行錯誤: {e}")
        print(f"錯誤輸出: {e.stderr}")
        raise
    except Exception as e:
        print(f"執行 whisper 時發生錯誤: {e}")
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


# 時間格式化函式已從 TimeFormatter 類別匯入


# 定義寫入逐字稿檔案的函式
@handle_errors("寫入逐字稿檔案時發生錯誤")
def write_transcript(
    segments_list: List[Segment], info: TranscriptionInfo, filename: str
) -> str:
    try:
        with resource_manager.manage_file(filename, "w", encoding="utf-8") as f:
            # 如果是中文，則初始化 OpenCC 物件，用於簡轉繁（台灣正體，包含詞彙轉換）
            cc = None
            if info.language == "zh":
                cc = opencc_manager.get_cc("s2twp.json")

            for segment in segments_list:  # 迭代處理每個語音片段
                segment_text: str = segment.text
                if info.language == "zh" and cc:  # 再次判斷語言是否為中文
                    # 如果是中文，則將片段文字轉換為繁體中文
                    segment_text = fix_s2twp(cc.convert(segment.text))  # 修正 s2twp 誤轉（如「只要」→「隻要」）
                # 將格式化後的時間戳記和文字寫入檔案
                f.write(
                    "[%s --> %s]\n %s\n\n"  # 時間戳記格式，並空一行
                    % (
                        TimeFormatter.format_time_vtt(segment.start),  # 格式化開始時間
                        TimeFormatter.format_time_vtt(segment.end),  # 格式化結束時間
                        segment_text,  # 語音片段文字
                    )
                )

        # 讀取整個檔案內容並回傳
        with resource_manager.manage_file(filename, "r", encoding="utf-8") as file:
            transcript = file.read()
        return transcript  # 回傳 逐字稿 內容

    except Exception as e:
        raise FileProcessingError(f"寫入逐字稿檔案失敗 {filename}: {str(e)}")


# 定義翻譯逐字稿檔案的函式
@handle_errors("翻譯逐字稿檔案時發生錯誤")
def write_translation(
    segments_list: List[Segment], info: TranscriptionInfo, filename: str
) -> str:
    try:
        with resource_manager.manage_file(filename, "w", encoding="utf-8") as f:
            for segment in segments_list:  # 迭代處理每個語音片段
                segment_text: str = segment.text
                # 使用安全執行來處理翻譯，避免單一失敗影響整個流程
                translated_text = safe_execute(
                    translate_text_ollama,
                    segment.text,
                    fallback_value=segment.text
                )
                if translated_text != segment.text:
                    segment_text = translated_text
                else:
                    logger.warning(f"翻譯失敗，使用原文: {segment.text[:50]}...")

                # 將格式化後的時間戳記和文字寫入檔案
                f.write(
                    "[%s --> %s]\n %s\n\n"  # 時間戳記格式，並空一行
                    % (
                        TimeFormatter.format_time_vtt(segment.start),  # 格式化開始時間
                        TimeFormatter.format_time_vtt(segment.end),  # 格式化結束時間
                        segment_text,  # 語音片段文字
                    )
                )

        # 讀取整個檔案內容並回傳
        with resource_manager.manage_file(filename, "r", encoding="utf-8") as file:
            transcript = file.read()
        return transcript  # 回傳 逐字稿 內容

    except Exception as e:
        raise TranslationError(f"翻譯逐字稿檔案失敗 {filename}: {str(e)}")


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
                    TimeFormatter.format_time_srt(segment.start),  # 格式化開始時間
                    TimeFormatter.format_time_srt(segment.end),  # 格式化結束時間
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
                    TimeFormatter.format_time_vtt(segment.start),  # 格式化開始時間
                    TimeFormatter.format_time_vtt(segment.end),  # 格式化結束時間
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
            fallback_value=f"摘要失敗: {segment.text[:50]}..."
        )
        summaries.append(part_summary)

    # 將各段落分開的總結合併成一個總結
    full_text = " ".join(summaries)
    # 總結會議摘要
    final_summary = safe_execute(
        summary_text_ollama,
        full_text,
        fallback_value="摘要功能暫時無法使用，請確認 Ollama 服務正在運行。"
    )

    return final_summary


def transcribe_file(audio_file: str) -> tuple[list[Segment], TranscriptionInfo]:
    """使用 whisper 命令行工具進行語音轉錄"""
    
    # 使用 whisper 命令行工具產生 SRT 檔案
    srt_file = run_whisper_cli(audio_file)
    
    # 解析 SRT 檔案
    segments_list, detected_language = parse_srt_file(srt_file)
    
    # 計算總時長
    duration = segments_list[-1].end if segments_list else 0.0
    
    # 建立 TranscriptionInfo 物件
    info_obj = TranscriptionInfo(
        language=detected_language,
        language_probability=1.0,  # 簡化的語言偵測，設為 1.0
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
        "Detected language '%s' with probability %f"
        % (info_obj.language, info_obj.language_probability)
    )

    # 根據輸入的音訊檔名產生輸出的檔名
    base_filename: str = FileHandler.safe_path_split(audio_file, ".", 1)[0]  # 取得不含副檔名的基本檔名
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
    try:
        summary = summary_text_ollama(transcript)
    except Exception as e:
        print(f"摘要生成失敗，使用預設訊息: {e}")
        summary = "摘要功能暫時無法使用，請確認 Ollama 服務正在運行。"

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
    transcribe_full(audio_file)
    # 不再需要釋放模型資源，因為我們使用命令行工具


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
