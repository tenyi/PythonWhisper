import argparse
import sys
import requests
import re
from opencc import OpenCC
from typing import Optional

# 匯入自定義異常和錯誤處理
from exceptions import (
    APIError, TranslationError, SummaryError, ErrorCode, FileError,
    create_api_connection_error
)
from error_handler import error_context, retry_on_exception, handle_file_operations
from http_client import HTTPClient
from logging_config import get_logger

logger = get_logger("translator")


@handle_file_operations
def translate_subtitle_file(input_file, output_file) -> None:
    """
    翻譯字幕檔案
    
    Args:
        input_file: 輸入字幕檔案路徑
        output_file: 輸出字幕檔案路徑
        
    Raises:
        FileError: 當檔案操作失敗時
        TranslationError: 當翻譯失敗時
    """
    try:
        with error_context("translate_subtitle_file", input_file=input_file, output_file=output_file):
            logger.info(f"開始翻譯字幕檔案: {input_file} -> {output_file}")
            
            with open(input_file, "r", encoding="utf-8") as f:
                content = f.read()
                
            if not content.strip():
                raise FileError(
                    message="字幕檔案內容為空",
                    error_code=ErrorCode.FILE_FORMAT_ERROR,
                    details={"file_path": input_file}
                )

            isWebVtt = False
            if content.startswith("WEBVTT"):
                isWebVtt = True
            
            # Split the content into subtitle blocks
            blocks = re.split(r"\n\s*\n", content.strip())

            merged_blocks = []
            for block in blocks:
                lines = block.split("\n")
                length = len(lines)
                if length >= 2:
                    minus_one = length - 1
                    # Keep the subtitle number and timestamp
                    merged_block = lines[:minus_one]
                    # Merge the text lines
                    merged_text = " ".join(lines[minus_one:])
                    
                    # 跳過空文字
                    if not merged_text.strip():
                        merged_block.append(merged_text)
                        merged_blocks.append("\n".join(merged_block))
                        continue
                    
                    # 翻譯
                    translated_text = translate_text_ollama(merged_text, "zh-TW")
                    merged_text = f"{merged_text}\n{translated_text}"
                    merged_block.append(merged_text)
                    merged_blocks.append("\n".join(merged_block))

            # Join the merged blocks with double newlines
            output_content = "\n\n".join(merged_blocks)

            if isWebVtt:
                output = f"WEBVTT\n\n{output_content}"
            else:
                output = output_content
                
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output)
                
            logger.info(f"字幕翻譯完成: {output_file}")
            
    except Exception as e:
        logger.error(f"字幕翻譯失敗: {str(e)}")
        raise


@retry_on_exception(max_retries=3, delay=1.0, exceptions=(APIError,))
def summary_text_ollama(text: str, url: str = "http://localhost:11434/api/chat", model: str = "gemma3:27b") -> str:
    """
    總結會議摘要

    Args:
        text: 要總結的文字
        url: Ollama API URL
        model: 使用的模型名稱

    Returns:
        摘要文字

    Raises:
        SummaryError: 當摘要生成失敗時
        APIError: 當 API 呼叫失敗時
    """
    if not text or not text.strip():
        raise SummaryError(
            message="輸入文字為空，無法生成摘要",
            error_code=ErrorCode.TEXT_TOO_SHORT,
            details={"text_length": len(text)}
        )

    # 檢查文字長度
    if len(text) > 50000:  # 設定合理的長度限制
        logger.warning(f"輸入文字過長 ({len(text)} 字符)，可能影響摘要品質")

    try:
        with error_context("summary_generation", text_length=len(text), model=model):
            client = HTTPClient(url.rsplit('/api/chat', 1)[0])

            data = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": """
1.  **摘要內容要求**：
    *   **主要議題**：識別並概述討論的核心主題。
    *   **關鍵成果**：重點描述重要的決定、達成的共識或結論。
    *   **行動項目**：列出具體的行動計劃及後續步驟；若有提及，須指明負責人或團隊。

2.  **語言與格式**：
    *   **僅限繁體中文**：所有輸出內容必須使用繁體中文。
    *   **保留英文術語**：若原文中包含英文專業術語，請在摘要中保留其英文原文。
    *   **專業簡潔**：使用精確、直接的語言，避免不必要的細節，確保摘要易於理解。

3.  **嚴格排除的內容**：
    *   **禁止任何引言、客套話、結論性陳述** (例如：「本次會議摘要如下」、「會議圓滿結束」、「以上是重點整理」等)。
    *   **禁止提及「摘要」、「紀錄」等詞語本身**。
    *   **禁止任何形式的自我評論、解釋或補充說明**。你的回應**只能是摘要本身**，不應包含摘要以外的任何文字。

4.  **特殊情況處理**：
    *   如果提供的文本資訊量過少，或內容零散難以歸納出有意義的摘要，請直接回答：「無法得出摘要」。請僅在確實無法摘要時使用此回覆。

5.  **最終輸出指示**：
    *   產生的摘要必須**條理分明、切中要點**，不含任何冗餘詞彙。
    *   請務必以 `zh-TW` (繁體中文) 回覆。
"""
                    },
                    {
                        "role": "user",
                        "content": text,
                    },
                ],
                "stream": False
            }

            # 發送 POST 請求以獲取總結
            json_data = client.post("/api/chat", data)

            # 檢查回應格式
            if "message" not in json_data or "content" not in json_data["message"]:
                raise SummaryError(
                    message="API 回應格式錯誤",
                    error_code=ErrorCode.API_RESPONSE_ERROR,
                    details={"response": json_data}
                )

            summary = json_data["message"]["content"].strip()

            # 移除思考標籤
            summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)

            if not summary:
                raise SummaryError(
                    message="API 回傳空摘要",
                    error_code=ErrorCode.SUMMARY_ERROR,
                    details={"original_response": json_data["message"]["content"]}
                )

            # 轉換為繁體中文
            try:
                cc = OpenCC("s2twp")
                summary = cc.convert(summary)
            except Exception as e:
                logger.warning(f"繁體中文轉換失敗: {str(e)}")
                # 轉換失敗不影響主要功能，繼續執行

            logger.info(f"摘要生成成功，長度: {len(summary)} 字符")
            return summary

    except APIError:
        raise  # 重新拋出 API 錯誤，讓重試機制處理
    except Exception as e:
        raise SummaryError(
            message=f"摘要生成失敗: {str(e)}",
            error_code=ErrorCode.SUMMARY_ERROR,
            details={"text_length": len(text), "model": model},
            original_exception=e
        )


@retry_on_exception(max_retries=3, delay=1.0, exceptions=(APIError,))
def correct_words_ollama(text: str, url: str = "http://localhost:11434/api/chat", model: str = "gemma3:27b") -> str:
    """
    修正錯字

    Args:
        text: 要修正的文字
        url: Ollama API URL
        model: 使用的模型名稱

    Returns:
        修正後的文字

    Raises:
        TranslationError: 當文字修正失敗時
        APIError: 當 API 呼叫失敗時
    """
    if not text:
        return text  # 空文字直接回傳

    try:
        with error_context("text_correction", text_length=len(text), model=model):
            client = HTTPClient(url.rsplit('/api/chat', 1)[0])

            data = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": """
# 角色
你是一位細心的文字校正專家。

# 任務
你的任務是校正由 Whisper 產生的逐字稿中的錯字與同音異字。你必須只回傳校正後的文字。

# 規則
嚴格遵守原始格式（包含換行、空格、標點符號）。只將錯誤的字元替換為正確的字元，絕對不要新增、刪除或改寫任何內容。
""",
                    },
                    {
                        "role": "user",
                        "content": text,
                    }
                ],
                "stream": False
            }

            json_data = client.post("/api/chat", data)

            # 檢查回應格式
            if "message" not in json_data or "content" not in json_data["message"]:
                raise TranslationError(
                    message="API 回應格式錯誤",
                    error_code=ErrorCode.API_RESPONSE_ERROR,
                    details={"response": json_data}
                )

            correct_text = json_data["message"]["content"].strip()

            if not correct_text:
                logger.warning("API 回傳空的修正結果，使用原始文字")
                return text

            logger.info(f"文字修正完成，原始長度: {len(text)}, 修正後長度: {len(correct_text)}")
            return correct_text

    except APIError:
        raise  # 重新拋出 API 錯誤，讓重試機制處理
    except Exception as e:
        raise TranslationError(
            message=f"文字修正失敗: {str(e)}",
            error_code=ErrorCode.TRANSLATION_ERROR,
            details={"text_length": len(text), "model": model},
            original_exception=e
        )


@retry_on_exception(max_retries=3, delay=1.0, exceptions=(APIError,))
def correct_words_ollama_fail(text: str, url: str = "http://localhost:11434/api/chat", model: str = "gemma3:27b") -> str:
    """
    修正錯字（備用版本）
    
    Args:
        text: 要修正的文字
        url: Ollama API URL
        model: 使用的模型名稱
        
    Returns:
        修正後的文字
        
    Raises:
        TranslationError: 當文字修正失敗時
        APIError: 當 API 呼叫失敗時
    """
    if not text:
        return text  # 空文字直接回傳

    try:
        with error_context("text_correction_fail", text_length=len(text), model=model):
            client = HTTPClient(url.rsplit('/api/chat', 1)[0])

            data = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": """
You are an AI proofreading engine specialized in correcting Whisper ASR transcripts.

Your single function is to identify and correct typos and homophone errors.

Follow these rules STRICTLY:
1.  **INPUT**: The user will provide a raw text transcript.
2.  **PROCESSING**:
    - Identify and correct all spelling mistakes and homophone errors (e.g., "的" vs. "得", "在" vs. "再").
    - PRESERVE the original document structure: All line breaks, spacing, and punctuation must remain identical.
    - DO NOT add any new words or sentences.
    - DO NOT remove any existing words or sentences.
    - DO NOT rephrase or paraphrase the content. Your only allowed action is character replacement for corrections.
3.  **OUTPUT**:
    - Return ONLY the corrected text.
    - DO NOT include any explanations, apologies, or introductory phrases like "這是校正後的文字：".

"""
                    },
                    {
                        "role": "user",
                        "content": text,
                    }
                ],
                "stream": False
            }

            # 發送 POST 請求
            json_data = client.post("/api/chat", data)

            # 檢查回應格式
            if "message" not in json_data or "content" not in json_data["message"]:
                raise TranslationError(
                    message="API 回應格式錯誤",
                    error_code=ErrorCode.API_RESPONSE_ERROR,
                    details={"response": json_data}
                )

            correct_text = json_data["message"]["content"].strip()
            
            if not correct_text:
                logger.warning("API 回傳空的修正結果，使用原始文字")
                return text

            logger.info(f"文字修正完成（備用版本），原始長度: {len(text)}, 修正後長度: {len(correct_text)}")
            return correct_text

    except APIError:
        raise  # 重新拋出 API 錯誤，讓重試機制處理
    except Exception as e:
        raise TranslationError(
            message=f"文字修正失敗（備用版本）: {str(e)}",
            error_code=ErrorCode.TRANSLATION_ERROR,
            details={"text_length": len(text), "model": model},
            original_exception=e
        )


@retry_on_exception(max_retries=3, delay=1.0, exceptions=(APIError,))
def translate_text_ollama(text: str, target_language: str = "zh-TW", model: str = "gemma3:27b") -> str:
    """
    翻譯文字

    Args:
        text: 要翻譯的文字
        target_language: 目標語言
        model: 使用的模型名稱

    Returns:
        翻譯後的文字

    Raises:
        TranslationError: 當翻譯失敗時
        APIError: 當 API 呼叫失敗時
    """
    if not text or not text.strip():
        return text  # 空文字直接回傳

    # 檢查語言支援
    supported_languages = ["zh-TW", "zh-CN", "en", "ja", "ko"]
    if target_language not in supported_languages:
        raise TranslationError(
            message=f"不支援的目標語言: {target_language}",
            error_code=ErrorCode.LANGUAGE_NOT_SUPPORTED,
            details={"target_language": target_language, "supported_languages": supported_languages}
        )

    try:
        with error_context("text_translation", text_length=len(text), target_language=target_language, model=model):
            url = "http://localhost:11434/api/chat"
            client = HTTPClient(url.rsplit('/api/chat', 1)[0])

            data = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are a highly disciplined translation model that strictly follows instructions. You are a highly skilled AI translator specializing in translating meeting records into {target_language}. Translate **only** the user's provided meeting notes according to the following strict guidelines:\n\n1. **Do NOT add any explanation, commentary, or extra content**.  \n   - Your task is strictly translation. Do not include summaries, comments, greetings, or interpretations.  \n   - If you add anything beyond the translation, it will be considered a mistake.\n\n2. **Preserve Professional Terminology in English**:  \n   - If the meeting notes contain technical terms, brand names, product names, or specialized industry jargon, retain them in their original English form.  \n   - Common English words that are part of professional terminology (e.g., 'AI model', 'Deep Learning', 'OpenAI API') should **not be translated** into Chinese.  \n\n3. **Ensure Natural and Fluent Translation**:  \n   - Use clear, professional, and natural formal Traditional Chinese.  \n   - Avoid overly literal translations that sound unnatural.  \n\n4. **Maintain Logical Flow and Readability**:  \n   - Adjust sentence structure to fit Chinese language conventions while preserving the original meaning.  \n\n5. **Keep Formatting and Structure Consistent**:  \n   - Retain original formatting: bullet points, lists, section headings, etc.  \n   - Smoothly integrate terms into the {target_language} context.  \n\n6. **Language Requirement**:  \n   - Entire output must be in {target_language}, except for English professional terminology.  \n\n7. 若文字太過簡短，直接 Translate into {target_language}.\n\nNo preambles. No commentary. If you include anything other than the direct translation, it will be considered an error.",
                    },
                    {
                        "role": "user",
                        "content": text,
                    },
                ],
                "stream": False
            }

            # 發送 POST 請求以獲取翻譯
            json_data = client.post("/api/chat", data)

            # 檢查回應格式
            if "message" not in json_data or "content" not in json_data["message"]:
                raise TranslationError(
                    message="API 回應格式錯誤",
                    error_code=ErrorCode.API_RESPONSE_ERROR,
                    details={"response": json_data}
                )

            translate_text = json_data["message"]["content"].strip()

            if not translate_text:
                logger.warning("API 回傳空的翻譯結果，使用原始文字")
                return text

            # 繁體中文轉換
            if target_language == "zh-TW":
                try:
                    cc = OpenCC("s2twp")  # 翻成台灣繁體中文，避免大語言模型誤寫簡體中文
                    translate_text = cc.convert(translate_text)
                except Exception as e:
                    logger.warning(f"繁體中文轉換失敗: {str(e)}")
                    # 轉換失敗不影響主要功能，繼續執行

            logger.info(f"翻譯完成，原始長度: {len(text)}, 翻譯後長度: {len(translate_text)}")
            return translate_text

    except APIError:
        raise  # 重新拋出 API 錯誤，讓重試機制處理
    except Exception as e:
        raise TranslationError(
            message=f"翻譯失敗: {str(e)}",
            error_code=ErrorCode.TRANSLATION_ERROR,
            details={"text_length": len(text), "target_language": target_language, "model": model},
            original_exception=e
        )


# 定義顯示使用說明的函式
def usage(script_name: str) -> None:
    print(f"Usage: python {script_name} <input_file> <output_file>")  # 印出使用語法
    print(f"Example: python {script_name} input.srt output.srt")  # 印出使用範例


def main():
    parser = argparse.ArgumentParser(
        description="將指定字幕檔翻譯成繁體中文，並將結果寫入新的檔案。"
    )
    parser.add_argument(
        "input_file",
        help="input file",
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default="output.srt",
        help="output file (default: output.srt)",
    )
    if len(sys.argv) > 1:
        args = parser.parse_args()
        translate_subtitle_file(args.input_file, args.output_file)
    else:
        usage(sys.argv[0])


if __name__ == "__main__":
    main()
