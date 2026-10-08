#!/usr/bin/env python3
"""
會議摘要產生工具

使用錯誤處理機制來處理檔案操作和 API 呼叫錯誤。
"""

from translator_ollama import summary_text_ollama, correct_words_ollama
from error_handler import handle_file_operations, error_context
from logging_config import get_logger
from exceptions import FileError

logger = get_logger("summarize")

@handle_file_operations
def read_transcript_file(filename: str) -> str:
    """讀取逐字稿檔案"""
    try:
        with error_context("read_transcript", filename=filename):
            with open(filename, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        logger.error(f"讀取逐字稿檔案失敗: {filename} - {str(e)}")
        raise

@handle_file_operations
def write_output_file(filename: str, content: str) -> None:
    """寫入輸出檔案"""
    try:
        with error_context("write_output", filename=filename):
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
                f.write("\n")
    except Exception as e:
        logger.error(f"寫入輸出檔案失敗: {filename} - {str(e)}")
        raise

def main():
    """主函數：處理會議摘要"""
    try:
        with error_context("summarize_main"):
            input_file = "transcript.txt"
            output_file = "correct_text.txt"
            
            # 檢查檔案是否存在
            if not os.path.exists(input_file):
                logger.error(f"輸入檔案不存在: {input_file}")
                return
            
            # 讀取逐字稿
            logger.info(f"開始讀取逐字稿: {input_file}")
            text = read_transcript_file(input_file)
            
            if not text.strip():
                logger.warning("逐字稿內容為空")
                return
            
            url = "http://10.10.10.201:11434/api/chat"
            
            # 修正錯字
            logger.info("開始修正錯字")
            corrected_text = correct_words_ollama(text, url)
            
            # 寫入修正後的文字
            write_output_file(output_file, corrected_text)
            logger.info(f"修正完成，結果已寫入: {output_file}")
            
            # 可選：產生摘要
            # logger.info("開始產生摘要")
            # summary = summary_text_ollama(text, url)
            # write_output_file("summary.txt", summary)
            # logger.info("摘要已產生")
            
    except Exception as e:
        logger.error(f"處理過程中發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    import os
    try:
        main()
    except KeyboardInterrupt:
        logger.info("使用者中斷操作")
    except Exception as e:
        logger.error(f"程式執行失敗: {str(e)}")
        exit(1)
