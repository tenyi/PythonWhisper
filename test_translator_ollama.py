"""
translator_ollama 的 LLM 呼叫測試（OpenAI 相容 API 格式）

LLM 服務（vLLM / Ollama）一律走 OpenAI 相容的 /v1/chat/completions，
回覆文字位於 choices[0].message.content。
所有請求都經由 http_client.HTTPClient 的 requests.Session 送出，
因此 mock 目標為 requests.Session.post（mock requests.post 攔不到）。
"""
import pytest
import requests
from unittest.mock import Mock, patch

from exceptions import SummaryError, TranslationError
from translator_ollama import (
    LLM_BASE_URL,
    LLM_MODEL,
    PUNCTUATE_PROMPT,
    correct_words_ollama,
    fix_s2twp,
    punctuate_text,
    summary_text_ollama,
    translate_text_ollama,
)

# 所有 LLM 請求都經過 HTTPClient 內的 requests.Session
SESSION_POST = "http_client.requests.Session.post"


def openai_response(content):
    """建立 OpenAI 相容格式的模擬回應。"""
    resp = Mock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]
    }
    return resp


def sent_url(mock_post):
    """取出實際送出的請求網址（patch 在類別上，呼叫參數不含 self）。"""
    return mock_post.call_args[0][0]


def sent_json(mock_post):
    """取出實際送出的請求內容。"""
    return mock_post.call_args[1]["json"]


class TestChatEndpoint:
    """共用呼叫邏輯：網址、端點與回應解析。換服務商時這些行為必須維持不變。"""

    @patch(SESSION_POST)
    def test_default_uses_configured_server_and_model(self, mock_post):
        """未指定參數時，必須打到設定的 LLM 服務與模型（避免誤打到不存在的舊模型）"""
        mock_post.return_value = openai_response("修正後")
        correct_words_ollama("測試文字")
        assert sent_url(mock_post) == f"{LLM_BASE_URL}/v1/chat/completions"
        assert sent_json(mock_post)["model"] == LLM_MODEL

    @pytest.mark.parametrize(
        "given_url",
        [
            "http://ollama-host:11434/api/chat",  # 舊版 Ollama 完整網址
            "http://ollama-host:11434/v1/chat/completions",  # 已是 OpenAI 端點
            "http://ollama-host:11434/",  # 服務根網址
        ],
    )
    @patch(SESSION_POST)
    def test_legacy_ollama_url_is_normalized(self, mock_post, given_url):
        """舊呼叫端仍傳 Ollama 的 /api/chat 網址，必須改打 OpenAI 端點才相容（Ollama 也支援）"""
        mock_post.return_value = openai_response("ok")
        correct_words_ollama("測試文字", url=given_url)
        assert sent_url(mock_post) == "http://ollama-host:11434/v1/chat/completions"

    @patch(SESSION_POST)
    def test_reads_openai_choices_content(self, mock_post):
        """回覆文字必須從 choices[0].message.content 取出，並去除前後空白"""
        mock_post.return_value = openai_response("  修正後的文字  ")
        assert correct_words_ollama("測試文字") == "修正後的文字"

    @patch(SESSION_POST)
    def test_ollama_native_format_is_rejected(self, mock_post):
        """收到非 OpenAI 格式（如 Ollama 原生 message.content）時要明確報錯，不可默默回傳錯誤內容"""
        resp = Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"message": {"content": "舊格式"}}
        mock_post.return_value = resp
        with pytest.raises(TranslationError):
            correct_words_ollama("測試文字")


class TestSummaryText:
    """summary_text_ollama：會議摘要"""

    @patch(SESSION_POST)
    def test_summary_success_with_custom_params(self, mock_post):
        """自訂網址與模型需原樣傳遞給服務"""
        mock_post.return_value = openai_response("討論新產品開發計畫，下月開始執行，負責人張三。")
        result = summary_text_ollama("會議內容", url="http://custom:8000", model="custom-model")
        assert result == "討論新產品開發計畫，下月開始執行，負責人張三。"
        assert sent_url(mock_post) == "http://custom:8000/v1/chat/completions"
        assert sent_json(mock_post)["model"] == "custom-model"
        assert sent_json(mock_post)["messages"][1]["content"] == "會議內容"

    @patch(SESSION_POST)
    def test_summary_converts_simplified_and_fixes_s2twp(self, mock_post):
        """LLM 可能誤寫簡體：摘要須轉台灣繁體，且「只要」不可被誤轉成「隻要」"""
        mock_post.return_value = openai_response("这是简体摘要，只要输入就会回答")
        result = summary_text_ollama("会议内容")
        assert "這是" in result
        assert "只要" in result and "隻要" not in result

    @patch(SESSION_POST)
    def test_summary_strips_think_tags(self, mock_post):
        """推理模型輸出的 <think> 區塊不可出現在摘要中"""
        mock_post.return_value = openai_response("<think>思考中</think>摘要內容")
        assert summary_text_ollama("會議內容") == "摘要內容"

    @patch(SESSION_POST)
    def test_summary_empty_response_is_error(self, mock_post):
        """LLM 回傳空白時不能回傳空摘要，必須報錯讓呼叫端走 fallback"""
        mock_post.return_value = openai_response("   ")
        with pytest.raises(SummaryError):
            summary_text_ollama("會議內容")

    def test_summary_empty_input_is_error(self):
        """空輸入不呼叫服務，直接報錯"""
        with pytest.raises(SummaryError):
            summary_text_ollama("   ")

    @patch(SESSION_POST)
    def test_summary_connection_error_wrapped(self, mock_post):
        """連線失敗要轉成 SummaryError，讓上層 safe_execute 能統一處理"""
        mock_post.side_effect = requests.exceptions.ConnectionError("連線失敗")
        with pytest.raises(SummaryError):
            summary_text_ollama("會議內容")

    @patch(SESSION_POST)
    def test_summary_system_prompt_requires_zh_tw(self, mock_post):
        """系統提示詞須要求繁體中文、保留英文術語，並定義無法摘要時的回覆"""
        mock_post.return_value = openai_response("摘要")
        summary_text_ollama("會議內容")
        system = sent_json(mock_post)["messages"][0]["content"]
        assert "繁體中文" in system
        assert "zh-TW" in system
        assert "保留英文術語" in system
        assert "無法得出摘要" in system


class TestCorrectWords:
    """correct_words_ollama：錯字校正"""

    @patch(SESSION_POST)
    def test_correction_returns_llm_text(self, mock_post):
        """回傳 LLM 校正後的文字，使用者原文須放在 user 訊息"""
        input_text = "的確需要在考慮一下。"
        mock_post.return_value = openai_response("的確需要再考慮一下。")
        assert correct_words_ollama(input_text) == "的確需要再考慮一下。"
        assert sent_json(mock_post)["messages"][1]["content"] == input_text

    @patch(SESSION_POST)
    def test_correction_empty_response_falls_back_to_original(self, mock_post):
        """LLM 回傳空白時保留原文，避免逐字稿內容被清空"""
        mock_post.return_value = openai_response("")
        assert correct_words_ollama("原始文字") == "原始文字"

    @patch(SESSION_POST)
    def test_correction_empty_input_skips_api(self, mock_post):
        """空輸入不應浪費一次 LLM 呼叫"""
        assert correct_words_ollama("") == ""
        mock_post.assert_not_called()

    @patch(SESSION_POST)
    def test_correction_system_prompt_forbids_rewriting(self, mock_post):
        """校正提示詞必須禁止新增、刪除或改寫內容"""
        mock_post.return_value = openai_response("測試文字")
        correct_words_ollama("測試文字")
        system = sent_json(mock_post)["messages"][0]["content"]
        assert "錯字與同音異字" in system
        assert "絕對不要新增、刪除或改寫任何內容" in system

    @patch(SESSION_POST)
    def test_correction_connection_error_wrapped(self, mock_post):
        """連線失敗要轉成 TranslationError"""
        mock_post.side_effect = requests.exceptions.ConnectionError("連線失敗")
        with pytest.raises(TranslationError):
            correct_words_ollama("測試文字")


class TestPunctuateText:
    """punctuate_text：為無標點的 ASR 文字補標點（取代原本的中翻中翻譯）"""

    @patch(SESSION_POST)
    def test_punctuate_uses_prompt_and_deterministic_sampling(self, mock_post):
        """使用補標點提示詞，且 temperature=0 降低 LLM 改寫內容的風險"""
        mock_post.return_value = openai_response("那很多時候，我們想要修改基礎模型。")
        result = punctuate_text("那很多時候 我們想要修改基礎模型")
        assert result == "那很多時候，我們想要修改基礎模型。"
        body = sent_json(mock_post)
        assert body["messages"][0]["content"] == PUNCTUATE_PROMPT
        assert body["messages"][1]["content"] == "那很多時候 我們想要修改基礎模型"
        assert body["temperature"] == 0

    @patch(SESSION_POST)
    def test_punctuate_does_not_apply_s2twp(self, mock_post):
        """輸入已是繁體，不可再做 s2twp：否則「參數」會被誤改成「引數」"""
        mock_post.return_value = openai_response("微調參數之後，只要輸入就好。")
        assert punctuate_text("微調參數之後 只要輸入就好") == "微調參數之後，只要輸入就好。"

    @patch(SESSION_POST)
    def test_punctuate_empty_response_falls_back_to_original(self, mock_post):
        """LLM 回傳空白時保留原文，字幕不可出現空段落"""
        mock_post.return_value = openai_response("  ")
        assert punctuate_text("原始文字") == "原始文字"

    @patch(SESSION_POST)
    def test_punctuate_blank_input_skips_api(self, mock_post):
        """空白輸入原樣回傳，不呼叫服務"""
        assert punctuate_text("  ") == "  "
        mock_post.assert_not_called()


class TestTranslateText:
    """translate_text_ollama：翻譯"""

    @patch(SESSION_POST)
    def test_translate_zh_tw_converts_and_fixes_s2twp(self, mock_post):
        """翻成 zh-TW 時要轉台灣繁體，且修正「隻要」誤轉"""
        mock_post.return_value = openai_response("只要输入问题")
        assert translate_text_ollama("Just enter the question", "zh-TW") == "只要輸入問題"

    @patch(SESSION_POST)
    def test_translate_non_zh_tw_keeps_output(self, mock_post):
        """非 zh-TW 目標語言不可做繁體轉換"""
        mock_post.return_value = openai_response("只要输入问题")
        assert translate_text_ollama("Just enter the question", "zh-CN") == "只要输入问题"

    def test_translate_unsupported_language_is_error(self):
        """不支援的語言在呼叫服務前就要拒絕"""
        with pytest.raises(TranslationError):
            translate_text_ollama("hello", "fr")


class TestFixS2twp:
    """fix_s2twp：修正 OpenCC s2twp 的已知誤轉"""

    def test_fix_only_targets_known_mistakes(self):
        """「隻要」修回「只要」，但量詞「一隻手」必須保留"""
        assert fix_s2twp("應該就是隻要輸入，他只有一隻手") == "應該就是只要輸入，他只有一隻手"


@pytest.mark.skip(reason="需要實際的 LLM 服務，用於手動驗證提示詞效果")
def test_real_service_punctuation():
    """手動測試：實際呼叫設定的 LLM 服務補標點"""
    result = punctuate_text("那很多時候 我們想要修改基礎模型 往往只想要改它的一個小地方")
    print(result)
    assert "，" in result or "。" in result


if __name__ == "__main__":
    pytest.main([__file__])
