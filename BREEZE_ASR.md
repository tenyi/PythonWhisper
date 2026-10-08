# Breeze-ASR-26 整合說明

本專案已將 ASR 模型從 OpenAI Whisper（`large-v3` / `turbo`）切換到
**MediaTek-Research/Breeze-ASR-26**。

## 關於 Breeze-ASR-26

- **性質**：台語（Taiwanese Hokkien / Taigi）ASR 模型，基於
  `openai/whisper-large-v2` 微調
- **用途**：將台語音訊轉寫為**國字（Mandarin Chinese characters）**
- **架構**：`WhisperForConditionalGeneration`（標準 Whisper，
  transformers 函式庫原生支援）
- **授權**：Apache 2.0
- **Hugging Face**：<https://huggingface.co/MediaTek-Research/Breeze-ASR-26>
- **大小**：約 5.8 GB（2 個 safetensors shard）
- **基準表現**：在 Breeze Taigi 基準上平均 CER 30.13%
  （vs. Whisper-large-v2 49.99%）

> ⚠️ **重要**：此模型**專為台語設計**。若輸入非台語音訊（普通話、
> 英語等），效果會明顯下降。輸出文字為繁體中文漢字。

## 檔案變更

| 檔案 | 變更 |
| --- | --- |
| `xfast.py` | **完全重寫**：改用 `transformers` + PyAV，原本的 OpenAI-Whisper 介面完全保留 |
| `models/Breeze-ASR-26/` | 新增模型權重目錄（從 HF 下載） |
| `pyproject.toml` | 新增 `transformers>=5.0`、`accelerate`、`av`、`tokenizers`、`huggingface-hub` |
| `requirements.txt` | 同步更新依賴 |
| `.gitignore` | 排除 `models/` 目錄（5.8 GB） |
| `BREEZE_ASR.md` | 本說明文件 |

## 主要設計決策

1. **繞過 `transformers` pipeline**：直接呼叫 `AutoModelForSpeechSeq2Seq`
   + `AutoProcessor`，因為 pipeline 會依賴 `torchcodec`，而 `torchcodec`
   對 FFmpeg 版本非常敏感，跨機器常因 FFmpeg minor 版本差異而無法載入。

2. **PyAV 載入音訊**：使用 `av`（PyAV）函式庫動態呼叫系統 FFmpeg，
   避免 `torchcodec` 的靜態綁定問題。

3. **30 秒分塊**：Whisper 模型單次最多處理 30 秒。我們在推論前手動
   分塊並記錄絕對時間偏移，再合併回傳。

4. **時間戳記解碼**：用 `processor.tokenizer.decode(output_offsets=True,
   output_offsets_unit="seconds")` 取得 `<|0.0|>` 等時間戳記 token，
   對應到 `<Segment start, end, text>` 結構。

## 使用方式

### 啟動服務（與原本相同）

```bash
./server.sh
# 或：
uv run gunicorn --workers=1 --worker-class=gthread --threads=4 \
  --bind=0.0.0.0:22434 'xflask-whisper:create_app()'
```

### 環境變數

| 變數 | 預設值 | 說明 |
| --- | --- | --- |
| `BREEZE_ASR_MODEL_PATH` | `./models/Breeze-ASR-26` | 模型路徑，可用絕對路徑覆寫 |

### 重新下載模型

```bash
hf download MediaTek-Research/Breeze-ASR-26 \
  --local-dir ./models/Breeze-ASR-26 \
  --exclude "training_args.bin"
```

## API

Flask 介面完全相容舊版（`xflask-whisper.py` 內部改用新的 `xfast.py`）：

- `POST /transcribe` — OpenAI Whisper API 格式回傳
- `POST /transcribe_full` — 完整輸出（JSON / 逐字稿 / SRT / VTT / 翻譯 / 摘要）
- `POST /translate` — 文字翻譯
- `POST /summary` — 文字摘要
- `GET  /health` — 健康檢查

## 已知限制

1. **torchcodec**：專案不再使用 `torchcodec`，但 `pyproject.toml` 仍
   列出 `torchaudio`（向後相容）。如果遇到 `libavutil.so.56` 找不到
   之類的錯誤，是 `transformers` import 期間的噪音警告，**不會影響
   實際推論**。
2. **首次載入時間**：模型載入約 2 秒，之後會常駐。
3. **靜音/極短音訊**：會回傳空 `segments`（模型正確識別為無語音）。
