# PythonWhisper — Breeze-ASR-26 整合版

> 一個 Flask + Gunicorn ASR（自動語音辨識）服務，原本基於 OpenAI Whisper，現已切換為
> [MediaTek-Research/Breeze-ASR-26](https://huggingface.co/MediaTek-Research/Breeze-ASR-26)
> 並整合 [Silero VAD](https://github.com/snakers4/silero-vad) 改善長音訊時間戳記。

---

## 目錄

- [專案簡介](#專案簡介)
- [架構概覽](#架構概覽)
- [環境需求](#環境需求)
- [安裝步驟](#安裝步驟)
- [下載模型](#下載模型)
- [模型檔案位置總覽](#模型檔案位置總覽)
- [部署到離線伺服器](#部署到離線伺服器)
- [啟動服務](#啟動服務)
- [API 規格](#api-規格)
- [環境變數總覽](#環境變數總覽)
- [技術細節](#技術細節)
  - [音訊載入 (load_audio_mono_16k)](#音訊載入-load_audio_mono_16k)
  - [VAD (VadProcessor)](#vad-vadprocessor)
  - [ASR 引擎 (BreezeASREngine)](#asr-引擎-breezeeasrengine)
  - [時間戳記推算](#時間戳記推算)
  - [字幕切段規則（Netflix 規範）](#字幕切段規則netflix-規範)
  - [錯誤處理與資源管理](#錯誤處理與資源管理)
- [輸出格式](#輸出格式)
- [檔案結構](#檔案結構)
- [故障排除](#故障排除)
- [已知限制](#已知限制)

---

## 專案簡介

本服務提供 OpenAI Whisper 兼容的 HTTP API，端點為 `POST /transcribe` 回傳
OpenAI 風格的 JSON。底層引擎為 **MediaTek-Research/Breeze-ASR-26** — 一個
針對**台灣台語（Taiwanese Hokkien / Taigi）**微調的 Whisper-large-v2 模型，
輸出固定為**繁體中文字**。

| 面向 | 說明 |
| --- | --- |
| 模型 | `MediaTek-Research/Breeze-ASR-26`（基於 `openai/whisper-large-v2`） |
| 模型授權 | Apache 2.0 |
| 模型大小 | 5.8 GB（2 個 safetensors shard） |
| 架構 | `transformers.AutoModelForSpeechSeq2Seq`（標準 Whisper seq2seq） |
| VAD | Silero VAD v4（~1.8 MB / 462K 參數，CPU 推論） |
| 音訊 I/O | PyAV（避免 torchcodec 對 FFmpeg minor 版本的耦合） |
| 推論裝置 | 自動偵測 CUDA → MPS → CPU（float16 / float32） |
| HTTP 伺服器 | Gunicorn（gthread worker，4 threads，1 worker） |
| Python | ≥ 3.12 |
| 依賴管理 | `uv` |

> ⚠️ **重要**：Breeze-ASR-26 **專為台語設計**。對非台語輸入，模型仍會嘗試
> 輸出繁體中文字（視覺上是「翻譯」效果）。若需要通用多語言 ASR，請改用
> `openai-whisper` 或 `faster-whisper` 的 `large-v3` 模型。

---

## 架構概覽

```
┌─────────────────┐  upload  ┌──────────────────┐
│  HTTP Client    │ ───────▶ │  Flask (gunicorn) │
└─────────────────┘          │  xflask-whisper.py│
                             └────────┬──────────┘
                                      │ temp file
                                      ▼
                             ┌──────────────────┐
                             │  xfast.py        │
                             │  transcribe_file │
                             └────────┬─────────┘
                                      │
                       ┌──────────────┴──────────────┐
                       ▼                             ▼
              ┌─────────────────┐         ┌──────────────────┐
              │  VadProcessor   │         │  BreezeASREngine │
              │  Silero VAD     │         │  (transformers)  │
              │  找語音段       │         │  30s sub-chunk   │
              └────────┬────────┘         └────────┬─────────┘
                       │ speech regions            │ segments
                       └────────────┬─────────────┘
                                    ▼
                       ┌────────────────────────┐
                       │  Segment + Info         │
                       │  → JSON / SRT / VTT     │
                       └────────────────────────┘
```

請求流程（`/transcribe`）：

1. Flask 接收 `multipart/form-data` 的音訊檔案，存成暫存檔
2. `xflask-whisper.transcribe()` 呼叫 `xfast.transcribe_openai_format()`
3. `xfast.transcribe_file()`：
   - **音訊載入** — `load_audio_mono_16k()` 透過 PyAV 讀檔 → 16 kHz mono float32
   - **VAD 偵測**（若啟用）— `VadProcessor.detect()` 回傳語音段列表
   - **ASR 推論** — 對每段語音內部以 30s 切塊 → Breeze 模型推論 → 取文字
   - **時間戳記推算** — 將 sub-chunk 內的文字按字元比例分配時間
4. Flask 加上 `Process-Time` header 回傳 JSON

---

## 環境需求

- **Python** ≥ 3.12
- **FFmpeg** 系統版本（任一版即可，PyAV 會動態呼叫）

  ```bash
  # macOS
  brew install ffmpeg
  
  # Debian / Ubuntu
  sudo apt-get install ffmpeg
  ```
- **GPU**（可選）：CUDA 12+ 可大幅加速推論；無 GPU 時自動 fallback CPU
- **磁碟空間**：模型 5.8 GB + Python venv 約 8 GB
- **RAM**：建議 ≥ 16 GB（CPU 推論時 1.5B 參數需要 ~6 GB）

---

## 安裝步驟

```bash
# 1. 進入專案
cd /home/tenyi/PythonWhisper

# 2. 同步依賴（uv 會自動建立 .venv/）
uv sync

# 3. 下載模型
hf download MediaTek-Research/Breeze-ASR-26 \
  --local-dir ./models/Breeze-ASR-26 \
  --exclude "training_args.bin"

# 4. 啟動服務
./server.sh
# 或：
uv run gunicorn --workers=1 --worker-class=gthread --threads=4 \
  --bind=0.0.0.0:22434 --timeout=300 \
  'xflask-whisper:create_app()'
```

### 依賴清單

| 套件 | 版本 | 用途 |
| --- | --- | --- |
| `transformers` | ≥ 5.0（實測 5.19.0） | 載入 Breeze-ASR-26 |
| `accelerate` | ≥ 1.0 | 加速模型載入 |
| `tokenizers` | 0.23.1 ~ 0.23.x | Whisper tokenizer |
| `huggingface-hub` | ≥ 0.34.0 | 模型下載 / hub 互動 |
| `av` (PyAV) | ≥ 12.0 | 音訊解碼（取代 torchcodec） |
| `torch` | ≥ 2.9.1 | Breeze 模型推論 |
| `torchaudio` | ≥ 2.9.1 | 音訊處理輔助 |
| `flask` | ≥ 3.1.2 | HTTP 框架 |
| `gunicorn` | ≥ 23.0.0 | WSGI 伺服器 |
| `openai-whisper` | 20250625 | 備援（未使用） |
| `faster-whisper` | ≥ 1.2.1 | 備援（未使用） |
| `opencc` / `opencc-python-reimplemented` | 1.4.2 / 0.1.7 | 簡轉繁 |

---

## 下載模型

```bash
hf download MediaTek-Research/Breeze-ASR-26 \
  --local-dir ./models/Breeze-ASR-26 \
  --exclude "training_args.bin"
```

下載完成後的目錄結構：

```
models/Breeze-ASR-26/
├── config.json
├── generation_config.json
├── model.safetensors          # 4.99 GB
├── model-00001-of-00002.safetensors  # (實際檔名)
├── model-00002-of-00002.safetensors  # 1.18 GB
├── preprocessor_config.json
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
└── (merges.txt 等)
```

> 📦 VAD 模型**不需手動下載** — 首次啟動時 `torch.hub.load('snakers4/silero-vad')`
> 會自動抓取到 `~/.cache/torch/hub/snakers4_silero-vad_master/`，之後啟動直接吃 cache。

---

## 模型檔案位置總覽

| 模型 | 路徑 | 大小 | 說明 |
| --- | --- | --- | --- |
| Breeze-ASR-26 | `./models/Breeze-ASR-26/` | **5.8 GB** | 透過 `hf download` 下載 |
| Silero VAD | `~/.cache/torch/hub/snakers4_silero-vad_master/` | 36 MB | 透過 `torch.hub.load` 自動下載（會快取整個 git repo） |
| Silero VAD 精簡 | `~/.cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/data/` | 14 MB | 實際推論用到的權重 |

### VAD 權重細節

`src/silero_vad/data/` 目錄內有多種格式的權重，推論時會根據 `onnx` 參數自動選擇：

| 檔案 | 大小 | 用途 |
| --- | --- | --- |
| `silero_vad.jit` | 2.3 MB | PyTorch JIT（`onnx=False`，預設使用） |
| `silero_vad.onnx` | 2.3 MB | ONNX（`onnx=True`） |
| `silero_vad_half.onnx` | 1.3 MB | ONNX fp16 |
| `silero_vad_16k.safetensors` | 1.2 MB | safetensors 格式（PyTorch 載入） |
| `silero_vad_16k_op15.onnx` | 1.3 MB | ONNX opset 15 |
| `silero_vad_16k_op18_ifless.onnx` | 2.8 MB | ONNX opset 18 (if-less) |
| `silero_vad_16k_sequence.onnx` | 1.2 MB | 序列式 ONNX |
| `silero_vad_openvino_16k.onnx` | 1.3 MB | OpenVINO IR |

我們用 PyTorch 版本（`silero_vad.jit` 或 `.safetensors`），其餘是給 ONNX / OpenVINO runtime 用的。

---

## 部署到離線伺服器

> 適用情境：開發機有網路，生產 server 無法連外（air-gapped）。

### 1. 複製模型（在線機器執行）

#### 方案 A：完整複製（最簡單，推薦）

```bash
# 在線機器打包
cd /home/tenyi
tar -czf breeze-asr-models.tar.gz \
  PythonWhisper/models/Breeze-ASR-26 \
  .cache/torch/hub/snakers4_silero-vad_master

# 傳到目標 server（用 scp / rsync / USB 等）
scp breeze-asr-models.tar.gz <user>@<server>:/tmp/

# 目標 server 解開（保持原來的相對路徑）
ssh <user>@<server>
mkdir -p ~/PythonWhisper/models
tar -xzf /tmp/breeze-asr-models.tar.gz -C /home/<user>/
```

#### 方案 B：精簡 VAD（只複製推論需要的權重，36 MB → ~3 MB）

```bash
# 在線機器
cd ~/.cache/torch/hub
tar -czf silero-vad-minimal.tar.gz \
  snakers4_silero-vad_master/src \
  snakers4_silero-vad_master/hubconf.py \
  snakers4_silero-vad_master/pyproject.toml \
  snakers4_silero-vad_master/LICENSE

# 目標 server
mkdir -p ~/.cache/torch/hub/snakers4_silero-vad_master
tar -xzf silero-vad-minimal.tar.gz -C ~/.cache/torch/hub/snakers4_silero-vad_master/
```

### 2. 複製 Python 環境（最重要！）

模型只是冰山一角，整個 Python venv 才是最大：

```bash
# 在線機器（保留所有依賴：transformers, accelerate, av, torch, ...）
rsync -avz --exclude='__pycache__' \
  /home/tenyi/PythonWhisper/.venv/ \
  <server>:/home/<user>/PythonWhisper/.venv/
```

### 3. 複製程式碼

```bash
# 排除 .venv 與 models/（都已另外處理）
rsync -avz \
  --exclude='.venv' \
  --exclude='models/' \
  --exclude='__pycache__' \
  --exclude='*.log' \
  --exclude='logs/' \
  /home/tenyi/PythonWhisper/ \
  <server>:/home/<user>/PythonWhisper/
```

### 4. 安裝系統依賴

目標 server 還需要 FFmpeg（PyAV 動態呼叫）：

```bash
# Debian / Ubuntu
sudo apt-get update && sudo apt-get install -y ffmpeg

# RHEL / CentOS
sudo yum install -y ffmpeg

# macOS
brew install ffmpeg
```

驗證：
```bash
ffmpeg -version | head -1
python -c "import av; print('PyAV OK', av.__version__)"
```

### 5. 設定離線環境變數

防止程式啟動時嘗試連外下載任何東西：

```bash
# 寫到 ~/.bashrc 或 server.sh 開頭
export HF_HUB_OFFLINE=1              # 阻止 huggingface_hub 連線
export TRANSFORMERS_OFFLINE=1        # 阻止 transformers 連線
export BREEZE_ASR_MODEL_PATH=./models/Breeze-ASR-26
```

`uv` 環境管理也需要鎖定（不要再試圖升級套件）：
```bash
# server.sh 開頭加
export UV_OFFLINE=1
```

### 6. 啟動驗證

```bash
# 啟動
cd ~/PythonWhisper
./server.sh

# 健康檢查
curl http://localhost:22434/health
# {"service":"whisper-api","status":"healthy"}

# 帶音訊測試（用本地檔案）
curl -X POST -F "file=@test.ogg" http://localhost:22434/transcribe
```

### 7. 檢查 log 確認沒嘗試連網

啟動時應該看到：

```
✓ Using cache found in /home/<user>/.cache/torch/hub/snakers4_silero-vad_master
✓ Breeze-ASR-26 模型載入完成
✓ 🚀 使用 CUDA GPU: <GPU 型號>
```

**絕對不該**看到的訊息：
- ❌ `Downloading: "https://github.com/..."`
- ❌ `HF_TOKEN` 相關
- ❌ `Connection refused` / `Timeout`
- ❌ `Could not download`

如果出現，模型沒放對位置或環境變數沒設。

### 8. 完整搬遷清單

打包時建議的檔案清單（依賴體積）：

| 項目 | 位置 | 大小 | 必要 |
| --- | --- | --- | --- |
| Breeze-ASR-26 權重 | `models/Breeze-ASR-26/` | 5.8 GB | ✅ |
| Silero VAD 完整 repo | `~/.cache/torch/hub/snakers4_silero-vad_master/` | 36 MB | ✅ |
| Python venv | `.venv/` | ~6-8 GB | ✅ |
| 程式碼 | `xfast.py`, `xflask-whisper.py` 等 | < 1 MB | ✅ |
| FFmpeg 系統套件 | `apt install ffmpeg` | ~50 MB | ✅ |
| GPU driver + CUDA | （系統已裝） | — | ✅ |

**總計約 12-14 GB**（含 Python venv）。

### 9. 用 rsync 增量同步（適合日後更新）

```bash
# 之後有新版本時，只同步變更的檔案
rsync -avzu --exclude='.venv' --exclude='models/' --exclude='__pycache__' \
  /home/tenyi/PythonWhisper/ \
  <server>:/home/<user>/PythonWhisper/

# 同步單一模型權重（單獨處理大檔）
rsync -avz --progress \
  /home/tenyi/PythonWhisper/models/ \
  <server>:/home/<user>/PythonWhisper/models/
```

`-u` 只同步較新的檔案；`--progress` 顯示進度。

---

## 啟動服務

### 使用 `server.sh`

```bash
./server.sh
```

這個腳本內容是：

```bash
uv run gunicorn --workers=1 --worker-class=gthread --threads=4 \
  --bind=0.0.0.0:22434 'xflask-whisper:create_app()'
```

### 直接 gunicorn

```bash
uv run gunicorn --workers=1 --worker-class=gthread --threads=4 \
  --bind=0.0.0.0:22434 --timeout=300 \
  'xflask-whisper:create_app()'
```

### 參數說明

| 參數 | 值 | 說明 |
| --- | --- | --- |
| `--workers` | `1` | 模型會常駐 GPU 記憶體，多 worker 浪費且需跨進程共享 |
| `--worker-class` | `gthread` | 支援同步 I/O + 多線程 |
| `--threads` | `4` | 同時處理 4 個轉錄請求 |
| `--bind` | `0.0.0.0:22434` | 監聽所有網卡 22434 埠 |
| `--timeout` | `300` | 單一請求最長 300 秒 |

### 健康檢查

```bash
curl http://localhost:22434/health
# {"service":"whisper-api","status":"healthy"}
```

---

## API 規格

### `GET /health`

健康檢查。

```bash
curl http://localhost:22434/health
```

**回應** `200 OK`：
```json
{"service": "whisper-api", "status": "healthy"}
```

---

### `POST /transcribe`

轉錄音訊，回傳 OpenAI Whisper API 格式 JSON。

```bash
curl -X POST -F "file=@audio.mp3" http://localhost:22434/transcribe
```

**回應** `200 OK`：
```json
{
  "text": "那很多時候 我們想要修改基礎模型",
  "language": "zh",
  "duration": 94.3411875,
  "segments": [
    {
      "start": 1.1,
      "end": 31.1,
      "text": "那很多時候 我們想要修改基礎模型 往往只想要改它的一個小地方 舉例來說 剛才說問誰是全世界最帥的人 GPT-4 mini 它不直接回答你 我要逼它回答 全世界最帥的人 就是李鴻義 那如果用微調模型的方法的話 那你就要準備訓練資資料 這個訓練資料 就是告訴模型說 輸入誰是全世界最帥的人 輸出就是李鴻義 微調參數之後 問它誰是全世界"
    },
    ...
  ]
}
```

**Headers**：
- `Content-Type: application/json; charset=utf-8`
- `Process-Time: <秒數>` — 伺服器端處理時間（不含 HTTP 傳輸）

**錯誤**：
- `400`：`{"error": "需要上傳檔案"}` / `{"error": "檔案名稱無效或為空"}` / 驗證失敗
- `500`：內部錯誤（會 log 在 `whisper-server.log`）

---

### `POST /transcribe_full`

完整轉錄流程：除了 `/transcribe` 的內容外，額外產生：
- **JSON 檔**（`{base}.json`）
- **逐字稿文字檔**（`{base}_transcript.txt`）
- **SRT 字幕**（`{base}.srt`）
- **VTT 字幕**（`{base}.vtt`）
- **翻譯檔**（`{base}_translation.txt`）— 透過 Ollama 翻譯
- **摘要**（`summary` 欄位）— 透過 Ollama 摘要

> 📁 輸出檔會寫到與輸入音訊相同的目錄。

```bash
curl -X POST -F "file=@audio.mp3" http://localhost:22434/transcribe_full
```

**回應** `200 OK`：
```json
{
  "transcript": "[0.000 --> 31.100]\n 那很多時候 我們想要修改基礎模型\n\n...",
  "file_name": "/tmp/tmpXXXXX.ogg",
  "summary": "這段影片介紹了 LLM 微調的副作用...",
  "srt": "1\n00:00:01,100 --> 00:00:31,100\n那很多時候 ...",
  "vtt": "WEBVTT\n\n00:00:01.100 --> 00:00:31.100\n那很多時候 ...",
  "translation": "...",
  "guid": "01694781-8f4a-423c-85bd-ffeca1b2bad8",
  "duration": 94.3411875,
  "whisper_api_cost": 0.0
}
```

---

### `POST /translate`

純文字翻譯（透過 Ollama）。

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"text": "你好世界", "language": "en"}' \
  http://localhost:22434/translate
```

---

### `POST /summary`

純文字摘要（透過 Ollama）。

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"text": "很長的文字..."}' \
  http://localhost:22434/summary
```

---

## 環境變數總覽

| 變數 | 預設值 | 說明 |
| --- | --- | --- |
| `BREEZE_ASR_MODEL_PATH` | `./models/Breeze-ASR-26` | Breeze 模型路徑（絕對或相對） |
| `VAD_ENABLED` | `1` | `0` / `false` / `False` 會關閉 VAD，fallback 為「整段音訊以 30s 切塊」 |
| `VAD_THRESHOLD` | `0.5` | Silero VAD 語音機率閾值（0~1） |
| `VAD_MIN_SPEECH_MS` | `250` | 短於此毫秒數的語音段會被丟棄（避免噪音觸發） |
| `VAD_MIN_SILENCE_MS` | `500` | 長於此毫秒數的靜音會切句 |
| `VAD_SPEECH_PAD_MS` | `100` | 語音段兩端的 padding（避免切到字首字尾） |
| `BREEZE_SEGMENT_MODE` | `chars` | 字幕切段模式：`chars`（依詞組與寬度切段）、`alignment`（實驗性 cross-attention 對齊）、`off`（每個 chunk 一段） |
| `BREEZE_SEGMENT_CHARS` | `16` | 每段字幕目標顯示寬度（中文 1、英數 0.5），參考 Netflix 繁中每行 16 字；無空白長詞組上限為其 2 倍 |
| `BREEZE_SEGMENT_MIN_CHARS` | `4` | 遇到標點時，段落至少達此寬度才斷句 |

> 💡 所有環境變數在 `initialize_model()` 呼叫時讀取；修改後需重啟服務。

### VAD 參數調校建議

| 場景 | 建議值 |
| --- | --- |
| Podcast / 長講者連續演說 | `VAD_MIN_SILENCE_MS=1000`（避免把句子中的逗號停頓當切點） |
| 多對話人會議錄音 | `VAD_MIN_SILENCE_MS=700`，`VAD_MIN_SPEECH_MS=300`（過濾咳嗽、笑聲） |
| 嘈雜背景（戶外、新聞現場） | `VAD_THRESHOLD=0.6` ~ `0.7`（降低誤報） |
| 安靜錄音室 | `VAD_THRESHOLD=0.4`（提高敏感度，避免漏掉輕聲） |
| 完全停用 VAD（連續獨白、ASR 文字足夠準確） | `VAD_ENABLED=0` |

---

## 技術細節

### 音訊載入 (`load_audio_mono_16k`)

**位置**：`xfast.py`，行 ~75

**為什麼不用 torchcodec？** `transformers` pipeline 內部依賴
`torchcodec` 做音訊 I/O，而 `torchcodec` 是預編譯的 wheel，
動態鏈結到 FFmpeg minor 56/57/58/59 — 跨機器時常因系統 FFmpeg
版本不同而 `OSError: libavutil.so.56: cannot open shared object file`。

**改用 PyAV 的原因**：

- PyAV 是 FFmpeg 的 Python binding，**動態**呼叫系統 FFmpeg 的 `.so` / `.dylib`
- 對 FFmpeg 版本寬容（任何 4.x / 5.x / 6.x / 7.x / 8.x 都能用）
- 已經在 venv 中（`av==19.0.1`）

**處理步驟**：

```python
container = av.open(path)              # 開啟容器
sr_in = container.streams.audio[0].rate # 取輸入取樣率
for frame in container.decode(audio=0): # 解碼所有 frame
    arr = frame.to_ndarray()            # shape: (channels, samples)
    if arr.shape[0] > 1:                # 多聲道 → 降混單聲道
        arr = arr.mean(axis=0)
    else:
        arr = arr.flatten()
    arr = arr.astype(np.float32)
    if fmt.startswith("s16"):           # int16 需除以 32768.0
        arr /= 32768.0
    elif fmt.startswith("s32"):         # int32 需除以 2147483648.0
        arr /= 2147483648.0
    # flt / fltp 已為 float32，在 [-1, 1] 範圍
    frames.append(arr)
container.close()
wav = np.concatenate(frames)
# 重採樣到 16 kHz（Whisper 要求）
if sr_in != 16000:
    wav = F.interpolate(...).squeeze().numpy()
```

**支援的輸入格式**：MP3、AAC、FLAC、WAV、OGG/Vorbis、Opus、M4A、WebM
（任何 FFmpeg 支援的格式都行）。

**取樣率**：自動重採樣到 16 kHz，使用 `torch.nn.functional.interpolate`
的 linear 模式（雖然語音該用 resample，但 linear 已足夠 ASR 用，且
不需要額外的 `torchaudio.functional.resample` 依賴）。

> 🐛 **歷史 bug 紀錄**：原本的實作無條件 `/ 32768.0`，但 `sample_fmt=fltp`
> 格式（MP3、OGG、Vorbis 常見）回傳的已經是 float32，導致所有音訊衰減成 0，
> 模型以為是靜音並幻覺輸出。修法見上方 `if fmt.startswith(...)` 分支。

---

### VAD (`VadProcessor`)

**位置**：`xfast.py`，行 ~115

**為什麼需要 VAD？** Breeze-ASR-26 訓練時**不學習** `<|0.0|>` 這類時間戳記
token，所以 `model.generate(return_timestamps=True)` + `decode(output_offsets=True)`
拿不到任何有時間的 chunk（`non-empty offsets: 0/0`）。

VAD 解決方案：
1. 用 Silero VAD 找出真實的語音段
2. 跳過靜音段（節省 ASR 算力）
3. 對每個語音段內部以 30s 切塊 → Breeze 推論
4. 把 sub-chunk 的絕對時間偏移加到 segments 上

**載入方式**：

```python
self.model, self.utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False,
    trust_repo=True,
)
```

- `torch.hub` 自動下載到 `~/.cache/torch/hub/snakers4_silero-vad_master/`
- 之後啟動直接吃 cache（< 0.1s 載入）
- 第一次會從 GitHub 下載 ~1.8 MB

**API 用法**：

```python
ts = vad.get_speech_timestamps(
    wav_tensor,                    # torch.Tensor, float32, 1D
    vad.model,
    sampling_rate=16000,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=100,
    threshold=0.5,
    return_seconds=True,           # True → 回傳 seconds；False → samples
)
# 範例回傳：[
#   {'start': 1.10, 'end': 49.70},
#   {'start': 50.50, 'end': 74.50},
#   {'start': 74.90, 'end': 91.40},
#   {'start': 91.90, 'end': 94.30},
# ]
```

**為什麼 VAD 跑 CPU？** Silero VAD 只有 462K 參數（~1.8 MB），CPU
推論一個 30s 音訊只要 ~50 ms，送到 GPU 反而有 host↔device 傳輸成本。
預設放在 CPU。

**VAD 失敗 fallback**：
若 `VadProcessor.__init__` 拋出任何 exception（例如無網路下載模型），
會被 `BreezeASREngine._load` 捕捉，記錄 error log，並把 `self.vad = None`、
`self.use_vad = False`，讓 `transcribe()` 改用「整段音訊 30s 切塊」模式。

---

### ASR 引擎 (`BreezeASREngine`)

**位置**：`xfast.py`，行 ~190

**核心方法**：`transcribe(audio_path, language="chinese", chunk_length_s=30)`

**完整流程**：

```
audio_path
  │
  ▼
load_audio_mono_16k() → wav (float32, 16kHz, mono)
  │
  ▼
VAD detect() → [(start, end), ...]   # 語音段列表
  │
  ▼
for each region (start, end):
    region_wav = wav[start*sr : end*sr]
    for each sub_start in range(0, len(region_wav), 30s*sr):
        chunk = region_wav[sub_start : sub_start+30s*sr]
        chunk_offset = region_start + sub_start / sr
        chunk_segs = _transcribe_chunk(chunk, "chinese", chunk_offset)
  │
  ▼
all_segments: List[Segment]
  │
  ▼
TranscriptionInfo(language="zh", probability=1.0, duration)
```

**`_transcribe_chunk` 細節**：

```python
def _transcribe_chunk(self, chunk_wav, language, offset):
    # 1. feature_extractor 把 wav 轉成 mel spectrogram
    inputs = self.processor.feature_extractor(
        chunk_wav, sampling_rate=16000, return_tensors="pt"
    )
    input_features = inputs.input_features.to(device).to(dtype)
    
    # 2. Breeze 推論
    with torch.no_grad():
        pred_ids = self.model.generate(
            input_features,
            max_new_tokens=440,        # < 448 (max_target_positions)
            language="chinese",        # <|zh|> token
            task="transcribe",
            # 不傳 return_timestamps — Breeze 不會產生時間戳
        )
    
    # 3. 拿文字
    text = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
    if not text:
        return []
    
    # 4. chunk 內按標點切句（如果文字有標點的話）
    parts = self._split_by_punctuation(text)
    
    # 5. 按字元比例分配時間
    chunk_duration = len(chunk_wav) / 16000
    segments = []
    cursor = offset
    for part in parts:
        portion = len(part) / total_chars
        seg_dur = chunk_duration * portion
        segments.append(Segment(start=cursor, end=cursor+seg_dur, text=part))
        cursor += seg_dur
    
    return segments
```

**為什麼 `language="chinese"`？** Breeze 在訓練時固定使用 `<|zh|>` token
（看 `generation_config.json` 與 `preprocessor_config.json`）。模型被設計
為「聽任何語言 → 輸出繁體中文」。不傳 `language` 參數時，transformers 會
自動偵測但效果不穩定。

**為什麼 `max_new_tokens=440`？** 模型設定的 `max_target_positions=448`，
而 prompt（`<|startoftranscript|><|zh|><|transcribe|>`）佔 3 個位置。
如果 `max_new_tokens=448`，會拋 `ValueError: 3 + 448 = 451 > 448`。
取 440 留一點 buffer。

**dtype 選擇**：
- `cuda` → `torch.float16`（省顯存，速度快）
- `mps` / `cpu` → `torch.float32`（fp16 在 CPU 會有精度問題）

---

### 時間戳記推算

Breeze 不產生時間戳，所以我們用**幾何推算**：

| 層級 | 推算方式 |
| --- | --- |
| VAD 語音段起點 | `VadProcessor.get_speech_timestamps(return_seconds=True)` 直接給秒數 |
| Sub-chunk 起點 | `region_start + sub_chunk_index * 30` |
| Sub-chunk 內部句子 | `chunk_duration * (段落寬度 / 總寬度)` 依顯示寬度比例分配（不含空白） |

VAD 會以 `max_speech_duration_s=29` 在靜音點自動切開長語音段，
因此每個語音段都 ≤ 30s，不會在句子中間硬切 30s 造成重複或漏字。

**例子**：test.ogg (94.34s)
- VAD 偵測 6 段，最長 24.0s，全部 ≤ 30s
- 第一段 `[1.10, 18.00]` 為一個 chunk：offset=1.10, duration=16.90
- 第二段 `[18.20, 37.00]` 為一個 chunk：offset=18.20, duration=18.80
- ...

**已知精度限制**：

- VAD 邊界本身有 ±100ms 誤差（受 `speech_pad_ms` 影響）
- 句子內部時間依寬度比例估算，**不是字級（word-level）對齊**

---

### 字幕切段規則（Netflix 規範）

`chars` 模式下，每個 chunk 的辨識文字會依下列規則切成多段字幕
（`BreezeASREngine._split_text_by_chars`）。數值參考
[Netflix 字幕規範](https://partnerhelp.netflixstudios.com/hc/en-us/articles/215986007)：

| 規範項目 | 繁體中文 | 英文 |
| --- | --- | --- |
| 每行最多 | 16 字 | 42 字元 |
| 每段最多 | 2 行 | 2 行 |
| 閱讀速度 | 約 9 字／秒 | 約 17–20 字元／秒 |

**顯示寬度**：中文、全形字元算 1，半形英數字元算 0.5。
42 個英文字元 ≈ 21 寬，與中文每行 16 字相近，中英混排時長度一致。

**切段規則**：

1. **只在詞組邊界斷句**：Breeze 輸出**沒有標點**，但詞組之間有空白，
   因此以空白（以及有標點時的標點）作為斷句點，不會把「微調」切成「微｜調」。
2. **合併成一行**：相鄰詞組合併，每段寬度 ≤ 16（`BREEZE_SEGMENT_CHARS`，一行字幕）。
3. **長詞組容忍兩行**：中間沒有空白的長詞組，寬度 ≤ 32（兩行）時整段保留；
   超過才**平均**切成等長片段，避免留下 1–2 字的短尾巴。
4. **英文單字不切**：純英數詞組（長單字、URL）一律完整保留，可能超過 32。
   連續英文單字（如 `GPT-4 Mini`）合計寬度 ≤ 16 時視為一個詞組，不會從中間斷開；
   更長的整句英文則逐字合併。
5. **標點優先**：若輸出有標點（如原版 Whisper），標點一律黏在前段結尾，
   且段落寬度 ≥ 4（`BREEZE_SEGMENT_MIN_CHARS`）時在標點處斷句。

**範例**（test.ogg）：

| 開始 | 結束 | 文字 |
| --- | --- | --- |
| 00:00:01.100 | 00:00:04.503 | 那很多時候 我們想要修改基礎模型 |
| 00:00:04.503 | 00:00:07.452 | 往往只想要改它的一個小地方 |
| 00:00:07.452 | 00:00:08.359 | 舉例來說 |
| 00:00:08.359 | 00:00:11.308 | 剛才說問誰是全世界最帥的人 |
| 00:00:11.308 | 00:00:13.917 | GPT-4 Mini 它不直接回答你 |

> 若想要更短的字幕（例如手機直式畫面），可設 `BREEZE_SEGMENT_CHARS=12`。

---

### 錯誤處理與資源管理

**錯誤裝飾器**：`@handle_errors`（`error_handler.py`）
- 把一般 `Exception` 轉成 `WhisperError`
- 自動 log `traceback`

**檔案操作裝飾器**：`@handle_file_operations`
- `FileNotFoundError` → `FileProcessingError`
- `PermissionError` → `FileProcessingError`
- `OSError` → `FileProcessingError`

**資源管理**：`xflask-whisper.py` 用 try/finally 確保暫存檔被刪除：

```python
finally:
    if temp_file_path and os.path.exists(temp_file_path):
        try:
            os.remove(temp_file_path)
        except Exception as e:
            logger.warning(f"清理臨時檔案失敗: {str(e)}")
```

**執行緒安全**：`asr_lock = threading.Lock()` 保護 `asr_engine` 全域變數，
避免 race condition。

**GPU 記憶體釋放**：`release_model()` 會：
1. `del asr_engine`
2. `torch.cuda.empty_cache()`（若可用）
3. `gc.collect()`

---

## 輸出格式

### OpenAI 風格 JSON（`/transcribe`）

```json
{
  "text": "整段合併文字",
  "language": "zh",
  "duration": 94.34,
  "segments": [
    {"start": 1.1, "end": 31.1, "text": "..."},
    {"start": 31.1, "end": 49.7, "text": "..."}
  ]
}
```

### SRT 字幕

```
1
00:00:01,100 --> 00:00:31,100
那很多時候 我們想要修改基礎模型 往往只想要改它的一個小地方

2
00:00:31,100 --> 00:00:49,700
世界最帥的人 他就回答李弘一 ...
```

### VTT 字幕

```
WEBVTT

00:00:01.100 --> 00:00:31.100
那很多時候 我們想要修改基礎模型 ...

00:00:31.100 --> 00:00:49.700
世界最帥的人 他就回答李弘一 ...
```

### 逐字稿

```
[00:00:01.100 --> 00:00:31.100]
 那很多時候 我們想要修改基礎模型 ...

[00:00:31.100 --> 00:00:49.700]
 世界最帥的人 他就回答李弘一 ...
```

### 完整結果（`/transcribe_full`）

見 [API 規格](#post-transcribe_full) 段落。

---

## 檔案結構

```
PythonWhisper/
├── BREEZE_ASR.md            # 舊版變更說明
├── README.md                # 本文件
├── server.sh                # 啟動腳本
├── pyproject.toml           # uv 依賴設定
├── requirements.txt         # pip 備用依賴
├── uv.lock                  # uv lockfile
├── .gitignore
│
├── xflask-whisper.py        # Flask HTTP 介面
├── xfast.py                 # Breeze-ASR-26 + VAD 整合（本專案核心）
├── xwhisper.py              # 舊版：openai-whisper CLI wrapper
├── fast.py                  # 舊版：faster-whisper wrapper
│
├── common.py                # 共用資料結構 (Segment, TranscriptionInfo)
├── exceptions.py            # 自訂異常
├── error_handler.py         # 統一錯誤處理
├── logging_config.py        # logging 設定
├── resource_manager.py      # 暫存檔追蹤
├── translator_ollama.py     # Ollama 翻譯/摘要
├── main.py                  # CLI 入口
│
├── models/                  # 模型權重（不在 git 中）
│   └── Breeze-ASR-26/
│       ├── config.json
│       ├── model-*.safetensors
│       ├── preprocessor_config.json
│       ├── tokenizer.json
│       └── ...
│
├── logs/                    # log 檔目錄
│   └── *.log
│
├── test.mp3                 # 測試音訊 (1.1s 英文 "You have new mail")
├── test.ogg                 # 測試音訊 (94.3s 國語 LLM 微調)
└── whisper-server.log       # Flask 服務 log
```

---

## 故障排除

### 1. `OSError: libavutil.so.56: cannot open shared object file`

`transformers` import 時的 torchcodec 警告，不影響功能（我們已繞過
pipeline 改用 `model.generate` + PyAV）。若 log 太吵可忽略。

若真的想消掉：

```bash
uv pip uninstall torchcodec
```

（但 `pyproject.toml` 裡沒有列 `torchcodec`，下次 `uv sync` 不會裝回來）

### 2. `The repository snakers4_silero-vad does not belong to the list of trusted repositories`

VAD 載入需要 `trust_repo=True`。已在程式碼中處理：
```python
torch.hub.load(..., trust_repo=True)
```
若你改了這行被擋，把 `trust_repo=True` 加回去。

### 3. `ValueError: 3 + 448 = 451 > 448`

`max_new_tokens=448` 太大。修法：使用 `MAX_NEW_TOKENS=440`（已 hard-code
在 `xfast.py`）。若你手動改過 generate 參數，請設為 440 或更小。

### 4. 模型輸出空 segments

可能原因：
- 音訊過短（< 0.1s）— log 會說「音訊過短，跳過推論」
- 音訊全部是靜音 — VAD 會回傳空，segments 自然空
- VAD 載入失敗 fallback — log 會有「Silero VAD 載入失敗」

### 5. 時間戳記看起來不對（不是 VAD 邊界）

檢查：
- `VAD_ENABLED` 是否設成 `0`？`echo $VAD_ENABLED`
- log 中是否有「VAD 偵測到 X 段語音」訊息？
- 試試調整 `VAD_MIN_SILENCE_MS`（加大 = 較少切點）

### 6. `transformers` 版本衝突

`huggingface_hub` 1.11 與 `transformers` 5.x 不相容。修法：
```bash
uv lock --upgrade-package tokenizers --upgrade-package huggingface-hub --upgrade-package transformers
uv sync
```

---

## 已知限制

1. **台語專用模型** — Breeze-ASR-26 對非台語輸入仍會輸出繁體中文字
   （效果近似翻譯）。需要多語言請改用 Whisper-large-v3。

2. **時間戳記精度** — VAD 邊界 ±100ms，句子內時間均分（非 word-level）。
   若需 word-level 對齊，建議用 `faster-whisper` + `word_timestamps=True`。

3. **單 GPU 單 worker** — gunicorn `workers=1` 是必要的（Breeze 模型常駐
   ~3 GB 顯存，多 worker 浪費）。若要提高吞吐量，改 `--threads`（已預設 4）。

4. **模型載入時間** — 首次啟動約 3-4 秒（含 VAD）。後續請求 < 0.1s 命中
   cache。

5. **Breeze 模型本身的轉寫不穩定** — 例如同一段音訊可能出現「李鴻義」/
   「李弘一」/「李宏毅」混用（同一個人名）。這是 Breeze 模型的特性，
   與 VAD / 音訊處理無關。

6. **無 language detection** — Breeze 強制使用 `language="chinese"`，
   沒有語言自動偵測。

7. **VAD 第一個 region 從 0 開始的假設** — 若 VAD 偵測結果第一個 region
   不從 0 開始（例如 [1.10, ...]），segments 會從 1.10 開始，開頭
   1.10s 的靜音會被跳過（這通常是想要的行為）。

8. **FFmpeg 安裝必要** — PyAV 動態呼叫系統 FFmpeg，若完全沒有 FFmpeg
   會 `av.open(path)` 失敗。
