"""
Breeze-ASR-26 語音轉錄服務

使用 MediaTek-Research/Breeze-ASR-26（Whisper-large-v2 為基礎的台語 ASR）。
- 透過 transformers 直接呼叫 AutoModelForSpeechSeq2Seq，繞過會依賴
  torchcodec 的 pipeline。
- 使用 PyAV 載入音訊，避免 FFmpeg 版本衝突。
- 自行做 30 秒切塊與時間戳記合併，支援長音訊。

對外介面（transcribe_file / transcribe_openai_format / transcribe_full /
initialize_model / release_model / write_*）與原本 OpenAI-Whisper 版本
保持相容，xflask-whisper.py 無需更動。
"""
import gc
import os
import re
import sys
import threading
import uuid
import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# 匯入共用模組
from common import (
    Segment,
    TranscriptionInfo,
    opencc_manager,
    safe_execute,
    handle_errors,
    log_error,
    log_info,
)
from error_handler import error_context, handle_file_operations
from exceptions import (
    ErrorCode,
    FileError,
    ModelError,
    SystemError,
    TranscriptionError,
    create_file_not_found_error,
    create_model_load_error,
    create_transcription_error,
)
from logging_config import get_logger
from translator_ollama import summary_text_ollama, translate_text_ollama

# 抑制 transformers 內部的 torchcodec 噪音警告（FFmpeg 版本衝突時會一直輸出）
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

logger = get_logger("xfast")  # 保留舊 logger 名稱以維持日誌/監控相容

# ---- 模型常數 ----
# 預設模型路徑：可透過環境變數 BREEZE_ASR_MODEL_PATH 覆寫
DEFAULT_MODEL_PATH = os.environ.get(
    "BREEZE_ASR_MODEL_PATH", "./models/Breeze-ASR-26"
)
MODEL_ID = "MediaTek-Research/Breeze-ASR-26 (Whisper-large-v2 為基礎的台語 ASR)"
TARGET_SR = 16000  # Whisper 要求 16 kHz
CHUNK_LENGTH_S = 30  # Whisper 單次最大輸入長度
MAX_NEW_TOKENS = 440  # < max_target_positions (448)

# 全域 ASR 引擎與鎖
asr_engine: Optional["BreezeASREngine"] = None
asr_lock = threading.Lock()


# =============================================================================
# 音訊載入
# =============================================================================
def load_audio_mono_16k(path: str) -> np.ndarray:
    """使用 PyAV 載入音訊並轉為單聲道 16 kHz float32 ndarray。

    PyAV 直接呼叫系統 FFmpeg 的動態庫，不會像 torchcodec 那樣綁定特定
    FFmpeg minor 版本，跨機器較穩定。
    """
    try:
        import av
    except ImportError as e:
        raise FileError(
            message="缺少 PyAV 套件，請執行: uv pip install av",
            error_code=ErrorCode.SYSTEM_ERROR,
            original_exception=e,
        )

    container = av.open(path)
    try:
        audio_stream = container.streams.audio[0]
        sr_in = audio_stream.rate or TARGET_SR
        frames: List[np.ndarray] = []
        for frame in container.decode(audio=0):
            # PyAV 對不同 sample format 會回傳不同 dtype：
            # - 'flt' / 'fltp'：float32，已在 [-1, 1] 範圍
            # - 's16' / 's16p'：int16，需除以 32768.0 才到 [-1, 1]
            # - 's32' / 's32p'：int32，需除以 2147483648.0
            fmt_name = frame.format.name
            arr = frame.to_ndarray()  # shape: (channels, samples) 或 (samples,)
            if arr.ndim == 2 and arr.shape[0] > 1:
                arr = arr.mean(axis=0)
            else:
                arr = arr.flatten()
            arr = arr.astype(np.float32)
            if fmt_name.startswith("s16"):
                arr = arr / 32768.0
            elif fmt_name.startswith("s32"):
                arr = arr / 2147483648.0
            # flt / fltp / fltp 等 float 格式：已經在 [-1, 1]，不處理
            frames.append(arr)
        wav = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
    finally:
        container.close()

    # 重採樣到 16 kHz
    if sr_in != TARGET_SR and len(wav) > 0:
        t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0)
        new_len = int(t.shape[-1] * TARGET_SR / sr_in)
        t = F.interpolate(t, size=new_len, mode="linear", align_corners=False)
        wav = t.squeeze().numpy()
    return wav


# =============================================================================
# Silero VAD (語音活動偵測)
# =============================================================================
class VadProcessor:
    """Silero VAD 封裝。用於偵測音訊中語音的時間區間，以便：

    1. 把長音訊切到「真實的語句邊界」而不是任意的 30s 區塊
    2. 跳過靜音段，節省 ASR 推論算力
    3. 補償 Breeze-ASR-26 不產生時間戳記 token 的限制
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 500,
        speech_pad_ms: int = 100,
    ):
        logger.info(
            f"載入 Silero VAD: threshold={threshold}, "
            f"min_speech={min_speech_ms}ms, min_silence={min_silence_ms}ms, "
            f"speech_pad={speech_pad_ms}ms"
        )
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.speech_pad_ms = speech_pad_ms

        # torch.hub 載入（第一次會下載 ~1.8MB 模型到 ~/.cache/torch/hub/）
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        (self.get_speech_timestamps, _, _, _, _) = self.utils
        # VAD 非常輕量，CPU 反而比 GPU 快
        self.model.eval()
        logger.info("Silero VAD 載入完成")

    def detect(self, wav: np.ndarray, sample_rate: int = TARGET_SR) -> List[Tuple[float, float]]:
        """回傳 (start_sec, end_sec) 列表，代表語音段。"""
        wav_tensor = torch.from_numpy(wav).float()
        timestamps = self.get_speech_timestamps(
            wav_tensor,
            self.model,
            sampling_rate=sample_rate,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            speech_pad_ms=self.speech_pad_ms,
            threshold=self.threshold,
            return_seconds=True,
        )
        return [(t["start"], t["end"]) for t in timestamps]


# =============================================================================
# ASR 引擎
# =============================================================================
class BreezeASREngine:
    """Breeze-ASR-26 推論引擎。"""

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype,
        use_vad: bool = True,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.use_vad = use_vad
        self.model: Optional[AutoModelForSpeechSeq2Seq] = None
        self.processor: Optional[AutoProcessor] = None
        self.vad: Optional[VadProcessor] = None
        self._load()

    def _load(self) -> None:
        logger.info(
            f"載入 {MODEL_ID}: path={self.model_path}, "
            f"device={self.device.type}, dtype={self.dtype}"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        if self.use_vad:
            try:
                self.vad = VadProcessor(
                    threshold=float(os.environ.get("VAD_THRESHOLD", "0.5")),
                    min_speech_ms=int(os.environ.get("VAD_MIN_SPEECH_MS", "250")),
                    min_silence_ms=int(os.environ.get("VAD_MIN_SILENCE_MS", "500")),
                    speech_pad_ms=int(os.environ.get("VAD_SPEECH_PAD_MS", "100")),
                )
            except Exception as e:
                import traceback
                logger.error(f"Silero VAD 載入失敗: {e}")
                logger.error(traceback.format_exc())
                logger.warning("將以無 VAD 模式運行")
                self.vad = None
                self.use_vad = False
        logger.info("Breeze-ASR-26 模型載入完成")

    def transcribe(
        self,
        audio_path: str,
        language: str = "chinese",
        chunk_length_s: int = CHUNK_LENGTH_S,
    ) -> Tuple[List[Segment], TranscriptionInfo]:
        """轉錄整段音訊，回傳 (segments, info)。

        流程：
        1. 載入音訊 → 16kHz mono
        2. （若有 VAD）偵測語音段、跳過靜音 → 取得 (start, end) 列表
        3. 針對每個語音段中切 30s sub-chunk（若 > 30s）→ ASR 推論
        4. 合併所有 segments，按絕對時間回傳
        """
        wav = load_audio_mono_16k(audio_path)
        duration = len(wav) / TARGET_SR if len(wav) > 0 else 0.0

        # 太短或空音訊：直接回空結果
        if len(wav) < int(TARGET_SR * 0.1):
            logger.warning(f"音訊過短（{duration:.2f}s），跳過推論")
            return [], TranscriptionInfo(
                language="zh", language_probability=0.0, duration=duration
            )

        # Step 1: 決定要處理的區間
        if self.vad is not None:
            regions = self.vad.detect(wav)
            if not regions:
                logger.info("VAD 未偵測到任何語音段")
                info = TranscriptionInfo(
                    language="zh", language_probability=0.0, duration=duration
                )
                return [], info
            logger.info(
                f"VAD 偵測到 {len(regions)} 段語音 "
                f"（總計 {sum(e - s for s, e in regions):.2f}s / 音訊 {duration:.2f}s）"
            )
        else:
            # 退路：整段音訊作為一個大區間
            regions = [(0.0, duration)]

        # Step 2: 對每個 region 內部切成 ≤ 30s sub-chunk
        chunk_samples = chunk_length_s * TARGET_SR
        all_segments: List[Segment] = []
        for region_start, region_end in regions:
            r_start_sample = int(region_start * TARGET_SR)
            r_end_sample = int(region_end * TARGET_SR)
            region_wav = wav[r_start_sample:r_end_sample]
            if len(region_wav) < int(TARGET_SR * 0.1):
                continue
            for sub_start in range(0, len(region_wav), chunk_samples):
                chunk = region_wav[sub_start : sub_start + chunk_samples]
                if len(chunk) < int(TARGET_SR * 0.1):  # < 0.1s 跳過
                    continue
                # 絕對偏移：region 起點 + sub-chunk 在 region 內的偏移
                chunk_offset = region_start + sub_start / TARGET_SR
                chunk_segs = self._transcribe_chunk(chunk, language, chunk_offset)
                all_segments.extend(chunk_segs)

        info = TranscriptionInfo(
            language="zh",
            language_probability=1.0,  # Breeze-ASR-26 不暴露 language 機率
            duration=duration,
        )
        logger.info(
            f"Breeze-ASR-26 轉錄完成：{len(all_segments)} 個片段, "
            f"音訊時長 {duration:.2f}s"
        )
        return all_segments, info

    def _transcribe_chunk(
        self, chunk_wav: np.ndarray, language: str, offset: float
    ) -> List[Segment]:
        """對單一 chunk 執行推論，依設定切成多個 segment。

        兩種切段模式（透過 BREEZE_SEGMENT_MODE 環境變數切換）：

        1. "chars"（預設）：按字元數切段，時間均分
           - 適合：無法做 cross-attention alignment 的模型（如 Breeze-ASR-26）
           - 環境變數：BREEZE_SEGMENT_CHARS=10（每段最大字數）

        2. "alignment"：用 cross-attention 找 token 對齊（需 Breeze 是
           Whisper 系且有時間戳記能力；目前 Breeze 沒學到對齊，會 fallback
           到 "chars"）

        3. "off"：整個 chunk 1 個 segment（向後相容舊行為）
        """
        assert self.model is not None and self.processor is not None

        mode = os.environ.get("BREEZE_SEGMENT_MODE", "chars").lower()

        # 1. 跑 ASR 拿文字
        inputs = self.processor.feature_extractor(
            chunk_wav, sampling_rate=TARGET_SR, return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device).to(self.dtype)

        with torch.no_grad():
            pred_ids = self.model.generate(
                input_features,
                max_new_tokens=MAX_NEW_TOKENS,
                language=language,
                task="transcribe",
            )

        text_out = self.processor.batch_decode(
            pred_ids, skip_special_tokens=True
        )[0].strip()

        if not text_out:
            return []

        chunk_duration = len(chunk_wav) / TARGET_SR

        # 2. 依 mode 切段
        if mode == "off":
            return [
                Segment(
                    start=round(offset, 3),
                    end=round(offset + chunk_duration, 3),
                    text=text_out,
                )
            ]
        elif mode == "alignment":
            try:
                aligned_segs = self._transcribe_chunk_aligned(
                    chunk_wav, language, offset, pred_ids, chunk_duration
                )
                if aligned_segs:
                    return aligned_segs
            except Exception as e:
                logger.warning(f"Cross-attention alignment 失敗，退用 chars: {e}")
                # 繼續用 chars 模式

        # 預設 chars 模式：按字元切段
        max_chars = int(os.environ.get("BREEZE_SEGMENT_CHARS", "10"))
        min_chars = int(os.environ.get("BREEZE_SEGMENT_MIN_CHARS", "4"))
        return self._split_text_by_chars(
            text_out, offset, chunk_duration, max_chars, min_chars
        )

    @staticmethod
    def _split_text_by_chars(
        text: str,
        offset: float,
        chunk_duration: float,
        max_chars: int = 10,
        min_chars: int = 4,
    ) -> List[Segment]:
        """按字元數切段，時間均分。

        規則：
        - 支援中文與台語漢字無空白連續文字切段
        - 英文與羅馬拼音按單詞切段並保留原空白
        - 標點符號優先作為自然斷句點
        - 切完每段字元數 <= max_chars（單詞超過者獨立成段）
        - 最後一段可短於 min_chars（保留尾部）
        """
        if not text or not text.strip():
            return []

        # Tokenize：區分 CJK 單字元、一般文字+空格、空白符
        pattern = (
            r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3000-\u303f\uff00-\uffef]'
            r'|[^\s\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3000-\u303f\uff00-\uffef]+(?:\s*)'
            r'|\s+'
        )
        tokens = re.findall(pattern, text)
        if not tokens:
            return []

        punct_set = set("，。！？、；：,!?.…")
        raw_segments: List[str] = []
        current_tokens: List[str] = []

        def cur_len() -> int:
            return len("".join(current_tokens).strip())

        for tok in tokens:
            # 若前一個 token 結尾為標點符號，且已達最短長度，優先切段
            if current_tokens and cur_len() >= min_chars:
                last_char = "".join(current_tokens).rstrip()[-1:]
                if last_char in punct_set:
                    raw_segments.append("".join(current_tokens).strip())
                    current_tokens = []

            tok_len = len(tok.strip())
            if tok_len > 0 and cur_len() + tok_len > max_chars and cur_len() >= min_chars:
                raw_segments.append("".join(current_tokens).strip())
                current_tokens = []

            current_tokens.append(tok)

        if current_tokens:
            txt = "".join(current_tokens).strip()
            if txt:
                raw_segments.append(txt)

        if not raw_segments:
            return []

        # 計算每段時間（按字元比例）
        total_chars = sum(len(s) for s in raw_segments)
        if total_chars == 0:
            return []

        result: List[Segment] = []
        cursor = offset
        end_boundary = offset + chunk_duration
        for i, txt in enumerate(raw_segments):
            portion = len(txt) / total_chars
            seg_dur = chunk_duration * portion
            seg_start = cursor
            seg_end = seg_start + seg_dur
            if i == len(raw_segments) - 1:
                # 最後一段對齊 chunk 結尾
                seg_end = end_boundary
            result.append(
                Segment(
                    start=round(seg_start, 3),
                    end=round(seg_end, 3),
                    text=txt,
                )
            )
            cursor = seg_end
        return result

    def _transcribe_chunk_aligned(
        self, chunk_wav, language, offset, pred_ids=None, chunk_duration=None
    ) -> List[Segment]:
        """使用 cross-attention 取得每個 token 的時間（Breeze 訓練時
        沒學時間對齊，效果差 — 大多數 token 的 attention 集中在
        attention sink；保留作為實驗用）。
        """
        assert self.model is not None and self.processor is not None

        if chunk_duration is None:
            chunk_duration = len(chunk_wav) / TARGET_SR

        if pred_ids is None:
            inputs = self.processor.feature_extractor(
                chunk_wav, sampling_rate=TARGET_SR, return_tensors="pt"
            )
            input_features = inputs.input_features.to(self.device).to(self.dtype)
            with torch.no_grad():
                pred_ids = self.model.generate(
                    input_features,
                    max_new_tokens=MAX_NEW_TOKENS,
                    language=language,
                    task="transcribe",
                )[0]
        elif hasattr(pred_ids, "ndim") and pred_ids.ndim == 2:
            pred_ids = pred_ids[0]

        if len(pred_ids) <= 4:
            return []

        inputs = self.processor.feature_extractor(
            chunk_wav, sampling_rate=TARGET_SR, return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device).to(self.dtype)

        with torch.no_grad():
            outputs = self.model(
                input_features=input_features,
                decoder_input_ids=pred_ids.unsqueeze(0),
                output_attentions=True,
                return_dict=True,
            )

        cross_attentions = outputs.cross_attentions
        if cross_attentions is None or len(cross_attentions) == 0:
            return []

        n_layers = len(cross_attentions)
        align_layer = int(os.environ.get("BREEZE_ALIGN_LAYER", "16"))
        if align_layer >= n_layers:
            alignment = None
            for layer_attn in cross_attentions:
                layer_mean = layer_attn[0].mean(dim=0)
                alignment = layer_mean if alignment is None else alignment + layer_mean
            alignment = alignment / n_layers
        else:
            alignment = cross_attentions[align_layer][0].mean(dim=0)

        # weighted mean
        n_frames = alignment.shape[-1]
        frame_indices = (alignment * torch.arange(n_frames, device=self.device).unsqueeze(0).float()).sum(dim=-1) / (alignment.sum(dim=-1) + 1e-8)
        token_times = frame_indices * 0.02 + offset

        special_ids = set(self.processor.tokenizer.all_special_ids)
        text_indices = [
            i for i in range(len(pred_ids))
            if pred_ids[i].item() not in special_ids
        ]
        if not text_indices:
            return []

        text_tokens = [
            self.processor.tokenizer.decode([pred_ids[i].item()], skip_special_tokens=True)
            for i in text_indices
        ]
        token_times_list = [
            max(offset, min(offset + chunk_duration, token_times[i].item()))
            for i in text_indices
        ]

        # 依時間間隔與長度將 token 組織為 Segment
        gap_threshold = float(os.environ.get("BREEZE_ALIGN_GAP_MS", "500")) / 1000.0
        aligned_segments: List[Segment] = []
        cur_tokens: List[str] = []
        cur_start = token_times_list[0]
        cur_end = token_times_list[0]

        for tok, t_time in zip(text_tokens, token_times_list):
            if not tok:
                continue
            if cur_tokens and (t_time - cur_end > gap_threshold or len("".join(cur_tokens)) >= 15):
                txt = "".join(cur_tokens).strip()
                if txt:
                    aligned_segments.append(
                        Segment(
                            start=round(cur_start, 3),
                            end=round(max(cur_end, cur_start + 0.1), 3),
                            text=txt,
                        )
                    )
                cur_tokens = [tok]
                cur_start = t_time
                cur_end = t_time
            else:
                cur_tokens.append(tok)
                cur_end = max(cur_end, t_time)

        if cur_tokens:
            txt = "".join(cur_tokens).strip()
            if txt:
                aligned_segments.append(
                    Segment(
                        start=round(cur_start, 3),
                        end=round(max(cur_end, offset + chunk_duration), 3),
                        text=txt,
                    )
                )

        return aligned_segments



# =============================================================================
# 裝置偵測
# =============================================================================
def get_best_device() -> torch.device:
    """檢查系統支援的最佳運算裝置。優先順序：CUDA > MPS > CPU。"""
    try:
        with error_context("device_detection"):
            if torch.cuda.is_available():
                device = torch.device("cuda")
                device_name = torch.cuda.get_device_name()
                logger.info(f"🚀 使用 CUDA GPU: {device_name}")
                return device
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("🚀 使用 Apple Silicon MPS GPU")
                return device
            device = torch.device("cpu")
            logger.info("💻 使用 CPU (未偵測到 GPU 支援)")
            return device
    except Exception as e:
        raise SystemError(
            message=f"裝置檢查失敗: {str(e)}",
            error_code=ErrorCode.DEVICE_ERROR,
            original_exception=e,
        )


def initialize_model() -> None:
    """初始化 Breeze-ASR-26 引擎（線程安全）。

    環境變數：
    - VAD_ENABLED=1/0（預設 1）：是否載入 Silero VAD
    - VAD_THRESHOLD=0.5：語音機率閾值
    - VAD_MIN_SPEECH_MS=250：最短語音長度
    - VAD_MIN_SILENCE_MS=500：最短靜音長度（超過則切句）
    - VAD_SPEECH_PAD_MS=100：語音邊界 padding
    """
    global asr_engine
    with asr_lock:
        if asr_engine is not None:
            log_info("模型已經初始化，跳過重複初始化。")
            return
        try:
            with error_context("breeze_asr_initialization"):
                device = get_best_device()
                dtype = torch.float16 if device.type == "cuda" else torch.float32
                use_vad = os.environ.get("VAD_ENABLED", "1") not in (
                    "0", "false", "False"
                )
                asr_engine = BreezeASREngine(
                    DEFAULT_MODEL_PATH, device, dtype, use_vad=use_vad
                )
        except Exception as e:
            log_error(f"模型初始化失敗: {e}")
            asr_engine = None
            raise create_model_load_error(MODEL_ID, str(e))


def release_model() -> None:
    """釋放 Breeze-ASR-26 引擎並清除 GPU 記憶體（線程安全）。"""
    global asr_engine
    with asr_lock:
        if asr_engine is None:
            log_info("模型尚未載入，無需釋放。")
            return
        try:
            with error_context("breeze_asr_release"):
                logger.info("正在釋放 Breeze-ASR-26 模型...")
                asr_engine = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                logger.info("Breeze-ASR-26 模型已成功釋放，GPU 記憶體已清除。")
        except Exception as e:
            logger.error(f"模型釋放失敗: {str(e)}")
            raise SystemError(
                message=f"模型釋放失敗: {str(e)}",
                error_code=ErrorCode.MEMORY_ERROR,
                original_exception=e,
            )


# =============================================================================
# 對外 API（與原 xfast.py 保持一致）
# =============================================================================
class OpenAI_Transcribe(BaseModel):
    text: str = ""
    language: str = ""
    duration: float = 0.0
    segments: List[Segment] = []


class Transcribe(BaseModel):
    """語音轉文字結果（支援自動型別檢查）"""

    transcript: str = ""
    file_name: str = ""
    summary: str = ""
    srt: str = ""
    vtt: str = ""
    translation: str = ""
    guid: str = str(uuid.uuid4())
    duration: float = 0.0
    whisper_api_cost: float = 0.0


# ---- 時間格式化 ----
def format_time_vtt(seconds_val: float) -> str:
    h = int(seconds_val // 3600)
    m = int((seconds_val % 3600) // 60)
    s = seconds_val % 60
    ms = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{int(s):02d}.{ms:03d}"


def format_time_srt(seconds_val: float) -> str:
    h = int(seconds_val // 3600)
    m = int((seconds_val % 3600) // 60)
    s = seconds_val % 60
    ms = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{int(s):02d},{ms:03d}"


# ---- 輸出檔案 ----
@handle_file_operations
def write_transcript(segments_list, info, filename):
    try:
        with error_context("write_transcript", filename=filename):
            with open(filename, "w", encoding="utf-8") as f:
                cc = None
                if info.language == "zh":
                    cc = opencc_manager.get_cc("s2twp.json")
                for segment in segments_list:
                    text = segment.text
                    if cc is not None:
                        text = cc.convert(text)
                    f.write(
                        "[%s --> %s]\n %s\n\n"
                        % (
                            format_time_vtt(segment.start),
                            format_time_vtt(segment.end),
                            text,
                        )
                    )
            with open(filename, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        raise FileError(
            message=f"寫入逐字稿失敗: {str(e)}",
            error_code=ErrorCode.FILE_WRITE_ERROR,
            details={"filename": filename},
            original_exception=e,
        )


@handle_file_operations
def write_translation(segments_list, info, filename):
    try:
        with error_context("write_translation", filename=filename):
            with open(filename, "w", encoding="utf-8") as f:
                cc = None
                if info.language == "zh":
                    cc = opencc_manager.get_cc("s2twp.json")
                for segment in segments_list:
                    text = segment.text
                    if not text.strip():
                        continue
                    translated = safe_execute(
                        translate_text_ollama,
                        text,
                        fallback_value=text,
                    )
                    if cc is not None and translated:
                        translated = cc.convert(translated)
                    f.write(
                        "[%s --> %s]\n %s\n\n"
                        % (
                            format_time_vtt(segment.start),
                            format_time_vtt(segment.end),
                            translated,
                        )
                    )
            with open(filename, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        logger.warning(f"翻譯逐字稿寫入失敗，略過翻譯檔案產製: {filename} - {str(e)}")
        return ""


def get_openai_json_text(segments, info):
    full_text = " ".join(s.text for s in segments)
    return OpenAI_Transcribe(
        text=full_text,
        language=info.language,
        duration=info.duration,
        segments=segments,
    ).model_dump_json()


@handle_file_operations
def write_json(segments, info, filename):
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


def write_srt(segments_list, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments_list):
            f.write(f"{i + 1}\n")
            f.write(
                "%s --> %s\n"
                % (format_time_srt(seg.start), format_time_srt(seg.end))
            )
            f.write(f"{seg.text}\n\n")
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


def write_vtt(segments_list, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments_list:
            f.write(
                "%s --> %s\n"
                % (format_time_vtt(seg.start), format_time_vtt(seg.end))
            )
            f.write(f"{seg.text}\n\n")
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


# ---- 摘要 ----
@handle_errors("總結會議摘要時發生錯誤")
def summary_text(segments):
    if not segments:
        return "沒有可用的語音片段來生成摘要"
    summaries = []
    for segment in segments:
        summaries.append(
            safe_execute(
                summary_text_ollama,
                segment.text,
                fallback_value=f"摘要失敗: {segment.text[:50]}...",
            )
        )
    return safe_execute(
        summary_text_ollama,
        " ".join(summaries),
        fallback_value="摘要功能暫時無法使用，請確認 Ollama 服務正在運行。",
    )


# ---- 主要轉錄函式 ----
def transcribe_file(audio_file: str) -> Tuple[List[Segment], TranscriptionInfo]:
    """使用 Breeze-ASR-26 轉錄音訊檔案。"""
    if not os.path.exists(audio_file):
        raise create_file_not_found_error(audio_file)
    file_size = os.path.getsize(audio_file)
    if file_size == 0:
        raise FileError(
            message=f"音訊檔案為空: {audio_file}",
            error_code=ErrorCode.FILE_SIZE_ERROR,
            details={"file_path": audio_file, "file_size": file_size},
        )
    try:
        with error_context(
            "transcribe_file", audio_file=audio_file, file_size=file_size
        ):
            initialize_model()
            if asr_engine is None:
                raise ModelError(
                    message="模型未正確初始化",
                    error_code=ErrorCode.MODEL_NOT_INITIALIZED,
                )
            segments, info = asr_engine.transcribe(audio_file)
            logger.info(f"轉錄完成，共 {len(segments)} 個片段")
            return segments, info
    except (FileError, TranscriptionError, ModelError):
        raise
    except Exception as e:
        raise create_transcription_error(audio_file, str(e))


def transcribe_openai_format(audio_file: str) -> str:
    """轉錄音訊並回傳 OpenAI Whisper API 格式的 JSON 字串。"""
    try:
        with error_context("transcribe_openai_format", audio_file=audio_file):
            segments, info = transcribe_file(audio_file)
            logger.info(
                f"偵測到語言 '{info.language}' 機率: {info.language_probability:.4f}"
            )
            return get_openai_json_text(segments, info)
    except (FileError, TranscriptionError, ModelError):
        raise
    except Exception as e:
        raise create_transcription_error(audio_file, str(e))


def transcribe_full(audio_file: str) -> str:
    """完整轉錄流程：產生 JSON / 逐字稿 / SRT / VTT / 翻譯 / 摘要。"""
    segments, info = transcribe_file(audio_file)
    base_filename = audio_file.rsplit(".", 1)[0]
    transcript_filename = f"{base_filename}_transcript.txt"
    srt_filename = f"{base_filename}.srt"
    vtt_filename = f"{base_filename}.vtt"
    json_filename = f"{base_filename}.json"
    translation_filename = f"{base_filename}_translation.txt"

    write_json(segments, info, json_filename)
    transcript = write_transcript(segments, info, transcript_filename)
    srt_text = write_srt(segments, srt_filename)
    vtt_text = write_vtt(segments, vtt_filename)
    translation = write_translation(segments, info, translation_filename)
    summary = safe_execute(
        summary_text_ollama,
        transcript,
        fallback_value="摘要功能暫時無法使用，請確認 Ollama 服務正在運行。",
    )

    return Transcribe(
        transcript=transcript,
        file_name=audio_file,
        summary=summary,
        srt=srt_text,
        vtt=vtt_text,
        translation=translation,
        duration=info.duration,
    ).model_dump_json()


# ---- CLI ----
def main(audio_file: str) -> Optional[str]:
    try:
        with error_context("main_process", audio_file=audio_file):
            result = transcribe_full(audio_file)
            logger.info("轉錄處理完成")
            return result
    finally:
        try:
            release_model()
        except Exception as e:
            logger.error(f"模型釋放失敗: {e}")


def usage(script_name: str) -> None:
    print(f"Usage: python {script_name} <audio_file>")
    print(f"Example: python {script_name} audio.mp3")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        usage(sys.argv[0])
