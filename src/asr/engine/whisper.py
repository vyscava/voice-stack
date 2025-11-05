from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import whisper

from asr.engine.base import (
    ASRBase,
    DetectLanguageResult,
    TranscribeResult,
    transcribe_effective_options,
)
from core.logging import logger_asr as logger
from core.settings import get_settings
from utils.audio.audio_helper import resample_to_16k_mono
from utils.language.language_codes import LanguageCode

settings = get_settings()


def _pick_device() -> str:
    """
    Prefer MPS on Apple Silicon, then CUDA, else CPU.
    Normalize as 'mps' | 'cuda' | 'cpu'.
    """
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ASRWhisperTorch(ASRBase):
    """
    Whisper (PyTorch) backend for macOS MPS / CUDA / CPU.

    - Respects your ASRBase pipeline (resample -> optional VAD -> ASR).
    - Returns TranscribeResult / DetectLanguageResult compatible with your code.
    """

    def __init__(self) -> None:
        super().__init__()

        # Encourage CPU fallback for missing MPS kernels
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        self.device = _pick_device()
        # fp16 on CUDA tends to be faster; MPS & CPU prefer fp32
        self.fp16 = True if self.device == "cuda" else False

        model_id = getattr(settings, "ASR_MODEL", "base") or "base"
        logger.info(f"Loading Whisper (PyTorch): model={model_id} device={self.device} fp16={self.fp16}")

        # Try to load on the chosen device; if MPS fails with NotImplementedError, fall back to CPU
        try:
            self.model = whisper.load_model(model_id, device=self.device)
        except NotImplementedError as e:
            if self.device == "mps":
                logger.warning(
                    "Whisper load on MPS failed due to missing kernel (%s). "
                    "Falling back to CPU. You can keep MPS fallback by setting "
                    "PYTORCH_ENABLE_MPS_FALLBACK=1 (currently: %s).",
                    e,
                    os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
                )
                self.device = "cpu"
                self.fp16 = False
                self.model = whisper.load_model(model_id, device="cpu")
            else:
                raise

    def _unload_model(self) -> None:
        """
        Safely unload PyTorch Whisper model.

        Moves the model to CPU (if on GPU/MPS) to free accelerator memory,
        then deletes the model reference and runs garbage collection.

        Note: Do NOT call torch.cuda.empty_cache() as it would affect other
        services (TTS, Open WebUI) sharing the same GPU.
        """
        import gc

        if hasattr(self, "model") and self.model is not None:
            logger.info("Unloading PyTorch Whisper model from memory...")

            # Move model to CPU to release GPU/MPS memory
            try:
                if self.device in ("cuda", "mps"):
                    self.model = self.model.to("cpu")
                    logger.info(f"Model moved from {self.device} to CPU")
            except Exception as e:
                logger.warning(f"Error moving model to CPU: {e}")

            # Delete model reference
            del self.model
            self.model = None  # type: ignore

            # Run garbage collection to free Python objects and torch memory
            gc.collect()

            # Synchronize CUDA operations if using CUDA
            if self.device == "cuda":
                torch.cuda.synchronize()

            logger.info("PyTorch Whisper model unloaded successfully")

    def _transcribe_core(
        self,
        *,
        audio_f32: npt.NDArray[np.float32],
        sr: int,
        raw_bytes: bytes | None,
        request_language: str | None,
        task: str | None,
        beam_size: int | None,
        temperature: float | None,
        best_of: int | None,
        word_timestamps: bool | None,
        vad: bool | None,
    ) -> TranscribeResult:
        t0 = time.time()
        props = transcribe_effective_options(
            request_language=request_language,
            task=task,
            beam_size=beam_size,
            temperature=temperature,
            best_of=best_of,
            word_timestamps=word_timestamps,
            vad=vad,
        )

        # Cache lookup (if available)
        if self.cache_enabled and raw_bytes is not None:
            hit, ok = self.helper_check_cache(opts=props, raw_bytes=raw_bytes)
            if ok and isinstance(hit, TranscribeResult):
                return hit

        # Standardize audio to 16 kHz mono float32 in [-1, 1]
        audio_f32, _ = resample_to_16k_mono(audio_f32=audio_f32, sr=sr)
        duration_input = float(audio_f32.size) / 16000.0

        # Optional VAD (your helper uses Silero and returns float32 @ 16k)
        vad_used = False
        if props.use_vad:
            audio_f32, vad_used = self.helper_apply_vad(audio_f32=audio_f32)

        # Ensure 1-D contiguous float32 for Whisper
        audio_f32 = np.ascontiguousarray(audio_f32.astype(np.float32).reshape(-1))
        duration_after_vad = float(audio_f32.size) / 16000.0

        # Build decode options for Whisper
        # See https://github.com/openai/whisper/blob/main/whisper/transcribe.py
        decode_opts: dict[str, Any] = {
            "task": props.task or "transcribe",
            "language": props.language,  # None lets Whisper auto-detect
            "beam_size": props.beam_size or 5,
            "temperature": props.temperature or 0.0,
            "best_of": props.best_of or 1,
            "fp16": self.fp16,  # important: False for MPS/CPU
        }
        # Official Whisper doesn't expose word-level timestamps in stable API;
        # ignore props.word_timestamps to keep parity without breaking.

        # ASR
        t_asr = time.time()
        result_whisper = self.model.transcribe(audio_f32, **decode_opts)
        asr_ms = int((time.time() - t_asr) * 1000)

        # Parse segments
        segments: list[dict[str, Any]] = []
        parts: list[str] = []
        for s in result_whisper.get("segments", []) or []:
            # s contains: id, start, end, text, tokens, avg_logprob, compression_ratio, no_speech_prob
            text_piece: str = (s.get("text") or "").strip()
            if not text_piece:
                continue
            parts.append(text_piece)
            segments.append(
                {
                    "start": float(s.get("start", 0.0)),
                    "end": float(s.get("end", 0.0)),
                    "text": text_piece,
                }
            )

        full_text = " ".join(parts).strip()

        # Language info: Whisper returns result_whisper["language"] if language was auto-detected
        lang_code = result_whisper.get("language", props.language or "UNKNOWN") or "UNKNOWN"
        lang = LanguageCode.from_string(lang_code)

        result = TranscribeResult(
            text=full_text,
            segments=segments,
            language_code=lang.canonical,
            language_name=lang.display_name,
            confidence=1.0,  # Whisper transcribe() doesn't return a global prob; keep 1.0 or compute separately
            duration_input_s=duration_input,
            duration_after_vad_s=duration_after_vad,
            processing_ms=int((time.time() - t0) * 1000),
            asr_ms=asr_ms,
            vad_used=vad_used,
            engine=getattr(settings, "ASR_ENGINE", "whisper"),
            model=getattr(settings, "ASR_MODEL", "base"),
        )

        # Save in cache
        if self.cache_enabled and raw_bytes is not None:
            self.helper_save_on_cache(raw_bytes=raw_bytes, result=result, opts=props)

        return result

    def _detect_language_core(
        self,
        *,
        audio_f32: npt.NDArray[np.float32],
        sr: int,
        raw_bytes: bytes | None,
        request_language: str | None,
        detect_lang_length: float | None = None,
        detect_lang_offset: float | None = None,
    ) -> DetectLanguageResult:
        t0 = time.time()
        # Re-use your transcribe options builder for consistency
        props = transcribe_effective_options(
            request_language=request_language,
            task="transcribe",
            beam_size=1,
            temperature=0.0,
            best_of=1,
            word_timestamps=False,
            vad=True,
        )

        # Normalize "ignore zero window"
        if detect_lang_offset == 0 and detect_lang_length == 0:
            detect_lang_offset = None
            detect_lang_length = None

        # Cache lookup
        if self.cache_enabled and raw_bytes is not None:
            hit, ok = self.helper_check_cache(opts=props, raw_bytes=raw_bytes)
            if ok and isinstance(hit, DetectLanguageResult):
                return hit

        # Standardize audio
        audio_f32, sr = resample_to_16k_mono(audio_f32=audio_f32, sr=sr)
        sr_hz = 16000

        # Optional slice for speed
        n = audio_f32.shape[0]
        if detect_lang_offset is not None or detect_lang_length is not None:
            start = int(max(0.0, (detect_lang_offset or 0.0)) * sr_hz)
            end = int(min(n, start + (int((detect_lang_length or 0.0) * sr_hz) if detect_lang_length else n)))
            if end > start:
                audio_f32 = audio_f32[start:end]
                n = audio_f32.shape[0]

        duration_input = float(n) / sr_hz

        # Optional VAD
        vad_used = False
        pre_vad = audio_f32
        if props.use_vad:
            audio_f32, vad_used = self.helper_apply_vad(audio_f32=audio_f32)

        # Ensure minimum duration for stability; else revert pre-VAD
        if audio_f32.shape[0] < int(0.5 * sr_hz):
            audio_f32 = pre_vad
            vad_used = False

        # 1-D contiguous float32
        audio_f32 = np.ascontiguousarray(audio_f32.astype(np.float32).reshape(-1))
        duration_after_vad = float(audio_f32.shape[0]) / sr_hz

        # Whisper language detection needs a log-mel spectrogram
        t_asr = time.time()
        mel = whisper.log_mel_spectrogram(audio_f32, padding=0)
        mel = mel.to(self.device) if hasattr(mel, "to") else mel  # make sure it’s on the model device

        # detect_language returns (lang, prob)
        lang_code, lang_prob = self.model.detect_language(mel)
        asr_ms = int((time.time() - t_asr) * 1000)

        lang = LanguageCode.from_string(lang_code)
        result = DetectLanguageResult(
            detected_language=lang.display_name,
            language_code=lang.canonical,
            language_name=lang.display_name,
            confidence=float(lang_prob or 0.0),
            duration_input_s=duration_input,
            duration_after_vad_s=duration_after_vad,
            processing_ms=int((time.time() - t0) * 1000),
            asr_ms=asr_ms,
            vad_used=vad_used,
            engine=getattr(settings, "ASR_ENGINE", "whisper-torch"),
            model=getattr(settings, "ASR_MODEL", "base"),
        )

        # Cache save
        if self.cache_enabled and raw_bytes is not None:
            self.helper_save_on_cache(raw_bytes=raw_bytes, result=result, opts=props)

        return result
