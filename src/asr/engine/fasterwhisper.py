from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo

from asr.engine.base import ASRBase, DetectLanguageResult, TranscribeResult, transcribe_effective_options
from core.logging import logger_asr as logger
from core.settings import get_settings
from utils.audio.audio_helper import (
    resample_to_16k_mono,
)
from utils.language.language_codes import LanguageCode

settings = get_settings()


class ASRFasterWhisper(ASRBase):
    def __init__(self) -> None:
        super().__init__()

        logger.info("Loading Faster-Whisper")
        logger.info(f"Model={settings.ASR_MODEL} Device={settings.ASR_DEVICE} Compute_Type={settings.ASR_COMPUTE_TYPE}")
        logger.info(f"CPU_Threads={settings.ASR_CPU_THREADS} Num_Workers={settings.ASR_NUM_OF_WORKERS}")

        self.model = WhisperModel(
            model_size_or_path=settings.ASR_MODEL,
            device=settings.ASR_DEVICE,
            compute_type=settings.ASR_COMPUTE_TYPE,
            download_root=settings.ASR_MODEL_LOCATION,
            cpu_threads=settings.ASR_CPU_THREADS,
            num_workers=settings.ASR_NUM_OF_WORKERS,
        )

    def _unload_model(self) -> None:
        """
        Safely unload faster-whisper model.

        faster-whisper uses CTranslate2 backend which manages its own memory.
        We delete the model reference and run garbage collection to free memory.

        Note: Do NOT call torch.cuda.empty_cache() as it would affect other
        services (TTS, Open WebUI) sharing the same GPU.
        """
        import gc

        if hasattr(self, "model") and self.model is not None:
            logger.info("Unloading faster-whisper model from memory...")

            # Delete model reference
            del self.model
            self.model = None  # type: ignore

            # Run garbage collection to free Python objects and CTranslate2 memory
            gc.collect()

            logger.info("Faster-whisper model successfully unloaded")

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

        # Before trying to transcribe lets see if exists in the cache
        if self.cache_enabled and raw_bytes is not None:
            # Search audio in the cache
            hit, cache_hit_flag = self.helper_check_cache(opts=props, raw_bytes=raw_bytes)

            # Lets check if there was a hit on the cache and if it is a TranscribeResult
            if cache_hit_flag and isinstance(hit, TranscribeResult):
                return hit

        # Lets make sure audio is float32 and is 16 kHz mono
        audio_f32, _ = resample_to_16k_mono(audio_f32=audio_f32, sr=sr)
        duration_input = float(audio_f32.size) / 16000.0

        # Apply VAD if requested
        vad_used = False
        if props.use_vad:
            audio_f32, vad_used = self.helper_apply_vad(audio_f32=audio_f32)

        # Now lets make sure the audio is in a single dimmension and is stored
        # contiguously in memory (no gaps or “strides”).
        audio_f32 = np.ascontiguousarray(audio_f32.astype(np.float32).reshape(-1))
        duration_after_vad = float(audio_f32.size) / 16000.0

        # Start Running time for Faster-Whisper
        t_asr = time.time()

        # Faster-Whisper accepts raw float32 NumPy 1-D arrays directly
        # Telling MyPy expected types
        segments_iter: Iterable[Segment]
        info: TranscriptionInfo

        segments_iter, info = self.model.transcribe(
            audio=audio_f32,
            language=props.language,
            task=props.task or "",
            beam_size=props.beam_size or 5,
            temperature=props.temperature or 0,
            best_of=props.best_of or 1,
            word_timestamps=props.word_timestamps if props.word_timestamps is not None else False,
        )

        # Stop Running time for Faster-Whisper
        asr_ms = int((time.time() - t_asr) * 1000)

        # Initializing transcription returns
        segments: list[dict[str, Any]] = []
        parts: list[str] = []

        # Putting together what the model returned
        for seg in segments_iter:
            # Each seg has .start, .end, .text
            text_piece = (seg.text or "").strip()
            parts.append(text_piece)
            segments.append({"start": float(seg.start), "end": float(seg.end), "text": text_piece})

        # Joining all text pieces to have a single string
        full_text = " ".join(parts).strip()

        # Language info (if available)
        lang = LanguageCode.from_string(getattr(info, "language", None))
        lang_prob = float(getattr(info, "language_probability", 0.0) or 0.0)

        result = TranscribeResult(
            text=full_text,
            segments=segments,
            language_code=lang.canonical,
            language_name=lang.display_name,
            confidence=lang_prob,
            duration_input_s=duration_input,
            duration_after_vad_s=duration_after_vad,
            processing_ms=int((time.time() - t0) * 1000),
            asr_ms=asr_ms,
            vad_used=vad_used,
            engine=getattr(settings, "ASR_ENGINE", "UNKNOWN"),
            model=getattr(settings, "ASR_MODEL", "base"),
        )

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
        props = transcribe_effective_options(
            request_language=request_language,
            task="transcribe",
            beam_size=1,  # Focusing on Speed
            temperature=0.0,  # Deterministic
            best_of=1,
            word_timestamps=False,
            vad=True,  # Better clean no voices sounds
        )

        # If both comes as zero it was probably an input error, lets ignore
        if detect_lang_offset == 0 and detect_lang_length == 0:
            detect_lang_offset = None
            detect_lang_length = None

        # Before trying to transcribe lets see if exists in the cache
        if self.cache_enabled and raw_bytes is not None:
            # Search audio in the cache
            hit, cache_hit_flag = self.helper_check_cache(opts=props, raw_bytes=raw_bytes)

            # Lets check if there was a hit on the cache and if it is a DetectLanguageResult
            if cache_hit_flag and isinstance(hit, DetectLanguageResult):
                return hit

        # Lets make sure audio is float32 and is 16 kHz mono
        audio_f32, sr = resample_to_16k_mono(audio_f32=audio_f32, sr=sr)
        sr_hz = 16000

        # Lets get just a portion of the audio to go faster
        n = audio_f32.shape[0]

        # Selecting part of the audio based on inputs
        if detect_lang_offset is not None or detect_lang_length is not None:
            start = int(max(0.0, (detect_lang_offset or 0.0)) * sr_hz)
            end = int(min(n, start + (int((detect_lang_length or 0.0) * sr_hz) if detect_lang_length else n)))
            if end > start:
                audio_f32 = audio_f32[start:end]
                n = audio_f32.shape[0]

        # Total audio duration before VAD and after slicing
        duration_input = float(n) / sr_hz

        # Apply VAD if requested
        vad_used = False
        pre_vad_audio = audio_f32
        if props.use_vad:
            audio_f32, vad_used = self.helper_apply_vad(audio_f32=audio_f32)

        # Enforce a tiny minimum duration to keep the detector robust (skip VAD if too short)
        if audio_f32.shape[0] < int(0.5 * sr_hz):  # < 0.5s
            audio_f32 = pre_vad_audio  # revert to pre-VAD if that helps
            vad_used = False

        # Now lets make sure the audio is in a single dimmension and is stored
        # contiguously in memory (no gaps or “strides”).
        audio_f32 = np.ascontiguousarray(audio_f32, dtype=np.float32).ravel()
        # Explicit casting to avoid MyPy warnings
        audio_f32 = cast(npt.NDArray[np.float32], audio_f32)
        duration_after_vad = float(audio_f32.shape[0]) / sr_hz

        # Start Running time for Faster-Whisper
        t_asr = time.time()
        lang_code: str
        lang_prob: float

        # Faster-Whisper accepts raw float32 NumPy 1-D arrays directly
        lang_code, lang_prob, _ = self.model.detect_language(audio=audio_f32)

        # Stop Running time for Faster-Whisper
        asr_ms = int((time.time() - t_asr) * 1000)

        # Language info (if available)
        lang = LanguageCode.from_string(lang_code)
        lang_prob = float(lang_prob or 0.0)

        result = DetectLanguageResult(
            detected_language=lang.display_name,
            language_code=lang.canonical,
            language_name=lang.display_name,
            confidence=lang_prob,
            duration_input_s=duration_input,
            duration_after_vad_s=duration_after_vad,
            processing_ms=int((time.time() - t0) * 1000),
            asr_ms=asr_ms,
            vad_used=vad_used,
            engine=getattr(settings, "ASR_ENGINE", "UNKNOWN"),
            model=getattr(settings, "ASR_MODEL", "base"),
        )

        if self.cache_enabled and raw_bytes is not None:
            self.helper_save_on_cache(raw_bytes=raw_bytes, result=result, opts=props)

        return result
