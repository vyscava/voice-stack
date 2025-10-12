"""
ASR engine: decode → resample → (optional VAD) → Faster-Whisper → transcript.

- Pure-Python class (not tied to FastAPI) so you can re-use it anywhere.
- Uses your helpers from `utils.audio_helper` (soundfile/ffmpeg decode, resample, VAD).
- Optional in-memory FIFO cache (enabled & sized via settings).
- Returns a rich dict (text, segments, language meta, timings, durations).
  The OpenAI router can slim this down to only {"text": "..."} for compatibility.

Notes on caching:
- Keyed by a SHA-256 of the **raw input bytes** + the effective options (language, task, etc.).
- Size is capped by `settings.ASR_MAX_CACHE_ITEMS` (FIFO eviction).

Timings & durations:
- `duration_input_s`: duration computed from the decoded/resampled audio (pre-VAD).
- `duration_after_vad_s`: duration after VAD (if enabled/available), else equals input.
- `processing_ms` & `asr_ms`: total and model-only timing.

Dependencies (imported):
- faster-whisper
- numpy
- your utils.audio_helper.{decode_with_soundfile, decode_with_ffmpeg, resample_to_16k_mono,
  load_silero_model, apply_vad_silero}
- your core.settings / core.logging
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from faster_whisper import WhisperModel

from core.logging import logger_asr as logger
from core.settings import get_settings
from utils.audio_helper import (
    apply_vad_silero,
    decode_with_ffmpeg,
    decode_with_soundfile,
    load_silero_model,
    resample_to_16k_mono,
)

settings = get_settings()


@dataclass
class _Options:
    """
    Effective per-request options with sane defaults.
    """

    language: str | None
    task: str
    beam_size: int
    temperature: float
    best_of: int
    word_timestamps: bool
    use_vad: bool


def _effective_options(
    *,
    request_language: str | None,
    task: str | None,
    beam_size: int | None,
    temperature: float | None,
    best_of: int | None,
    word_timestamps: bool | None,
    vad: bool | None,
) -> _Options:
    """
    Merge request-time overrides with settings defaults.
    Only minimal validation here; rely on Faster-Whisper for deeper checks.
    """
    return _Options(
        language=request_language or getattr(settings, "ASR_LANGUAGE", None),
        task=task or "transcribe",
        beam_size=int(beam_size or settings.ASR_TRANSCRIBE_BEAM_SIZE or 5),
        temperature=float(temperature or settings.ASR_TRANSCRIBE_TEMPERATURE or 0.0),
        best_of=int(best_of or settings.ASR_TRANSCRIBE_BEST_OF or 1),
        word_timestamps=bool(word_timestamps if word_timestamps is not None else False),
        use_vad=bool(vad if vad is not None else settings.ASR_VAD_ENABLED or False),
    )


def _hash_cache_key(*, audio_bytes: bytes, opts: _Options) -> str:
    """
    Stable cache key: audio bytes + the slimmed options footprint.
    """
    h = hashlib.sha256(audio_bytes)
    h.update(
        json.dumps(
            {
                "language": opts.language,
                "task": opts.task,
                "beam_size": opts.beam_size,
                "temperature": opts.temperature,
                "best_of": opts.best_of,
                "word_timestamps": opts.word_timestamps,
                "use_vad": opts.use_vad,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return h.hexdigest()


class AudioEngine:
    """
    Single shared Faster-Whisper model with helper methods:

      - `transcribe_file(file_bytes, filename)`
      - `transcribe_bytes(raw_bytes)`
      - `transcribe_array(audio_f32, sr)`
      - `transcribe_stream(chunks_iterable_of_bytes)`  # buffered for now

    All of them funnel into `_transcribe_core()`.
    """

    _WHISPER_VARIANTS = [
        "tiny",
        "base",
        "small",
        "medium",
        "large-v1",
        "large-v2",
        "large-v3",
    ]

    def __init__(self) -> None:

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

        # Optional Silero VAD (CPU)
        if settings.ASR_VAD_ENABLED:
            self.silero_model, self.silero_ts_fn = load_silero_model(logger)

        if self.silero_model is not None:
            logger.info("Silero VAD is available for transcribing")
        else:
            logger.info("Silero VAD is NOT available for transcribing")

        # Tiny in-memory cache
        self.cache_enabled = bool(settings.ASR_CACHE_ENABLED or False)
        self.cache_max = int(settings.ASR_CACHE_MAX_ITEMS or 64)
        self._cache: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []  # FIFO keys

    def _transcribe_core(
        self,
        *,
        audio_f32: np.ndarray[Any, Any],
        sr: int,
        raw_bytes: bytes | None,  # only for caching; can be None for array inputs
        request_language: str | None,
        task: str | None,
        beam_size: int | None,
        temperature: float | None,
        best_of: int | None,
        word_timestamps: bool | None,
        vad: bool | None,
    ) -> dict[str, Any]:
        """
        Core pipeline:
          1) Resample to 16 kHz mono float32
          2) (Optional) VAD with Silero (on CPU)
          3) Faster-Whisper inference on a BytesIO (WAV not required; FW can read raw float32)
          4) Build segments & full text
          5) Return rich dict (router can slim to {"text": ...})

        Returns (example):
        {
          "text": "hello world",
          "segments": [{"start": 0.0, "end": 1.2, "text": "hello"}, ...],
          "language": "en",
          "language_probability": 0.98,
          "duration_input_s": 12.34,
          "duration_after_vad_s": 10.87,
          "asr_ms": 1234,
          "processing_ms": 1456,
          "vad_used": true,
          "model": "...",
        }
        """
        t0 = time.time()
        opts = _effective_options(
            request_language=request_language,
            task=task,
            beam_size=beam_size,
            temperature=temperature,
            best_of=best_of,
            word_timestamps=word_timestamps,
            vad=vad,
        )

        # 1) Standardize to 16 kHz mono float32 for robust behavior
        audio_f32, _ = resample_to_16k_mono(audio_f32=audio_f32, sr=sr)
        duration_input = float(audio_f32.size) / 16000.0

        # 2) VAD (optional, Silero on CPU) - operates on PCM16
        # Note: If silero_model or silereo_ts_fn it means the resources are not available to use
        if opts.use_vad and self.silero_model is not None and self.silero_ts_fn is not None:
            pcm16 = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
            pcm16_vad = apply_vad_silero(
                pcm16=pcm16,
                sr=16000,
                silero_model=self.silero_model,
                get_speech_timestamps_fn=self.silero_ts_fn,
                log=logger,
            )

            audio_f32 = (pcm16_vad.astype(np.float32) / 32768.0) if pcm16_vad.size else audio_f32
            vad_used = True
        else:
            vad_used = False

        duration_after_vad = float(audio_f32.size) / 16000.0

        # -------------- cache (optional) --------------
        cache_key = None
        if self.cache_enabled and raw_bytes is not None:
            try:
                cache_key = _hash_cache_key(audio_bytes=raw_bytes, opts=opts)

                if cache_key in self._cache:
                    hit = self._cache[cache_key]

                    # move key to tail (fresh) in FIFO
                    try:
                        self._order.remove(cache_key)
                    except ValueError:
                        pass

                    self._order.append(cache_key)
                    return hit

            except Exception:
                # cache is best-effort; never fail the request
                # even if it fail to retrieve or store we will continue
                cache_key = None

        # 3) Run Faster-Whisper
        t_asr = time.time()
        # Faster-Whisper accepts raw float32 NumPy 1-D arrays directly
        segments_iter, info = self.model.transcribe(
            audio=audio_f32,
            language=opts.language,
            task=opts.task,
            beam_size=opts.beam_size,
            temperature=opts.temperature,
            best_of=opts.best_of,
            word_timestamps=opts.word_timestamps,
        )
        asr_ms = int((time.time() - t_asr) * 1000)

        # 4) Materialize segments & full text
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
        lang = getattr(info, "language", None)
        lang_prob = float(getattr(info, "language_probability", 0.0) or 0.0)

        result: dict[str, Any] = {
            "text": full_text,
            "segments": segments,
            "language": lang,
            "language_probability": lang_prob,
            "duration_input_s": duration_input,
            "duration_after_vad_s": duration_after_vad,
            "asr_ms": asr_ms,
            "processing_ms": int((time.time() - t0) * 1000),
            "vad_used": vad_used,
            "model": getattr(settings, "ASR_MODEL", "base"),
        }

        # 5) Store in FIFO cache
        if self.cache_enabled and cache_key:
            try:
                self._cache[cache_key] = result
                self._order.append(cache_key)

                while len(self._order) > self.cache_max:
                    old = self._order.pop(0)
                    self._cache.pop(old, None)

            except Exception:
                pass

        return result

    def transcribe_file(
        self,
        *,
        file_bytes: bytes,
        filename: str | None = None,
        request_language: str | None = None,
        task: str | None = None,
        beam_size: int | None = None,
        temperature: float | None = None,
        best_of: int | None = None,
        word_timestamps: bool | None = None,
        vad: bool | None = None,
    ) -> dict[str, Any]:
        """
        Decode from `file_bytes` (using soundfile fast-path, then ffmpeg fallback).
        """
        # Try libsndfile first
        try:
            audio_f32, sr = decode_with_soundfile(raw_bytes=file_bytes)
            source = "libsndfile"
        except Exception:
            ext = (filename or "").split(".")[-1].lower() or None
            audio_f32, sr = decode_with_ffmpeg(raw_bytes=file_bytes, fmt_hint=ext)
            source = "ffmpeg"

        logger.info("Decoded via %s: sr=%d samples=%d", source, sr, int(audio_f32.shape[0]))
        return self._transcribe_core(
            audio_f32=audio_f32,
            sr=sr,
            raw_bytes=file_bytes,  # for cache key
            request_language=request_language,
            task=task,
            beam_size=beam_size,
            temperature=temperature,
            best_of=best_of,
            word_timestamps=word_timestamps,
            vad=vad,
        )

    def transcribe_bytes(
        self,
        *,
        raw_bytes: bytes,
        request_language: str | None = None,
        task: str | None = None,
        beam_size: int | None = None,
        temperature: float | None = None,
        best_of: int | None = None,
        word_timestamps: bool | None = None,
        vad: bool | None = None,
    ) -> dict[str, Any]:
        """
        Convenience alias for file-style input (same as `transcribe_file`).
        """
        return self.transcribe_file(
            file_bytes=raw_bytes,
            filename=None,
            request_language=request_language,
            task=task,
            beam_size=beam_size,
            temperature=temperature,
            best_of=best_of,
            word_timestamps=word_timestamps,
            vad=vad,
        )

    def transcribe_array(
        self,
        *,
        audio_f32: np.ndarray[Any, Any],
        sr: int,
        request_language: str | None = None,
        task: str | None = None,
        beam_size: int | None = None,
        temperature: float | None = None,
        best_of: int | None = None,
        word_timestamps: bool | None = None,
        vad: bool | None = None,
        cache_seed_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        """
        Accept an already-decoded waveform. If you pass `cache_seed_bytes`,
        it will be used in the cache key (so you still get memoization).
        """
        return self._transcribe_core(
            audio_f32=audio_f32,
            sr=sr,
            raw_bytes=cache_seed_bytes,
            request_language=request_language,
            task=task,
            beam_size=beam_size,
            temperature=temperature,
            best_of=best_of,
            word_timestamps=word_timestamps,
            vad=vad,
        )

    def transcribe_stream(
        self,
        *,
        chunks: Iterable[bytes],
        request_language: str | None = None,
        task: str | None = None,
        beam_size: int | None = None,
        temperature: float | None = None,
        best_of: int | None = None,
        word_timestamps: bool | None = None,
        vad: bool | None = None,
    ) -> dict[str, Any]:
        """
        Buffered streaming: we accumulate all chunks and run once.
        """
        raw = b"".join(chunks)
        return self.transcribe_file(
            file_bytes=raw,
            filename=None,
            request_language=request_language,
            task=task,
            beam_size=beam_size,
            temperature=temperature,
            best_of=best_of,
            word_timestamps=word_timestamps,
            vad=vad,
        )

    def _infer_active_variant(self) -> str | None:
        """
        Infer the active Whisper variant from the configured model string.
        Works for Faster-Whisper names like:
          - "tiny", "base", "small", "medium"
          - "large-v1", "large-v2", "large-v3"
          - "distil-large-v2"
        and for HuggingFace/ctranslate paths that include those substrings.
        """
        m = (settings.ASR_MODEL or "").lower()
        if not m:
            return None

        # Exact matches first
        for v in self._WHISPER_VARIANTS:
            if m == v:
                return v

        # Substring matches (handles paths like "Systran/faster-whisper-large-v3")
        for v in self._WHISPER_VARIANTS:
            if v in m:
                return v

        # Common aliases
        if m in {"large", "large-v0"}:
            return "large-v1"

        return None

    def list_models(self) -> dict[str, Any]:
        """
        Return an OpenAI-compatible models payload.
        We advertise:
          - "whisper-1" (canonical ASR model)
          - Child variants: whisper-1-{variant}, with "parent": "whisper-1"
          - The currently active variant is marked with {"active": true} for convenience.
        """
        base = {
            "id": "whisper-1",
            "object": "model",
            "owned_by": "voice-stack",
            "supported_tasks": ["transcriptions", "translations"],
        }

        active_variant = self._infer_active_variant()

        variants = []
        for v in self._WHISPER_VARIANTS:
            item = {
                "id": f"whisper-1-{v}",
                "object": "model",
                "parent": "whisper-1",
                "owned_by": "voice-stack",
                "supported_tasks": ["transcriptions", "translations"],
                "active": False,
            }
            if v == active_variant:
                item["active"] = True  # non-OpenAI field; helpful for UIs/logs
            variants.append(item)

        return {"object": "list", "data": [base, *variants]}


audio_engine = AudioEngine()
