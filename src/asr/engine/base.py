from __future__ import annotations

import hashlib
import json
import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
from fastapi import UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse

from asr.schemas.audio_engine import Output
from core.logging import logger_asr as logger
from core.settings import get_settings
from utils.audio.audio_helper import apply_vad_silero, decode_with_soundfile, load_silero_model
from utils.audio.ffmpeg_helper import decode_with_ffmpeg
from utils.language.language_codes import LanguageCode

settings = get_settings()

OUTPUT_HEADERS = {
    Output.TXT: ("text/plain; charset=utf-8", "txt"),
    Output.JSON: ("application/json", "json"),
    Output.SRT: ("application/x-subrip", "srt"),
    Output.VTT: ("text/vtt; charset=utf-8", "vtt"),
    Output.TSV: ("text/tab-separated-values; charset=utf-8", "tsv"),
    Output.JSONL: ("application/jsonl; charset=utf-8", "jsonl"),
}


@dataclass
class CacheConf:
    """
    Utilized to cache results
    Effective per-request options with sane defaults.
    """

    language: str | None = None
    task: str | None = None
    beam_size: int | None = None
    temperature: float | None = None
    best_of: int | None = None
    word_timestamps: bool | None = None
    use_vad: bool | None = None
    detect_lang_offset: float | None = None
    detect_lang_length: float | None = None


@dataclass
class _BaseReturn:
    language_code: str
    language_name: str
    confidence: float
    duration_input_s: float  # Seconds
    duration_after_vad_s: float  # Seconds
    processing_ms: int  # Milliseconds
    asr_ms: int  # Milliseconds
    vad_used: bool
    engine: str
    model: str
    vad_segments: list[dict[str, float]]  # Speech segments from VAD: [{"start": float, "end": float}, ...]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dict__)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TranscribeResult(_BaseReturn):
    """
    Transcription result with minimal, model-agnostic segments.

    segments: list[{"start": float, "end": float, "text": str}]
    """

    text: str
    segments: list[dict[str, Any]]

    def to_txt(self) -> str:
        """
        Plain text transcript: concatenates segment texts with spaces.
        Falls back to `self.text` if segments are empty.
        """
        pieces = [self._normalize_text(s.get("text", "")) for s in self.segments]
        pieces = [p for p in pieces if p]
        return " ".join(pieces) if pieces else (self.text or "")

    def to_srt(self, *, max_line_len: int | None = None) -> str:
        """
        SubRip (SRT) formatter.

        Parameters
        ----------
        max_line_len : Optional[int]
            If provided, lines are soft-wrapped at this character length.
        """
        lines: list[str] = []
        idx = 1
        for seg in self.segments:
            text = self._normalize_text(seg.get("text", ""))
            if not text:
                continue
            if max_line_len:
                text = self._soft_wrap(text, max_line_len)

            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", max(start, 0.0)))
            if end <= start:
                end = start + 0.001  # keep players happy

            lines.append(str(idx))
            lines.append(f"{self._ts_srt(start)} --> {self._ts_srt(end)}")
            lines.append(text)
            lines.append("")  # blank line
            idx += 1

        return "\n".join(lines).rstrip() + ("\n" if lines else "")

    def to_vtt(self, *, max_line_len: int | None = None) -> str:
        """
        WebVTT formatter (UTF-8). Includes the 'WEBVTT' header.
        """
        cues: list[str] = ["WEBVTT", ""]
        for seg in self.segments:
            text = self._normalize_text(seg.get("text", ""))
            if not text:
                continue
            if max_line_len:
                text = self._soft_wrap(text, max_line_len)

            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", max(start, 0.0)))
            if end <= start:
                end = start + 0.001

            cues.append(f"{self._ts_vtt(start)} --> {self._ts_vtt(end)}")
            cues.append(text)
            cues.append("")

        return "\n".join(cues).rstrip() + ("\n" if cues else "")

    def to_tsv(self) -> str:
        """
        Tab-separated values: start(s) \t end(s) \t text
        Times printed with 3 decimal places (seconds).
        """
        rows = ["start\tend\ttext"]
        for seg in self.segments:
            text = (seg.get("text", "") or "").replace("\t", " ")
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            rows.append(f"{start:.3f}\t{end:.3f}\t{text}")
        return "\n".join(rows) + ("\n" if rows else "")

    def to_segments_jsonl(self) -> str:
        """
        One JSON object per line: {"start": ..., "end": ..., "text": "..."}.
        Only uses fields present in your segments.
        """
        lines: list[str] = []
        for seg in self.segments:
            obj = {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": seg.get("text", "") or "",
            }
            lines.append(json.dumps(obj, ensure_ascii=False))
        return "\n".join(lines) + ("\n" if lines else "")

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Minimal cleanup to avoid blank/whitespace-only cues."""
        return (text or "").strip()

    @staticmethod
    def _soft_wrap(text: str, width: int) -> str:
        """
        Wrap text to approximately `width` characters per line without hard hyphenation.
        Keeps words intact; may slightly exceed width for long words.
        """
        words = text.split()
        if not words:
            return ""

        lines: list[str] = []
        cur: list[str] = []
        cur_len = 0
        for w in words:
            if cur and (cur_len + 1 + len(w)) > width:
                lines.append(" ".join(cur))
                cur = [w]
                cur_len = len(w)
            else:
                if cur:
                    cur_len += 1 + len(w)
                    cur.append(w)
                else:
                    cur = [w]
                    cur_len = len(w)
        if cur:
            lines.append(" ".join(cur))
        return "\n".join(lines)

    @staticmethod
    def _ts_vtt(seconds: float) -> str:
        """
        Format timestamp as WebVTT 'HH:MM:SS.mmm' (dot milliseconds).
        """
        if seconds < 0:
            seconds = 0.0

        ms = int(round(seconds * 1000))
        h, rem = divmod(ms, 3_600_000)
        m, rem = divmod(rem, 60_000)
        s, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    @staticmethod
    def _ts_srt(seconds: float) -> str:
        """
        Format timestamp as SRT 'HH:MM:SS,mmm' (comma milliseconds).
        """
        return TranscribeResult._ts_vtt(seconds).replace(".", ",")


@dataclass
class DetectLanguageResult(_BaseReturn):
    detected_language: str


def _normalize_enum(*, v: str | Enum | None, default: str) -> str:
    if v is None:
        return default
    if isinstance(v, Enum):
        return str(v.value)
    return str(v)


def transcribe_effective_options(
    *,
    request_language: str | None,
    task: str | None,
    beam_size: int | None,
    temperature: float | None,
    best_of: int | None,
    word_timestamps: bool | None,
    vad: bool | None,
) -> CacheConf:
    """
    Merge request-time overrides with settings defaults.
    Only minimal validation here; rely on Faster-Whisper for deeper checks.
    """
    return CacheConf(
        language=request_language or settings.ASR_TRANSCRIBE_LANG or None,
        task=_normalize_enum(v=task, default="transcribe"),
        beam_size=int(beam_size or settings.ASR_TRANSCRIBE_BEAM_SIZE or 5),
        temperature=float(temperature or settings.ASR_TRANSCRIBE_TEMPERATURE or 0.0),
        best_of=int(best_of or settings.ASR_TRANSCRIBE_BEST_OF or 1),
        word_timestamps=bool(word_timestamps if word_timestamps is not None else False),
        use_vad=bool(vad if vad is not None else settings.ASR_VAD_ENABLED or False),
        detect_lang_length=None,
        detect_lang_offset=None,
    )


class ASRBase(ABC):
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
        self._cache: dict[str, TranscribeResult | DetectLanguageResult] = {}
        self._order: list[str] = []  # FIFO keys

        # Idle timeout management (optional feature)
        from datetime import datetime

        self.idle_timeout_minutes = int(getattr(settings, "ASR_IDLE_TIMEOUT_MINUTES", 0))
        self.last_used = datetime.now()
        self._model_loaded = True

        if self.idle_timeout_minutes > 0:
            logger.info(f"ASR idle timeout enabled: {self.idle_timeout_minutes} minutes")

    def _touch(self) -> None:
        """Update last used timestamp - call this on every inference request."""
        from datetime import datetime

        self.last_used = datetime.now()
        self._model_loaded = True

    @abstractmethod
    def _load_model(self) -> None:
        """
        Subclass-specific model loading/initialization.

        This method should load the model into memory. It will be called:
        1. During __init__() (initial load)
        2. After the model has been unloaded due to idle timeout (reload)

        The implementation should set self.model to the loaded model instance.
        """
        raise NotImplementedError

    @abstractmethod
    def _unload_model(self) -> None:
        """
        Subclass-specific model cleanup.

        This method should safely free GPU/CPU memory for the model without
        affecting other services. Typically involves:
        1. Deleting model references
        2. Running gc.collect()
        3. For GPU models: torch.cuda.synchronize() (NOT empty_cache!)

        DO NOT call torch.cuda.empty_cache() as it affects all processes
        sharing the GPU.
        """
        raise NotImplementedError

    def ensure_model_loaded(self) -> None:
        """
        Ensure the model is loaded before inference.

        If the model was unloaded due to idle timeout, this method reloads it.
        This should be called at the beginning of every inference operation.
        """
        if not self._model_loaded:
            logger.info("Model was unloaded, reloading...")
            self._load_model()
            self._model_loaded = True
            logger.info("Model successfully reloaded")

    def check_and_unload_if_idle(self) -> bool:
        """
        Check if model has been idle and unload if past timeout.

        Returns:
            bool: True if model was unloaded, False otherwise
        """
        if self.idle_timeout_minutes <= 0 or not self._model_loaded:
            return False

        from datetime import datetime, timedelta

        idle_time = datetime.now() - self.last_used
        timeout = timedelta(minutes=self.idle_timeout_minutes)

        if idle_time > timeout:
            logger.info(f"Model idle for {idle_time}, unloading...")
            self._unload_model()
            self._model_loaded = False
            return True

        return False

    @abstractmethod
    def _transcribe_core(
        self,
        *,
        audio_f32: npt.NDArray[np.float32],
        sr: int,
        # Only for caching; can be None for array inputs
        # if not passed the cache implementation is ignored
        raw_bytes: bytes | None,
        request_language: str | None,
        task: str | None,
        beam_size: int | None,
        temperature: float | None,
        best_of: int | None,
        word_timestamps: bool | None,
        vad: bool | None,
    ) -> TranscribeResult:
        raise NotImplementedError

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
    ) -> TranscribeResult:
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

        # Cleaning some bad inputs
        request_language = request_language if request_language is not LanguageCode.UNKNOWN else None

        logger.info("Decoded via %s: sr=%d samples=%d", source, sr, int(audio_f32.shape[0]))

        # Update last used timestamp for idle timeout tracking
        self._touch()

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

    @abstractmethod
    def _detect_language_core(
        self,
        *,
        audio_f32: npt.NDArray[np.float32],
        sr: int,
        # Only for caching; can be None for array inputs
        # if not passed the cache implementation is ignored
        raw_bytes: bytes | None,
        request_language: str | None,
        detect_lang_length: float | None = None,
        detect_lang_offset: float | None = None,
    ) -> DetectLanguageResult:
        raise NotImplementedError

    def detect_language_file(
        self,
        *,
        file_bytes: bytes,
        filename: str | None = None,
        request_language: str | None = None,
        detect_lang_length: float | None = None,
        detect_lang_offset: float | None = None,
    ) -> DetectLanguageResult:
        """
        Detect Language from `file_bytes` (using soundfile fast-path, then ffmpeg fallback).
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

        # Update last used timestamp for idle timeout tracking
        self._touch()

        return self._detect_language_core(
            audio_f32=audio_f32,
            sr=sr,
            raw_bytes=file_bytes,  # for cache key
            request_language=request_language,
            detect_lang_length=detect_lang_length,
            detect_lang_offset=detect_lang_offset,
        )

    @staticmethod
    def create_hash_cache_key(*, audio_bytes: bytes, conf: CacheConf) -> str:
        """
        Stable cache key: audio bytes + the slimmed options/conf footprint.
        """
        h = hashlib.sha256(audio_bytes)
        h.update(
            json.dumps(
                asdict(conf),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return h.hexdigest()

    def helper_apply_vad(
        self, *, audio_f32: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], bool, list[dict[str, float]]]:
        """
        Apply VAD to audio and return processed audio, flag, and timestamps.

        Returns
        -------
        tuple[np.ndarray, bool, list[dict]]
            - Processed audio (float32)
            - Whether VAD was applied
            - List of speech segments with original timeline timestamps: [{"start": float, "end": float}, ...]
              Empty list if VAD was not applied.
        """
        # Check if model was loaded and is available
        if self.silero_model is not None and self.silero_ts_fn is not None:
            # Lets convert the narray to integer as SILERO needs a PCM16 Integer
            # Clip between -1 and 1 to avoid integer overflow
            pcm16 = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16)

            pcm16_vad, vad_segments = apply_vad_silero(
                pcm16=pcm16,
                sr=16000,  # Yes we expect it is in 16000
                silero_model=self.silero_model,
                get_speech_timestamps_fn=self.silero_ts_fn,
                log=logger,
            )

            # Now lets convert it back to float
            audio_f32 = (pcm16_vad.astype(np.float32) / 32768.0) if pcm16_vad.size else audio_f32
            vad_used = True
            return audio_f32, vad_used, vad_segments
        else:
            return audio_f32, False, []

    @staticmethod
    def helper_map_compressed_to_original_timeline(
        compressed_time: float, vad_segments: list[dict[str, float]]
    ) -> float:
        """
        Map a timestamp from VAD-compressed audio back to the original timeline.

        When VAD removes silence, the audio is compressed by concatenating speech segments.
        This function maps a timestamp in the compressed audio to its position in the original.

        Parameters
        ----------
        compressed_time : float
            Timestamp in seconds from the compressed (VAD-processed) audio.
        vad_segments : list[dict[str, float]]
            Speech segments from VAD with original timeline positions.
            Format: [{"start": float, "end": float}, ...]

        Returns
        -------
        float
            Timestamp in the original (uncompressed) timeline.

        Examples
        --------
        If VAD found speech at: [{start: 1.0, end: 2.0}, {start: 5.0, end: 6.0}]
        - Compressed audio is 2 seconds long (1s + 1s of speech)
        - compressed_time=0.5 → maps to 1.5 (within first segment)
        - compressed_time=1.5 → maps to 5.5 (within second segment)
        """
        if not vad_segments:
            # No VAD was applied, timestamps are already correct
            return compressed_time

        # Calculate cumulative duration of each segment in compressed audio
        cumulative_duration = 0.0

        for seg in vad_segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_duration = seg_end - seg_start

            # Check if compressed_time falls within this segment
            if compressed_time <= cumulative_duration + seg_duration:
                # Time is within this segment
                # Calculate offset within segment
                offset_in_segment = compressed_time - cumulative_duration
                # Map to original timeline
                return seg_start + offset_in_segment

            cumulative_duration += seg_duration

        # If we've gone past all segments, assume it's at the end of the last segment
        if vad_segments:
            last_seg = vad_segments[-1]
            # Extrapolate beyond last segment (shouldn't happen in practice)
            overflow = compressed_time - cumulative_duration
            return last_seg["end"] + overflow

        # Fallback (should never reach here if vad_segments is not empty)
        return compressed_time

    def _write_output_body(
        self, *, result: TranscribeResult, output: Output, max_line_len: int = 42
    ) -> str | dict[str, Any]:
        if output is Output.JSON:
            # Bazarr-style JSON (language + segments)
            return {
                "language": result.language_code,
                "segments": [
                    {"start": float(s["start"]), "end": float(s["end"]), "text": s["text"] or ""}
                    for s in result.segments
                ],
            }
        elif output is Output.SRT:
            return result.to_srt(max_line_len=max_line_len)
        elif output is Output.VTT:
            return result.to_vtt(max_line_len=max_line_len)
        elif output is Output.TXT:
            return result.to_txt()
        elif output is Output.TSV:
            return result.to_tsv()
        elif output is Output.JSONL:
            return result.to_segments_jsonl()

        raise ValueError(f"Unsupported output format: {output}")

    def helper_write_output(
        self, *, file: UploadFile, result: TranscribeResult, output: Output, max_line_len: int = 42
    ) -> StreamingResponse | JSONResponse:
        """
        Create appropriate response for the requested output and
        also persist a temp file for debugging / hand-off.
        """

        # Choosing the Stream Headers
        media_type, ext = OUTPUT_HEADERS[output]
        body = self._write_output_body(result=result, output=output, max_line_len=max_line_len)

        # Creating a TempFile and setting its name
        tmp_dir = tempfile.gettempdir()
        safe_name = (file.filename or "audio").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        base_no_ext = os.path.splitext(safe_name)[0] or "audio"
        tmp_path = os.path.join(tmp_dir, f"{base_no_ext}.{ext}")

        try:
            if isinstance(body, dict):
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(body, f, ensure_ascii=False)
            else:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(body)

            logger.info("ASR/Bazarr wrote temp file: %s", tmp_path)
        except Exception as e_write:
            logger.warning("ASR/Bazarr could not write temp file (%s): %s", tmp_path, e_write)

        # Non-JSON formats: stream as text
        headers = {
            "Asr-Engine": settings.ASR_ENGINE or "UNKNOWN",
            "Content-Disposition": f'attachment; filename="{base_no_ext}.{ext}"',
            "X-Temp-Path": tmp_path,
        }

        # Output is Output.JSON; JSON will return JSONResponse (schema-friendly)
        if isinstance(body, dict):
            return JSONResponse(content=body, media_type=media_type, headers=headers, status_code=status.HTTP_200_OK)

        # Small strings are fine to stream from memory
        return StreamingResponse(
            content=iter([body]),
            media_type=media_type,
            headers=headers,
            status_code=status.HTTP_200_OK,
        )

    def helper_check_cache(
        self, *, opts: CacheConf, raw_bytes: bytes | None
    ) -> tuple[TranscribeResult | DetectLanguageResult | None, bool]:
        # Check if cache is enable and raw bytes has content
        if self.cache_enabled and raw_bytes is not None:
            try:
                cache_key = ASRBase.create_hash_cache_key(audio_bytes=raw_bytes, conf=opts)

                if cache_key in self._cache:
                    hit = self._cache[cache_key]

                    # move key to tail (fresh) in FIFO
                    try:
                        self._order.remove(cache_key)
                    except ValueError:
                        pass

                    self._order.append(cache_key)
                    return hit, True
                # Didnt find anything in the cache
                return None, False

            except Exception:
                # cache is best-effort; never fail the request
                # even if it fail to retrieve or store we will continue
                return None, False
        else:
            return None, False

    def helper_save_on_cache(
        self, *, raw_bytes: bytes, result: TranscribeResult | DetectLanguageResult, opts: CacheConf
    ) -> None:
        # Check if cache is enable otherwise return
        if not self.cache_enabled:
            return
        try:
            cache_key = ASRBase.create_hash_cache_key(audio_bytes=raw_bytes, conf=opts)

            self._cache[cache_key] = result
            self._order.append(cache_key)

            while len(self._order) > self.cache_max:
                old = self._order.pop(0)
                self._cache.pop(old, None)

        except Exception:
            # cache is best-effort; never fail the request
            # even if it fail to retrieve or store we will continue
            pass

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
