from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
from fastapi.responses import Response, StreamingResponse

from core.settings import get_settings
from tts.schemas.audio_engine import AudioFormat, ModelResponse, ModelsResponse, StreamFormat, VoicesResponse
from utils.audio.audio_helper import pick_torch_device
from utils.language.language_codes import LanguageCode

settings = get_settings()


def languange_canonical_str(code: str | None) -> str | None:
    """Return canonical short code ('en','pt','zh-cn') or None."""
    if not code:
        return None
    lc = LanguageCode.from_string(code)
    return None if lc == LanguageCode.UNKNOWN else lc.value


def _mime_for(fmt: str, *, sample_rate: int = 24000, channels: int = 1) -> str:
    fmt = fmt.lower()
    if fmt == "mp3":
        return "audio/mpeg"
    if fmt == "opus":
        # Usually in OGG container in our pipeline
        return "audio/ogg; codecs=opus"
    if fmt == "aac":
        # ADTS framing
        return "audio/aac"
    if fmt == "flac":
        return "audio/flac"
    if fmt == "wav":
        return "audio/wav"
    if fmt == "pcm":
        # Raw 16-bit little-endian PCM (no header)
        return f"audio/L16; rate={sample_rate}; channels={channels}"
    # Fallback
    return "application/octet-stream"


def _ext_for(fmt: str) -> str:
    fmt = fmt.lower()
    return {
        "mp3": "mp3",
        "opus": "ogg",  # ogg container
        "aac": "aac",  # adts
        "flac": "flac",
        "wav": "wav",
        "pcm": "s16le",  # raw pcm 16-bit LE
    }.get(fmt, "bin")


def _disposition(filename: str) -> dict[str, str]:
    # Most browsers respect this for "download"
    return {"Content-Disposition": f'attachment; filename="{filename}"'}


@dataclass
class PropsConf:
    """
    Utilized validate props for the model
    """

    input: str
    voice: str = "default"
    response_format: str = "mp3"
    speed: float = 1.0
    stream_format: str = "audio"
    requested_language: str | None = None
    language_hint: str | None = None


def _normalize_enum(*, v: str | Enum | None, default: str) -> str:
    if v is None:
        return default
    if isinstance(v, Enum):
        return str(v.value)
    return str(v)


def speech_effective_options(
    *,
    input: str | None = None,
    voice: str | None = None,
    response_format: str | None = None,
    speed: float | None = None,
    stream_format: str | None = None,
    requested_language: str | None = None,
    language_hint: str | None = None,
) -> PropsConf:
    """
    Merge request-time overrides with settings defaults.
    Only minimal validation here; rely on Downstream Classes for deeper checks.
    """
    return PropsConf(
        input=input or "You need to provide an input!",
        voice=voice or "default",
        response_format=_normalize_enum(v=response_format, default="mp3").lower(),
        speed=speed or 1.0,
        stream_format=_normalize_enum(v=stream_format, default="audio").lower(),
        requested_language=requested_language,
        language_hint=language_hint,
    )


class TTSBase(ABC):
    # Engines whose model must not run two syntheses at once set this to True:
    # in_use() then serialises synthesis on the engine. Synthesis left the event
    # loop (so /health stays responsive), which removed the accidental serialisation the loop provided.
    serialize_synthesis: bool = False

    def _state(self) -> dict[str, Any]:
        """Lifecycle lock, synthesis lock and in-flight count, created on first use.

        Created lazily (dict.setdefault is atomic in CPython) so engines built
        with __new__ in tests, which skip __init__, still get them.
        """
        state: dict[str, Any] = self.__dict__.setdefault(
            "_lifecycle_state",
            {"lock": threading.RLock(), "synth": threading.Lock(), "inflight": 0},
        )
        return state

    @contextmanager
    def in_use(self) -> Iterator[None]:
        """Hold the model for one synthesis.

        Marks the engine in use, (re)loads the model under the lifecycle lock so
        two requests after an idle unload cannot both load it, and keeps the
        idle unloader away until the synthesis finishes. Engines wrap their
        model calls in this instead of calling _touch()/ensure_model_loaded().
        """
        state = self._state()
        with state["lock"]:
            state["inflight"] += 1
            self._touch()
        try:
            with state["lock"]:
                self.ensure_model_loaded()
            if self.serialize_synthesis:
                with state["synth"]:
                    yield
            else:
                yield
        finally:
            with state["lock"]:
                state["inflight"] -= 1
                self._touch()  # idle time counts from the END of the last synthesis

    def __init__(self) -> None:
        # Retrieving requested settings
        self.model_id = settings.TTS_MODEL
        self.sample_rate: int = settings.TTS_SAMPLE_RATE or 24000
        self.voices_dir = settings.TTS_VOICES_DIR
        self.max_chars = settings.TTS_MAX_CHARS
        self.min_chars = settings.TTS_MIN_CHARS
        self.retry_steps = settings.TTS_RETRY_STEPS

        self.auto_language = settings.TTS_AUTO_LANG
        self.default_languange = languange_canonical_str(settings.TTS_DEFAULT_LANG) or "en"
        self.language_hint = languange_canonical_str(settings.TTS_LANG_HINT)
        self.force_language = languange_canonical_str(settings.TTS_FORCE_LANG)

        # Checking device requested
        self.model_device = settings.TTS_DEVICE.lower().strip()
        if self.model_device == "auto":
            self.model_device = pick_torch_device()

        # Idle timeout management (optional feature)
        from datetime import datetime

        from core.logging import logger_tts as logger

        self.idle_timeout_minutes = int(getattr(settings, "TTS_IDLE_TIMEOUT_MINUTES", 0))
        self.last_used = datetime.now()
        self._model_loaded = True

        if self.idle_timeout_minutes > 0:
            logger.info(f"TTS idle timeout enabled: {self.idle_timeout_minutes} minutes")

    def _touch(self) -> None:
        """Update last used timestamp - call this on every synthesis request.

        It must not mark the model as loaded. Engines call _touch() before
        ensure_model_loaded(), so setting the flag here made the reload check
        always pass: after an idle unload the next request found no model.
        Only ensure_model_loaded() (and __init__) may set _model_loaded.
        """
        from datetime import datetime

        self.last_used = datetime.now()

    @abstractmethod
    def _load_model(self) -> None:
        """
        Subclass-specific model loading/initialization.

        This method should load the model into memory. It will be called:
        1. During __init__() (initial load)
        2. After the model has been unloaded due to idle timeout (reload)

        The implementation should set self.tts (or equivalent) to the loaded model instance.
        """
        raise NotImplementedError

    @abstractmethod
    def _unload_model(self) -> None:
        """
        Subclass-specific model cleanup.

        This method should safely free GPU/CPU memory for the model without
        affecting other services. Typically involves:
        1. Moving model to CPU (for GPU models)
        2. Deleting model references
        3. Running gc.collect()
        4. torch.cuda.synchronize() (NOT empty_cache!)

        DO NOT call torch.cuda.empty_cache() as it affects all processes
        sharing the GPU.
        """
        raise NotImplementedError

    def ensure_model_loaded(self) -> None:
        """
        Ensure the model is loaded before synthesis.

        If the model was unloaded due to idle timeout, this method reloads it.
        This should be called at the beginning of every synthesis operation.
        """
        from core.logging import logger_tts as logger

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
        if self.idle_timeout_minutes <= 0:
            return False

        from datetime import datetime, timedelta

        from core.logging import logger_tts as logger

        state = self._state()
        with state["lock"]:
            # Under the lock, and never while a synthesis holds the model: the
            # idle checker runs concurrently with synthesis threads, and
            # unloading mid-synthesis left the engine calling a None model.
            if not self._model_loaded or state["inflight"] > 0:
                return False

            idle_time = datetime.now() - self.last_used
            if idle_time <= timedelta(minutes=self.idle_timeout_minutes):
                return False

            logger.info(f"Model idle for {idle_time}, unloading...")
            self._unload_model()
            self._model_loaded = False
            return True

    def get_model(self, model_id: str) -> ModelResponse:
        return ModelResponse(id=model_id)

    def list_models(self) -> ModelsResponse:
        return ModelsResponse(data=[ModelResponse(id=f"model_{i}") for i in range(1, 11)])

    @abstractmethod
    def list_voices(self) -> VoicesResponse:
        raise NotImplementedError

    @abstractmethod
    def speech(
        self,
        *,
        input: str,
        voice: str | None = None,
        response_format: AudioFormat | None = AudioFormat.MP3,
        speed: float | None = 1.0,
        stream_format: StreamFormat | None = StreamFormat.AUDIO,
        requested_language: str | None = None,
        language_hint: str | None = None,
    ) -> Any:
        raise NotImplementedError

    def package_audio(self, wav: npt.NDArray[np.float32], props: PropsConf) -> Any:
        """Encode synthesized samples to the requested format and transport.

        Shared by every engine: an engine produces float32 samples at
        self.sample_rate, this turns them into the bytes / chunk iterator the
        API layer returns. Kept out of the engines so a new engine cannot get
        the formats subtly different from the others.
        """
        import io

        import soundfile as sf

        from utils.audio.audio_helper import wav_bytes_to_pcm16le_bytes
        from utils.audio.ffmpeg_helper import encode_audio_from_wav_bytes

        buf = io.BytesIO()
        sf.write(buf, wav, self.sample_rate, subtype="PCM_16", format="WAV")
        wav_bytes = buf.getvalue()

        if props.response_format == "pcm":
            encoded = wav_bytes_to_pcm16le_bytes(wav_bytes)  # fast path via soundfile
        else:
            encoded = encode_audio_from_wav_bytes(
                wav_bytes=wav_bytes,
                target_format=props.response_format,
                sample_rate=self.sample_rate,
                channels=1,
            )

        if props.stream_format == "audio":
            return encoded
        if props.stream_format == "sse":
            return self._sse_base64_chunks(encoded)
        return self._chunked(encoded)

    @staticmethod
    def _chunked(b: bytes, *, chunk_size: int = 65536) -> Iterable[bytes]:
        """
        Binary chunk generator for streaming audio transport.
        """
        view = memoryview(b)
        n = len(view)
        i = 0
        while i < n:
            yield bytes(view[i : i + chunk_size])
            i += chunk_size

    @staticmethod
    def _sse_base64_chunks(b: bytes, *, b64_chunk_size: int = 65536) -> Iterable[bytes]:
        """
        SSE-friendly generator: yields lines like b"data: <base64>\n\n".
        Caller sets the HTTP headers:
          Content-Type: text/event-stream
          Cache-Control: no-cache
          Connection: keep-alive
        """
        import base64

        # We base64-encode in chunks to keep each SSE event small-ish.
        i = 0
        n = len(b)
        while i < n:
            chunk = b[i : i + b64_chunk_size]
            i += b64_chunk_size
            yield b"data: " + base64.b64encode(chunk) + b"\n\n"

    def helper_return_sse_stream(
        self, *, audio: bytes | bytearray | Iterable[bytes], already_sse: bool = True
    ) -> StreamingResponse:
        """
        Return an SSE stream. If `audio` is bytes, we wrap into "data: <base64>\\n\\n".
        If `audio` is an Iterable[bytes]:
        - when already_sse=True (default): pass-through (engine formatted SSE lines)
        - when already_sse=False: base64-wrap each binary chunk into SSE events
        """

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Nginx: disable proxy buffering
        }

        # Case 1: blob -> wrap to SSE
        if isinstance(audio, bytes | bytearray):

            def _to_sse_from_bytes(b: bytes) -> Iterator[bytes]:
                import base64

                CH = 64 * 1024
                for i in range(0, len(b), CH):
                    yield b"data: " + base64.b64encode(b[i : i + CH]) + b"\n\n"

            return StreamingResponse(_to_sse_from_bytes(bytes(audio)), media_type="text/event-stream", headers=headers)

        # Case 2: Iterable[bytes]
        if already_sse:
            # Engine already yields b"data: ...\\n\\n" lines — pass-through
            return StreamingResponse(audio, media_type="text/event-stream", headers=headers)

        # Case 3: Iterable[bytes] but raw -> wrap each chunk
        def _to_sse_from_chunks(chunks: Iterable[bytes]) -> Iterator[bytes]:
            import base64

            for ch in chunks:
                yield b"data: " + base64.b64encode(ch) + b"\n\n"

        return StreamingResponse(_to_sse_from_chunks(audio), media_type="text/event-stream", headers=headers)

    def helper_return_audio_file(
        self, *, audio: bytes | bytearray | Iterable[bytes], response_format: AudioFormat = AudioFormat.MP3
    ) -> Response | StreamingResponse:
        """
        Return audio either as a full file (bytes) or as a streaming response (Iterable[bytes]).
        """
        # Pick content type + filename
        mime = _mime_for(response_format, sample_rate=self.sample_rate, channels=1)
        filename = f"tts-{int(time.time())}.{_ext_for(response_format)}"

        # If engine returned bytes -> regular Response; if it returned an iterable -> stream it.
        if isinstance(audio, bytes | bytearray):
            # Return as a downloadable file (or use inline by omitting Content-Disposition)
            return Response(
                content=bytes(audio),
                media_type=mime,
                headers=_disposition(filename),
            )
        else:
            # Streaming binary audio (chunked)
            return StreamingResponse(
                audio,  # Iterable[bytes]
                media_type=mime,
                headers=_disposition(filename),
            )
