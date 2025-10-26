from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

from fastapi.responses import Response, StreamingResponse

from core.settings import get_settings
from tts.schemas.audio_engine import AudioFormat, ModelResponse, ModelsResponse, StreamFormat
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

    def get_model(self, model_id: str) -> ModelResponse:
        return ModelResponse(id=model_id)

    def list_models(self) -> ModelsResponse:
        return ModelsResponse(data=[ModelResponse(id=f"model_{i}") for i in range(1, 11)])

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
        if isinstance(audio, (bytes, bytearray)):

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
        if isinstance(audio, (bytes, bytearray)):
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
