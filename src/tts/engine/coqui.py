from __future__ import annotations

import base64
import io
import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt
import soundfile as sf
from torch.serialization import add_safe_globals
from TTS.api import TTS
from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig

from core.logging import logger_tts as logger
from core.settings import get_settings
from tts.engine.base import TTSBase, languange_canonical_str, speech_effective_options
from tts.schemas.audio_engine import AudioFormat, ModelResponse, ModelsResponse, StreamFormat
from utils.audio.audio_helper import wav_bytes_to_pcm16le_bytes
from utils.audio.ffmpeg_helper import encode_audio_from_wav_bytes
from utils.language.language_codes import LanguageCode
from utils.language.language_helper import detect_lang, latin_heuristic, script_heuristic
from utils.text import build_safe_chunks, normalize_text

settings = get_settings()

add_safe_globals([BaseAudioConfig, BaseDatasetConfig, XttsConfig, XttsArgs, XttsAudioConfig])


class TTSCoqui(TTSBase):

    def __init__(self) -> None:
        super().__init__()

        logger.info("Loading Coqui TTS")
        logger.info("Loading model_id=%s on %s", self.model_id, self.model_device)
        self.tts = TTS(model_name=self.model_id, progress_bar=False).to(self.model_device)

        # Loading Available Models in memoery
        self.available_models = self.tts.list_models()

        # Builtin speakers (if the model exposes them)
        self._load_voices_presets()

        # Supported languages reported by the model (preferred)
        self._load_supported_languages()

    def _load_voices_presets(self) -> None:
        """
        Build a map of available builtin speakers.

        Example:
        {
            'Claribel Dervla' = 'Claribel Dervla',
            'Daisy Studious' = 'Daisy Studious',
            ...
        }
        """
        self.voice_to_preset: dict[str, str] = {}
        if hasattr(self.tts, "speakers") and isinstance(self.tts.speakers, list):
            for name in self.tts.speakers:
                self.voice_to_preset[name] = name

    def _load_supported_languages(self) -> None:
        """
        Prefer model-config languages; otherwise fall back to full LanguageCode set.
        Always store as canonical *strings*.
        """
        try:
            cfg_langs = getattr(getattr(self.tts, "tts_model", None), "config", None).languages  # type: ignore
            if cfg_langs:
                self.supported_langs: set[str] = {("zh-cn" if x == "zh" else str(x)).lower() for x in cfg_langs}
            else:
                raise RuntimeError("empty languages")
        except Exception:
            # Fallback to all known canonical codes from LanguageCode (except UNKNOWN)
            self.supported_langs = {m.value for m in LanguageCode if m is not LanguageCode.UNKNOWN}

    def _choose_lang(self, chunk: str, requested: str | None) -> str:
        """
        Decide a supported language code (canonical string) for a text chunk.
        Priority:
          1) FORCE_LANG (env) if set & supported
          2) explicit 'requested' argument if supported
          3) LANG_HINT (env) if supported
          4) AUTO detection → clamp to supported (script/Latin heuristics fallbacks)
          5) DEFAULT_LANG if supported else first supported
        """
        # 1) Force override
        if self.force_language and self.force_language in self.supported_langs:
            if requested and languange_canonical_str(requested) != self.force_language:
                logger.info(
                    "tts.xtts | overriding requested lang=%s with FORCE_LANG=%s",
                    requested,
                    self.force_language,
                )
            return "zh-cn" if self.force_language == "zh" else self.force_language

        # 2) Request argument
        requested_canonical = languange_canonical_str(requested)
        if requested_canonical and requested_canonical in self.supported_langs:
            return "zh-cn" if requested_canonical == "zh" else requested_canonical

        # 3) Hint
        if self.language_hint and self.language_hint in self.supported_langs:
            return self.language_hint

        # 4) Autodetect
        if self.auto_language:
            detected = detect_lang(chunk)
            detected_can: str | None = detected.value if detected and detected != LanguageCode.UNKNOWN else None
            if detected_can and detected_can in self.supported_langs:
                return detected_can

            # Script heuristic (robust for non-Latin)
            script_guess = script_heuristic(chunk)
            sc_can: str | None = script_guess.value if script_guess and script_guess != LanguageCode.UNKNOWN else None
            if sc_can and sc_can in self.supported_langs:
                logger.info("tts.coqui | langdetect=%s unsupported -> script_guess=%s", detected_can, sc_can)
                return sc_can

            # Latin heuristic (quick PT/ES/FR/DE/IT cues)
            latin_guess = latin_heuristic(chunk)
            la_can: str | None = latin_guess.value if latin_guess and latin_guess != LanguageCode.UNKNOWN else None
            if la_can and la_can in self.supported_langs:
                logger.info("tts.coqui | langdetect=%s unsupported -> latin_guess=%s", detected_can, la_can)
                return la_can

            if detected_can:
                logger.info("tts.xtts | langdetect=%s unsupported -> falling back", detected_can)

        # 5) Default
        if self.default_languange in self.supported_langs:
            return self.default_languange

        # Last resort: any supported
        # Need to check the function that creates that variable
        return next(iter(self.supported_langs))

    def list_models(self) -> ModelsResponse:
        return ModelsResponse(data=[ModelResponse(id=model) for model in self.available_models])

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

        props = speech_effective_options(
            input=input,
            voice=voice,
            response_format=response_format,
            speed=speed,
            stream_format=stream_format,
            requested_language=requested_language,
            language_hint=language_hint,
        )

        if (props.speed) != 1.0:
            logger.warning(
                "speed=%.2f requested; high-quality time-stretch not configured. Returning original tempo.",
                props.speed,
            )

        # Voice preset selection (builtin speakers)
        if not self.voice_to_preset:
            raise RuntimeError("No builtin speakers exposed by this model.")

        # Checking if requested voice exists in preset
        if props.voice not in self.voice_to_preset:
            # It does not exists lets get the first one available
            props.voice = next(iter(self.voice_to_preset))

        # Split long text into manageable chunks
        cur_size = max(1, int(self.max_chars))  # current chunk budget
        chunks = build_safe_chunks(props.input, cur_size)

        attempts = 0
        while True:
            try:
                wavs: list[npt.NDArray[np.float32]] = []
                for chunk in chunks:
                    chunk = normalize_text(chunk)
                    lang_for_chunk = self._choose_lang(chunk, props.requested_language)

                    # Generate chunk audio
                    wav = self.tts.tts(
                        text=chunk,
                        speaker=props.voice,
                        language=lang_for_chunk,
                    )
                    wavs.append(np.asarray(wav, dtype=np.float32))

                # Merge chunks
                wav = wavs[0] if len(wavs) == 1 else np.concatenate(wavs, axis=0)
                break

            except Exception as e:
                msg = str(e)
                retriable = any(t in msg for t in ("index out of range", "device-side assert", "CUDA error", "cudnn"))
                if (not retriable) or attempts >= self.retry_steps or cur_size <= int(self.min_chars):
                    raise
                attempts += 1
                new_size = max(self.min_chars, math.floor(self.max_chars * 0.66))
                if new_size == cur_size:
                    new_size = max(self.min_chars, cur_size - 5)
                cur_size = new_size
                logger.warning(
                    "tts.coqui | retry: shrinking chunk size to %d (attempt %d)",
                    cur_size,
                    attempts,
                )
                chunks = build_safe_chunks(props.input, cur_size)

        # Always produce WAV internally, then encode to target format
        buf = io.BytesIO()
        sf.write(buf, wav, self.sample_rate, subtype="PCM_16", format="WAV")
        wav_bytes = buf.getvalue()

        # Encode via utils
        if props.response_format == "pcm":
            encoded = wav_bytes_to_pcm16le_bytes(wav_bytes)  # fast path via soundfile
        else:
            encoded = encode_audio_from_wav_bytes(
                wav_bytes=wav_bytes,
                target_format=props.response_format,
                sample_rate=self.sample_rate,
                channels=1,
            )

        # Stream selection
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
        # We base64-encode in chunks to keep each SSE event small-ish.
        i = 0
        n = len(b)
        while i < n:
            chunk = b[i : i + b64_chunk_size]
            i += b64_chunk_size
            enc = base64.b64encode(chunk)
            yield b"data: " + enc + b"\n\n"
