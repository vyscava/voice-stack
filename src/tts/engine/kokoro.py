"""Kokoro-82M TTS engine on ONNX Runtime (CPU).

Kokoro is small (82M parameters) and fast enough on a CPU to answer in well under
real time, which is why it is the default engine. It has no voice cloning: each
voice is a fixed style vector bound to one language. XTTS (TTS_ENGINE=coqui)
remains the engine for cloned voices.

Language handling differs from XTTS on purpose. An XTTS speaker can read any
supported language; a Kokoro voice cannot -- pf_dora reading English sounds like
a Brazilian accent, not like an English voice. So the language comes from, in
order: TTS_FORCE_LANG, the request, then the voice's own language. There is no
per-chunk auto-detection, because a detected language that does not match the
voice produces worse audio than the voice's own language would.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from core.logging import logger_tts as logger
from core.settings import get_settings
from tts.engine.base import TTSBase, languange_canonical_str, speech_effective_options
from tts.exceptions import ModelLoadError
from tts.schemas.audio_engine import (
    AudioFormat,
    ModelResponse,
    ModelsResponse,
    StreamFormat,
    VoiceResponse,
    VoicesResponse,
)

settings = get_settings()

# Kokoro voice ids start with a language letter. Only languages espeak-ng can
# phonemize are listed: Japanese (j) and Mandarin (z) voices need the misaki G2P,
# which kokoro-onnx does not ship, so they are not offered rather than offered
# and mispronounced.
VOICE_PREFIX_TO_LANG: dict[str, str] = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "p": "pt-br",
}

# Canonical request language (LanguageCode value) -> espeak-ng tag Kokoro takes.
CANONICAL_TO_ESPEAK: dict[str, str] = {
    "en": "en-us",
    "es": "es",
    "fr": "fr-fr",
    "hi": "hi",
    "it": "it",
    "pt": "pt-br",
}

# Kokoro rejects speeds outside this range with a ValueError.
MIN_SPEED = 0.5
MAX_SPEED = 2.0


def voice_language(voice: str) -> str | None:
    """espeak tag for a Kokoro voice id, or None if the voice's language is unsupported."""
    if len(voice) < 2 or voice[1] not in "fm" or "_" not in voice:
        return None
    return VOICE_PREFIX_TO_LANG.get(voice[0])


class TTSKokoro(TTSBase):
    def __init__(self) -> None:
        super().__init__()
        self.model_path = Path(settings.TTS_KOKORO_MODEL_PATH)
        self.voices_path = Path(settings.TTS_KOKORO_VOICES_PATH)
        self.default_voice = settings.TTS_KOKORO_DEFAULT_VOICE
        self.threads = int(settings.TTS_KOKORO_THREADS)
        self.kokoro: Any = None
        self.voices: list[str] = []
        self._load_model()

    # ------------------------------------------------------------------ model

    def _load_model(self) -> None:
        for path, what in ((self.model_path, "model"), (self.voices_path, "voices archive")):
            if not path.is_file():
                raise ModelLoadError(
                    message=f"Kokoro {what} not found",
                    details=f"Expected a file at {path}",
                )

        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        opts = ort.SessionOptions()
        if self.threads > 0:
            opts.intra_op_num_threads = self.threads
        session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.kokoro = Kokoro.from_session(session, str(self.voices_path))

        self.voices = sorted(v for v in self.kokoro.get_voices() if voice_language(v) is not None)
        if self.default_voice not in self.voices:
            raise ModelLoadError(
                message="Kokoro default voice not available",
                details=f"TTS_KOKORO_DEFAULT_VOICE={self.default_voice!r} is not one of {self.voices}",
            )
        logger.info(
            "Kokoro loaded model=%s voices=%d default=%s threads=%s",
            self.model_path.name,
            len(self.voices),
            self.default_voice,
            self.threads or "auto",
        )

    def _unload_model(self) -> None:
        if self.kokoro is not None:
            logger.info("Unloading Kokoro model")
            self.kokoro = None
            gc.collect()

    # ----------------------------------------------------------------- voices

    def resolve_voice(self, voice: str | None) -> str:
        """Voice id to synthesize with. 'default' (or empty) maps to the configured default.

        An unknown voice raises KeyError, which the API layer turns into a 404.
        It must never fall back to another voice: a typo returning 200 and audio
        in somebody else's voice is the bug the Coqui engine already had once.
        """
        if not voice or voice == "default":
            return self.default_voice
        if voice in self.voices:
            return voice
        raise KeyError(voice)

    def list_voices(self) -> VoicesResponse:
        return VoicesResponse(data=[VoiceResponse(id=v, name=v) for v in self.voices])

    def list_models(self) -> ModelsResponse:
        return ModelsResponse(data=[ModelResponse(id=self.model_path.stem)])

    # --------------------------------------------------------------- language

    def choose_lang(self, voice: str, requested: str | None) -> str:
        """espeak tag to phonemize with: TTS_FORCE_LANG, then the request, then the voice's own language."""
        for source in (self.force_language, languange_canonical_str(requested)):
            if source and source in CANONICAL_TO_ESPEAK:
                return CANONICAL_TO_ESPEAK[source]
        lang = voice_language(voice)
        if lang is None:  # resolve_voice only returns voices with a known language
            raise KeyError(voice)
        return lang

    # ----------------------------------------------------------------- speech

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
        self._touch()
        self.ensure_model_loaded()

        voice_id = self.resolve_voice(props.voice)
        lang = self.choose_lang(voice_id, props.requested_language)
        clamped = min(MAX_SPEED, max(MIN_SPEED, float(props.speed)))
        if clamped != props.speed:
            logger.warning(
                "speed=%.2f outside Kokoro's %.1f-%.1f range; using %.2f", props.speed, MIN_SPEED, MAX_SPEED, clamped
            )

        samples, sample_rate = self.kokoro.create(props.input, voice=voice_id, speed=clamped, lang=lang)
        wav: npt.NDArray[np.float32] = np.asarray(samples, dtype=np.float32)
        if sample_rate != self.sample_rate:
            # Kokoro always produces 24 kHz; TTS_SAMPLE_RATE describes the output,
            # so a mismatch would mislabel the WAV header rather than resample.
            raise ModelLoadError(
                message="Kokoro sample rate mismatch",
                details=f"Model produced {sample_rate} Hz but TTS_SAMPLE_RATE={self.sample_rate}",
            )
        return self.package_audio(wav, props)
