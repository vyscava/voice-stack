from __future__ import annotations

import base64
import io
import math
from collections.abc import Iterable
from pathlib import Path
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
from tts.exceptions import ModelLoadError
from tts.schemas.audio_engine import (
    AudioFormat,
    ModelResponse,
    ModelsResponse,
    StreamFormat,
    VoiceResponse,
    VoicesResponse,
)
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
        self._load_model()

    def _load_model(self) -> None:
        """Load or reload the Coqui TTS model."""
        logger.info("Loading Coqui TTS")
        logger.info("Loading model_id=%s on %s", self.model_id, self.model_device)

        local = self._resolve_local_model(self.model_id)
        if local is not None:
            model_path, config_path = local
            logger.info("Loading local checkpoint model_path=%s config_path=%s", model_path, config_path)
            self.tts = TTS(model_path=str(model_path), config_path=str(config_path), progress_bar=False).to(
                self.model_device
            )
        else:
            self.tts = TTS(model_name=self.model_id, progress_bar=False).to(self.model_device)

        # Loading Available Models in memory
        self.available_models = self.tts.list_models()

        # Builtin speakers (if the model exposes them)
        self._load_voices_presets()

        # Custom voices cloned from reference audio in TTS_VOICES_DIR
        self._load_custom_voices()

        # Supported languages reported by the model (preferred)
        self._load_supported_languages()

    @staticmethod
    def _resolve_local_model(model_id: str) -> tuple[Path, Path] | None:
        """Resolve a local fine-tuned checkpoint, or None for a Coqui model id.

        A fine-tuned XTTS model is served by pointing TTS_MODEL at either the
        checkpoint file or the directory holding it; anything else (the default
        `tts_models/...` form) falls through to Coqui's model manager.

        Both a model file and a config are required -- Coqui cannot infer the
        architecture from weights alone, and passing model_path without
        config_path fails deep inside the loader with a much less obvious error.
        """
        if not model_id or model_id.startswith(("tts_models/", "voice_conversion_models/")):
            return None

        p = Path(model_id)
        if not p.exists():
            return None

        model_path = p if p.is_file() else p / "model.pth"
        config_path = model_path.parent / "config.json"

        if not model_path.is_file():
            raise ModelLoadError(
                message="Local TTS model not found",
                details=f"Expected a checkpoint at {model_path}",
            )
        if not config_path.is_file():
            raise ModelLoadError(
                message="Local TTS model config not found",
                details=f"Expected config.json next to the checkpoint at {config_path}",
            )
        return model_path, config_path

    def _unload_model(self) -> None:
        """
        Safely unload Coqui TTS model from GPU.

        Moves the model to CPU first to free GPU memory, then deletes the model
        reference and runs garbage collection.

        Note: Do NOT call torch.cuda.empty_cache() as it would affect other
        services (ASR, Open WebUI) sharing the same GPU.
        """
        import gc

        import torch

        if hasattr(self, "tts") and self.tts is not None:
            logger.info("Unloading Coqui TTS model from GPU...")

            # Move model to CPU to release GPU memory
            try:
                self.tts.to("cpu")
                logger.info("Model moved to CPU")
            except Exception as e:
                logger.warning(f"Error moving model to CPU: {e}")

            # Delete model reference
            del self.tts
            self.tts = None  # type: ignore

            # Run garbage collection to free Python objects and PyTorch tensors
            gc.collect()

            # Synchronize CUDA to ensure GPU operations complete
            # This does NOT clear the cache - it just waits for pending ops
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            logger.info("Coqui TTS model successfully unloaded")

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

    def _load_custom_voices(self) -> None:
        """Map custom voice ids to the reference audio that clones them.

        TTS_VOICES_DIR was read into settings and never used, so the documented
        "drop a .wav in voices/ and get a new voice" feature did not exist: the
        engine only ever offered the model's builtin speakers.

        Two layouts are supported:
          voices/alice.wav          -> voice "alice",  one reference clip
          voices/alice/*.wav        -> voice "alice",  several reference clips

        Several clips are worth supporting because XTTS averages the speaker
        embedding over them, which is more stable than a single sample. More is
        not automatically better though -- 3 good clips beat 6 mixed ones.
        """
        self.voice_to_wavs: dict[str, list[str]] = {}

        root = Path(self.voices_dir)
        if not root.is_dir():
            logger.info("No voices dir at %s; only builtin speakers available", root)
            return

        for entry in sorted(root.iterdir()):
            if entry.is_file() and entry.suffix.lower() == ".wav":
                self.voice_to_wavs[entry.stem] = [str(entry)]
            elif entry.is_dir():
                wavs = sorted(str(w) for w in entry.glob("*.wav"))
                if wavs:
                    self.voice_to_wavs[entry.name] = wavs

        overlap = set(self.voice_to_wavs) & set(self.voice_to_preset)
        if overlap:
            # A custom voice shadowing a builtin one is almost certainly a
            # mistake, and a silent win for either side is worse than saying so.
            logger.warning("Custom voices shadow builtin speakers: %s", sorted(overlap))

        if self.voice_to_wavs:
            logger.info(
                "Loaded %d custom voice(s) from %s: %s",
                len(self.voice_to_wavs),
                root,
                sorted(self.voice_to_wavs),
            )

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

    def resolve_voice(self, voice: str) -> list[str] | None:
        """Resolve a requested voice id.

        Returns None for a builtin speaker (selected by name), or the list of
        reference clips for a custom voice (cloned via speaker_wav).

        Raises KeyError if the voice is unknown. That matters: this used to
        silently substitute the first builtin speaker, so a typo -- or a voice
        that failed to load -- returned HTTP 200 and confident audio in
        somebody else's voice. The API layer already maps KeyError to a 404
        VoiceNotFoundError; that branch was simply unreachable.

        Kept as its own method so the behaviour is testable without loading a
        model. Inlining it in speech() is what let the bug hide.
        """
        if not self.voice_to_preset and not self.voice_to_wavs:
            raise RuntimeError("No builtin speakers or custom voices available for this model.")

        if voice in self.voice_to_preset:
            return None
        if voice in self.voice_to_wavs:
            return self.voice_to_wavs[voice]
        raise KeyError(voice)

    def list_voices(self) -> VoicesResponse:
        """
        List all available voices for this TTS model.
        Returns builtin speakers plus any custom voices cloned from TTS_VOICES_DIR.
        """
        data = [VoiceResponse(id=voice_id, name=voice_name) for voice_id, voice_name in self.voice_to_preset.items()]
        data += [
            VoiceResponse(id=name, name=name) for name in sorted(self.voice_to_wavs) if name not in self.voice_to_preset
        ]
        return VoicesResponse(data=data)

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

        # Update last used timestamp for idle timeout tracking
        self._touch()

        # Ensure model is loaded (reload if it was unloaded due to idle timeout)
        self.ensure_model_loaded()

        if (props.speed) != 1.0:
            logger.warning(
                "speed=%.2f requested; high-quality time-stretch not configured. Returning original tempo.",
                props.speed,
            )

        # Voice selection: builtin speaker, or a custom voice cloned from wavs.
        speaker_wav = self.resolve_voice(props.voice)

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

                    # Generate chunk audio. A custom voice is cloned from its
                    # reference clips (speaker_wav); a builtin one is selected
                    # by name. Passing both would be ambiguous to Coqui.
                    if speaker_wav is not None:
                        wav = self.tts.tts(
                            text=chunk,
                            speaker_wav=speaker_wav,
                            language=lang_for_chunk,
                        )
                    else:
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
