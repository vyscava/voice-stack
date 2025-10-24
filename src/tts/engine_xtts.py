# engine_xtts.py
from __future__ import annotations

import glob
import io
import logging
import math
import os
import re

import numpy as np
import soundfile as sf
from torch.serialization import add_safe_globals
from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig

# --- Types from Coqui TTS that may appear in pickles (safe to register) ---
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig

add_safe_globals(
    [
        XttsConfig,
        XttsAudioConfig,
        XttsArgs,
        BaseDatasetConfig,
        BaseAudioConfig,
    ]
)

from TTS.api import TTS
from utils.language_helper import (
    CANONICAL_LANGS,
    canon_lang,
    detect_lang,
    latin_heuristic,
    script_heuristic,
)

# Shared utils
from utils.text import build_safe_chunks, normalize_text

# --------------------------
# Tunables (via environment)
# --------------------------
MAX_CHARS = int(os.getenv("TTS_MAX_CHARS", "180"))  # conservative under XTTS ~250 warn
MIN_CHARS = int(os.getenv("TTS_MIN_CHARS", "70"))  # can dip below 80 if needed
RETRY_STEPS = int(os.getenv("TTS_RETRY_STEPS", "2"))

DEFAULT_LANG = (os.getenv("TTS_DEFAULT_LANG", "en") or "en").strip()
AUTO_LANG = os.getenv("TTS_AUTO_LANG", "0").lower() in {"1", "true", "yes"}
LANG_HINT = os.getenv("TTS_LANG_HINT", "").strip()  # soft preference if supported
FORCE_LANG = os.getenv("TTS_FORCE_LANG", "").strip()  # hard override for all chunks


class XTTSSynth:
    """
    XTTS wrapper providing:
      - Robust text cleanup, conservative sentence splitting, and chunking (utils.text).
      - GPU-assert backoff by shrinking chunk size automatically.
      - Per-chunk language selection:
          * FORCE_LANG (env) > request `language` > LANG_HINT (env) > autodetect > DEFAULT_LANG.
          * Detection uses langdetect (if installed) with script/Latin heuristics fallback (utils.lang).
      - Language-specific **voice cloning samples**:
          * Place files in voices/ as:
              voices/<VOICE>.wav                  (generic)
              voices/<VOICE>_<lang>.wav           (e.g., _pt, _en, _zh-cn)
            The code prefers the exact language match, else falls back to generic.
          * You can also **force a variant** with voice_id like "Bethania:pt".
      - Builtin speakers supported if no cloning sample is found.
    """

    # ------------------------------ init -----------------------------------------
    def __init__(self, device: str, model_name: str, voices_dir: str, sample_rate: int, log: logging.Logger):
        self.log = log.getChild("xtts")
        self.device = device
        self.sample_rate = sample_rate
        self.voices_dir = voices_dir
        os.makedirs(self.voices_dir, exist_ok=True)

        model_id = model_name
        if model_id in {"xtts_v2", "XTTS-v2"}:
            model_id = "tts_models/multilingual/multi-dataset/xtts_v2"

        self.log.info("Loading XTTS model_id=%s on %s", model_id, device)
        self.tts = TTS(model_name=model_id, progress_bar=False).to(self.device)

        # Builtin speakers (if the model exposes them)
        self.voice_to_preset: dict[str, str] = {}
        if hasattr(self.tts, "speakers") and isinstance(self.tts.speakers, list):
            for name in self.tts.speakers:
                self.voice_to_preset[name] = name

        # Supported languages reported by the model (preferred),
        # else fall back to a known list.
        try:
            cfg_langs = getattr(getattr(self.tts, "tts_model", None), "config", None).languages  # type: ignore
            self.supported_langs = {("zh-cn" if x == "zh" else x) for x in (cfg_langs or [])}
            if not self.supported_langs:
                self.supported_langs = set(CANONICAL_LANGS)
        except Exception:
            self.supported_langs = set(CANONICAL_LANGS)

        # Pre-scan custom voice WAVs (including per-language variants)
        self.custom_map = self._scan_custom_voices()

        self.log.info(
            "XTTS ready | auto_lang=%s | default=%s | hint=%s | force=%s | supported=%s",
            AUTO_LANG,
            DEFAULT_LANG,
            LANG_HINT or "-",
            FORCE_LANG or "-",
            sorted(self.supported_langs),
        )

    # ------------------------ voice file discovery -------------------------------
    def _scan_custom_voices(self) -> dict[str, dict[str | None, str]]:
        """
        Build a map: { base_id: {None: base_wav, 'en': wav_en, 'pt': wav_pt, ... } }
        Accepts file name patterns like:
        - Bethania.wav                -> base (None)
        - Bethania_pt.wav             -> 'pt'
        - Bethania-pt.wav             -> 'pt'
        - Bethania.pt.wav             -> 'pt'
        - Bethania (pt).wav           -> 'pt'
        Language codes are lowercased and we normalize 'zh' -> 'zh-cn'.
        """
        patterns = [
            r"^(?P<base>.+?)[ _\-\.](?P<lang>[A-Za-z]{2}(?:-[A-Za-z]{2})?)\.wav$",
            r"^(?P<base>.+?)\s*\((?P<lang>[A-Za-z]{2}(?:-[A-Za-z]{2})?)\)\.wav$",
        ]
        rx = [re.compile(p) for p in patterns]
        out: dict[str, dict[str | None, str]] = {}
        for path in glob.glob(os.path.join(self.voices_dir, "*.wav")):
            fname = os.path.basename(path)
            base, lang = None, None
            m = None
            for r in rx:
                m = r.match(fname)
                if m:
                    break
            if m:
                base = m.group("base").strip()
                tag = m.group("lang").lower()
                lang = "zh-cn" if tag.startswith("zh") else tag.split("-")[0]
            else:
                base = os.path.splitext(fname)[0]
                lang = None

            bucket = out.setdefault(base, {})
            if lang not in bucket:
                bucket[lang] = os.path.join(self.voices_dir, fname)
        return out

    def _resolve_sample_for_lang(self, voice_id: str, language: str | None) -> tuple[str | None, str | None]:
        """
        Choose the best cloning sample path for (voice_id, language).
        Priority:
          1) exact language match (canonicalized)
          2) base file without language suffix
        Returns (path, matched_lang) or (None, None) if not found.
        """
        lang = canon_lang(language) if language else None
        variants = self.custom_map.get(voice_id) or {}
        if lang and lang in variants:
            return variants[lang], lang
        if None in variants:
            return variants[None], None
        return None, None

    # ----------------------------- public API ------------------------------------
    def list_voices(self) -> list[dict[str, str]]:
        """
        List available voices:
        - Custom voices (base + per-language) from voices_dir.
        - Builtin model speakers (if any).
        Variants appear as IDs like "Bethania:pt".
        """
        voices: list[dict[str, str]] = []

        # Custom: base first, then variants (avoid sorting None with str)
        for vid in sorted(self.custom_map.keys()):
            variants = self.custom_map[vid]
            if None in variants:
                voices.append(
                    {
                        "id": vid,
                        "name": f"{vid} (custom)",
                        "sample": variants[None],
                        "type": "custom",
                    }
                )
            lang_keys = [k for k in variants.keys() if k is not None]
            lang_keys.sort()
            for lang in lang_keys:
                voices.append(
                    {
                        "id": f"{vid}:{lang}",
                        "name": f"{vid} ({lang})",
                        "sample": variants[lang],
                        "type": "custom",
                    }
                )

        # Builtins
        built_in = getattr(self.tts, "speakers", None)
        if built_in:
            for speaker in built_in:
                voices.append({"id": speaker, "name": speaker, "sample": "", "type": "builtin"})

        if not voices:
            voices.append({"id": "en_US_generic", "name": "en_US_generic", "sample": "", "type": "default"})
        return voices

    # ----------------------- language selection ----------------------------------
    def _choose_lang(self, chunk: str, requested: str | None) -> str:
        """
        Decide a supported language code for a text chunk.
        Priority:
          1) FORCE_LANG (env) if set & supported
          2) explicit 'requested' argument if supported (request language)
          3) LANG_HINT (env) if supported
          4) AUTO detection (langdetect) → clamp to supported
             - if unsupported: script heuristic (non-Latin)
             - else Latin heuristic (PT/ES/FR/DE/IT cues)
          5) DEFAULT_LANG if supported else first supported language
        """
        # 1) Force override
        if FORCE_LANG and FORCE_LANG in self.supported_langs:
            if requested and requested != FORCE_LANG:
                self.log.info(
                    "tts.xtts | overriding requested lang=%s with FORCE_LANG=%s",
                    requested,
                    FORCE_LANG,
                )
            return "zh-cn" if FORCE_LANG == "zh" else FORCE_LANG

        # 2) Request argument
        if requested and requested in self.supported_langs:
            return "zh-cn" if requested == "zh" else requested

        # 3) Hint
        if LANG_HINT and LANG_HINT in self.supported_langs:
            return LANG_HINT

        # 4) Autodetect
        if AUTO_LANG:
            detected = detect_lang(chunk)
            if detected in self.supported_langs:
                return detected

            # Script heuristic (robust for non-Latin)
            script_guess = script_heuristic(chunk)
            if script_guess in self.supported_langs:
                self.log.info(
                    "tts.xtts | langdetect=%s unsupported -> script_guess=%s",
                    detected,
                    script_guess,
                )
                return script_guess

            # Latin heuristic (quick PT/ES/FR/DE/IT cues)
            latin_guess = latin_heuristic(chunk)
            if latin_guess in self.supported_langs:
                self.log.info("tts.xtts | langdetect=%s unsupported -> latin_guess=%s", detected, latin_guess)
                return latin_guess

            if detected:
                self.log.info("tts.xtts | langdetect=%s unsupported -> falling back", detected)

        # 5) Default
        if DEFAULT_LANG in self.supported_langs:
            return DEFAULT_LANG
        # Last resort: any supported
        return next(iter(self.supported_langs))

    # ----------------------- single inference call --------------------------------
    def _tts_once(self, chunk: str, preset_list: list[str], sample_path: str | None, language: str) -> np.ndarray:
        if sample_path and os.path.isfile(sample_path):
            wav = self.tts.tts(text=chunk, speaker_wav=sample_path, language=language)
            return np.asarray(wav, dtype=np.float32)

        if not preset_list:
            raise ValueError(
                "No cloning sample and no preset speakers available. "
                "Add voices/<voice>.wav or choose a listed builtin voice."
            )

        speaker = self.voice_to_preset.get(self.active_voice_id, preset_list[0])
        wav = self.tts.tts(text=chunk, speaker=speaker, language=language)
        return np.asarray(wav, dtype=np.float32)

    def _resolve_sample_path(self, voice_id: str) -> str | None:
        """
        Map a voice_id to a concrete WAV path if it's a custom voice.
        Supports:
        - "Bethania"        -> custom_map['Bethania'][None] (if present)
        - "Bethania:pt"     -> custom_map['Bethania']['pt'] (if present)
        If not found, returns None (i.e., use builtin preset if available).
        """
        base, lang = (voice_id.split(":", 1) + [None])[:2]
        variants = self.custom_map.get(base)
        if not variants:
            return None
        if lang:
            return variants.get(lang)
        return variants.get(None)

    # ----------------------------- Synthesize --------------------------------------
    def synth(self, text: str, voice_id: str, fmt: str, language: str | None = None) -> bytes:
        """
        Synthesize speech.

        Args:
            text: Raw input text (can be multilingual).
            voice_id:
                - Custom voice ID (e.g., "Bethania") or composite "Bethania:pt".
                - Builtin speaker name (if model exposes presets).
            fmt: Output format. Currently only 'wav' (PCM_16).
            language: Optional requested language code for all chunks.
                      If None and AUTO_LANG=1, language is picked per chunk.

        Behavior:
            - Text is normalized, sentence-split, and chunked (<= MAX_CHARS).
            - For each chunk:
                * Choose language via _choose_lang (unless FORCE_LANG).
                * Resolve the best cloning sample for that language:
                    voices/<VOICE>_<lang>.wav > voices/<VOICE>.wav
                  or fall back to builtin speaker if no sample exists.
            - If CUDA 'index out of range' / device-side asserts occur,
              chunk budget is reduced and the synthesis is retried (up to RETRY_STEPS).
        """
        self.active_voice_id = voice_id
        _ = self._resolve_sample_path(voice_id)  # pre-warm existence checks
        preset_list = getattr(self.tts, "speakers", None) or []

        # Support composite ID: "Voice:pt" to hard-pick a variant
        base_vid, forced_lang = (voice_id.split(":", 1) + [None])[:2]
        forced_lang = canon_lang(forced_lang)

        # Build initial chunks
        size = MAX_CHARS
        chunks = build_safe_chunks(text, size)

        attempts = 0
        while True:
            try:
                wavs: list[np.ndarray] = []
                for chunk in chunks:
                    chunk = normalize_text(chunk)
                    # pick language for this chunk
                    lang_for_chunk = forced_lang or self._choose_lang(chunk, language)
                    if lang_for_chunk == "zh":  # normalize just in case
                        lang_for_chunk = "zh-cn"

                    # resolve best sample for (base_vid, lang_for_chunk)
                    sample_path, _matched = self._resolve_sample_for_lang(base_vid, lang_for_chunk)

                    # if no custom sample -> try builtin speaker using the original voice_id
                    wav = self._tts_once(
                        chunk=chunk,
                        preset_list=preset_list,
                        sample_path=sample_path,
                        language=lang_for_chunk,
                    )
                    wavs.append(wav)

                # success
                wav = wavs[0] if len(wavs) == 1 else np.concatenate(wavs, axis=0)
                break

            except Exception as e:
                msg = str(e)
                retriable = any(t in msg for t in ("index out of range", "device-side assert", "CUDA error", "cudnn"))
                if (not retriable) or attempts >= RETRY_STEPS or size <= MIN_CHARS:
                    raise
                attempts += 1
                new_size = max(MIN_CHARS, math.floor(size * 0.66))
                if new_size == size:
                    new_size = max(MIN_CHARS, size - 5)
                size = new_size
                self.log.warning(
                    "tts.xtts | XTTS retry: shrinking char budget to %d (attempt %d)",
                    size,
                    attempts,
                )
                chunks = build_safe_chunks(text, size)

        if fmt.lower() != "wav":
            raise ValueError("Only 'wav' is supported right now.")
        buf = io.BytesIO()
        sf.write(buf, wav, self.sample_rate, subtype="PCM_16", format="WAV")
        return buf.getvalue()
