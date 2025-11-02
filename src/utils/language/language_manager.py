"""
language_manager.py — Glue layer that unifies language handling across the stack.

Exports:
- normalize_code(value) -> LanguageCode
- detect_language_from_text(text) -> LanguageCode | None
- detect_language_from_audio(audio_engine, ...) -> dict (schema-like result)

Notes:
- The audio-based detector is a *thin forwarder* to avoid circular imports.
  We accept any object with `detect_language(...) -> dict` (e.g., ASR engine).
"""

from __future__ import annotations

from typing import Any, cast

from utils.language_codes import LanguageCode
from utils.language_helper import detect_lang, latin_heuristic, script_heuristic


def normalize_code(value: str | None) -> LanguageCode:
    """
    Convert arbitrary user input into a LanguageCode. UNKNOWN if unsure.
    """
    return LanguageCode.from_string(value)


def detect_language_from_text(text: str) -> LanguageCode | None:
    """
    Try langdetect → script heuristic → Latin heuristic.
    Returns a LanguageCode or None when we really can't guess.
    """
    return detect_lang(text) or script_heuristic(text) or latin_heuristic(text)


def detect_language_from_audio(
    *,
    audio_engine: Any,
    file_bytes: bytes | None = None,
    audio_array: Any,
    sr: int | None = None,
    detect_lang_length: float | None = None,
    detect_lang_offset: float | None = None,
) -> dict[str, Any]:
    """
    Delegate to an engine that exposes: detect_language(file_bytes|audio_array, ...)

    This indirection keeps utils free from ASR engine imports while allowing
    the app layer to share a single function signature everywhere.
    """
    result = audio_engine.detect_language(
        file_bytes=file_bytes,
        audio_array=audio_array,
        sr=sr,
        detect_lang_length=detect_lang_length,
        detect_lang_offset=detect_lang_offset,
    )
    if not isinstance(result, dict):
        raise TypeError("audio_engine.detect_language must return a dict")
    return cast(dict[str, Any], result)
