"""
Language helper functions (script & Latin heuristics + optional langdetect).

This module **does not** depend on ASR/TTS. It only exposes pure utilities
that return **LanguageCode** enums to keep the rest of the stack consistent.
"""

from __future__ import annotations

from langdetect import detect as _ld_detect

from utils.language.language_codes import LanguageCode


def detect_lang(text: str) -> LanguageCode | None:
    """
    Try langdetect on text and map to LanguageCode.

    Returns:
        LanguageCode or None if detector is unavailable/indecisive.
    """
    if not text or len(text) < 8:
        return None
    try:
        code: str = str(_ld_detect(text) or "").lower()
        # normalize zh* → zh-cn, strip region suffixes (en-us → en)
        code = "zh-cn" if code.startswith("zh") else code.split("-")[0]
        return LanguageCode.from_string(code)
    except Exception:
        return None


def script_heuristic(s: str) -> LanguageCode | None:
    """
    Heuristic by Unicode script (robust for non-Latin texts).
    """
    for ch in s:
        o = ord(ch)
        # CJK
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            return LanguageCode.ZH_CN
        # Japanese (hiragana/katakana)
        if 0x3040 <= o <= 0x309F or 0x30A0 <= o <= 0x30FF:
            return LanguageCode.JA
        # Korean (Hangul)
        if 0x1100 <= o <= 0x11FF or 0xAC00 <= o <= 0xD7AF:
            return LanguageCode.KO
        # Arabic
        if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F:
            return LanguageCode.AR
        # Cyrillic (ru)
        if 0x0400 <= o <= 0x04FF:
            return LanguageCode.RU
        # Devanagari (hi)
        if 0x0900 <= o <= 0x097F:
            return LanguageCode.HI
    return None


def latin_heuristic(s: str) -> LanguageCode | None:
    """
    Rough cues for PT/ES/FR/DE/IT within Latin script.
    """
    st = (s or "").lower()
    # PT cues
    if " você" in st or "você " in st or "ção" in st or "ções" in st or "quê" in st:
        return LanguageCode.PT
    # ES cues
    if "¿" in s or "¡" in s or "ñ" in st or " por qué" in st or "qué " in st:
        return LanguageCode.ES
    # FR cues
    if "ç" in st or "œ" in st or (" aux " in st) or (" des " in st and " de " in st):
        return LanguageCode.FR
    # DE cues
    if "ß" in st or " ä" in st or " ö" in st or " ü" in st:
        return LanguageCode.DE
    # IT cues
    if (" gli " in st or " che " in st) and "zione" in st:
        return LanguageCode.IT
    return None
