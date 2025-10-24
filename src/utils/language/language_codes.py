"""
LanguageCode — a shared canonical enum for ASR/TTS.

Why this exists
---------------
Both XTTS (TTS) and Whisper-style ASR accept short language codes, but not
always with the same expectations. We standardize here and expose helpers
to convert arbitrary inputs (e.g., "EN-US", "pt-br", "zh") into a single
canonical form (e.g., "en", "pt", "zh-cn").

Usage
-----
from utils.language_codes import LanguageCode
LanguageCode.from_string("en-us").canonical == "en"
LanguageCode.from_string("ZH").canonical == "zh-cn"
"""

from __future__ import annotations

from enum import Enum

DISPLAY_BY_CODE: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "tr": "Turkish",
    "ru": "Russian",
    "nl": "Dutch",
    "cs": "Czech",
    "ar": "Arabic",
    "zh-cn": "Chinese (Simplified)",
    "hu": "Hungarian",
    "ko": "Korean",
    "ja": "Japanese",
    "hi": "Hindi",
    "unknown": "Unknown",
}
LANG_ALIASES: dict[str, str] = {
    # English
    "en": "en",
    "en-us": "en",
    "en_uk": "en",
    "en-uk": "en",
    "eng": "en",
    "us": "en",
    "uk": "en",
    # Portuguese
    "pt": "pt",
    "pt-br": "pt",
    "ptbr": "pt",
    "pt_br": "pt",
    "br": "pt",
    # Spanish / French / etc.
    "es": "es",
    "spa": "es",
    "fr": "fr",
    "fra": "fr",
    "de": "de",
    "ger": "de",
    "it": "it",
    "ita": "it",
    "pl": "pl",
    "tr": "tr",
    "ru": "ru",
    "nl": "nl",
    "cs": "cs",
    "ar": "ar",
    # Chinese
    "zh": "zh-cn",
    "zh-cn": "zh-cn",
    "zhcn": "zh-cn",
    "zho": "zh-cn",
    "cn": "zh-cn",
    # Others
    "hu": "hu",
    "ko": "ko",
    "ja": "ja",
    "hi": "hi",
}


class LanguageCode(str, Enum):
    # Canonical XTTS & Whisper-friendly codes
    EN = "en"
    ES = "es"
    FR = "fr"
    DE = "de"
    IT = "it"
    PT = "pt"
    PL = "pl"
    TR = "tr"
    RU = "ru"
    NL = "nl"
    CS = "cs"
    AR = "ar"
    ZH_CN = "zh-cn"
    HU = "hu"
    KO = "ko"
    JA = "ja"
    HI = "hi"

    UNKNOWN = "unknown"  # when detection fails or is unset

    @property
    def canonical(self) -> str:
        """Canonical short code as string (what models expect)."""
        return str(self.value)

    @property
    def display_name(self) -> str:
        """Human-friendly display name."""
        # return self.__class__._DISPLAY.get(self, "Unknown")
        return DISPLAY_BY_CODE.get(self.value, "Unknown")

    @classmethod
    def from_string(cls, value: str | None) -> LanguageCode:
        """
        Convert arbitrary user/model input into a canonical LanguageCode.

        Returns:
            LanguageCode.UNKNOWN if parsing fails or input is None.
        """
        if not value:
            return cls.UNKNOWN
        # Normalize zh → zh-cn, en-us → en, etc.
        norm = LANG_ALIASES.get(value.strip().lower(), value.strip().lower())
        for member in cls:
            if member.value == norm:
                return member
        return cls.UNKNOWN
