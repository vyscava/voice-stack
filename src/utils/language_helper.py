"""
language_utils.py

Language utilities shared by TTS/ASR (primarily TTS):

Goals
-----
1) **Canonicalize language tags** so downstream components (e.g., XTTS v2)
   get exactly the codes they expect (e.g., "zh-cn" instead of "zh").
2) Provide **safe, optional language detection** using `langdetect` if it's
   installed. We never hard-require it; production systems often trim
   dependencies for size.
3) Add **fallback heuristics** when the detector is missing or unsure:
   - A Unicode **script heuristic** that is robust for non-Latin scripts
     (CJK, Arabic, Cyrillic, Devanagari, etc.).
   - A lightweight **Latin heuristic** to distinguish common Western languages
     (PT/ES/FR/DE/IT) based on characteristic glyphs and substrings.

Why this matters
----------------
- XTTS and similar multilingual models typically support a *fixed* list of
  language codes. Passing an unexpected or alias code (e.g., "en-us") can raise
  an error or yield incorrect prosody.
- `langdetect` sometimes returns broader tags (like "zh") or region variants
  ("pt-BR"). We normalize those to the canonical form required downstream.
- For short/ambiguous text, detectors can fail. Heuristics keep things usable.
"""

from __future__ import annotations

# Optional language detection
# We treat langdetect as an optional dependency:
# - If available, we'll try it for a first-pass guess.
# - If missing or inconclusive, we fall back to heuristics.
try:
    from langdetect import detect as _ld_detect  # type: ignore

    _LANGDETECT_OK = True
except Exception:  # pragma: no cover - optional dependency
    _LANGDETECT_OK = False


# ---------------------------------------------------------------------------
# Canonical codes and alias mapping
# ---------------------------------------------------------------------------

# The canonical codes the downstream TTS expects. Keep this aligned with the
# model's documented list (XTTS v2 as of writing).
CANONICAL_LANGS: set[str] = {
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "pl",
    "tr",
    "ru",
    "nl",
    "cs",
    "ar",
    "zh-cn",  # NOTE: XTTS expects 'zh-cn' instead of bare 'zh'
    "hu",
    "ko",
    "ja",
    "hi",
}

# A compact alias table that maps common user/file inputs to our canonical set.
# This helps:
#   - file naming (voices/MyVoice_pt-BR.wav -> "pt")
#   - user input (language="EN-US" -> "en")
LANG_ALIASES: dict[str, str] = {
    # English
    "en": "en",
    "eng": "en",
    "en-us": "en",
    "en_gb": "en",
    "en-uk": "en",
    "enus": "en",
    "us": "en",
    "uk": "en",
    # Portuguese
    "pt": "pt",
    "ptbr": "pt",
    "pt-br": "pt",
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
    # Important: normalize *any* zh-ish tag to 'zh-cn' because that’s what XTTS expects.
    "zh": "zh-cn",
    "zh-cn": "zh-cn",
    "zho": "zh-cn",
    "cn": "zh-cn",
    "zhcn": "zh-cn",
    # Others in the canonical set
    "hu": "hu",
    "ko": "ko",
    "ja": "ja",
    "hi": "hi",
}


def canon_lang(tag: str | None) -> str | None:
    """
    Normalize an arbitrary language tag to the canonical code expected by the model.

    Parameters
    ----------
    tag : str | None
        A user-provided or file-derived tag (e.g., 'EN-us', 'pt-BR', 'zh').

    Returns
    -------
    str | None
        - Canonical code (e.g., 'en', 'pt', 'zh-cn') if we can recognize it.
        - None if the input is missing or not recognized.

    Behavior
    --------
    - Lowercases and trims the input.
    - Applies alias mapping first (covers common variants like 'pt-br').
    - If not in aliases, accepts the tag only if it’s already one of the
      canonical codes in `CANONICAL_LANGS`.
    """
    if not tag:
        return None
    t = tag.strip().lower()
    return LANG_ALIASES.get(t, t if t in CANONICAL_LANGS else None)


def detect_lang(chunk: str) -> str | None:
    """
    Try to detect a language for a given text `chunk`, normalized to canonical form.

    Parameters
    ----------
    chunk : str
        The text to analyze. Very short inputs are unreliable.

    Returns
    -------
    str | None
        Canonical language code if detection succeeds and maps cleanly, else None.

    Notes
    -----
    - Requires `langdetect`; if unavailable or the text is too short (< 8 chars),
      returns None.
    - Normalizes 'zh*' to 'zh-cn' and trims region parts (e.g., 'pt-br' -> 'pt')
      because models typically expect 2-letter codes (with 'zh-cn' as the
      special-case exception).
    """
    if not _LANGDETECT_OK or len(chunk) < 8:
        return None
    try:
        # langdetect may yield region codes like 'pt-BR' or 'zh-cn'.
        code = (_ld_detect(chunk) or "").lower()

        # Normalize zh* -> zh-cn; otherwise cut region to just the base (e.g., 'pt-BR' -> 'pt').
        code = "zh-cn" if code.startswith("zh") else code.split("-")[0]

        # Accept only 2-letter codes *or* the special 'zh-cn'.
        return code if (len(code) == 2 or code == "zh-cn") else None
    except Exception:
        # Detection can fail on pathological inputs; treat as unknown.
        return None


def script_heuristic(s: str) -> str | None:
    """
    Guess language by Unicode script. Robust for non-Latin languages.

    Parameters
    ----------
    s : str
        The input text (ideally a full sentence or more).

    Returns
    -------
    str | None
        Canonical code for a *likely* language family tied to a script,
        or None if no script pattern matched.

    How it works
    ------------
    - Scans characters and checks their code points against known script ranges:
      * CJK Unified Ideographs       -> zh-cn
      * Hiragana/Katakana            -> ja
      * Hangul (Jamo/Syllables)      -> ko
      * Arabic                       -> ar
      * Cyrillic                     -> ru
      * Devanagari                   -> hi

    Caveats
    -------
    - Script != language. We pick a *likely* canonical target that maps well
      to model support, not a definitive linguistic label.
    """
    for ch in s:
        o = ord(ch)
        # CJK (Chinese/Japanese, but here we prefer 'zh-cn' as default target)
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            return "zh-cn"
        # Japanese (Hiragana/Katakana)
        if 0x3040 <= o <= 0x309F or 0x30A0 <= o <= 0x30FF:
            return "ja"
        # Korean (Hangul Jamo & Syllables)
        if 0x1100 <= o <= 0x11FF or 0xAC00 <= o <= 0xD7AF:
            return "ko"
        # Arabic
        if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F:
            return "ar"
        # Cyrillic (map to Russian as a practical default for XTTS)
        if 0x0400 <= o <= 0x04FF:
            return "ru"
        # Devanagari (map to Hindi)
        if 0x0900 <= o <= 0x097F:
            return "hi"
    return None


def latin_heuristic(s: str) -> str | None:
    """
    Rough cues for distinguishing PT/ES/FR/DE/IT in Latin script text.

    Parameters
    ----------
    s : str
        The input text.

    Returns
    -------
    str | None
        Canonical code ("pt", "es", "fr", "de", "it") if a cue matches, else None.

    Rationale
    ---------
    For Latin script, language detection can be shaky on short inputs.
    These lightweight cues are fast and good-enough for picking a TTS voice/phoneme set.

    Heuristics
    ----------
    - PT: diacritics and common substrings (você, ção, quê, …).
    - ES: inverted punctuation (¿ ¡), 'ñ', and phrasal cues ('por qué', 'qué ').
    - FR: 'ç', 'œ', and common stop-phrases (' aux ', ' des ' with ' de ').
    - DE: 'ß' and umlauts (ä, ö, ü).
    - IT: articles/particles (gli, che) combined with 'zione' endings.
    """
    st = s.lower()

    # Portuguese cues
    if " você" in st or "você " in st or "ção" in st or "ções" in st or "quê" in st:
        return "pt"

    # Spanish cues
    if "¿" in s or "¡" in s or "ñ" in st or " por qué" in st or "qué " in st:
        return "es"

    # French cues
    if "ç" in st or "œ" in st or (" aux " in st) or (" des " in st and " de " in st):
        return "fr"

    # German cues
    if "ß" in st or " ä" in st or " ö" in st or " ü" in st:
        return "de"

    # Italian cues
    if (" gli " in st or " che " in st) and "zione" in st:
        return "it"

    return None
