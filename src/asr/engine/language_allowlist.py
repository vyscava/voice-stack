"""Restrict Whisper's language detection to the languages the speakers actually use.

Whisper picks one of ~100 languages for every utterance. On short or quiet turns
("thanks", "obrigado") the small models regularly pick a language nobody spoke,
and the transcript comes back in that language. With ``ASR_LANGUAGES`` set, a
detection outside the list is replaced by the most probable listed language and
the audio is decoded again in it. Turns detected correctly cost nothing extra.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from prometheus_client import Counter

LANGUAGE_OVERRIDES = Counter(
    "asr_language_overrides_total",
    "Transcriptions whose detected language was outside ASR_LANGUAGES and was replaced.",
    ["detected", "chosen"],
)

_CODE = re.compile(r"^[a-z]{2,3}$")


def parse_allowed_languages(raw: str | None) -> tuple[str, ...]:
    """``"en, PT"`` gives ``("en", "pt")``; blank gives ``()`` (any language).

    Raises ValueError on a code that is not a bare ISO 639-1/639-3 code, so a typo
    fails at startup instead of silently allowing nothing.
    """
    codes = tuple(dict.fromkeys(code.strip().lower() for code in (raw or "").split(",") if code.strip()))
    bad = [code for code in codes if not _CODE.match(code)]
    if bad:
        raise ValueError(f"ASR_LANGUAGES has invalid code(s) {bad}; use comma-separated ISO codes, e.g. en,pt")
    return codes


def best_allowed_language(
    probabilities: Iterable[tuple[str, float]] | None, allowed: tuple[str, ...]
) -> tuple[str, float]:
    """The listed language with the highest detection probability, and that probability.

    Falls back to the first listed language (probability 0.0) when the engine gave
    no probabilities or none of them is listed.
    """
    best: tuple[str, float] | None = None
    for code, prob in probabilities or ():
        if code in allowed and (best is None or prob > best[1]):
            best = (code, float(prob))
    return best if best is not None else (allowed[0], 0.0)


def needs_override(detected: str | None, allowed: tuple[str, ...]) -> bool:
    """True when a list is set and the detected language is not on it."""
    return bool(allowed) and (detected or "").lower() not in allowed
