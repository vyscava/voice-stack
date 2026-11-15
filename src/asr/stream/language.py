"""
Text-based language identification for streaming finals.

The streaming model emits a language tag only after a sentence's final
punctuation, and it never punctuates the last sentence of an utterance, so a
one-sentence turn usually ends without a tag. For those, the final transcript's
text is classified with lingua (lingua-language-detector, Apache-2.0),
restricted to the configured languages (ASR_STREAM_LANGUAGES).

lingua is imported lazily: it lives in the optional ``stream`` extra, and
validating the configuration must work without it.
"""

from __future__ import annotations

from typing import Any

# ISO 639-1 codes lingua 2.x can detect (lingua.Language.all()). Kept as data so
# ASR_STREAM_LANGUAGES can be validated without importing lingua; a unit test
# checks it against the installed lingua.
LINGUA_ISO_639_1 = frozenset(
    "af ar az be bg bn bs ca cs cy da de el en eo es et eu fa fi fr ga gu he hi hr hu hy id is it ja ka kk ko "
    "la lg lt lv mi mk mn mr ms nb nl nn pa pl pt ro ru sk sl sn so sq sr st sv sw ta te th tl tn tr ts uk ur "
    "vi xh yo zh zu".split()
)

# A text verdict counts only when the winning language's confidence reaches this.
# Measured on short turns: "What time is it in Lisbon" scores 0.77 English and
# "Oi Iris, tudo bem" 0.87 Portuguese, while "Iris" (0.52), "no" (0.63) and
# "Por favor" (0.57) are too ambiguous to trust over the previous turn's language.
MIN_TEXT_CONFIDENCE = 0.7


def parse_languages(raw: str) -> tuple[str, ...]:
    """``"en, PT"`` gives ``("en", "pt")``. Raises ValueError on an empty list or a code lingua cannot detect."""
    codes = tuple(dict.fromkeys(code.strip().lower() for code in raw.split(",") if code.strip()))
    if not codes:
        raise ValueError("ASR_STREAM_LANGUAGES is empty; list at least one ISO 639-1 code, e.g. en,pt")
    unknown = [code for code in codes if code not in LINGUA_ISO_639_1]
    if unknown:
        raise ValueError(f"ASR_STREAM_LANGUAGES has unsupported code(s) {unknown}; use ISO 639-1 codes lingua knows")
    return codes


class TextLanguageDetector:
    """lingua restricted to ``languages``; built once per process (about 0.1 s, models preloaded)."""

    def __init__(self, languages: tuple[str, ...], min_confidence: float = MIN_TEXT_CONFIDENCE) -> None:
        self.languages = languages
        self.min_confidence = min_confidence
        self._detector: Any = None
        if len(languages) > 1:  # lingua needs two candidates; with one there is nothing to decide
            from lingua import IsoCode639_1, Language, LanguageDetectorBuilder

            members = [Language.from_iso_code_639_1(IsoCode639_1.from_str(code)) for code in languages]
            self._detector = LanguageDetectorBuilder.from_languages(*members).with_preloaded_language_models().build()

    def detect(self, text: str) -> str | None:
        """The configured language ``text`` is written in, or None when it is empty or ambiguous."""
        if not text or not any(ch.isalpha() for ch in text):
            return None
        if self._detector is None:
            return self.languages[0]
        values = self._detector.compute_language_confidence_values(text)
        if not values or values[0].value < self.min_confidence:
            return None
        return str(values[0].language.iso_code_639_1.name).lower()
