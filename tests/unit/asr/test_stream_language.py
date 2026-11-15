"""Unit tests for the text language fallback of streaming finals, against the REAL lingua (fast: ~0.1 s build)."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("lingua")

from asr.stream.language import LINGUA_ISO_639_1, TextLanguageDetector, parse_languages  # noqa: E402
from tests.unit.asr.stream_fakes import text_detector  # noqa: E402


@pytest.mark.unit
@pytest.mark.asr
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What time is it in Lisbon", "en"),
        ("Please turn off the lights in the kitchen", "en"),
        ("Que horas são", "pt"),
        ("Oi Iris, tudo bem", "pt"),
        ("horas são em Lisboa", "pt"),
    ],
)
def test_detects_en_and_pt_on_short_turns(text: str, expected: str) -> None:
    assert text_detector().detect(text) == expected


@pytest.mark.unit
@pytest.mark.asr
@pytest.mark.parametrize("text", ["", "   ", ". ,", "Iris", "no", "Por favor"])
def test_empty_or_ambiguous_text_gives_no_verdict(text: str) -> None:
    assert text_detector().detect(text) is None


@pytest.mark.unit
@pytest.mark.asr
def test_only_the_configured_languages_can_be_answered() -> None:
    detector = TextLanguageDetector(("en", "de"))
    for text in ["Que horas são", "Oi Iris, tudo bem", "What time is it in Lisbon", "Wie spät ist es"]:
        assert detector.detect(text) in {"en", "de", None}
    assert detector.detect("Wie spät ist es in Lissabon") == "de"


@pytest.mark.unit
@pytest.mark.asr
def test_a_single_configured_language_answers_any_non_empty_text() -> None:
    detector = TextLanguageDetector(("pt",))
    assert detector.detect("What time is it") == "pt"
    assert detector.detect("") is None


@pytest.mark.unit
@pytest.mark.asr
def test_detection_is_fast() -> None:
    detector = text_detector()
    detector.detect("warm up")
    start = time.perf_counter()
    for _ in range(100):
        detector.detect("What time is it in Lisbon")
    assert (time.perf_counter() - start) / 100 < 0.005


@pytest.mark.unit
@pytest.mark.asr
def test_parse_languages_normalises_and_dedupes() -> None:
    assert parse_languages(" en, PT ,en,") == ("en", "pt")


@pytest.mark.unit
@pytest.mark.asr
@pytest.mark.parametrize("raw", ["", " , ", "en,xx", "english", "pt-BR"])
def test_parse_languages_rejects_unknown_or_empty(raw: str) -> None:
    with pytest.raises(ValueError, match="ASR_STREAM_LANGUAGES"):
        parse_languages(raw)


@pytest.mark.unit
@pytest.mark.asr
def test_supported_codes_match_the_installed_lingua() -> None:
    from lingua import Language

    assert {str(lang.iso_code_639_1.name).lower() for lang in Language.all()} == LINGUA_ISO_639_1
