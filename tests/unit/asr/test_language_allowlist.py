"""ASR_LANGUAGES: language detection limited to the languages the speakers use."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest

from asr.engine.fasterwhisper import ASRFasterWhisper
from asr.engine.language_allowlist import LANGUAGE_OVERRIDES, best_allowed_language, parse_allowed_languages

pytestmark = [pytest.mark.unit, pytest.mark.asr]

# A short "obrigado" the CPU model heard as Russian, with Portuguese a close second.
RU_FIRST = [("ru", 0.48), ("pt", 0.31), ("en", 0.12), ("uk", 0.05)]


def _segment(text: str) -> Mock:
    seg = Mock(start=0.0, end=1.0, text=text)
    seg.words = None
    return seg


def _info(language: str, probability: float, probs: list[tuple[str, float]] | None = None) -> Mock:
    return Mock(language=language, language_probability=probability, all_language_probs=probs)


@contextmanager
def _engine(languages: str, model: Mock) -> Iterator[ASRFasterWhisper]:
    with (
        patch("asr.engine.fasterwhisper.WhisperModel", return_value=model),
        patch("asr.engine.fasterwhisper.settings") as fw_settings,
        patch("asr.engine.base.settings") as base_settings,
    ):
        for s in (fw_settings, base_settings):
            s.ASR_LANGUAGES = languages
            s.ASR_DEVICE = "cpu"
            s.ASR_VAD_ENABLED = False
            s.ASR_CACHE_ENABLED = False
            s.ASR_TRANSCRIBE_LANG = None
            s.ASR_TRANSCRIBE_BEAM_SIZE = 1
            s.ASR_TRANSCRIBE_TEMPERATURE = 0.0
            s.ASR_TRANSCRIBE_BEST_OF = 1
            s.ASR_TRANSCRIBE_HOTWORDS = None
            s.ASR_ENGINE = "fasterwhisper"
            s.ASR_MODEL = "small"
        yield ASRFasterWhisper()


def _transcribe(engine: ASRFasterWhisper, request_language: str | None = None) -> Any:
    return engine._transcribe_core(
        audio_f32=np.zeros(16000, dtype=np.float32),
        sr=16000,
        raw_bytes=None,
        request_language=request_language,
        task=None,
        beam_size=None,
        temperature=None,
        best_of=None,
        word_timestamps=None,
        vad=None,
    )


def test_an_unlisted_detection_is_decoded_again_in_the_most_likely_listed_language() -> None:
    model = Mock()
    model.transcribe.side_effect = [
        ([_segment(" Спасибо")], _info("ru", 0.48, RU_FIRST)),
        ([_segment(" Obrigado")], _info("pt", 1.0)),
    ]
    before = LANGUAGE_OVERRIDES.labels(detected="ru", chosen="pt")._value.get()
    with _engine("en,pt", model) as engine:
        result = _transcribe(engine)

    assert [c.kwargs["language"] for c in model.transcribe.call_args_list] == [None, "pt"]
    assert result.text == "Obrigado"
    assert result.language_code == "pt"
    assert result.confidence == pytest.approx(0.31)  # detection's view of pt, not the forced decode's 1.0
    assert LANGUAGE_OVERRIDES.labels(detected="ru", chosen="pt")._value.get() == before + 1


def test_a_listed_detection_decodes_once() -> None:
    model = Mock()
    model.transcribe.return_value = ([_segment(" Thanks")], _info("en", 0.9, [("en", 0.9), ("pt", 0.05)]))
    with _engine("en,pt", model) as engine:
        result = _transcribe(engine)

    assert model.transcribe.call_count == 1
    assert result.text == "Thanks"
    assert result.language_code == "en"


def test_a_blank_list_keeps_any_language() -> None:
    model = Mock()
    model.transcribe.return_value = ([_segment(" Спасибо")], _info("ru", 0.48, RU_FIRST))
    with _engine("", model) as engine:
        result = _transcribe(engine)

    assert model.transcribe.call_count == 1
    assert result.language_code == "ru"


def test_a_language_the_caller_asked_for_is_never_replaced() -> None:
    model = Mock()
    model.transcribe.return_value = ([_segment(" Hola")], _info("es", 1.0))
    with _engine("en,pt", model) as engine:
        _transcribe(engine, request_language="es")

    assert [c.kwargs["language"] for c in model.transcribe.call_args_list] == ["es"]


def test_detect_language_answers_with_the_most_likely_listed_language() -> None:
    model = Mock()
    model.detect_language.return_value = ("ru", 0.48, RU_FIRST)
    with _engine("en,pt", model) as engine:
        result = engine._detect_language_core(
            audio_f32=np.zeros(16000, dtype=np.float32),
            sr=16000,
            raw_bytes=None,
            request_language=None,
        )

    assert result.language_code == "pt"
    assert result.confidence == pytest.approx(0.31)


@pytest.mark.parametrize("languages", ["en,portuguese", "en,jp", "xx"])
def test_a_bad_code_stops_the_service_at_startup(languages: str) -> None:
    with pytest.raises(ValueError, match="ASR_LANGUAGES"), _engine(languages, Mock()):
        pass


def test_parse_lowercases_dedupes_and_blank_means_any() -> None:
    assert parse_allowed_languages(" en, PT ,en,") == ("en", "pt")
    assert parse_allowed_languages("") == ()
    assert parse_allowed_languages(None) == ()


def test_best_allowed_falls_back_to_the_first_listed_language() -> None:
    assert best_allowed_language(None, ("pt", "en")) == ("pt", 0.0)
    assert best_allowed_language([("ru", 0.9), ("uk", 0.1)], ("pt", "en")) == ("pt", 0.0)
    assert best_allowed_language([("en", 0.2), ("pt", 0.3)], ("en", "pt")) == ("pt", 0.3)


@pytest.mark.parametrize("compose", ["asr.yml", "asr-gpu.yml"])
def test_both_batch_deploys_limit_detection_to_english_and_portuguese(compose: str) -> None:
    from pathlib import Path

    import yaml

    deploy = Path(__file__).resolve().parents[3] / "compose" / "deploy" / compose
    (service,) = yaml.safe_load(deploy.read_text(encoding="utf-8"))["services"].values()
    assert parse_allowed_languages(service["environment"]["ASR_LANGUAGES"]) == ("en", "pt")
