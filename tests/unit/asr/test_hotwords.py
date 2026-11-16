"""ASR_TRANSCRIBE_HOTWORDS: vocabulary biasing for faster-whisper."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from asr.engine import base
from asr.engine.base import normalize_hotwords, transcribe_effective_options
from asr.engine.fasterwhisper import ASRFasterWhisper

pytestmark = [pytest.mark.unit, pytest.mark.asr]


def _options() -> base.CacheConf:
    return transcribe_effective_options(
        request_language=None,
        task=None,
        beam_size=None,
        temperature=None,
        best_of=None,
        word_timestamps=None,
        vad=None,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("   \n\t ", None),
        ("Iris, Plex", "Iris, Plex"),
        ("  Iris,\n  Plex,\tNAS  ", "Iris, Plex, NAS"),
    ],
)
def test_normalize_hotwords(raw: str | None, expected: str | None) -> None:
    assert normalize_hotwords(raw) == expected


def test_options_carry_the_configured_hotwords(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base.settings, "ASR_TRANSCRIBE_HOTWORDS", "Iris,\n Plex")
    assert _options().hotwords == "Iris, Plex"


def test_options_have_no_hotwords_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base.settings, "ASR_TRANSCRIBE_HOTWORDS", None)
    assert _options().hotwords is None


def test_hotwords_are_part_of_the_cache_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # A cached transcript made without the word list must not answer a request made with it.
    monkeypatch.setattr(base.settings, "ASR_TRANSCRIBE_HOTWORDS", None)
    without = _options()
    monkeypatch.setattr(base.settings, "ASR_TRANSCRIBE_HOTWORDS", "Iris")
    assert _options() != without


def _transcribe_kwargs(monkeypatch: pytest.MonkeyPatch, hotwords: str | None) -> dict[str, object]:
    monkeypatch.setattr(base.settings, "ASR_TRANSCRIBE_HOTWORDS", hotwords)
    with (
        patch("asr.engine.fasterwhisper.WhisperModel") as model_cls,
        patch("asr.engine.fasterwhisper.settings") as engine_settings,
    ):
        engine_settings.ASR_CACHE_ENABLED = False
        model = Mock()
        model.transcribe.return_value = ([], Mock(language="en", language_probability=0.9))
        model_cls.return_value = model
        engine = ASRFasterWhisper()
        engine._transcribe_core(
            audio_f32=np.zeros(16000, dtype=np.float32),
            sr=16000,
            raw_bytes=None,
            request_language=None,
            task=None,
            beam_size=None,
            temperature=None,
            best_of=None,
            word_timestamps=None,
            vad=False,
        )
        return dict(model.transcribe.call_args.kwargs)


def test_engine_passes_hotwords_to_faster_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _transcribe_kwargs(monkeypatch, "Iris, Plex")["hotwords"] == "Iris, Plex"


def test_engine_passes_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # None, not "": faster-whisper only skips the prompt block for a falsy value,
    # and None keeps the call identical to the pre-hotwords behaviour.
    assert _transcribe_kwargs(monkeypatch, "  ")["hotwords"] is None
