"""TTS engine selection: the registry behind TTS_ENGINE.

The old factory had two branches that both built Coqui, so any value -- a typo,
"kokoro" before it existed, an empty string -- silently booted XTTS. These tests
pin the replacement: the default is Kokoro, aliases resolve, an unknown name
fails, and an engine whose dependencies are missing from the image fails with
the image that has them. Nothing ever substitutes a different engine.
"""

from __future__ import annotations

import pytest

from tts import engine_factory
from tts.exceptions import EngineUnavailableError


@pytest.fixture(autouse=True)
def _reset_engine_singleton():
    engine_factory._engine = None
    engine_factory._semaphore = None
    yield
    engine_factory._engine = None
    engine_factory._semaphore = None


def _set_engine_setting(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    settings = engine_factory.get_settings()
    monkeypatch.setattr(settings, "TTS_ENGINE", value)


@pytest.mark.unit
@pytest.mark.tts
@pytest.mark.parametrize("raw", ["", None, "   "])
def test_empty_engine_means_kokoro(raw: str | None) -> None:
    assert engine_factory.resolve_engine_name(raw) == "kokoro"


@pytest.mark.unit
@pytest.mark.tts
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("kokoro", "kokoro"), ("KOKORO", "kokoro"), ("coqui", "coqui"), ("xtts", "coqui"), ("xtts-2", "coqui")],
)
def test_names_and_aliases_resolve(raw: str, expected: str) -> None:
    assert engine_factory.resolve_engine_name(raw) == expected


@pytest.mark.unit
@pytest.mark.tts
def test_unknown_engine_raises_and_lists_the_known_ones() -> None:
    with pytest.raises(EngineUnavailableError) as exc:
        engine_factory.resolve_engine_name("piper")
    assert "piper" in exc.value.message
    assert "kokoro" in (exc.value.details or "")
    assert "coqui" in (exc.value.details or "")


@pytest.mark.unit
@pytest.mark.tts
def test_unknown_engine_is_never_replaced_by_coqui(monkeypatch: pytest.MonkeyPatch) -> None:
    """The previous factory returned TTSCoqui() for any unrecognised value."""
    _set_engine_setting(monkeypatch, "kokro")  # typo
    imported: list[str] = []
    monkeypatch.setattr(engine_factory.importlib, "import_module", lambda name: imported.append(name))
    with pytest.raises(EngineUnavailableError):
        engine_factory.get_audio_engine()
    assert imported == []


@pytest.mark.unit
@pytest.mark.tts
def test_missing_dependency_names_the_image_that_has_it(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_engine_setting(monkeypatch, "coqui")

    def _no_torch(name: str):
        raise ImportError("No module named 'torch'", name="torch")

    monkeypatch.setattr(engine_factory.importlib, "import_module", _no_torch)
    with pytest.raises(EngineUnavailableError) as exc:
        engine_factory.get_audio_engine()
    assert "coqui" in exc.value.message
    assert "voice-stack:xtts" in (exc.value.details or "")
    assert "tts-xtts" in (exc.value.details or "")
    assert "torch" in (exc.value.details or "")


@pytest.mark.unit
@pytest.mark.tts
def test_selected_engine_class_is_built(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_engine_setting(monkeypatch, "xtts")

    class FakeEngine:
        pass

    class FakeModule:
        TTSCoqui = FakeEngine

    seen: list[str] = []

    def _import(name: str):
        seen.append(name)
        return FakeModule

    monkeypatch.setattr(engine_factory.importlib, "import_module", _import)
    engine = engine_factory.get_audio_engine()
    assert isinstance(engine, FakeEngine)
    assert seen == ["tts.engine.coqui"]
