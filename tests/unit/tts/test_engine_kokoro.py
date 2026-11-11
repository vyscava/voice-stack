"""Kokoro engine: voices, language choice, speech, and loading.

The engine is driven with a stub in place of the ONNX model, the same way the
Coqui tests avoid loading XTTS: TTSKokoro.__init__ calls _load_model, so most
tests build the object with __new__ and set the few attributes a method reads.
_load_model itself is tested against fake onnxruntime / kokoro_onnx modules.
"""

from __future__ import annotations

import io
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from tts.engine.kokoro import TTSKokoro, voice_language
from tts.exceptions import ModelLoadError

VOICES = ["af_heart", "am_adam", "bf_emma", "pf_dora", "pm_alex", "jf_alpha", "zf_xiaobei"]


class StubKokoro:
    """Records create() calls and returns a short 24 kHz tone."""

    def __init__(self, voices: list[str] | None = None, sample_rate: int = 24000) -> None:
        self._voices = voices if voices is not None else VOICES
        self.sample_rate = sample_rate
        self.calls: list[dict[str, Any]] = []

    def get_voices(self) -> list[str]:
        return list(self._voices)

    def create(self, text: str, voice: str, speed: float = 1.0, lang: str = "en-us") -> tuple[np.ndarray, int]:
        self.calls.append({"text": text, "voice": voice, "speed": speed, "lang": lang})
        t = np.linspace(0, 0.25, int(self.sample_rate * 0.25), endpoint=False)
        return (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), self.sample_rate


def _engine(stub: StubKokoro | None = None, *, force: str | None = None, default: str = "af_heart") -> TTSKokoro:
    eng = TTSKokoro.__new__(TTSKokoro)
    eng.kokoro = stub or StubKokoro()
    eng.voices = sorted(v for v in eng.kokoro.get_voices() if voice_language(v) is not None)
    eng.default_voice = default
    eng.force_language = force
    eng.sample_rate = 24000
    eng.model_path = Path("/models/kokoro-v1.0.onnx")
    eng.idle_timeout_minutes = 0
    eng._model_loaded = True
    return eng


# --------------------------------------------------------------------- voices


@pytest.mark.unit
@pytest.mark.tts
@pytest.mark.parametrize(
    ("voice", "lang"),
    [("af_heart", "en-us"), ("bm_george", "en-gb"), ("pf_dora", "pt-br"), ("ef_dora", "es"), ("if_sara", "it")],
)
def test_voice_language_from_prefix(voice: str, lang: str) -> None:
    assert voice_language(voice) == lang


@pytest.mark.unit
@pytest.mark.tts
@pytest.mark.parametrize("voice", ["jf_alpha", "zf_xiaobei", "default", "x", "pxdora", ""])
def test_voice_language_unsupported_or_malformed(voice: str) -> None:
    assert voice_language(voice) is None


@pytest.mark.unit
@pytest.mark.tts
def test_list_voices_hides_languages_espeak_cannot_phonemize() -> None:
    ids = [v.id for v in _engine().list_voices().data]
    assert "pf_dora" in ids and "af_heart" in ids
    assert "jf_alpha" not in ids and "zf_xiaobei" not in ids


@pytest.mark.unit
@pytest.mark.tts
@pytest.mark.parametrize("voice", [None, "", "default"])
def test_default_voice_alias(voice: str | None) -> None:
    assert _engine(default="pf_dora").resolve_voice(voice) == "pf_dora"


@pytest.mark.unit
@pytest.mark.tts
def test_unknown_voice_raises_instead_of_substituting() -> None:
    with pytest.raises(KeyError):
        _engine().resolve_voice("Claribel Dervla")  # an XTTS speaker name


@pytest.mark.unit
@pytest.mark.tts
def test_hidden_voice_is_unknown_even_though_the_model_has_it() -> None:
    with pytest.raises(KeyError):
        _engine().resolve_voice("jf_alpha")


# ------------------------------------------------------------------- language


@pytest.mark.unit
@pytest.mark.tts
def test_language_defaults_to_the_voices_own() -> None:
    assert _engine().choose_lang("pf_dora", None) == "pt-br"


@pytest.mark.unit
@pytest.mark.tts
def test_requested_language_wins_over_the_voice() -> None:
    assert _engine().choose_lang("pf_dora", "en") == "en-us"


@pytest.mark.unit
@pytest.mark.tts
def test_force_language_wins_over_everything() -> None:
    assert _engine(force="pt").choose_lang("af_heart", "en") == "pt-br"


@pytest.mark.unit
@pytest.mark.tts
def test_unsupported_requested_language_falls_back_to_the_voice() -> None:
    assert _engine().choose_lang("af_heart", "ja") == "en-us"


# --------------------------------------------------------------------- speech


@pytest.mark.unit
@pytest.mark.tts
def test_speech_passes_voice_language_and_speed_to_kokoro() -> None:
    stub = StubKokoro()
    _engine(stub).speech(input="Olá Vitor", voice="pf_dora", response_format="wav", speed=1.25)
    assert stub.calls == [{"text": "Olá Vitor", "voice": "pf_dora", "speed": 1.25, "lang": "pt-br"}]


@pytest.mark.unit
@pytest.mark.tts
@pytest.mark.parametrize(("asked", "sent"), [(4.0, 2.0), (0.25, 0.5)])
def test_speed_is_clamped_to_kokoros_range(asked: float, sent: float) -> None:
    stub = StubKokoro()
    _engine(stub).speech(input="hi", voice="af_heart", response_format="wav", speed=asked)
    assert stub.calls[0]["speed"] == sent


@pytest.mark.unit
@pytest.mark.tts
def test_speech_returns_a_real_wav_at_24k() -> None:
    audio = _engine().speech(input="hello", voice="af_heart", response_format="wav")
    data, sr = sf.read(io.BytesIO(audio))
    assert sr == 24000
    assert len(data) == 6000  # the stub's 0.25 s


@pytest.mark.unit
@pytest.mark.tts
def test_speech_pcm_is_headerless() -> None:
    audio = _engine().speech(input="hello", voice="af_heart", response_format="pcm")
    assert len(audio) == 6000 * 2  # int16 mono, no RIFF header
    assert not audio.startswith(b"RIFF")


@pytest.mark.unit
@pytest.mark.tts
def test_speech_sse_yields_base64_events() -> None:
    events = list(_engine().speech(input="hello", voice="af_heart", response_format="wav", stream_format="sse"))
    assert events and all(e.startswith(b"data: ") and e.endswith(b"\n\n") for e in events)


@pytest.mark.unit
@pytest.mark.tts
def test_speech_with_unknown_voice_raises_keyerror() -> None:
    stub = StubKokoro()
    with pytest.raises(KeyError):
        _engine(stub).speech(input="hello", voice="alloy", response_format="wav")
    assert stub.calls == []


@pytest.mark.unit
@pytest.mark.tts
def test_sample_rate_mismatch_is_an_error_not_a_mislabelled_file() -> None:
    with pytest.raises(ModelLoadError):
        _engine(StubKokoro(sample_rate=22050)).speech(input="hello", voice="af_heart", response_format="wav")


@pytest.mark.unit
@pytest.mark.tts
def test_unloaded_model_is_reloaded_before_speech(monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine()
    stub = eng.kokoro
    eng._unload_model()
    eng._model_loaded = False
    assert eng.kokoro is None

    def _reload() -> None:
        eng.kokoro = stub

    monkeypatch.setattr(eng, "_load_model", _reload)
    eng.speech(input="hello", voice="af_heart", response_format="wav")
    assert stub.calls and eng._model_loaded


# ---------------------------------------------------------------------- loading


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fake onnxruntime + kokoro_onnx so _load_model runs without the real model."""
    seen: dict[str, Any] = {}

    class SessionOptions:
        intra_op_num_threads = 0

    def InferenceSession(path: str, sess_options: Any = None, providers: Any = None) -> object:
        seen["session"] = {"path": path, "threads": sess_options.intra_op_num_threads, "providers": providers}
        return object()

    class Kokoro:
        @classmethod
        def from_session(cls, session: object, voices_path: str) -> StubKokoro:
            seen["voices_path"] = voices_path
            return StubKokoro()

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        types.SimpleNamespace(SessionOptions=SessionOptions, InferenceSession=InferenceSession),
    )
    monkeypatch.setitem(sys.modules, "kokoro_onnx", types.SimpleNamespace(Kokoro=Kokoro))
    return seen


def _unloaded(
    tmp_path: Path, *, model: bool = True, voices: bool = True, default: str = "af_heart", threads: int = 0
) -> TTSKokoro:
    eng = TTSKokoro.__new__(TTSKokoro)
    eng.model_path = tmp_path / "kokoro.onnx"
    eng.voices_path = tmp_path / "voices.bin"
    if model:
        eng.model_path.write_bytes(b"x")
    if voices:
        eng.voices_path.write_bytes(b"x")
    eng.default_voice = default
    eng.threads = threads
    eng.kokoro = None
    eng.voices = []
    return eng


@pytest.mark.unit
@pytest.mark.tts
def test_load_builds_a_cpu_session_with_the_configured_threads(tmp_path: Path, fake_runtime: dict[str, Any]) -> None:
    eng = _unloaded(tmp_path, threads=4)
    eng._load_model()
    assert fake_runtime["session"] == {"path": str(eng.model_path), "threads": 4, "providers": ["CPUExecutionProvider"]}
    assert fake_runtime["voices_path"] == str(eng.voices_path)
    assert "pf_dora" in eng.voices and "jf_alpha" not in eng.voices


@pytest.mark.unit
@pytest.mark.tts
def test_zero_threads_leaves_the_runtime_default(tmp_path: Path, fake_runtime: dict[str, Any]) -> None:
    _unloaded(tmp_path, threads=0)._load_model()
    assert fake_runtime["session"]["threads"] == 0


@pytest.mark.unit
@pytest.mark.tts
@pytest.mark.parametrize(("model", "voices", "what"), [(False, True, "model"), (True, False, "voices")])
def test_missing_files_fail_with_the_path(
    tmp_path: Path, fake_runtime: dict[str, Any], model: bool, voices: bool, what: str
) -> None:
    eng = _unloaded(tmp_path, model=model, voices=voices)
    with pytest.raises(ModelLoadError) as exc:
        eng._load_model()
    assert what in exc.value.message
    assert "session" not in fake_runtime


@pytest.mark.unit
@pytest.mark.tts
def test_default_voice_must_exist(tmp_path: Path, fake_runtime: dict[str, Any]) -> None:
    with pytest.raises(ModelLoadError) as exc:
        _unloaded(tmp_path, default="Claribel Dervla")._load_model()
    assert "Claribel Dervla" in (exc.value.details or "")
