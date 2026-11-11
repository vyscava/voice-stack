"""Silero VAD on faster-whisper's bundled ONNX model (voice-stack#13).

These run the REAL model where they can: the point of the change is that VAD no
longer needs torch, and a mocked VAD would pass whether or not that is true.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest

from utils.audio.audio_helper import SILERO_VAD_DEFAULTS, apply_vad_silero, load_silero_model

SRC = Path(__file__).resolve().parents[4] / "src"
log = logging.getLogger("test-vad")


@pytest.mark.unit
def test_load_silero_model_returns_a_working_onnx_session() -> None:
    model, ts_fn = load_silero_model(log)

    assert model is not None and ts_fn is not None
    # The real session runs, and one second of silence is not speech.
    assert ts_fn(np.zeros(16000, dtype=np.float32), model, sampling_rate=16000) == []


@pytest.mark.unit
def test_vad_path_never_imports_torch() -> None:
    """The slim image has no torch. Prove the VAD path does not need it.

    A fresh interpreter with torch made unimportable (sys.modules[name] = None
    makes `import torch` raise ImportError) must still load and run VAD. Before
    #13 load_silero_model imported torch and silero_vad, and returned (None,
    None) here: transcription silently lost VAD.
    """
    code = textwrap.dedent(f"""
        import logging, sys
        for name in ("torch", "torchaudio", "silero_vad"):
            sys.modules[name] = None
        sys.path.insert(0, {str(SRC)!r})
        import numpy as np
        from utils.audio.audio_helper import load_silero_model
        model, fn = load_silero_model(logging.getLogger("x"))
        assert model is not None and fn is not None, "VAD did not load without torch"
        assert fn(np.zeros(16000, dtype=np.float32), model, sampling_rate=16000) == []
        assert "torch" not in [m for m in sys.modules if sys.modules[m] is not None]
        print("ok")
        """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


@pytest.mark.unit
def test_timestamps_fn_passes_silero_defaults_not_faster_whisper_defaults() -> None:
    """faster-whisper's own VadOptions default to 2 s silence / 400 ms padding.

    The service used silero-vad's defaults before #13; swapping the runtime must
    not silently change what gets trimmed.
    """
    _, ts_fn = load_silero_model(log)
    captured: dict[str, Any] = {}

    def fake(audio: Any, vad_options: Any = None, sampling_rate: int = 16000) -> list[dict[str, int]]:
        captured["opts"] = vad_options
        return []

    with patch("faster_whisper.vad.get_speech_timestamps", side_effect=fake):
        ts_fn(np.zeros(16000, dtype=np.float32), None, sampling_rate=16000)

    opts = captured["opts"]
    assert opts.min_silence_duration_ms == SILERO_VAD_DEFAULTS["min_silence_duration_ms"] == 100
    assert opts.speech_pad_ms == SILERO_VAD_DEFAULTS["speech_pad_ms"] == 30
    assert opts.min_speech_duration_ms == SILERO_VAD_DEFAULTS["min_speech_duration_ms"] == 250
    assert opts.threshold == SILERO_VAD_DEFAULTS["threshold"] == 0.5


@pytest.mark.unit
def test_apply_vad_stitches_speech_and_reports_seconds() -> None:
    # 3 s at 16 kHz: silence, "speech" (constant 0.5), silence.
    audio = np.zeros(48000, dtype=np.float32)
    audio[16000:32000] = 0.5
    pcm16 = (audio * 32767).astype(np.int16)
    ts_fn = Mock(return_value=[{"start": 16000, "end": 32000}])

    out, segs = apply_vad_silero(pcm16=pcm16, sr=16000, silero_model=object(), get_speech_timestamps_fn=ts_fn, log=log)

    assert segs == [{"start": 1.0, "end": 2.0}]
    assert out.dtype == np.int16
    assert out.shape == (16000,)
    assert abs(int(out[0]) - int(pcm16[16000])) <= 1
    # The VAD function receives float32 in [-1, 1] at 16 kHz.
    passed = ts_fn.call_args.args[0]
    assert passed.dtype == np.float32 and passed.shape == (48000,)
    assert ts_fn.call_args.kwargs == {"sampling_rate": 16000}


@pytest.mark.unit
def test_apply_vad_resamples_to_16k_first() -> None:
    pcm16 = np.zeros(22050, dtype=np.int16)  # 1 s at 22.05 kHz
    ts_fn = Mock(return_value=[])

    apply_vad_silero(pcm16=pcm16, sr=22050, silero_model=object(), get_speech_timestamps_fn=ts_fn, log=log)

    assert ts_fn.call_args.args[0].shape == (16000,)


@pytest.mark.unit
def test_apply_vad_returns_input_unchanged_on_error() -> None:
    pcm16 = np.arange(1000, dtype=np.int16)
    ts_fn = Mock(side_effect=RuntimeError("onnx exploded"))

    out, segs = apply_vad_silero(pcm16=pcm16, sr=16000, silero_model=object(), get_speech_timestamps_fn=ts_fn, log=log)

    assert out is pcm16
    assert segs == []


@pytest.mark.unit
def test_apply_vad_without_model_is_a_no_op() -> None:
    pcm16 = np.arange(10, dtype=np.int16)
    out, segs = apply_vad_silero(pcm16=pcm16, sr=16000, silero_model=None, get_speech_timestamps_fn=None, log=log)
    assert out is pcm16 and segs == []


@pytest.mark.unit
def test_load_failure_is_logged_at_error_and_returns_none(caplog: pytest.LogCaptureFixture) -> None:
    with patch("faster_whisper.vad.get_vad_model", side_effect=RuntimeError("asset missing")):
        with caplog.at_level(logging.ERROR, logger="test-vad"):
            model, fn = load_silero_model(log)
    assert (model, fn) == (None, None)
    assert any("WITHOUT VAD" in r.message and r.levelno == logging.ERROR for r in caplog.records)


@pytest.mark.unit
@pytest.mark.asr
def test_asr_engine_constructs_with_vad_disabled() -> None:
    """Regression: with ASR_VAD_ENABLED=false the engine raised AttributeError
    (silero_model was only assigned inside the `if`, then read unconditionally)."""
    from asr.engine.fasterwhisper import ASRFasterWhisper

    with (
        patch("asr.engine.fasterwhisper.WhisperModel"),
        patch("asr.engine.base.settings") as base_settings,
    ):
        base_settings.ASR_VAD_ENABLED = False
        base_settings.ASR_CACHE_ENABLED = False
        base_settings.ASR_CACHE_MAX_ITEMS = 8
        base_settings.ASR_IDLE_TIMEOUT_MINUTES = 0
        engine = ASRFasterWhisper()

    assert engine.silero_model is None and engine.silero_ts_fn is None
    audio = np.ones(100, dtype=np.float32) * 0.1
    out, used, segs = engine.helper_apply_vad(audio_f32=audio)
    assert used is False and segs == [] and out is audio
