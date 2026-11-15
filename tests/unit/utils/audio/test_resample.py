"""resample_to_16k_mono: resampy only, never torchaudio (end of line at 2.11)."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from utils.audio.audio_helper import resample_to_16k_mono


def _tone(freq: float, sr: int, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _peak_hz(x: np.ndarray, sr: int) -> float:
    spectrum = np.abs(np.fft.rfft(x))
    return float(np.fft.rfftfreq(len(x), 1 / sr)[int(np.argmax(spectrum))])


@pytest.mark.parametrize("sr", [8000, 22050, 24000, 44100, 48000])
def test_resamples_to_16k_keeping_length_and_pitch(sr):
    y, out_sr = resample_to_16k_mono(audio_f32=_tone(440.0, sr), sr=sr)
    assert out_sr == 16000
    assert y.dtype == np.float32
    assert abs(len(y) - 16000) <= 2  # one second stays one second
    assert abs(_peak_hz(y, 16000) - 440.0) < 2.0  # same pitch, no aliasing to another bin


def test_stereo_is_averaged_to_mono():
    left = _tone(440.0, 48000)
    stereo = np.stack([left, left], axis=1)
    y, _ = resample_to_16k_mono(audio_f32=stereo, sr=48000)
    assert y.ndim == 1 and abs(len(y) - 16000) <= 2


def test_16k_input_passes_through_unchanged():
    x = _tone(300.0, 16000)
    y, sr = resample_to_16k_mono(audio_f32=x, sr=16000)
    assert sr == 16000 and np.array_equal(y, x)


def test_torchaudio_is_never_imported(monkeypatch):
    """A broken or missing torchaudio (the end-of-line risk) cannot affect ASR resampling."""
    monkeypatch.setitem(sys.modules, "torchaudio", None)  # any import now raises ImportError
    y, sr = resample_to_16k_mono(audio_f32=_tone(440.0, 44100), sr=44100)
    assert sr == 16000 and abs(len(y) - 16000) <= 2
