"""MP3 output carries a real bitrate: ffmpeg's own default for 24 kHz mono is 32 kbps."""

from __future__ import annotations

import io
import json
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from core.settings import Settings
from utils.audio.ffmpeg_helper import _encode_with_cli, encode_audio_from_wav_bytes

pytestmark = pytest.mark.skipif(not shutil.which("ffprobe"), reason="ffmpeg/ffprobe not installed")


def _wav(seconds: float = 2.0, sr: int = 24000) -> bytes:
    rng = np.random.default_rng(0)
    x = (0.3 * rng.standard_normal(int(sr * seconds))).astype(np.float32)  # broadband, so the encoder uses its budget
    buf = io.BytesIO()
    sf.write(buf, x, sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def _kbps(mp3: bytes) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=bit_rate", "-of", "json", "-"],
        input=mp3,
        capture_output=True,
        check=True,
    )
    return int(json.loads(out.stdout)["streams"][0]["bit_rate"]) / 1000


def test_mp3_defaults_to_128_kbps_not_ffmpegs_32():
    assert _kbps(encode_audio_from_wav_bytes(wav_bytes=_wav(), target_format="mp3")) == pytest.approx(128, abs=8)


def test_mp3_bitrate_is_configurable():
    mp3 = encode_audio_from_wav_bytes(wav_bytes=_wav(), target_format="mp3", mp3_bitrate="64k")
    assert _kbps(mp3) == pytest.approx(64, abs=6)


def test_the_cli_fallback_uses_the_same_bitrate():
    assert _kbps(_encode_with_cli(_wav(), "mp3", 24000, 1, "96k")) == pytest.approx(96, abs=8)


def test_settings_default_and_validation():
    assert Settings().TTS_MP3_BITRATE == "128k"
    assert Settings(TTS_MP3_BITRATE="96k").TTS_MP3_BITRATE == "96k"
    for bad in ("128", "abc", "0k", "1000k"):
        with pytest.raises(ValidationError):
            Settings(TTS_MP3_BITRATE=bad)
