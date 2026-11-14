"""Unit tests for the streaming ASR helpers: language-tag parsing, PCM decoding, config validation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from asr.stream.base import StreamConfig, parse_language_tag, pcm16_to_float32


@pytest.mark.unit
@pytest.mark.asr
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What time? <en-US>", "en"),
        ("Que horas são? <pt-BR>", "pt"),
        ("<pt-BR>", "pt"),
        ("What time?", None),
        ("", None),
        (None, None),
        # Special tokens around the tag are not languages; the last real tag wins.
        ("Hallo. <de-DE></s><pad>", "de"),
        ("<unk> mixed <en-US> then <es-ES>", "es"),
        ("<pad> no locale here <unk>", None),
    ],
)
def test_parse_language_tag(text: str | None, expected: str | None) -> None:
    assert parse_language_tag(text) == expected


@pytest.mark.unit
@pytest.mark.asr
def test_pcm16_to_float32_scales_little_endian_samples() -> None:
    data = np.array([0, 16384, -32768, 32767], dtype="<i2").tobytes()
    samples = pcm16_to_float32(data)
    assert samples.dtype == np.float32
    assert samples.tolist() == pytest.approx([0.0, 0.5, -1.0, 32767 / 32768])


def _settings(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "ASR_STREAM_ENABLED": False,
        "ASR_STREAM_MODEL": "nvidia/nemotron-3.5-asr-streaming-0.6b",
        "ASR_STREAM_LOOKAHEAD": 3,
        "ASR_STREAM_DTYPE": "bfloat16",
        "ASR_STREAM_THREADS": 0,
        "ASR_CPU_THREADS": 8,
        "ASR_DEVICE": "cpu",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.unit
@pytest.mark.asr
def test_stream_config_defaults_come_from_real_settings() -> None:
    """The shipped defaults: off, the Nemotron model, 320 ms lookahead, float32, ASR_CPU_THREADS."""
    from core.settings import Settings

    settings = Settings()
    config = StreamConfig.from_settings(settings)
    assert config.enabled is False
    assert config.model == "nvidia/nemotron-3.5-asr-streaming-0.6b"
    assert config.lookahead == 3
    assert config.dtype == "float32"
    assert config.threads == settings.ASR_CPU_THREADS


@pytest.mark.unit
@pytest.mark.asr
def test_stream_config_threads_fall_back_to_asr_cpu_threads() -> None:
    assert StreamConfig.from_settings(_settings(ASR_STREAM_THREADS=0, ASR_CPU_THREADS=6)).threads == 6
    assert StreamConfig.from_settings(_settings(ASR_STREAM_THREADS=2, ASR_CPU_THREADS=6)).threads == 2


@pytest.mark.unit
@pytest.mark.asr
@pytest.mark.parametrize("lookahead", [0, 3, 6, 13])
def test_stream_config_accepts_supported_lookahead(lookahead: int) -> None:
    assert StreamConfig.from_settings(_settings(ASR_STREAM_LOOKAHEAD=lookahead)).lookahead == lookahead


@pytest.mark.unit
@pytest.mark.asr
@pytest.mark.parametrize("lookahead", [1, 2, 4, 14])
def test_stream_config_rejects_unsupported_lookahead(lookahead: int) -> None:
    with pytest.raises(ValueError, match="ASR_STREAM_LOOKAHEAD"):
        StreamConfig.from_settings(_settings(ASR_STREAM_LOOKAHEAD=lookahead))


@pytest.mark.unit
@pytest.mark.asr
def test_stream_config_dtype_validation() -> None:
    assert StreamConfig.from_settings(_settings(ASR_STREAM_DTYPE="Float32")).dtype == "float32"
    with pytest.raises(ValueError, match="ASR_STREAM_DTYPE"):
        StreamConfig.from_settings(_settings(ASR_STREAM_DTYPE="float16"))
