"""
Streaming ASR: the small interface the WebSocket handler talks to.

The handler in ``asr.api.api_v1.endpoints.stream`` only ever sees the two
protocols below, never a model. That keeps the protocol testable with a fake
engine (no download, no torch) and keeps the model code in one place
(``asr.stream.nemotron``).

Threading contract: every method on these protocols is BLOCKING and may do
model work. The handler always calls them through a thread executor, never on
the event loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

# The protocol is fixed at 16 kHz mono PCM16, the rate the model was trained on.
STREAM_SAMPLE_RATE = 16000

# Values processor.set_num_lookahead_tokens() accepts for the default model
# ("supported_num_lookahead_tokens" in its processor config). 3 = 320 ms chunks.
SUPPORTED_LOOKAHEAD = frozenset({0, 3, 6, 13})
SUPPORTED_DTYPES = frozenset({"bfloat16", "float32"})

# A trailing locale tag the model appends in language="auto" mode, e.g.
# "What time is it? <en-US>". Only tags WITH a region are accepted, so special
# tokens such as <pad> or <unk> can never be mistaken for a language.
_LANG_TAG_RE = re.compile(r"<([A-Za-z]{2,3})-[A-Za-z0-9]{2,4}>")

AudioArray = npt.NDArray[np.float32]


def parse_language_tag(text: str | None) -> str | None:
    """Return the primary subtag of the last ``<xx-YY>`` tag in ``text``.

    ``"What time? <en-US>"`` gives ``"en"``, ``"Olá. <pt-BR>"`` gives ``"pt"``,
    and text without a tag gives ``None``. Pure, so it is tested directly.
    """
    if not text:
        return None
    matches = _LANG_TAG_RE.findall(text)
    if not matches:
        return None
    return str(matches[-1]).lower()


def pcm16_to_float32(data: bytes) -> AudioArray:
    """Little-endian PCM16 bytes to float32 samples in [-1, 1). Needs an even length."""
    samples: AudioArray = np.frombuffer(data, dtype="<i2").astype(np.float32)
    samples /= np.float32(32768.0)
    return samples


@dataclass(frozen=True)
class StreamConfig:
    """Effective streaming settings, validated once from the ASR_STREAM_* variables."""

    enabled: bool
    model: str
    lookahead: int
    dtype: str
    threads: int
    device: str

    @classmethod
    def from_settings(cls, settings: Any) -> StreamConfig:
        """Build and validate a config from the application settings.

        ASR_STREAM_THREADS <= 0 falls back to ASR_CPU_THREADS, so one knob sizes
        both engines unless the operator splits them on purpose.
        """
        lookahead = int(settings.ASR_STREAM_LOOKAHEAD)
        if lookahead not in SUPPORTED_LOOKAHEAD:
            raise ValueError(
                f"ASR_STREAM_LOOKAHEAD={lookahead} is not supported; use one of {sorted(SUPPORTED_LOOKAHEAD)}"
            )
        dtype = str(settings.ASR_STREAM_DTYPE).strip().lower()
        if dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"ASR_STREAM_DTYPE={dtype!r} is not supported; use one of {sorted(SUPPORTED_DTYPES)}")
        threads = int(settings.ASR_STREAM_THREADS)
        if threads <= 0:
            threads = int(settings.ASR_CPU_THREADS)
        return cls(
            enabled=bool(settings.ASR_STREAM_ENABLED),
            model=str(settings.ASR_STREAM_MODEL),
            lookahead=lookahead,
            dtype=dtype,
            threads=threads,
            device=str(settings.ASR_DEVICE).lower(),
        )


class StreamSession(Protocol):
    """One utterance being transcribed incrementally."""

    def feed(self, samples: AudioArray) -> str:
        """Append 16 kHz float32 samples; return the full hypothesis so far."""
        ...

    def finish(self) -> str:
        """Pad silence so the last chunk is decoded, end the utterance, return the final text."""
        ...

    def close(self) -> None:
        """Abandon the utterance (client went away). Must be safe to call more than once."""
        ...


class StreamEngine(Protocol):
    """A loaded streaming model. One per process."""

    def new_session(self) -> StreamSession:
        """Start a new utterance."""
        ...

    def detect_language(self, samples: AudioArray) -> str | None:
        """Batch-decode ``samples`` with automatic language detection; return e.g. ``"en"`` or None."""
        ...
