"""A soft ticking sound while the caller waits for the brain's answer.

After the acknowledgement ("Okay.") the brain may need another second or
two. Silence there sounds like a dropped call, and invites the caller to
talk over the reply. A quiet "tuk ... tuk ... tuk" says something is
loading.

LiveKit's built-in thinking sound only plays in the ``thinking`` state; once
the ack has been spoken the agent is ``speaking`` for the whole reply, so it
would never fill this gap. The ticks therefore play on LiveKit's background
audio track, started and stopped by the worker around the brain's first
words. The sound is generated here: no audio file to ship.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any, Protocol

import numpy as np
from livekit import rtc

SAMPLE_RATE = 48000  # BackgroundAudioPlayer mixes at 48 kHz mono
FRAME_MS = 20


def tick_samples(*, interval_s: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """One period: a short muted knock (a decaying low tone), then silence."""
    period = max(int(interval_s * sample_rate), int(0.05 * sample_rate))
    knock_n = int(0.035 * sample_rate)
    t = np.arange(knock_n) / sample_rate
    knock = np.sin(2 * np.pi * 520 * t) * np.exp(-t / 0.008)
    out = np.zeros(period, dtype=np.float32)
    out[:knock_n] = knock.astype(np.float32)
    return out


async def tick_frames(*, interval_s: float, sample_rate: int = SAMPLE_RATE) -> AsyncIterator[rtc.AudioFrame]:
    """Endless 20 ms int16 frames of the tick period, looped (the mixer paces them)."""
    pcm = (tick_samples(interval_s=interval_s, sample_rate=sample_rate) * 0.9 * 32767).astype(np.int16)
    step = sample_rate * FRAME_MS // 1000
    pos = 0
    while True:
        chunk = np.take(pcm, range(pos, pos + step), mode="wrap")
        pos = (pos + step) % len(pcm)
        yield rtc.AudioFrame(chunk.tobytes(), sample_rate, 1, step)
        await asyncio.sleep(0)


class Ticker(Protocol):
    def start(self, delay_s: float | None = None) -> None: ...

    def stop(self) -> None: ...

    def stop_later(self, delay_s: float) -> None: ...


class NullTicker:
    """Ticks turned off (and the default in tests)."""

    def start(self, delay_s: float | None = None) -> None:
        return None

    def stop(self) -> None:
        return None

    def stop_later(self, delay_s: float) -> None:
        return None


class BackgroundTicker:
    """Starts the ticks ``delay_s`` after ``start()`` (so they do not overlap the
    ack's own audio) and stops them on ``stop()``. Both are idempotent."""

    def __init__(self, player: Any, *, delay_s: float, volume: float, interval_s: float) -> None:
        self._player = player
        self._delay_s = delay_s
        self._volume = volume
        self._interval_s = interval_s
        self._pending: asyncio.Task[None] | None = None
        self._handle: Any = None
        self._deadline: asyncio.TimerHandle | None = None

    def start(self, delay_s: float | None = None) -> None:
        """Tick after ``delay_s`` (default: the configured delay), unless already on."""
        self._cancel_deadline()
        if self._pending is not None or self._handle is not None:
            return
        self._pending = asyncio.ensure_future(self._start_later(self._delay_s if delay_s is None else delay_s))

    async def _start_later(self, delay_s: float) -> None:
        await asyncio.sleep(delay_s)
        from livekit.agents.voice.background_audio import AudioConfig

        cfg = AudioConfig(tick_frames(interval_s=self._interval_s), volume=self._volume, fade_out=0.08)
        self._handle = self._player.play(cfg)
        self._pending = None

    def stop_later(self, delay_s: float) -> None:
        """Safety net: stop after ``delay_s`` unless stopped (or restarted) first."""
        self._cancel_deadline()
        self._deadline = asyncio.get_running_loop().call_later(delay_s, self.stop)

    def _cancel_deadline(self) -> None:
        if self._deadline is not None:
            self._deadline.cancel()
            self._deadline = None

    def stop(self) -> None:
        self._cancel_deadline()
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
        if self._handle is not None:
            with contextlib.suppress(Exception):
                self._handle.stop()
            self._handle = None
