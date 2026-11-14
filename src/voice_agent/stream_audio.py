"""Audio plumbing for the streaming STT: PCM16 conversion, the replay buffer, the stream URL."""

from __future__ import annotations

import sys
from array import array
from collections import deque

from livekit import rtc

SAMPLE_RATE = 16000
# Audio kept for replay into the batch STT if the stream fails: only what no
# final has answered yet, and never more than this, so a fallback does not
# replay stale room audio from long before the utterance.
REPLAY_SECONDS = 10.0
_REPLAY_BYTES = int(REPLAY_SECONDS * SAMPLE_RATE) * 2


def pcm16_mono(frame: rtc.AudioFrame) -> bytes:
    """The frame as mono PCM16 little-endian (channels averaged)."""
    samples = array("h")
    samples.frombytes(bytes(frame.data))
    n = frame.num_channels
    if n > 1:
        samples = array("h", (sum(samples[i : i + n]) // n for i in range(0, len(samples), n)))
    if sys.byteorder == "big":
        samples.byteswap()
    return samples.tobytes()


def frame_from_pcm16(data: bytes) -> rtc.AudioFrame:
    """A 16 kHz mono frame from PCM16 little-endian bytes."""
    if sys.byteorder == "big":
        samples = array("h")
        samples.frombytes(data)
        samples.byteswap()
        data = samples.tobytes()
    return rtc.AudioFrame(data=data, sample_rate=SAMPLE_RATE, num_channels=1, samples_per_channel=len(data) // 2)


def mono_frame(frame: rtc.AudioFrame) -> rtc.AudioFrame:
    """``frame`` downmixed to mono (unchanged if it already is), so replayed and live audio match."""
    return frame if frame.num_channels == 1 else frame_from_pcm16(pcm16_mono(frame))


class ReplayBuffer:
    """Audio not yet answered by a final: one segment per flush, REPLAY_SECONDS at most in total."""

    def __init__(self) -> None:
        self._segments: deque[bytearray] = deque([bytearray()])

    def add(self, data: bytes) -> None:
        self._segments[-1].extend(data)
        excess = sum(len(s) for s in self._segments) - _REPLAY_BYTES
        while excess > 0:  # drop the oldest audio first
            head = self._segments[0]
            cut = min(excess, len(head))
            del head[:cut]
            excess -= cut
            if not head and len(self._segments) > 1:
                self._segments.popleft()

    def flushed(self) -> None:
        self._segments.append(bytearray())

    def finalized(self) -> None:
        if len(self._segments) > 1:
            self._segments.popleft()
        else:
            self._segments[0].clear()

    def segments(self) -> list[tuple[bytes, bool]]:
        """Each segment's audio, and whether a flush (still unanswered) followed it."""
        last = len(self._segments) - 1
        return [(bytes(seg), i < last) for i, seg in enumerate(self._segments)]


def stream_url(asr_url: str, override: str = "") -> str:
    """The ``/v1/stream`` WebSocket URL: ``override`` if set, else derived from the ASR base URL."""
    if override.strip():
        return override.strip()
    base = asr_url.strip().rstrip("/")
    for http, ws in (("https://", "wss://"), ("http://", "ws://")):
        if base.startswith(http):
            base = ws + base[len(http) :]
            break
    return base + "/v1/stream"
