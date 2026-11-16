"""FallbackBatchSTT: primary first, the same audio to the fallback on failure, then a cooldown."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from livekit.agents import stt
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer

from voice_agent import metrics
from voice_agent.fallback_stt import FallbackBatchSTT
from voice_agent.stream_audio import frame_from_pcm16


class FakeSTT(stt.STT[Any]):
    def __init__(self, name: str, *, fail: Exception | None = None, delay_s: float = 0.0) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self.name = name
        self.fail = fail
        self.delay_s = delay_s
        self.calls: list[tuple[AudioBuffer, APIConnectOptions]] = []

    async def _recognize_impl(
        self, buffer: AudioBuffer, *, language: NotGivenOr[str] = NOT_GIVEN, conn_options: APIConnectOptions
    ) -> stt.SpeechEvent:
        self.calls.append((buffer, conn_options))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail is not None:
            raise self.fail
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language="en", text=f"from {self.name}")],
        )


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _audio() -> AudioBuffer:
    return frame_from_pcm16((1).to_bytes(2, "little", signed=True) * 1600)


def _count(counter: Any, **labels: str) -> float:
    return counter.labels(**labels)._value.get()


def _adapter(primary: FakeSTT, fallback: FakeSTT, clock: Clock | None = None, **kw: Any) -> FallbackBatchSTT:
    return FallbackBatchSTT(primary=primary, fallback=fallback, clock=clock or Clock(), **kw)


async def _text(adapter: FallbackBatchSTT, audio: AudioBuffer | None = None) -> str:
    event = await adapter.recognize(audio if audio is not None else _audio())
    return event.alternatives[0].text


@pytest.mark.asyncio
async def test_the_primary_answers_when_healthy() -> None:
    primary, fallback = FakeSTT("gpu"), FakeSTT("cpu")
    before = _count(metrics.STT_REQUESTS, service="primary")
    assert await _text(_adapter(primary, fallback)) == "from gpu"
    assert fallback.calls == []
    assert _count(metrics.STT_REQUESTS, service="primary") == before + 1


@pytest.mark.asyncio
async def test_a_failing_primary_sends_the_same_audio_to_the_fallback() -> None:
    primary, fallback = FakeSTT("gpu", fail=RuntimeError("connection refused")), FakeSTT("cpu")
    audio = _audio()
    before = _count(metrics.STT_PRIMARY_FALLBACKS, reason="error")
    assert await _text(_adapter(primary, fallback), audio) == "from cpu"
    assert fallback.calls[0][0] is audio  # the caller never repeats themselves
    assert _count(metrics.STT_PRIMARY_FALLBACKS, reason="error") == before + 1


@pytest.mark.asyncio
async def test_a_slow_primary_is_abandoned_after_its_timeout() -> None:
    primary, fallback = FakeSTT("gpu", delay_s=5.0), FakeSTT("cpu")
    before = _count(metrics.STT_PRIMARY_FALLBACKS, reason="timeout")
    started = asyncio.get_running_loop().time()
    assert await _text(_adapter(primary, fallback, primary_timeout_s=0.05)) == "from cpu"
    assert asyncio.get_running_loop().time() - started < 1.0
    assert _count(metrics.STT_PRIMARY_FALLBACKS, reason="timeout") == before + 1


@pytest.mark.asyncio
async def test_the_primary_gets_one_attempt_and_no_retries() -> None:
    primary, fallback = FakeSTT("gpu", fail=RuntimeError("boom")), FakeSTT("cpu")
    await _text(_adapter(primary, fallback, primary_timeout_s=2.5))
    assert len(primary.calls) == 1
    assert primary.calls[0][1].max_retry == 0
    assert primary.calls[0][1].timeout == 2.5


@pytest.mark.asyncio
async def test_after_a_failure_the_primary_is_skipped_until_the_cooldown_ends() -> None:
    clock = Clock()
    primary, fallback = FakeSTT("gpu", fail=RuntimeError("down")), FakeSTT("cpu")
    adapter = _adapter(primary, fallback, clock=clock, cooldown_s=30.0)
    await _text(adapter)
    assert len(primary.calls) == 1

    primary.fail = None  # it recovers, but the cooldown is still running
    clock.now += 29.0
    assert await _text(adapter) == "from cpu"
    assert len(primary.calls) == 1

    clock.now += 1.0  # cooldown over: the next utterance tries the primary again
    assert await _text(adapter) == "from gpu"
    assert len(primary.calls) == 2
