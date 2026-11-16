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


@pytest.mark.asyncio
async def test_a_primary_just_under_its_timeout_still_answers() -> None:
    clock = Clock()
    primary, fallback = FakeSTT("gpu", delay_s=0.05), FakeSTT("cpu")
    adapter = _adapter(primary, fallback, clock=clock, primary_timeout_s=0.5)
    assert await _text(adapter) == "from gpu"
    assert fallback.calls == []
    assert await _text(adapter) == "from gpu"  # no cooldown was armed
    assert len(primary.calls) == 2


@pytest.mark.asyncio
async def test_a_timed_out_primary_request_is_cancelled() -> None:
    started, cancelled = asyncio.Event(), asyncio.Event()

    class Hanging(FakeSTT):
        async def _recognize_impl(self, buffer, *, language=NOT_GIVEN, conn_options):  # type: ignore[override]
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            raise AssertionError("unreachable")

    assert await _text(_adapter(Hanging("gpu"), FakeSTT("cpu"), primary_timeout_s=0.05)) == "from cpu"
    assert started.is_set()
    await asyncio.wait_for(cancelled.wait(), 1.0)


@pytest.mark.asyncio
async def test_the_fallback_gets_one_attempt_per_pass() -> None:
    primary, fallback = FakeSTT("gpu", fail=RuntimeError("down")), FakeSTT("cpu")
    await _text(_adapter(primary, fallback))
    assert fallback.calls[0][1].max_retry == 0


@pytest.mark.asyncio
async def test_a_fallback_failure_reaches_the_caller() -> None:
    primary = FakeSTT("gpu", fail=RuntimeError("gpu down"))
    fallback = FakeSTT("cpu", fail=RuntimeError("cpu down"))
    adapter = _adapter(primary, fallback)
    with pytest.raises(Exception, match="cpu down"):
        await adapter.recognize(_audio(), conn_options=APIConnectOptions(max_retry=0))


@pytest.mark.asyncio
async def test_a_skipped_primary_does_not_extend_its_cooldown() -> None:
    clock = Clock()
    primary, fallback = FakeSTT("gpu", fail=RuntimeError("down")), FakeSTT("cpu")
    adapter = _adapter(primary, fallback, clock=clock, cooldown_s=30.0)
    await _text(adapter)  # fails at t=1000: skipped until t=1030
    primary.fail = None
    for _ in range(5):  # turns during the cooldown answer from the fallback ...
        clock.now += 5.0
        await _text(adapter)
    clock.now = 1030.0  # ... and do not push the retry later
    assert await _text(adapter) == "from gpu"


def test_the_fallback_series_exist_before_the_first_event() -> None:
    # increase() needs a sample before the first event: a series born at 1
    # loses that event, and the "2 in 30m" alert would fire on the third.
    from prometheus_client import generate_latest

    text = generate_latest().decode()
    for reason in ("timeout", "error"):
        assert f'voice_agent_stt_primary_fallbacks_total{{reason="{reason}"}}' in text
    for service in ("primary", "fallback"):
        assert f'voice_agent_stt_requests_total{{service="{service}"}}' in text
