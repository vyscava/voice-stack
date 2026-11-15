"""Streaming STT fallbacks that need more than a refusal: stalls, and the real StreamAdapter path."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from aiohttp import WSMsgType, web
from livekit import rtc
from livekit.agents import stt

from tests.unit.voice_agent.stream_fakes import FakeBatch, FakeVad, NonStreamingBatch, collect_until
from voice_agent import metrics
from voice_agent.stream_audio import REPLAY_SECONDS, ReplayBuffer, frame_from_pcm16
from voice_agent.stream_stt import AsrStreamSTT

TIMEOUT = 0.3


def _pcm(fill: int, samples: int = 320) -> bytes:
    return fill.to_bytes(2, "little", signed=True) * samples


def _fallbacks(reason: str) -> float:
    return metrics.STT_STREAM_FALLBACKS.labels(reason=reason)._value.get()


async def test_no_final_after_a_flush_falls_back_within_the_deadline(serve, caplog):
    caplog.set_level(logging.INFO, logger="voice_agent")

    async def silent(ws: web.WebSocketResponse, received: list[Any]) -> None:
        async for _ in ws:  # reads (and answers pings) but never transcribes
            pass

    _, url = await serve(silent)
    batch = FakeBatch()
    plugin = AsrStreamSTT(url=url, batch=batch, vad=None, final_timeout_s=TIMEOUT)
    before = _fallbacks("stall")
    stream = plugin.stream()
    stream.push_frame(frame_from_pcm16(_pcm(9)))
    started = time.monotonic()
    stream.flush()
    events = await collect_until(stream, "final_transcript")
    elapsed = time.monotonic() - started
    await stream.aclose()

    assert events[-1].alternatives[0].text == "from batch"
    assert plugin.fallback_reason == "stall"
    assert _fallbacks("stall") == before + 1
    assert TIMEOUT <= elapsed < TIMEOUT + 1.5
    assert bytes(batch.audio) == _pcm(9), "the unanswered segment is replayed, once"
    assert "no final within" in caplog.text


async def test_a_server_that_stops_reading_falls_back(serve, caplog):
    caplog.set_level(logging.INFO, logger="voice_agent")
    release = asyncio.Event()

    async def deaf(ws: web.WebSocketResponse, received: list[Any]) -> None:
        await release.wait()  # never reads: the client's sends back up

    _, url = await serve(deaf)
    batch = FakeBatch()
    plugin = AsrStreamSTT(url=url, batch=batch, vad=None, final_timeout_s=TIMEOUT, heartbeat_s=60)
    stream = plugin.stream()
    second = _pcm(1, samples=16000)
    try:
        for _ in range(600):  # 10 minutes of audio: far more than the socket buffers hold
            stream.push_frame(frame_from_pcm16(second))
        stream.flush()
        events = await collect_until(stream, "final_transcript", timeout=10)
        await stream.aclose()
    finally:
        release.set()

    assert events[-1].alternatives[0].text == "from batch"
    assert plugin.fallback_reason == "stall"
    # The replayed tail plus everything never sent: never more than was spoken.
    assert 0 < len(batch.audio) <= len(second) * 600
    assert "send blocked" in caplog.text, "the bounded send, not the final deadline, must catch this"


async def test_unanswered_pings_fall_back(serve, caplog):
    caplog.set_level(logging.INFO, logger="voice_agent")
    release = asyncio.Event()

    async def frozen(ws: web.WebSocketResponse, received: list[Any]) -> None:
        await release.wait()  # never reads, so never answers a ping

    _, url = await serve(frozen)
    plugin = AsrStreamSTT(url=url, batch=FakeBatch(), vad=None, final_timeout_s=30, heartbeat_s=0.2)
    stream = plugin.stream()
    try:
        await asyncio.sleep(0.05)
        for _ in range(100):
            if plugin.fallback_reason:
                break
            await asyncio.sleep(0.05)
        stream.push_frame(frame_from_pcm16(_pcm(2)))
        stream.flush()
        events = await collect_until(stream, "final_transcript")
        await stream.aclose()
    finally:
        release.set()
    assert plugin.fallback_reason == "stall"
    assert events[-1].alternatives[0].text == "from batch"
    assert "stopped answering pings" in caplog.text


async def test_fallback_to_a_non_streaming_batch_goes_through_stream_adapter_and_vad(serve):
    """The production shape: openai.STT cannot stream, so the fallback segments with a VAD."""

    async def busy(ws: web.WebSocketResponse, received: list[Any]) -> None:
        await ws.send_json({"type": "error", "code": "busy"})
        await ws.close(code=1013)

    _, url = await serve(busy)
    batch = NonStreamingBatch()
    plugin = AsrStreamSTT(url=url, batch=batch, vad=FakeVad())
    stream = plugin.stream()
    stream.push_frame(frame_from_pcm16(_pcm(4)))
    stream.push_frame(frame_from_pcm16(_pcm(4)))
    stream.flush()  # the session's flush at VAD end of speech: must not discard the speech
    stream.push_frame(frame_from_pcm16(_pcm(0)))  # silence ends the segment
    events = await collect_until(stream, "final_transcript")
    await stream.aclose()
    await plugin.aclose()

    assert plugin.fallback_reason == "busy"
    assert events[-1].alternatives[0].text == "from batch"
    assert stt.SpeechEventType.START_OF_SPEECH in [e.type for e in events]
    assert batch.buffers == [_pcm(4) * 2]


async def test_live_frames_after_a_fallback_are_mono_like_the_replay(serve):
    async def busy(ws: web.WebSocketResponse, received: list[Any]) -> None:
        await ws.send_json({"type": "error", "code": "busy"})
        await ws.close(code=1013)

    _, url = await serve(busy)
    batch = FakeBatch()
    plugin = AsrStreamSTT(url=url, batch=batch, vad=None)
    stream = plugin.stream()
    stream.push_frame(rtc.AudioFrame(data=_pcm(8) * 2, sample_rate=16000, num_channels=2, samples_per_channel=320))
    stream.flush()
    await collect_until(stream, "final_transcript")
    stream.push_frame(rtc.AudioFrame(data=_pcm(6) * 2, sample_rate=16000, num_channels=2, samples_per_channel=320))
    stream.flush()
    await collect_until(stream, "final_transcript")
    await stream.aclose()
    assert bytes(batch.audio) == _pcm(8) + _pcm(6)


def test_replay_buffer_keeps_only_unanswered_audio_and_at_most_the_cap():
    buf = ReplayBuffer()
    buf.add(b"a" * 4)
    buf.flushed()
    buf.finalized()  # the final answered it
    assert buf.segments() == [(b"", False)]

    cap = int(REPLAY_SECONDS * 16000) * 2
    buf.add(b"x" * cap)
    buf.flushed()
    buf.add(b"y" * 10)  # over the cap: the oldest audio goes first
    assert buf.segments() == [(b"x" * (cap - 10), True), (b"y" * 10, False)]


async def test_a_replayed_flush_never_reaches_the_stream_adapter(serve):
    """The server fails after the flush was sent: the replay must not flush the VAD and lose the speech."""

    async def fails_on_flush(ws: web.WebSocketResponse, received: list[Any]) -> None:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT and '"flush"' in msg.data:
                await ws.send_json({"type": "error", "code": "decode_failed"})
                await ws.close(code=1011)
                return

    _, url = await serve(fails_on_flush)
    batch = NonStreamingBatch()
    plugin = AsrStreamSTT(url=url, batch=batch, vad=FakeVad())
    stream = plugin.stream()
    stream.push_frame(frame_from_pcm16(_pcm(4)))
    stream.flush()
    for _ in range(100):
        if plugin.fallback_reason:
            break
        await asyncio.sleep(0.02)
    stream.push_frame(frame_from_pcm16(_pcm(0)))
    events = await collect_until(stream, "final_transcript")
    await stream.aclose()
    await plugin.aclose()

    assert plugin.fallback_reason == "error"
    assert events[-1].alternatives[0].text == "from batch"
    assert batch.buffers == [_pcm(4)]
