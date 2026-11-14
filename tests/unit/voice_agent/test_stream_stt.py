"""The streaming STT plugin against a fake ASR /v1/stream server (aiohttp, local, offline)."""

from __future__ import annotations

import json
import socket
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer
from livekit import rtc
from livekit.agents import stt

from tests.unit.voice_agent.stream_fakes import FakeBatch, collect_until
from voice_agent import metrics
from voice_agent.stream_stt import AsrStreamSTT, frame_from_pcm16, pcm16_mono, stream_url

ET = stt.SpeechEventType
Handler = Callable[[web.WebSocketResponse, list[Any]], Awaitable[None]]


class FakeAsr:
    """A /v1/stream server whose behaviour is a per-test coroutine; records what it receives."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.received: list[Any] = []  # dicts for JSON frames, bytes for audio
        self.connections = 0

    async def ws(self, request: web.Request) -> web.WebSocketResponse:
        self.connections += 1
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await self.handler(ws, self.received)
        return ws


@pytest.fixture
async def serve() -> AsyncIterator[Callable[[Handler], Awaitable[tuple[FakeAsr, str]]]]:
    servers: list[TestServer] = []

    async def start(handler: Handler) -> tuple[FakeAsr, str]:
        fake = FakeAsr(handler)
        app = web.Application()
        app.router.add_get("/v1/stream", fake.ws)
        server = TestServer(app)
        await server.start_server()
        servers.append(server)
        return fake, str(server.make_url("/v1/stream")).replace("http://", "ws://")

    yield start
    for server in servers:
        await server.close()


async def transcribing(ws: web.WebSocketResponse, received: list[Any]) -> None:
    """Partials on audio, a final (with language_source) on flush, until the client closes."""
    async for msg in ws:
        if msg.type == WSMsgType.BINARY:
            received.append(msg.data)
            heard = sum(1 for r in received if isinstance(r, bytes))
            await ws.send_json({"type": "partial", "text": "olá" if heard == 1 else "olá mundo"})
        elif msg.type == WSMsgType.TEXT:
            control = json.loads(msg.data)
            received.append(control)
            if control["type"] == "flush":
                await ws.send_json({"type": "final", "text": "olá mundo", "language": "pt", "language_source": "probe"})
            elif control["type"] == "close":
                await ws.close()


def refusing(code: str, close_code: int) -> Handler:
    async def handler(ws: web.WebSocketResponse, received: list[Any]) -> None:
        await ws.send_json({"type": "error", "code": code})
        await ws.close(code=close_code)

    return handler


async def dies_after_audio(ws: web.WebSocketResponse, received: list[Any]) -> None:
    async for msg in ws:
        if msg.type == WSMsgType.BINARY:
            received.append(msg.data)
            await ws.send_json({"type": "partial", "text": "half"})
            await ws.close(code=1011)
            return


def _frame(fill: int = 1, samples: int = 320) -> rtc.AudioFrame:
    return frame_from_pcm16(fill.to_bytes(2, "little", signed=True) * samples)


async def _until(stream: stt.RecognizeStream, kind: stt.SpeechEventType) -> list[stt.SpeechEvent]:
    return await collect_until(stream, kind.value)


def _fallbacks(reason: str) -> float:
    return metrics.STT_STREAM_FALLBACKS.labels(reason=reason)._value.get()


async def test_partials_become_interims_and_the_final_carries_the_language(serve):
    fake, url = await serve(transcribing)
    plugin = AsrStreamSTT(url=url, batch=FakeBatch(), vad=None)
    stream = plugin.stream()
    stream.push_frame(_frame())
    stream.push_frame(_frame())
    events = await _until(stream, ET.INTERIM_TRANSCRIPT)
    events += await _until(stream, ET.INTERIM_TRANSCRIPT)
    stream.flush()
    events += await _until(stream, ET.END_OF_SPEECH)
    await stream.aclose()

    assert [e.type for e in events] == [ET.START_OF_SPEECH, ET.INTERIM_TRANSCRIPT, ET.INTERIM_TRANSCRIPT] + [
        ET.FINAL_TRANSCRIPT,
        ET.END_OF_SPEECH,
    ]
    assert [e.alternatives[0].text for e in events[1:4]] == ["olá", "olá mundo", "olá mundo"]
    final = events[3].alternatives[0]
    assert final.language == "pt"
    assert final.metadata == {"language_source": "probe"}
    assert plugin.fallback_reason is None


async def test_flush_sends_a_flush_message_after_the_audio(serve):
    fake, url = await serve(transcribing)
    stream = AsrStreamSTT(url=url, batch=FakeBatch(), vad=None).stream()
    stream.push_frame(_frame(fill=7))
    stream.flush()
    await _until(stream, ET.FINAL_TRANSCRIPT)
    await stream.aclose()

    assert fake.received[0] == {"type": "start", "sample_rate": 16000}
    assert fake.received[1] == (7).to_bytes(2, "little") * 320  # 16 kHz mono PCM16 LE, unchanged
    assert fake.received[2] == {"type": "flush"}


async def test_final_without_language_source_is_accepted(serve):
    async def old_server(ws: web.WebSocketResponse, received: list[Any]) -> None:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT and json.loads(msg.data)["type"] == "flush":
                await ws.send_json({"type": "final", "text": "hi", "language": None})

    _, url = await serve(old_server)
    stream = AsrStreamSTT(url=url, batch=FakeBatch(), vad=None).stream()
    stream.push_frame(_frame())
    stream.flush()
    (final,) = await _until(stream, ET.FINAL_TRANSCRIPT)
    await stream.aclose()
    assert final.alternatives[0].text == "hi"
    assert final.alternatives[0].language == ""
    assert final.alternatives[0].metadata is None


async def test_end_input_waits_for_the_last_final(serve):
    fake, url = await serve(transcribing)
    stream = AsrStreamSTT(url=url, batch=FakeBatch(), vad=None).stream()
    stream.push_frame(_frame())
    stream.end_input()
    events = [ev async for ev in stream]
    assert ET.FINAL_TRANSCRIPT in [e.type for e in events]
    assert fake.received[-1] == {"type": "close"}


@pytest.mark.parametrize(("code", "close_code"), [("busy", 1013), ("disabled", 1008)])
async def test_a_refusal_falls_back_to_batch_for_the_session(serve, code, close_code):
    fake, url = await serve(refusing(code, close_code))
    batch = FakeBatch()
    plugin = AsrStreamSTT(url=url, batch=batch, vad=None)
    before = _fallbacks(code)

    stream = plugin.stream()
    stream.push_frame(_frame(fill=3))
    stream.flush()
    (final,) = await _until(stream, ET.FINAL_TRANSCRIPT)
    await stream.aclose()

    assert final.alternatives[0].text == "from batch"
    assert plugin.fallback_reason == code
    assert _fallbacks(code) == before + 1
    assert bytes(batch.audio) == (3).to_bytes(2, "little") * 320, "the caller's audio must reach the batch STT"

    # The rest of the call goes straight to batch: no new connection, no second count.
    again = plugin.stream()
    again.push_frame(_frame())
    again.flush()
    await _until(again, ET.FINAL_TRANSCRIPT)
    await again.aclose()
    assert fake.connections == 1
    assert _fallbacks(code) == before + 1


async def test_connect_failure_falls_back_to_batch():
    with socket.socket() as s:  # a port that was free a moment ago: nothing listens on it
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    plugin = AsrStreamSTT(url=f"ws://127.0.0.1:{port}/v1/stream", batch=FakeBatch(), vad=None)
    before = _fallbacks("connect")
    stream = plugin.stream()
    stream.push_frame(_frame())
    stream.flush()
    (final,) = await _until(stream, ET.FINAL_TRANSCRIPT)
    await stream.aclose()
    assert final.alternatives[0].text == "from batch"
    assert plugin.fallback_reason == "connect"
    assert _fallbacks("connect") == before + 1


async def test_server_dying_mid_utterance_replays_the_utterance_into_batch(serve):
    _, url = await serve(dies_after_audio)
    batch = FakeBatch()
    plugin = AsrStreamSTT(url=url, batch=batch, vad=None)
    before = _fallbacks("error")
    stream = plugin.stream()
    stream.push_frame(_frame(fill=5))
    await _until(stream, ET.INTERIM_TRANSCRIPT)
    stream.push_frame(_frame(fill=6))
    stream.flush()
    events = await _until(stream, ET.FINAL_TRANSCRIPT)
    await stream.aclose()
    assert events[-1].alternatives[0].text == "from batch"
    assert plugin.fallback_reason == "error"
    assert _fallbacks("error") == before + 1
    assert bytes(batch.audio).startswith((5).to_bytes(2, "little") * 320)
    assert bytes(batch.audio).endswith((6).to_bytes(2, "little") * 320)


async def test_the_final_records_stt_latency(serve):
    def count() -> float:
        return sum(
            s.value
            for m in metrics.STAGE_SECONDS.collect()
            for s in m.samples
            if s.name.endswith("_count") and s.labels.get("stage") == "stt"
        )

    _, url = await serve(transcribing)
    before = count()
    stream = AsrStreamSTT(url=url, batch=FakeBatch(), vad=None).stream()
    stream.push_frame(_frame())
    stream.flush()
    await _until(stream, ET.FINAL_TRANSCRIPT)
    await stream.aclose()
    assert count() == before + 1


def test_the_plugin_asks_livekit_to_flush_it():
    caps = AsrStreamSTT(url="ws://x/v1/stream", batch=FakeBatch(), vad=None).capabilities
    # Without manual_flush the session never calls flush(), so no final would ever come.
    assert caps.streaming and caps.interim_results and caps.manual_flush


def test_stereo_frames_are_downmixed():
    stereo = rtc.AudioFrame(
        data=b"".join(v.to_bytes(2, "little", signed=True) for v in (100, 300, -10, -30)),
        sample_rate=16000,
        num_channels=2,
        samples_per_channel=2,
    )
    assert pcm16_mono(stereo) == b"".join(v.to_bytes(2, "little", signed=True) for v in (200, -20))


@pytest.mark.parametrize(
    ("asr", "override", "expected"),
    [
        ("http://asr:5001", "", "ws://asr:5001/v1/stream"),
        ("https://asr.example/", "", "wss://asr.example/v1/stream"),
        ("http://asr:5001", "ws://other:9/v1/stream", "ws://other:9/v1/stream"),
    ],
)
def test_stream_url(asr, override, expected):
    assert stream_url(asr, override) == expected
