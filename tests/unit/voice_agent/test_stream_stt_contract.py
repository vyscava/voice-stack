"""The streaming STT plugin against the REAL /v1/stream endpoint, with the ASR tests' fake engine.

test_stream_stt.py pins the plugin's behaviour against a scripted server; this
pins the protocol itself: if the endpoint changes a message, a field or a
refusal, these fail. The endpoint runs under uvicorn on a free local port, so
the plugin speaks real WebSocket frames to it. No model, no network.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
import uvicorn
from fastapi import FastAPI

from asr.api.api_v1.endpoints import stream as stream_endpoint
from asr.stream import factory as stream_factory
from tests.unit.asr.stream_fakes import FakeEngine, config, pcm
from tests.unit.voice_agent.stream_fakes import FakeBatch, collect_until
from voice_agent.stream_audio import frame_from_pcm16
from voice_agent.stream_stt import AsrStreamSTT

pytestmark = pytest.mark.unit


@pytest.fixture
async def endpoint(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[FakeEngine, str]]:
    engine = FakeEngine()  # its default tags the stream <pt-BR> after the second word
    monkeypatch.setattr(stream_factory, "get_stream_config", lambda: config(enabled=True))
    stream_factory.set_stream_engine(engine)
    app = FastAPI()
    app.include_router(stream_endpoint.router, prefix="/v1")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="off"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield engine, f"ws://127.0.0.1:{port}/v1/stream"
    server.should_exit = True
    await task
    stream_factory.set_stream_engine(None)


async def test_partials_and_final_with_language_from_the_real_endpoint(endpoint):
    engine, url = endpoint
    plugin = AsrStreamSTT(url=url, batch=FakeBatch(), vad=None)
    stream = plugin.stream()
    stream.push_frame(frame_from_pcm16(pcm(3.0)))
    stream.flush()
    events = await collect_until(stream, "final_transcript")
    await stream.aclose()

    interims = [e.alternatives[0].text for e in events if e.type == "interim_transcript"]
    final = events[-1].alternatives[0]
    assert interims[-1] == "one two three four five six"
    assert final.text == "One two three four five six."
    assert final.language == "pt"
    assert plugin.fallback_reason is None
    assert engine.sessions[0].finished


async def test_disabled_endpoint_falls_back(endpoint, monkeypatch):
    _, url = endpoint
    monkeypatch.setattr(stream_factory, "get_stream_config", lambda: config(enabled=False))
    plugin = AsrStreamSTT(url=url, batch=FakeBatch(), vad=None)
    stream = plugin.stream()
    stream.push_frame(frame_from_pcm16(pcm(0.5)))
    stream.flush()
    events = await collect_until(stream, "final_transcript")
    await stream.aclose()
    assert events[-1].alternatives[0].text == "from batch"
    assert plugin.fallback_reason == "disabled"


async def test_busy_endpoint_falls_back(endpoint):
    _, url = endpoint
    assert stream_factory.try_acquire_stream_slot()  # someone else holds the one stream
    try:
        plugin = AsrStreamSTT(url=url, batch=FakeBatch(), vad=None)
        stream = plugin.stream()
        stream.push_frame(frame_from_pcm16(pcm(0.5)))
        stream.flush()
        events = await collect_until(stream, "final_transcript")
        await stream.aclose()
    finally:
        stream_factory.release_stream_slot()
    assert events[-1].alternatives[0].text == "from batch"
    assert plugin.fallback_reason == "busy"
