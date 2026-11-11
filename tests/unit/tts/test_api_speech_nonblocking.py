"""Regression for #8: a synthesis in progress must not stop the server answering.

/audio/speech is an async handler, and it called the synchronous, CPU-bound
engine.speech() directly. That held the event loop for the whole synthesis, so
the liveness probe went unanswered and every long synthesis paged EndpointDown.

The test runs a synthesis that blocks until released, then asks a second route
for an answer while it is still running. With synthesis on the event loop the
second request cannot be served until the first finishes, so it times out.
"""

from __future__ import annotations

import asyncio
import threading

import httpx
import pytest
from fastapi import FastAPI

from tts import engine_factory
from tts.api.api_v1.endpoints import openai as openai_endpoints


class BlockingEngine:
    """speech() blocks the calling thread until release is set."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def speech(self, **kwargs: object) -> bytes:
        self.started.set()
        assert self.release.wait(timeout=3), "synthesis was never released"
        return b"RIFF-fake-audio"

    def helper_return_audio_file(self, *, audio: bytes, response_format: object) -> object:
        from fastapi.responses import Response

        return Response(content=audio, media_type="audio/wav")


@pytest.fixture
def app_with_blocking_engine(monkeypatch: pytest.MonkeyPatch):
    engine = BlockingEngine()
    monkeypatch.setattr(engine_factory, "_engine", engine)
    monkeypatch.setattr(engine_factory, "_semaphore", asyncio.Semaphore(2))

    app = FastAPI()
    app.include_router(openai_endpoints.router)

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    return app, engine


@pytest.mark.unit
@pytest.mark.tts
@pytest.mark.asyncio
async def test_server_answers_while_a_synthesis_is_running(app_with_blocking_engine) -> None:
    app, engine = app_with_blocking_engine
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://tts") as client:
        speech = asyncio.create_task(
            client.post(
                "/audio/speech", json={"model": "tts-1", "input": "hello", "voice": "default", "response_format": "wav"}
            )
        )
        # Wait until the synthesis is actually running (polling yields the loop to it).
        for _ in range(200):
            if engine.started.is_set():
                break
            await asyncio.sleep(0.01)
        assert engine.started.is_set(), "synthesis never started"

        try:
            ping = await asyncio.wait_for(client.get("/ping"), timeout=2)
            assert ping.status_code == 200
        finally:
            engine.release.set()

        response = await asyncio.wait_for(speech, timeout=10)
        assert response.status_code == 200
        assert response.content == b"RIFF-fake-audio"


@pytest.mark.unit
@pytest.mark.tts
@pytest.mark.asyncio
async def test_cancelled_request_keeps_its_slot_until_the_synthesis_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    """A thread cannot be cancelled. If the handler is cancelled (client gone),
    the synthesis keeps running, so its concurrency slot must stay taken until
    the thread actually finishes. Releasing it on cancellation let real
    concurrency exceed TTS_MAX_CONCURRENT_REQUESTS."""
    engine = BlockingEngine()
    semaphore = asyncio.Semaphore(1)
    monkeypatch.setattr(engine_factory, "_engine", engine)
    monkeypatch.setattr(engine_factory, "_semaphore", semaphore)

    from tts.schemas.openai import OpenAICreateSpeechRequest

    request = OpenAICreateSpeechRequest(model="tts-1", input="hello", voice="default", response_format="wav")
    handler = asyncio.create_task(openai_endpoints.tts(request))
    for _ in range(200):
        if engine.started.is_set():
            break
        await asyncio.sleep(0.01)
    assert engine.started.is_set()

    handler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handler
    await asyncio.sleep(0.05)
    assert semaphore.locked(), "slot was released while the synthesis thread was still running"

    engine.release.set()
    for _ in range(200):
        if not semaphore.locked():
            break
        await asyncio.sleep(0.01)
    assert not semaphore.locked(), "slot was never released after the synthesis ended"


@pytest.mark.unit
@pytest.mark.tts
@pytest.mark.asyncio
async def test_unspeakable_input_is_a_422_not_a_500(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.error_handler import register_exception_handlers
    from tts.exceptions import TextValidationError

    class NothingToSay:
        def speech(self, **kwargs: object) -> bytes:
            raise TextValidationError(message="Nothing to synthesize")

    monkeypatch.setattr(engine_factory, "_engine", NothingToSay())
    monkeypatch.setattr(engine_factory, "_semaphore", asyncio.Semaphore(2))
    app = FastAPI()
    app.include_router(openai_endpoints.router)
    register_exception_handlers(app)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://tts") as client:
        r = await client.post("/audio/speech", json={"model": "tts-1", "input": "...", "voice": "default"})
    assert r.status_code == 422
