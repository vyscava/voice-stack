"""Streaming ASR metrics, slot release after a still-running close, and model warmup."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

import asr.app as asr_app
from asr.api.api_v1.endpoints import stream as stream_endpoint
from asr.api.api_v1.endpoints.stream import _StreamConnection
from asr.stream import factory as stream_factory
from tests.unit.asr.stream_fakes import FakeEngine, send_audio, slot_is_free, until_final, wait_until


def sample(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def connection_outcomes() -> float:
    """All connection outcomes together: one per connection."""
    return sum(
        s.value
        for m in REGISTRY.collect()
        if m.name == "voice_asr_stream_connections"
        for s in m.samples
        if s.name.endswith("_total")
    )


@pytest.mark.unit
@pytest.mark.asr
def test_a_streamed_utterance_is_counted_and_timed(asr_client: TestClient, fake_engine: FakeEngine) -> None:
    accepted = sample("voice_asr_stream_connections_total", outcome="accepted")
    flushes = sample("voice_asr_stream_flush_seconds_count")
    feeds = sample("voice_asr_stream_feed_seconds_count")
    tagged = sample("voice_asr_stream_finals_total", language_source="stream_tag")

    with asr_client.websocket_connect("/v1/stream") as ws:
        send_audio(ws, 1.5)
        ws.send_json({"type": "flush"})
        until_final(ws)
        assert sample("voice_asr_stream_active") == 1

    assert sample("voice_asr_stream_connections_total", outcome="accepted") == accepted + 1
    assert sample("voice_asr_stream_flush_seconds_count") == flushes + 1
    assert sample("voice_asr_stream_feed_seconds_count") >= feeds + 1
    assert sample("voice_asr_stream_finals_total", language_source="stream_tag") == tagged + 1
    assert wait_until(lambda: sample("voice_asr_stream_active") == 0)


@pytest.mark.unit
@pytest.mark.asr
def test_a_busy_refusal_is_counted_and_leaves_the_active_gauge_alone(
    asr_client: TestClient, fake_engine: FakeEngine
) -> None:
    busy = sample("voice_asr_stream_connections_total", outcome="busy")
    assert stream_factory.try_acquire_stream_slot()
    try:
        with asr_client.websocket_connect("/v1/stream") as ws:
            ws.receive_json()
    except Exception:  # noqa: BLE001,S110 - the close after the refusal; only the counter matters here
        pass
    finally:
        stream_factory.release_stream_slot()
    assert sample("voice_asr_stream_connections_total", outcome="busy") == busy + 1
    assert sample("voice_asr_stream_active") == 0


@pytest.mark.unit
@pytest.mark.asr
def test_metrics_endpoint_serves_the_stream_metrics(asr_client: TestClient) -> None:
    body = asr_client.get("/metrics").text
    assert "voice_asr_stream_flush_seconds" in body
    assert "voice_asr_stream_engine_loaded" in body


class _SlowCloseSession:
    def __init__(self, gate: threading.Event) -> None:
        self.gate = gate
        self.closed = False

    def close(self) -> None:
        self.gate.wait(5)
        self.closed = True


@pytest.mark.unit
@pytest.mark.asr
async def test_the_slot_waits_for_a_close_the_cancelled_handler_stopped_awaiting() -> None:
    gate = threading.Event()
    session = _SlowCloseSession(gate)
    connection = _StreamConnection(object(), FakeEngine())  # type: ignore[arg-type]
    task = asyncio.create_task(connection._close_session(session))  # type: ignore[arg-type]
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    released = threading.Event()
    connection.release_when_closed(released.set)
    assert not released.is_set(), "the slot was handed back while the decode thread was still closing"

    gate.set()
    for _ in range(200):
        if released.is_set():
            break
        await asyncio.sleep(0.01)
    assert session.closed
    assert released.is_set()


@pytest.mark.unit
@pytest.mark.asr
async def test_the_slot_is_released_at_once_when_nothing_is_closing() -> None:
    connection = _StreamConnection(object(), FakeEngine())  # type: ignore[arg-type]
    released: list[bool] = []
    connection.release_when_closed(lambda: released.append(True))
    assert released == [True]


@pytest.mark.unit
@pytest.mark.asr
def test_the_slot_is_free_after_a_normal_stream(asr_client: TestClient, fake_engine: FakeEngine) -> None:
    with asr_client.websocket_connect("/v1/stream") as ws:
        send_audio(ws, 0.5)
    assert wait_until(slot_is_free)


@pytest.mark.unit
@pytest.mark.asr
async def test_warmup_loads_the_engine_only_when_enabled_and_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    loads: list[bool] = []

    def load() -> Any:
        loads.append(True)
        return object()

    monkeypatch.setattr(stream_factory, "get_stream_engine", load)

    for enabled, warmup in [(False, True), (True, False), (False, False)]:
        monkeypatch.setattr(asr_app.settings, "ASR_STREAM_ENABLED", enabled)
        monkeypatch.setattr(asr_app.settings, "ASR_STREAM_WARMUP", warmup)
        assert asr_app.start_stream_warmup() is None
    assert loads == []

    monkeypatch.setattr(asr_app.settings, "ASR_STREAM_ENABLED", True)
    monkeypatch.setattr(asr_app.settings, "ASR_STREAM_WARMUP", True)
    future = asr_app.start_stream_warmup()
    assert future is not None
    await future
    assert loads == [True]


@pytest.mark.unit
@pytest.mark.asr
def test_a_bad_sample_rate_is_a_refusal_not_a_second_connection_outcome(
    asr_client: TestClient, fake_engine: FakeEngine
) -> None:
    accepted = sample("voice_asr_stream_connections_total", outcome="accepted")
    outcomes_before = sum(
        s.value
        for m in REGISTRY.collect()
        if m.name == "voice_asr_stream_connections"
        for s in m.samples
        if s.name.endswith("_total")
    )
    refused = sample("voice_asr_stream_refusals_total", code="unsupported_sample_rate")
    try:
        with asr_client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"type": "start", "sample_rate": 8000})
            ws.receive_json()
    except Exception:  # noqa: BLE001,S110 - the close after the refusal
        pass
    outcomes_after = sum(
        s.value
        for m in REGISTRY.collect()
        if m.name == "voice_asr_stream_connections"
        for s in m.samples
        if s.name.endswith("_total")
    )
    assert sample("voice_asr_stream_refusals_total", code="unsupported_sample_rate") == refused + 1
    assert sample("voice_asr_stream_connections_total", outcome="accepted") == accepted + 1
    assert outcomes_after == outcomes_before + 1, "one connection, one outcome"


@pytest.mark.unit
@pytest.mark.asr
def test_an_empty_flush_is_not_timed(asr_client: TestClient, fake_engine: FakeEngine) -> None:
    flushes = sample("voice_asr_stream_flush_seconds_count")
    with asr_client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "flush"})
        until_final(ws)
    assert sample("voice_asr_stream_flush_seconds_count") == flushes


@pytest.mark.unit
@pytest.mark.asr
def test_detailed_health_reports_the_stream_model(fake_engine: FakeEngine) -> None:
    assert asr_app._stream_engine_loaded() is True
    stream_factory.set_stream_engine(None)
    assert asr_app._stream_engine_loaded() is False


class _FakeWebSocket:
    """Just enough of a WebSocket for the handler: one audio frame, then silence until cancelled."""

    def __init__(self) -> None:
        self.sent: list[Any] = []
        self._frames = [{"type": "websocket.receive", "bytes": b"\x01\x00" * 1600}]

    async def accept(self) -> None:
        return None

    async def receive(self) -> dict[str, Any]:
        if self._frames:
            return self._frames.pop(0)
        await asyncio.Event().wait()  # until cancelled
        raise AssertionError("unreachable")

    async def send_json(self, data: Any) -> None:
        self.sent.append(data)

    async def close(self, code: int = 1000) -> None:
        return None


@pytest.mark.unit
@pytest.mark.asr
async def test_a_cancelled_handler_keeps_the_slot_until_its_session_closes(
    monkeypatch: pytest.MonkeyPatch, fake_engine: FakeEngine
) -> None:
    gate = threading.Event()
    real_new_session = fake_engine.new_session

    def slow_close_session() -> Any:
        session = real_new_session()

        def close() -> None:
            gate.wait(5)
            session.closed = True

        session.close = close  # type: ignore[method-assign]
        return session

    monkeypatch.setattr(fake_engine, "new_session", slow_close_session)
    active = sample("voice_asr_stream_active")
    ws = _FakeWebSocket()
    handler = asyncio.create_task(stream_endpoint.stream(ws))  # type: ignore[arg-type]
    for _ in range(200):
        if fake_engine.sessions:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    # anyio's cancel scopes (uvicorn, the test client) re-deliver a
    # cancellation at every await, so the handler's finally is cancelled too.
    handler.cancel()
    await asyncio.sleep(0.05)
    handler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(handler, 2)

    assert not slot_is_free(), "the slot was handed on while the decode thread was still closing"
    assert sample("voice_asr_stream_active") == active + 1
    gate.set()
    for _ in range(200):
        if slot_is_free():
            break
        await asyncio.sleep(0.01)
    assert fake_engine.sessions[0].closed
    assert slot_is_free()
    assert sample("voice_asr_stream_active") == active


@pytest.mark.unit
@pytest.mark.asr
async def test_a_session_started_under_a_cancel_is_closed_not_leaked(fake_engine: FakeEngine) -> None:
    started = threading.Event()
    release = threading.Event()
    real_new_session = fake_engine.new_session

    def slow_new_session() -> Any:
        started.set()
        release.wait(5)
        return real_new_session()

    fake_engine.new_session = slow_new_session  # type: ignore[method-assign]
    connection = _StreamConnection(object(), fake_engine)  # type: ignore[arg-type]
    task = asyncio.create_task(connection._new_session())
    while not started.is_set():
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    released = threading.Event()
    connection.release_when_closed(released.set)
    assert not released.is_set(), "the slot was handed on while the orphan session was still starting"

    release.set()
    for _ in range(300):
        if released.is_set():
            break
        await asyncio.sleep(0.01)
    assert fake_engine.sessions and fake_engine.sessions[0].closed
    assert released.is_set()


@pytest.mark.unit
@pytest.mark.asr
def test_streams_up_to_the_session_limit_are_served_and_the_next_is_busy(
    asr_client: TestClient, fake_engine: FakeEngine
) -> None:
    stream_factory.set_max_sessions(2)
    with asr_client.websocket_connect("/v1/stream") as one, asr_client.websocket_connect("/v1/stream") as two:
        send_audio(one, 1.0)
        send_audio(two, 1.0)
        one.send_json({"type": "flush"})
        two.send_json({"type": "flush"})
        _, final_one = until_final(one)
        _, final_two = until_final(two)
        assert final_one["text"] and final_two["text"]
        with asr_client.websocket_connect("/v1/stream") as three:
            assert three.receive_json() == {"type": "error", "code": "busy"}
    assert wait_until(slot_is_free)


@pytest.mark.unit
@pytest.mark.asr
def test_a_release_without_a_claim_is_an_error() -> None:
    stream_factory.set_max_sessions(1)
    try:
        with pytest.raises(RuntimeError):
            stream_factory.release_stream_slot()
    finally:
        stream_factory.set_max_sessions(None)
