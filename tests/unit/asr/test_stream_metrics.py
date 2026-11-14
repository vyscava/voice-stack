"""Streaming ASR metrics, slot release after a still-running close, and model warmup."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

import asr.app as asr_app
from asr.api.api_v1.endpoints.stream import _StreamConnection
from asr.stream import factory as stream_factory
from tests.unit.asr.stream_fakes import FakeEngine, send_audio, slot_is_free, until_final, wait_until


def sample(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


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
