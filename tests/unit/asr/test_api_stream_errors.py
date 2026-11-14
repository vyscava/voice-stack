"""Error paths of the /v1/stream WebSocket: every failure must close with its code AND free the slot.

Each test ends by opening a NEW connection and getting partials from it. A
leaked slot would answer that connection with "busy" instead.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from asr.stream import factory as stream_factory
from tests.unit.asr.stream_fakes import SR, FakeEngine, recv_json, send_audio, until_final


def _expect_refusal(ws: Any, code: str, close_code: int) -> None:
    assert recv_json(ws) == {"type": "error", "code": code}
    with pytest.raises(WebSocketDisconnect) as closed:
        recv_json(ws)
    assert closed.value.code == close_code


def _assert_next_connection_streams(client: TestClient) -> None:
    """The slot is free: a new connection is served (partials and a final), not refused as busy."""
    with client.websocket_connect("/v1/stream") as ws:
        send_audio(ws, 1.0)
        ws.send_json({"type": "flush"})
        partials, final = until_final(ws)
    assert partials == ["one", "one two"]
    assert final["text"] == "One two."


@pytest.mark.unit
@pytest.mark.asr
def test_engine_load_failure_is_unavailable_and_frees_the_slot(
    asr_client: TestClient, fake_engine: FakeEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_load() -> Any:
        raise RuntimeError("no transformers in this image")

    monkeypatch.setattr(stream_factory, "get_stream_engine", failing_load)
    with asr_client.websocket_connect("/v1/stream") as ws:
        _expect_refusal(ws, "unavailable", 1011)

    monkeypatch.setattr(stream_factory, "get_stream_engine", lambda: fake_engine)
    _assert_next_connection_streams(asr_client)


@pytest.mark.unit
@pytest.mark.asr
def test_invalid_settings_are_misconfigured_and_hold_no_slot(
    asr_client: TestClient, fake_engine: FakeEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reachable for real: an unsupported ASR_STREAM_LOOKAHEAD makes StreamConfig raise ValueError."""
    from types import SimpleNamespace

    from asr.stream.base import StreamConfig

    bad = SimpleNamespace(
        ASR_STREAM_ENABLED=True,
        ASR_STREAM_MODEL="fake",
        ASR_STREAM_LOOKAHEAD=4,
        ASR_STREAM_DTYPE="float32",
        ASR_STREAM_THREADS=0,
        ASR_CPU_THREADS=4,
        ASR_DEVICE="cpu",
    )
    good = stream_factory.get_stream_config()
    monkeypatch.setattr(stream_factory, "get_stream_config", lambda: StreamConfig.from_settings(bad))
    with asr_client.websocket_connect("/v1/stream") as ws:
        _expect_refusal(ws, "misconfigured", 1011)

    monkeypatch.setattr(stream_factory, "get_stream_config", lambda: good)
    _assert_next_connection_streams(asr_client)


@pytest.mark.unit
@pytest.mark.asr
def test_feed_failure_mid_stream_is_decode_failed_and_frees_the_slot(
    asr_client: TestClient, fake_engine: FakeEngine
) -> None:
    fake_engine.feed_fails_after = SR  # fails 1 s into the utterance
    with asr_client.websocket_connect("/v1/stream") as ws:
        send_audio(ws, 2.0)
        assert recv_json(ws) == {"type": "partial", "text": "one"}
        assert recv_json(ws) == {"type": "partial", "text": "one two"}
        _expect_refusal(ws, "decode_failed", 1011)
    assert fake_engine.sessions[0].closed  # the failed session was released, not leaked

    fake_engine.feed_fails_after = None
    _assert_next_connection_streams(asr_client)


@pytest.mark.unit
@pytest.mark.asr
def test_finish_failure_still_closes_the_session(asr_client: TestClient, fake_engine: FakeEngine) -> None:
    """finish() raising (e.g. its join timing out) must not leak the decode thread."""
    fake_engine.finish_fails = True
    with asr_client.websocket_connect("/v1/stream") as ws:
        send_audio(ws, 0.5)
        assert recv_json(ws) == {"type": "partial", "text": "one"}
        ws.send_json({"type": "flush"})
        _expect_refusal(ws, "decode_failed", 1011)
    session = fake_engine.sessions[0]
    assert session.finished
    assert session.closed

    fake_engine.finish_fails = False
    _assert_next_connection_streams(asr_client)


@pytest.mark.unit
@pytest.mark.asr
def test_abrupt_disconnect_mid_stream_frees_slot_and_session(asr_client: TestClient, fake_engine: FakeEngine) -> None:
    with asr_client.websocket_connect("/v1/stream") as ws:
        send_audio(ws, 0.5)
        assert recv_json(ws) == {"type": "partial", "text": "one"}
        # Leave mid-utterance: no flush, no close message.
    assert fake_engine.sessions[0].closed
    assert not fake_engine.sessions[0].finished

    _assert_next_connection_streams(asr_client)


@pytest.mark.unit
@pytest.mark.asr
def test_stale_language_detection_is_not_adopted_by_the_next_utterance(
    asr_client: TestClient, fake_engine: FakeEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A detection still running when its utterance ends belongs to that utterance only."""
    from asr.api.api_v1.endpoints import stream as stream_endpoint

    monkeypatch.setattr(stream_endpoint, "LANGUAGE_GRACE_SECONDS", 0.05)
    fake_engine.release_detection.clear()
    try:
        with asr_client.websocket_connect("/v1/stream") as ws:
            send_audio(ws, 2.5)
            ws.send_json({"type": "flush"})
            _, first = until_final(ws)
            assert first["language"] is None  # not ready in time

            # The late result lands while the next (short) utterance is under way.
            fake_engine.release_detection.set()
            assert fake_engine.detect_returned.wait(5)
            send_audio(ws, 0.5)
            ws.send_json({"type": "flush"})
            _, second = until_final(ws)
    finally:
        fake_engine.release_detection.set()

    assert second["language"] is None  # the stale "pt" was discarded, not adopted
    assert fake_engine.detect_calls == [2 * SR]
