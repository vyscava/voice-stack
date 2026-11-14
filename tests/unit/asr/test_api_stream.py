"""Unit tests for the streaming ASR WebSocket endpoint (/v1/stream), driven by a fake engine.

No model is downloaded or loaded: the endpoint only talks to the StreamEngine
protocol, and these tests inject a fake through asr.stream.factory.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from asr.stream import factory as stream_factory
from tests.unit.asr.stream_fakes import SR, WORDS, FakeEngine, config, pcm, recv_json, send_audio, until_final


@pytest.mark.unit
@pytest.mark.asr
def test_stream_happy_path_partials_then_final_with_language(asr_client: TestClient, fake_engine: FakeEngine) -> None:
    with asr_client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "start", "sample_rate": SR})
        send_audio(ws, 3.0)
        ws.send_json({"type": "flush"})
        partials, final = until_final(ws)

    assert partials == [" ".join(WORDS[:n]) for n in range(1, 7)]
    assert final == {"type": "final", "text": "One two three four five six.", "language": "pt"}
    # Exactly ONE detection, on exactly the first 2 s of the utterance.
    assert fake_engine.detect_calls == [2 * SR]
    assert fake_engine.sessions[0].finished


@pytest.mark.unit
@pytest.mark.asr
def test_stream_flush_resets_for_the_next_utterance(asr_client: TestClient, fake_engine: FakeEngine) -> None:
    with asr_client.websocket_connect("/v1/stream") as ws:
        send_audio(ws, 2.5)
        ws.send_json({"type": "flush"})
        _, first = until_final(ws)

        send_audio(ws, 1.0)
        ws.send_json({"type": "flush"})
        partials, second = until_final(ws)

    assert first["text"] == "One two three four five."
    # A fresh session: the hypothesis restarts instead of continuing "...five".
    assert len(fake_engine.sessions) == 2
    assert partials == ["one", "one two"]
    assert second["text"] == "One two."
    # The 1 s utterance never reached 2 s, so no second detection ran.
    assert fake_engine.detect_calls == [2 * SR]


@pytest.mark.unit
@pytest.mark.asr
def test_stream_short_utterance_reuses_previous_language(asr_client: TestClient, fake_engine: FakeEngine) -> None:
    with asr_client.websocket_connect("/v1/stream") as ws:
        send_audio(ws, 2.5)
        ws.send_json({"type": "flush"})
        _, first = until_final(ws)
        send_audio(ws, 0.5)
        ws.send_json({"type": "flush"})
        _, second = until_final(ws)

    assert first["language"] == "pt"
    assert second["language"] == "pt"


@pytest.mark.unit
@pytest.mark.asr
def test_stream_short_first_utterance_has_null_language(asr_client: TestClient, fake_engine: FakeEngine) -> None:
    with asr_client.websocket_connect("/v1/stream") as ws:
        send_audio(ws, 1.0)
        ws.send_json({"type": "flush"})
        _, final = until_final(ws)

    assert final["language"] is None
    assert fake_engine.detect_calls == []


@pytest.mark.unit
@pytest.mark.asr
def test_stream_detection_not_ready_by_flush_falls_back(
    asr_client: TestClient, fake_engine: FakeEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    from asr.api.api_v1.endpoints import stream as stream_endpoint

    monkeypatch.setattr(stream_endpoint, "LANGUAGE_GRACE_SECONDS", 0.05)
    fake_engine.release_detection.clear()  # detection hangs until the test lets it go
    try:
        with asr_client.websocket_connect("/v1/stream") as ws:
            send_audio(ws, 2.5)
            ws.send_json({"type": "flush"})
            _, final = until_final(ws)
    finally:
        fake_engine.release_detection.set()

    assert fake_engine.detect_calls == [2 * SR]  # it was started ...
    assert final["language"] is None  # ... but not ready, and there is no previous language


@pytest.mark.unit
@pytest.mark.asr
def test_stream_second_connection_is_refused_busy(asr_client: TestClient, fake_engine: FakeEngine) -> None:
    with asr_client.websocket_connect("/v1/stream") as first:
        # A full round trip proves the first handler is running and holds the slot.
        first.send_json({"type": "flush"})
        until_final(first)

        with asr_client.websocket_connect("/v1/stream") as second:
            assert recv_json(second) == {"type": "error", "code": "busy"}
            with pytest.raises(WebSocketDisconnect) as closed:
                recv_json(second)
            assert closed.value.code == 1013

    # The slot is released when the first connection ends.
    with asr_client.websocket_connect("/v1/stream") as third:
        third.send_json({"type": "flush"})
        _, final = until_final(third)
        assert final["text"] == ""  # a flush with no audio is an empty utterance, not an error


@pytest.mark.unit
@pytest.mark.asr
def test_stream_disabled_refuses_with_1008(asr_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stream_factory, "get_stream_config", lambda: config(enabled=False))

    def must_not_load() -> Any:
        raise AssertionError("a disabled stream must not load the model")

    monkeypatch.setattr(stream_factory, "get_stream_engine", must_not_load)

    with asr_client.websocket_connect("/v1/stream") as ws:
        assert recv_json(ws) == {"type": "error", "code": "disabled"}
        with pytest.raises(WebSocketDisconnect) as closed:
            recv_json(ws)
        assert closed.value.code == 1008


@pytest.mark.unit
@pytest.mark.asr
def test_stream_close_message_ends_connection_and_drops_session(
    asr_client: TestClient, fake_engine: FakeEngine
) -> None:
    with asr_client.websocket_connect("/v1/stream") as ws:
        send_audio(ws, 0.6)
        assert recv_json(ws) == {"type": "partial", "text": "one"}
        ws.send_json({"type": "close"})
        with pytest.raises(WebSocketDisconnect) as closed:
            recv_json(ws)
        assert closed.value.code == 1000

    assert fake_engine.sessions[0].closed
    assert not fake_engine.sessions[0].finished


@pytest.mark.unit
@pytest.mark.asr
def test_stream_rejects_other_sample_rates(asr_client: TestClient, fake_engine: FakeEngine) -> None:
    with asr_client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "start", "sample_rate": 48000})
        assert recv_json(ws) == {"type": "error", "code": "unsupported_sample_rate"}
        with pytest.raises(WebSocketDisconnect) as closed:
            recv_json(ws)
        assert closed.value.code == 1003


@pytest.mark.unit
@pytest.mark.asr
def test_stream_odd_length_frames_keep_sample_alignment(asr_client: TestClient, fake_engine: FakeEngine) -> None:
    data = pcm(0.5)  # exactly one word's worth of samples
    with asr_client.websocket_connect("/v1/stream") as ws:
        ws.send_bytes(data[:3])  # 1.5 samples: one now, the odd byte carried over
        ws.send_bytes(data[3:])
        assert recv_json(ws) == {"type": "partial", "text": "one"}
        ws.send_json({"type": "bad"})
        assert recv_json(ws) == {"type": "error", "code": "bad_message"}
        ws.send_json({"type": "flush"})
        until_final(ws)

    assert fake_engine.sessions[0].samples == SR // 2
