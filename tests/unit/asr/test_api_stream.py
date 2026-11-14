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
from tests.unit.asr.stream_fakes import (
    SR,
    WORDS,
    FakeEngine,
    config,
    pcm,
    recv_json,
    send_audio,
    slot_is_free,
    until_final,
    wait_until,
)


@pytest.mark.unit
@pytest.mark.asr
def test_stream_happy_path_partials_then_final_with_language(asr_client: TestClient, fake_engine: FakeEngine) -> None:
    with asr_client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "start", "sample_rate": SR})
        send_audio(ws, 3.0)
        ws.send_json({"type": "flush"})
        partials, final = until_final(ws)

    assert partials == [" ".join(WORDS[:n]) for n in range(1, 7)]
    # The tag after "two" never reaches the text, but it sets the language.
    assert final == {"type": "final", "text": "One two three four five six.", "language": "pt"}
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


@pytest.mark.unit
@pytest.mark.asr
def test_stream_utterance_without_a_tag_reuses_previous_language(
    asr_client: TestClient, fake_engine: FakeEngine
) -> None:
    with asr_client.websocket_connect("/v1/stream") as ws:
        send_audio(ws, 2.5)  # reaches the tag after "two"
        ws.send_json({"type": "flush"})
        _, first = until_final(ws)
        send_audio(ws, 0.5)  # "one" only: no sentence end, no tag
        ws.send_json({"type": "flush"})
        _, second = until_final(ws)

    assert first["language"] == "pt"
    assert second["language"] == "pt"


@pytest.mark.unit
@pytest.mark.asr
def test_stream_first_utterance_without_a_tag_has_null_language(
    asr_client: TestClient, fake_engine: FakeEngine
) -> None:
    with asr_client.websocket_connect("/v1/stream") as ws:
        send_audio(ws, 0.5)
        ws.send_json({"type": "flush"})
        _, final = until_final(ws)

    assert final["language"] is None


@pytest.mark.unit
@pytest.mark.asr
def test_stream_last_tag_of_the_utterance_wins(asr_client: TestClient, fake_engine: FakeEngine) -> None:
    fake_engine.tags = [(1, "<en-US>"), (3, "<pt-BR>")]
    with asr_client.websocket_connect("/v1/stream") as ws:
        send_audio(ws, 2.0)
        ws.send_json({"type": "flush"})
        _, final = until_final(ws)

    assert final["text"] == "One two three four."
    assert final["language"] == "pt"


@pytest.mark.unit
@pytest.mark.asr
def test_stream_a_new_tag_replaces_the_previous_language(asr_client: TestClient, fake_engine: FakeEngine) -> None:
    with asr_client.websocket_connect("/v1/stream") as ws:
        send_audio(ws, 1.0)
        ws.send_json({"type": "flush"})
        _, first = until_final(ws)
        fake_engine.tags = [(2, "<en-US>")]
        send_audio(ws, 1.0)
        ws.send_json({"type": "flush"})
        _, second = until_final(ws)

    assert (first["language"], second["language"]) == ("pt", "en")


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
    assert wait_until(slot_is_free)
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

    assert wait_until(lambda: fake_engine.sessions[0].closed)
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
