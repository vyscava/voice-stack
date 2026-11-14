"""Shared fakes and helpers for the /v1/stream WebSocket tests.

The endpoint only talks to the StreamEngine protocol, so these fakes stand in
for the model: no download, no torch.
"""

from __future__ import annotations

import json
import threading
from typing import Any

import anyio
import numpy as np

from asr.stream.base import StreamConfig

SR = 16000
WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight"]
SAMPLES_PER_WORD = SR // 2  # the fake hypothesis grows by one word per 0.5 s


class FakeSession:
    def __init__(self, engine: FakeEngine) -> None:
        self.engine = engine
        self.samples = 0
        self.finished = False
        self.closed = False

    def hypothesis(self) -> str:
        return " ".join(WORDS[: self.samples // SAMPLES_PER_WORD])

    def feed(self, samples: Any) -> str:
        assert samples.dtype == np.float32
        self.samples += int(samples.size)
        if self.engine.feed_fails_after is not None and self.samples > self.engine.feed_fails_after:
            raise RuntimeError("fake decode failure")
        return self.hypothesis()

    def finish(self) -> str:
        self.finished = True
        if self.engine.finish_fails:
            raise TimeoutError("fake: streaming decode did not finish in time")
        return self.hypothesis().capitalize() + "."

    def close(self) -> None:
        self.closed = True


class FakeEngine:
    def __init__(self, language: str | None = "pt") -> None:
        self.language = language
        self.sessions: list[FakeSession] = []
        self.detect_calls: list[int] = []
        self.release_detection = threading.Event()
        self.release_detection.set()  # set = detection answers at once
        self.detect_returned = threading.Event()
        self.feed_fails_after: int | None = None  # samples into a session before feed() raises
        self.finish_fails = False

    def new_session(self) -> FakeSession:
        session = FakeSession(self)
        self.sessions.append(session)
        return session

    def detect_language(self, samples: Any) -> str | None:
        self.detect_calls.append(int(samples.size))
        self.release_detection.wait(5)
        self.detect_returned.set()
        return self.language


def config(enabled: bool = True) -> StreamConfig:
    return StreamConfig(enabled=enabled, model="fake", lookahead=3, dtype="float32", threads=1, device="cpu")


def pcm(seconds: float) -> bytes:
    """Non-silent PCM16 so nothing downstream could special-case zeros."""
    n = int(seconds * SR)
    return (np.sin(np.arange(n) / 10.0) * 8000).astype("<i2").tobytes()


def send_audio(ws: Any, seconds: float, frame_ms: int = 100) -> None:
    data = pcm(seconds)
    step = SR * frame_ms // 1000 * 2
    for i in range(0, len(data), step):
        ws.send_bytes(data[i : i + step])


def until_final(ws: Any) -> tuple[list[str], dict[str, Any]]:
    partials: list[str] = []
    while True:
        msg = recv_json(ws)
        if msg["type"] == "partial":
            partials.append(msg["text"])
        elif msg["type"] == "final":
            return partials, msg
        else:
            raise AssertionError(f"unexpected message {msg}")


# Upper bound on any single wait for a server message. A regression that leaves
# the server silent (a missing refusal, a stuck handler) fails here instead of
# hanging the suite: the repo has no pytest-timeout.
RECV_TIMEOUT_S = 5.0


def recv_json(ws: Any, timeout: float = RECV_TIMEOUT_S) -> Any:
    """recv_json(WebSocketTestSession) with a deadline.

    Starlette's own receive blocks forever. This runs the same receive on the
    session's portal under anyio.fail_after, so it raises TimeoutError; leaving
    the ``with`` block then disconnects the client and the server handler ends,
    so nothing is left hanging. Uses the session's private ``_send_rx`` and
    ``_raise_on_close`` (starlette 1.x); if those move, this fails loudly.
    """

    async def receive() -> Any:
        with anyio.fail_after(timeout):
            return await ws._send_rx.receive()

    message = ws.portal.call(receive)
    ws._raise_on_close(message)  # WebSocketDisconnect(code) on a close frame
    return json.loads(message["text"])
