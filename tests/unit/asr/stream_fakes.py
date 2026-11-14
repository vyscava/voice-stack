"""Shared fakes and helpers for the /v1/stream WebSocket tests.

The endpoint only talks to the StreamEngine protocol, so these fakes stand in
for the model: no download, no torch.
"""

from __future__ import annotations

import functools
import json
import time
from collections.abc import Callable
from typing import Any

import anyio
import numpy as np

from asr.stream.base import StreamConfig, parse_language_tag
from asr.stream.language import TextLanguageDetector

SR = 16000
WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight"]
SAMPLES_PER_WORD = SR // 2  # the fake hypothesis grows by one word per 0.5 s


class FakeSession:
    """Streams one fake word per 0.5 s, with the engine's language tags interleaved like the model's."""

    def __init__(self, engine: FakeEngine) -> None:
        self.engine = engine
        self.samples = 0
        self.finished = False
        self.closed = False

    def stream_output(self) -> list[str]:
        """Raw decoded tokens so far: words, plus an ``<xx-YY>`` tag wherever the engine scripts one."""
        out: list[str] = []
        for n, word in enumerate(WORDS[: self.samples // SAMPLES_PER_WORD], start=1):
            out.append(word)
            out.extend(tag for after, tag in self.engine.tags if after == n)
        return out

    def hypothesis(self) -> str:
        return " ".join(t for t in self.stream_output() if parse_language_tag(t) is None)

    @property
    def language(self) -> str | None:
        return parse_language_tag(" ".join(self.stream_output()))

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
        if self.engine.final_text is not None:
            return self.engine.final_text
        return self.hypothesis().capitalize() + "."

    def close(self) -> None:
        self.closed = True


class FakeEngine:
    def __init__(self) -> None:
        # (after word N, tag): the model's tag after a sentence. Default: "one two." then <pt-BR>,
        # so any utterance of at least 1 s carries a Portuguese tag.
        self.tags: list[tuple[int, str]] = [(2, "<pt-BR>")]
        self.sessions: list[FakeSession] = []
        self.feed_fails_after: int | None = None  # samples into a session before feed() raises
        self.finish_fails = False
        self.final_text: str | None = None  # overrides the final transcript when set
        self.text_language_calls: list[str] = []

    def new_session(self) -> FakeSession:
        session = FakeSession(self)
        self.sessions.append(session)
        return session

    def detect_text_language(self, text: str) -> str | None:
        """The REAL text language ID (lingua, en+pt), as the Nemotron engine runs it."""
        self.text_language_calls.append(text)
        return text_detector().detect(text)


@functools.lru_cache(maxsize=1)
def text_detector() -> TextLanguageDetector:
    """Built once for the whole test session (about 0.1 s)."""
    return TextLanguageDetector(("en", "pt"))


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


def wait_until(predicate: Callable[[], bool], timeout: float = RECV_TIMEOUT_S) -> bool:
    """Poll until predicate() is true or the deadline passes; return its last value.

    The server finishes a connection (closing the session, releasing the slot) on
    its own task after the client sees the close frame, so state checked right
    after the client context exits races that cleanup. A real leak still fails:
    the predicate stays false and the caller's assert sees it.
    """
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    return predicate()


def slot_is_free() -> bool:
    """True when the process's stream slot can be claimed (claims and releases it)."""
    from asr.stream import factory

    if factory.try_acquire_stream_slot():
        factory.release_stream_slot()
        return True
    return False
