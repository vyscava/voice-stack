"""
Streaming transcription over a WebSocket: ``/v1/stream``.

Protocol (JSON text frames for control, binary frames for audio):

- client -> ``{"type":"start","sample_rate":16000}`` (optional; only 16000 is accepted)
- client -> binary frames: PCM16 little-endian, mono, 16 kHz
- server -> ``{"type":"partial","text":"<full hypothesis so far>"}`` whenever it changes
- client -> ``{"type":"flush"}``: end of utterance
- server -> ``{"type":"final","text":"...","language":"en"|"pt"|...|null,
  "language_source":"stream_tag"|"text"|"previous"|null}``, then the connection is
  ready for the next utterance
- client -> ``{"type":"close"}`` (or just disconnect) ends the connection
- server -> ``{"type":"error","code":"..."}`` then a close, for: ``disabled`` (1008),
  ``busy`` (1013, one stream per process), ``unsupported_sample_rate`` (1003),
  ``unavailable``/``misconfigured``/``decode_failed`` (1011). ``bad_message`` keeps
  the connection open.

Language, first match wins (``language_source`` says which):

1. ``stream_tag``: the model emits a ``<xx-YY>`` tag after each completed
   sentence; the last one of the utterance. Free, but the model never
   punctuates an utterance's final sentence, so one-sentence turns rarely
   carry one.
2. ``text``: the final transcript classified by lingua, restricted to
   ASR_STREAM_LANGUAGES, when the verdict is confident.
3. ``previous``: the connection's previous final language, when the text is
   empty or ambiguous.
4. null (with ``language_source`` null) on a first utterance that gave nothing.

Off by default (ASR_STREAM_ENABLED=false). The route stays registered and
refuses with ``disabled`` so a client gets a reason rather than a bare 403.
Every engine call runs in a thread executor; the event loop never blocks on the model.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from asr.stream import factory as stream_factory
from asr.stream import metrics as stream_metrics
from asr.stream.base import STREAM_SAMPLE_RATE, StreamEngine, StreamSession, pcm16_to_float32
from core.logging import logger_asr as logger

router = APIRouter()

T = TypeVar("T")

# WebSocket close codes (RFC 6455 section 7.4.1).
CLOSE_NORMAL = 1000
CLOSE_UNSUPPORTED_DATA = 1003
CLOSE_POLICY_VIOLATION = 1008
CLOSE_INTERNAL_ERROR = 1011
CLOSE_TRY_AGAIN_LATER = 1013


async def _refuse(websocket: WebSocket, code: str, close_code: int) -> None:
    """Tell the client why, then close. Tolerates a client that already left."""
    try:
        await websocket.send_json({"type": "error", "code": code})
        await websocket.close(code=close_code)
    except (WebSocketDisconnect, RuntimeError):
        pass


def _log_close_failure(future: asyncio.Future[Any]) -> None:
    """Log a failed session close; also runs when nobody awaits the close any more (cancelled caller)."""
    if not future.cancelled() and future.exception() is not None:
        logger.error("Closing an abandoned streaming session failed", exc_info=future.exception())


class _StreamConnection:
    """State for one accepted, slot-holding connection. Utterances follow each other on it."""

    def __init__(self, websocket: WebSocket, engine: StreamEngine) -> None:
        self.ws = websocket
        self.engine = engine
        self.loop = asyncio.get_running_loop()
        self.session: StreamSession | None = None
        self.last_language: str | None = None
        # The latest session close, which may still be running when the handler ends.
        self._closing: asyncio.Future[None] | None = None
        self._reset_utterance()

    def _reset_utterance(self) -> None:
        self.last_sent = ""
        self.carry = b""  # odd trailing byte of a frame, completed by the next one

    async def _call(self, fn: Callable[..., T], *args: Any) -> T:
        return await self.loop.run_in_executor(None, fn, *args)

    async def run(self) -> None:
        try:
            while True:
                message = await self.ws.receive()
                if message["type"] == "websocket.disconnect":
                    return
                if message.get("bytes") is not None:
                    await self._on_audio(message["bytes"])
                elif message.get("text") is not None and not await self._on_control(message["text"]):
                    return
        except WebSocketDisconnect:
            return
        except Exception:
            logger.exception("Streaming ASR connection failed")
            stream_metrics.ERRORS.inc()
            await _refuse(self.ws, "decode_failed", CLOSE_INTERNAL_ERROR)
        finally:
            await self._drop_session()

    async def _on_control(self, raw: str) -> bool:
        """Handle a JSON control frame. False means the connection is over."""
        try:
            msg = json.loads(raw)
            kind = msg.get("type") if isinstance(msg, dict) else None
        except json.JSONDecodeError:
            kind = None

        if kind == "start":
            rate = msg.get("sample_rate", STREAM_SAMPLE_RATE)
            if rate != STREAM_SAMPLE_RATE:
                stream_metrics.REFUSALS.labels(code="unsupported_sample_rate").inc()
                await _refuse(self.ws, "unsupported_sample_rate", CLOSE_UNSUPPORTED_DATA)
                return False
            return True
        if kind == "flush":
            await self._flush()
            return True
        if kind == "close":
            with contextlib.suppress(WebSocketDisconnect, RuntimeError):  # the client may already be gone
                await self.ws.close(code=CLOSE_NORMAL)
            return False
        await self.ws.send_json({"type": "error", "code": "bad_message"})
        return True

    async def _on_audio(self, data: bytes) -> None:
        data = self.carry + data
        if len(data) % 2:
            self.carry, data = data[-1:], data[:-1]
        else:
            self.carry = b""
        if not data:
            return
        if self.session is None:
            self.session = await self._new_session()
        started = time.perf_counter()
        text = await self._call(self.session.feed, pcm16_to_float32(data))
        stream_metrics.FEED_SECONDS.observe(time.perf_counter() - started)
        if text != self.last_sent:
            self.last_sent = text
            await self.ws.send_json({"type": "partial", "text": text})

    async def _new_session(self) -> StreamSession:
        """Start a session; if we are cancelled while it starts, close it once it exists.

        new_session() starts a decode thread. Cancelling a plain executor await
        drops the result while that thread runs on, blocked on its audio queue
        with nobody to close it. Shielded, the session still arrives; the
        cancelled caller hands it to a closer instead.
        """
        work = self.loop.run_in_executor(None, self.engine.new_session)
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            # The slot waits for this too (release_when_closed): the orphan's
            # decode thread is still starting, then closing.
            self._closing = self._close_when_started(work)
            raise

    def _close_when_started(self, work: asyncio.Future[StreamSession]) -> asyncio.Future[None]:
        """A future that resolves once ``work``'s session (if any) is closed.

        close() joins a thread, so it runs on a thread of its own, never on the loop.
        """
        closed: asyncio.Future[None] = self.loop.create_future()

        def signal() -> None:
            if not closed.done():
                closed.set_result(None)

        def close_then_signal(session: StreamSession) -> None:
            try:
                session.close()
            except Exception:
                logger.exception("Closing an orphaned streaming session failed")
            finally:
                with contextlib.suppress(RuntimeError):  # loop already closed: nobody is waiting
                    self.loop.call_soon_threadsafe(signal)

        def on_started(future: asyncio.Future[StreamSession]) -> None:
            if future.cancelled() or future.exception() is not None:
                if not future.cancelled():
                    logger.error("A cancelled stream's session failed to start", exc_info=future.exception())
                signal()
                return
            threading.Thread(
                target=close_then_signal, args=(future.result(),), name="stream-orphan-close", daemon=True
            ).start()

        work.add_done_callback(on_started)
        return closed

    async def _flush(self) -> None:
        started = time.perf_counter()
        text = ""
        language: str | None = None
        source: str | None = None
        had_audio = self.session is not None
        if self.session is not None:
            session, self.session = self.session, None
            try:
                text = await self._call(session.finish)
                # Read after finish(): the padded tail can still emit the last sentence's tag.
                if session.language is not None:
                    language, source = session.language, "stream_tag"
            finally:
                # Always release the decode thread, also when finish() raised
                # (e.g. its join timed out); close() is safe after finish().
                await self._close_session(session)
        if language is None and text:
            language = await self._call(self.engine.detect_text_language, text)
            source = "text" if language is not None else None
        if language is None and self.last_language is not None:
            language, source = self.last_language, "previous"
        self.last_language = language
        await self.ws.send_json({"type": "final", "text": text, "language": language, "language_source": source})
        if had_audio:
            stream_metrics.FLUSH_SECONDS.observe(time.perf_counter() - started)
        stream_metrics.FINALS.labels(language_source=source or "none").inc()
        self._reset_utterance()

    async def _drop_session(self) -> None:
        if self.session is None:
            return
        session, self.session = self.session, None
        await self._close_session(session)

    async def _close_session(self, session: StreamSession) -> None:
        """Close ``session`` in the executor, and make sure that close RUNS even if we are cancelled.

        This is cleanup, so it usually runs in a ``finally`` while the task is
        being cancelled (server shutdown, or a test client that cancels the
        handler right after sending its disconnect). Awaiting a plain
        run_in_executor() future there is not enough: cancelling the awaiting
        task cancels that future, and asyncio then cancels the queued
        concurrent.futures work item, so if no worker thread had picked it up
        yet, close() never runs and the session's decode thread leaks.
        asyncio.shield() keeps the work item alive; the cancellation itself
        still propagates to the caller at once.
        """
        try:
            work = self.loop.run_in_executor(None, session.close)
        except RuntimeError:  # executor already shut down (loop closing)
            logger.exception("Could not schedule the stream session close")
            return
        work.add_done_callback(_log_close_failure)
        self._closing = work
        with contextlib.suppress(Exception):  # logged by the callback; never masks the error being handled
            await asyncio.shield(work)

    def release_when_closed(self, release: Callable[[], None]) -> None:
        """Call ``release`` now, or once the last session close finishes if it is still running.

        A cancelled handler stops awaiting its close (see ``_close_session``)
        while the decode thread is still winding down. Handing the slot to the
        next stream then would overlap two decodes on the same cores.
        """
        closing = self._closing
        if closing is None or closing.done():
            release()
        else:
            closing.add_done_callback(lambda _done: release())


def _release_slot() -> None:
    stream_metrics.ACTIVE.dec()
    stream_factory.release_stream_slot()


@router.websocket("/stream")
async def stream(websocket: WebSocket) -> None:
    """Streaming transcription. See the module docstring for the protocol."""
    await websocket.accept()
    try:
        config = stream_factory.get_stream_config()
    except ValueError:
        logger.exception("Streaming ASR is misconfigured")
        stream_metrics.CONNECTIONS.labels(outcome="misconfigured").inc()
        await _refuse(websocket, "misconfigured", CLOSE_INTERNAL_ERROR)
        return
    if not config.enabled:
        stream_metrics.CONNECTIONS.labels(outcome="disabled").inc()
        await _refuse(websocket, "disabled", CLOSE_POLICY_VIOLATION)
        return
    if not stream_factory.try_acquire_stream_slot():
        stream_metrics.CONNECTIONS.labels(outcome="busy").inc()
        await _refuse(websocket, "busy", CLOSE_TRY_AGAIN_LATER)
        return
    stream_metrics.ACTIVE.inc()
    connection: _StreamConnection | None = None
    try:
        try:
            engine = await asyncio.get_running_loop().run_in_executor(None, stream_factory.get_stream_engine)
        except Exception:
            logger.exception("Streaming ASR engine failed to load")
            stream_metrics.CONNECTIONS.labels(outcome="unavailable").inc()
            await _refuse(websocket, "unavailable", CLOSE_INTERNAL_ERROR)
            return
        stream_metrics.CONNECTIONS.labels(outcome="accepted").inc()
        connection = _StreamConnection(websocket, engine)
        await connection.run()
    finally:
        if connection is None:
            _release_slot()
        else:
            connection.release_when_closed(_release_slot)
