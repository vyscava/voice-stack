"""
Streaming transcription over a WebSocket: ``/v1/stream``.

Protocol (JSON text frames for control, binary frames for audio):

- client -> ``{"type":"start","sample_rate":16000}`` (optional; only 16000 is accepted)
- client -> binary frames: PCM16 little-endian, mono, 16 kHz
- server -> ``{"type":"partial","text":"<full hypothesis so far>"}`` whenever it changes
- client -> ``{"type":"flush"}``: end of utterance
- server -> ``{"type":"final","text":"...","language":"en"|"pt"|...|null}``, then the
  connection is ready for the next utterance
- client -> ``{"type":"close"}`` (or just disconnect) ends the connection
- server -> ``{"type":"error","code":"..."}`` then a close, for: ``disabled`` (1008),
  ``busy`` (1013, one stream per process), ``unsupported_sample_rate`` (1003),
  ``unavailable``/``misconfigured``/``decode_failed`` (1011). ``bad_message`` keeps
  the connection open.

Language: the model emits a ``<xx-YY>`` tag after each completed sentence
("Que horas são? <pt-BR>"). The final carries the primary subtag of the LAST
tag seen in the utterance; an utterance without one (no sentence finished,
typically a short one) gets the connection's previous language, else null.
No extra decode runs for it.

Off by default (ASR_STREAM_ENABLED=false). The route stays registered and
refuses with ``disabled`` so a client gets a reason rather than a bare 403.
Every engine call runs in a thread executor; the event loop never blocks on the model.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable
from typing import Any, TypeVar

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from asr.stream import factory as stream_factory
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
                await _refuse(self.ws, "unsupported_sample_rate", CLOSE_UNSUPPORTED_DATA)
                return False
            return True
        if kind == "flush":
            await self._flush()
            return True
        if kind == "close":
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
            self.session = await self._call(self.engine.new_session)
        text = await self._call(self.session.feed, pcm16_to_float32(data))
        if text != self.last_sent:
            self.last_sent = text
            await self.ws.send_json({"type": "partial", "text": text})

    async def _flush(self) -> None:
        text = ""
        if self.session is not None:
            session, self.session = self.session, None
            try:
                text = await self._call(session.finish)
                # Read after finish(): the padded tail can still emit the last sentence's tag.
                if session.language is not None:
                    self.last_language = session.language
            finally:
                # Always release the decode thread, also when finish() raised
                # (e.g. its join timed out); close() is safe after finish().
                await self._close_session(session)
        await self.ws.send_json({"type": "final", "text": text, "language": self.last_language})
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
        with contextlib.suppress(Exception):  # logged by the callback; never masks the error being handled
            await asyncio.shield(work)


@router.websocket("/stream")
async def stream(websocket: WebSocket) -> None:
    """Streaming transcription. See the module docstring for the protocol."""
    await websocket.accept()
    try:
        config = stream_factory.get_stream_config()
    except ValueError:
        logger.exception("Streaming ASR is misconfigured")
        await _refuse(websocket, "misconfigured", CLOSE_INTERNAL_ERROR)
        return
    if not config.enabled:
        await _refuse(websocket, "disabled", CLOSE_POLICY_VIOLATION)
        return
    if not stream_factory.try_acquire_stream_slot():
        await _refuse(websocket, "busy", CLOSE_TRY_AGAIN_LATER)
        return
    try:
        try:
            engine = await asyncio.get_running_loop().run_in_executor(None, stream_factory.get_stream_engine)
        except Exception:
            logger.exception("Streaming ASR engine failed to load")
            await _refuse(websocket, "unavailable", CLOSE_INTERNAL_ERROR)
            return
        await _StreamConnection(websocket, engine).run()
    finally:
        stream_factory.release_stream_slot()
