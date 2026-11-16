"""Streaming STT: this project's ASR ``/v1/stream`` WebSocket as a LiveKit STT plugin.

The ASR server has no endpointing of its own: it transcribes until the client
says ``{"type":"flush"}``. So the turn is still found by VAD, and this plugin
declares ``manual_flush``. In this LiveKit version that capability is what makes
the session call ``RecognizeStream.flush()`` when VAD reports end of speech;
without it the flush never comes and no final transcript ever arrives.

Per utterance:

* audio frames (resampled to 16 kHz by the base class) go out as binary PCM16;
* each ``partial`` becomes an INTERIM_TRANSCRIPT (the first one also opens the
  utterance with START_OF_SPEECH);
* ``flush()`` sends ``{"type":"flush"}``; the ``final`` that answers it becomes a
  FINAL_TRANSCRIPT carrying the detected language, then END_OF_SPEECH.

**Fallback.** The stream endpoint serves one caller per process and can be
switched off. When it refuses (``busy``, ``disabled``), cannot be reached,
stalls, or fails mid-call, the call carries on with the batch STT for the rest
of the session: the audio no final has answered yet is replayed into it, so
the caller does not have to repeat themselves, and no error ever reaches the
session (three STT errors would end the call).

A stall is any of: no final within ``final_timeout_s`` of a flush, a send
blocked that long (a server that stopped reading), or unanswered WebSocket
pings (``heartbeat_s``).

After a fallback ``manual_flush`` stays declared (capabilities are read by the
session, not renegotiated). A batch STT that cannot stream runs behind
``stt.StreamAdapter``, which segments with its own VAD, exactly as in batch
mode; the session's flushes are dropped there rather than forwarded, because a
VAD flush discards the speech it has accumulated and batch mode never sends
one. The session still behaves: with no transcript after its flush it waits,
and the batch final re-enters end-of-turn detection.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections import deque
from collections.abc import Awaitable
from typing import Any, TypeVar

import aiohttp
from livekit import rtc
from livekit.agents import stt
from livekit.agents.language import LanguageCode
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer, aio

from voice_agent import metrics
from voice_agent.aio_util import within as _within
from voice_agent.stream_audio import SAMPLE_RATE, ReplayBuffer, frame_from_pcm16, mono_frame, pcm16_mono

logger = logging.getLogger("voice_agent")
T = TypeVar("T")

DEFAULT_FINAL_TIMEOUT_S = 2.0
DEFAULT_HEARTBEAT_S = 10.0
# Bounds the polite close of a connection being abandoned (a stalled server never answers it).
CLOSE_TIMEOUT_S = 1.0
# Fallback reasons: a fixed set, so the metric label stays low-cardinality.
REFUSALS = ("busy", "disabled")


class _Stalled(Exception):
    """A send did not complete in time: the server stopped reading."""


class AsrStreamSTT(stt.STT[Any]):
    """Streams to the ASR service; falls back to ``batch`` for the session when it cannot."""

    def __init__(
        self,
        *,
        url: str,
        batch: stt.STT[Any],
        vad: Any,
        final_timeout_s: float = DEFAULT_FINAL_TIMEOUT_S,
        heartbeat_s: float = DEFAULT_HEARTBEAT_S,
    ) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True, manual_flush=True))
        self._url = url
        self._batch = batch
        self._vad = vad
        self.final_timeout_s = final_timeout_s
        self.heartbeat_s = heartbeat_s
        self._batch_streamer: stt.STT[Any] | None = None
        self.fallback_reason: str | None = None
        self._batch.on("metrics_collected", self._forward_metrics)

    @property
    def url(self) -> str:
        return self._url

    @property
    def batch(self) -> stt.STT[Any]:
        return self._batch

    @property
    def model(self) -> str:
        return "asr-stream"

    @property
    def provider(self) -> str:
        return "voice-stack"

    def _forward_metrics(self, *args: Any, **kwargs: Any) -> None:
        self.emit("metrics_collected", *args, **kwargs)

    async def _recognize_impl(
        self, buffer: AudioBuffer, *, language: NotGivenOr[str] = NOT_GIVEN, conn_options: APIConnectOptions
    ) -> stt.SpeechEvent:
        # One-shot recognition has nothing to stream: the batch STT does it.
        return await self._batch.recognize(buffer, language=language, conn_options=conn_options)

    def stream(
        self, *, language: NotGivenOr[str] = NOT_GIVEN, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> stt.RecognizeStream:
        return AsrSpeechStream(stt=self, conn_options=conn_options)

    def fall_back(self, reason: str) -> None:
        """Use the batch STT for the rest of this session. Counted and logged once."""
        if self.fallback_reason is not None:
            return
        self.fallback_reason = reason
        metrics.STT_STREAM_FALLBACKS.labels(reason=reason).inc()
        logger.warning(f"streaming ASR unavailable ({reason}); using batch STT for the rest of this call")

    def batch_stream(self, conn_options: APIConnectOptions) -> tuple[stt.RecognizeStream, bool]:
        """A stream from the batch STT, and whether it takes flushes.

        A batch STT that cannot stream is wrapped in VAD segmentation, as the
        session does in batch mode, and is never flushed (see the module docstring).
        """
        if self._batch.capabilities.streaming:
            return self._batch.stream(conn_options=conn_options), True
        if self._batch_streamer is None:
            if self._vad is None:
                raise RuntimeError("batch STT fallback needs a VAD")
            self._batch_streamer = stt.StreamAdapter(stt=self._batch, vad=self._vad)
        return self._batch_streamer.stream(conn_options=conn_options), False

    async def aclose(self) -> None:
        self._batch.off("metrics_collected", self._forward_metrics)
        if self._batch_streamer is not None:
            await self._batch_streamer.aclose()
        await self._batch.aclose()


class AsrSpeechStream(stt.RecognizeStream):
    def __init__(self, *, stt: AsrStreamSTT, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._parent = stt
        self._timeout_s = stt.final_timeout_s
        self._replay = ReplayBuffer()
        self._language = ""
        self._in_speech = False
        self._pending_flushes: deque[float] = deque()  # send time of each unanswered flush
        self._flush_sent = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._usage_samples = 0  # audio streamed since the last usage event

    async def _run(self) -> None:
        if self._parent.fallback_reason is None:
            reason = await self._run_stream()
            if reason is None:
                return
            self._parent.fall_back(reason)
        await self._run_batch(self._replay.segments())

    async def _run_stream(self) -> str | None:
        """Stream until the input ends (None) or the server cannot serve us (the reason)."""
        async with aiohttp.ClientSession() as http:
            try:
                ws = await _within(
                    http.ws_connect(self._parent.url, heartbeat=self._parent.heartbeat_s),
                    self._conn_options.timeout,
                )
                await _within(ws.send_json({"type": "start", "sample_rate": SAMPLE_RATE}), self._timeout_s)
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                logger.info(f"streaming ASR connect failed: {e!r}")
                return "connect"
            send = asyncio.create_task(self._send(ws), name="asr_stream_send")
            recv = asyncio.create_task(self._recv(ws), name="asr_stream_recv")
            watch = asyncio.create_task(self._watch_finals(), name="asr_stream_watch")
            try:
                return await self._outcome(send, recv, watch)
            finally:
                await aio.cancel_and_wait(send, recv, watch)
                with contextlib.suppress(Exception):
                    await _within(ws.close(), CLOSE_TIMEOUT_S)

    async def _outcome(self, send: asyncio.Task[None], recv: asyncio.Task[str], watch: asyncio.Task[str]) -> str | None:
        done, _ = await asyncio.wait({send, recv, watch}, return_when=asyncio.FIRST_COMPLETED)
        if send in done and send.exception() is None:
            return None  # input ended and the last final came back
        if watch in done:
            return watch.result()
        if recv not in done:
            error = send.exception()
            if isinstance(error, _Stalled):
                logger.info(f"streaming ASR send blocked for {self._timeout_s}s: the server stopped reading")
                return "stall"
            # A failed send usually means the server closed on us: its error message, if any, says why.
            logger.info(f"streaming ASR send failed: {error!r}")
            await asyncio.wait({recv}, timeout=CLOSE_TIMEOUT_S)
        if recv.done() and not recv.cancelled() and recv.exception() is None:
            return recv.result()
        return "error"

    async def _bounded(self, send: Awaitable[None]) -> None:
        try:
            await _within(send, self._timeout_s)
        except asyncio.TimeoutError as e:
            raise _Stalled from e

    async def _send(self, ws: aiohttp.ClientWebSocketResponse[bool]) -> None:
        async for item in self._input_ch:
            if isinstance(item, rtc.AudioFrame):
                data = pcm16_mono(item)
                self._replay.add(data)
                self._usage_samples += len(data) // 2
                await self._bounded(ws.send_bytes(data))
            else:
                self._replay.flushed()
                self._pending_flushes.append(time.monotonic())
                self._idle.clear()
                self._flush_sent.set()
                await self._bounded(ws.send_json({"type": "flush"}))
        # end_input() flushes before closing: let that final arrive (the watchdog bounds the wait).
        await self._idle.wait()
        with contextlib.suppress(aiohttp.ClientError, ConnectionError, _Stalled):
            await self._bounded(ws.send_json({"type": "close"}))

    async def _watch_finals(self) -> str:
        """Returns "stall" once a flush has gone unanswered for the timeout."""
        while True:
            if self._pending_flushes:
                remaining = self._pending_flushes[0] + self._timeout_s - time.monotonic()
                if remaining <= 0:
                    logger.info(f"streaming ASR sent no final within {self._timeout_s}s of a flush")
                    return "stall"
                await asyncio.sleep(remaining)
            else:
                self._flush_sent.clear()
                await self._flush_sent.wait()

    async def _recv(self, ws: aiohttp.ClientWebSocketResponse[bool]) -> str:
        """Turn server messages into speech events; returns why the stream can no longer serve."""
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                event = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.warning("streaming ASR sent a frame that is not JSON")
                continue
            kind = event.get("type") if isinstance(event, dict) else None
            if kind == "partial":
                self._on_partial(str(event.get("text") or ""))
            elif kind == "final":
                self._on_final(event)
            elif kind == "error":
                code = str(event.get("code") or "")
                if code == "bad_message":  # the connection stays usable
                    logger.warning("streaming ASR rejected a control message")
                    continue
                logger.info(f"streaming ASR error: {code}")
                return code if code in REFUSALS else "error"
        # The connection ended while we still had a call. Unanswered pings end it with a timeout.
        if isinstance(ws.exception(), asyncio.TimeoutError):
            logger.info("streaming ASR stopped answering pings")
            return "stall"
        return "error"

    def _speech(self, kind: stt.SpeechEventType, text: str = "", **extra: Any) -> None:
        alternatives = (
            [stt.SpeechData(language=LanguageCode(self._language), text=text, **extra)] if kind in _TRANSCRIPTS else []
        )
        self._event_ch.send_nowait(stt.SpeechEvent(type=kind, alternatives=alternatives))

    def _emit_usage(self) -> None:
        """LiveKit usage accounting: the audio streamed since the last usage event, in seconds."""
        if self._usage_samples <= 0:
            return
        seconds, self._usage_samples = self._usage_samples / SAMPLE_RATE, 0
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                recognition_usage=stt.RecognitionUsage(audio_duration=seconds),
            )
        )

    def _on_partial(self, text: str) -> None:
        if not text:
            return
        if not self._in_speech:
            self._in_speech = True
            self._speech(stt.SpeechEventType.START_OF_SPEECH)
        self._speech(stt.SpeechEventType.INTERIM_TRANSCRIPT, text)

    def _on_final(self, event: dict[str, Any]) -> None:
        if self._pending_flushes:
            sent_at = self._pending_flushes.popleft()
            metrics.STAGE_SECONDS.labels(stage="stt").observe(time.monotonic() - sent_at)
        if not self._pending_flushes:
            self._idle.set()
        self._replay.finalized()
        language = event.get("language")
        if isinstance(language, str) and language:
            self._language = language
        source = event.get("language_source")  # optional: older servers do not send it
        meta = {"language_source": source} if isinstance(source, str) else None
        self._speech(stt.SpeechEventType.FINAL_TRANSCRIPT, str(event.get("text") or "").strip(), metadata=meta)
        if self._in_speech:
            self._in_speech = False
            self._speech(stt.SpeechEventType.END_OF_SPEECH)
        self._emit_usage()

    async def _run_batch(self, replay: list[tuple[bytes, bool]]) -> None:
        """Serve the rest of this stream from the batch STT, starting with the replayed audio."""
        inner, takes_flush = self._parent.batch_stream(self._conn_options)

        async def forward_input() -> None:
            for audio, flushed in replay:
                if audio:
                    inner.push_frame(frame_from_pcm16(audio))
                if flushed and takes_flush:
                    inner.flush()
            async for item in self._input_ch:
                if isinstance(item, rtc.AudioFrame):
                    inner.push_frame(mono_frame(item))  # mono like the replay: one channel count per stream
                elif takes_flush:
                    inner.flush()
            inner.end_input()

        feeder = asyncio.create_task(forward_input(), name="asr_batch_feed")
        try:
            async for ev in inner:
                self._event_ch.send_nowait(ev)
        finally:
            await aio.cancel_and_wait(feeder)
            await inner.aclose()


_TRANSCRIPTS = (stt.SpeechEventType.INTERIM_TRANSCRIPT, stt.SpeechEventType.FINAL_TRANSCRIPT)
