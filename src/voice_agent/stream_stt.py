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
switched off. When it refuses (``busy``, ``disabled``), cannot be reached, or
fails mid-call, the call carries on with the batch STT for the rest of the
session: the audio of the utterance in progress is replayed into it, so the
caller does not have to repeat themselves, and no error ever reaches the
session (three STT errors would end the call).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
import time
from array import array
from collections import deque
from typing import Any

import aiohttp
from livekit import rtc
from livekit.agents import stt
from livekit.agents.language import LanguageCode
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer, aio

from voice_agent import metrics

logger = logging.getLogger("voice_agent")

SAMPLE_RATE = 16000
# Audio kept for replay into the batch STT if the stream fails mid-utterance.
REPLAY_SECONDS = 30.0
_REPLAY_BYTES = int(REPLAY_SECONDS * SAMPLE_RATE) * 2
# After the input ends, how long to wait for the final of the last flush.
FINAL_WAIT_S = 5.0
# Fallback reasons: a fixed set, so the metric label stays low-cardinality.
REFUSALS = ("busy", "disabled")


class AsrStreamSTT(stt.STT[Any]):
    """Streams to the ASR service; falls back to ``batch`` for the session when it cannot."""

    def __init__(self, *, url: str, batch: stt.STT[Any], vad: Any) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True, manual_flush=True))
        self._url = url
        self._batch = batch
        self._vad = vad
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

    def batch_stream(self, conn_options: APIConnectOptions) -> stt.RecognizeStream:
        """A stream from the batch STT, wrapped in VAD segmentation when it cannot stream itself."""
        if self._batch.capabilities.streaming:
            return self._batch.stream(conn_options=conn_options)
        if self._batch_streamer is None:
            if self._vad is None:
                raise RuntimeError("batch STT fallback needs a VAD")
            self._batch_streamer = stt.StreamAdapter(stt=self._batch, vad=self._vad)
        return self._batch_streamer.stream(conn_options=conn_options)

    async def aclose(self) -> None:
        self._batch.off("metrics_collected", self._forward_metrics)
        if self._batch_streamer is not None:
            await self._batch_streamer.aclose()
        await self._batch.aclose()


def pcm16_mono(frame: rtc.AudioFrame) -> bytes:
    """The frame as mono PCM16 little-endian (channels averaged)."""
    samples = array("h")
    samples.frombytes(bytes(frame.data))
    n = frame.num_channels
    if n > 1:
        samples = array("h", (sum(samples[i : i + n]) // n for i in range(0, len(samples), n)))
    if sys.byteorder == "big":
        samples.byteswap()
    return samples.tobytes()


def frame_from_pcm16(data: bytes) -> rtc.AudioFrame:
    if sys.byteorder == "big":
        samples = array("h")
        samples.frombytes(data)
        samples.byteswap()
        data = samples.tobytes()
    return rtc.AudioFrame(data=data, sample_rate=SAMPLE_RATE, num_channels=1, samples_per_channel=len(data) // 2)


class _ReplayBuffer:
    """Audio not yet covered by a final, one segment per flush, capped at REPLAY_SECONDS."""

    def __init__(self) -> None:
        self._segments: deque[bytearray] = deque([bytearray()])

    def add(self, data: bytes) -> None:
        self._segments[-1].extend(data)
        excess = sum(len(s) for s in self._segments) - _REPLAY_BYTES
        while excess > 0 and self._segments:
            head = self._segments[0]
            cut = min(excess, len(head))
            del head[:cut]
            excess -= cut
            if not head and len(self._segments) > 1:
                self._segments.popleft()

    def flushed(self) -> None:
        self._segments.append(bytearray())

    def finalized(self) -> None:
        if len(self._segments) > 1:
            self._segments.popleft()
        else:
            self._segments[0].clear()

    def segments(self) -> list[tuple[bytes, bool]]:
        """Each segment's audio, and whether a flush (still unanswered) followed it."""
        last = len(self._segments) - 1
        return [(bytes(seg), i < last) for i, seg in enumerate(self._segments)]


class AsrSpeechStream(stt.RecognizeStream):
    def __init__(self, *, stt: AsrStreamSTT, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._parent = stt
        self._replay = _ReplayBuffer()
        self._language = ""
        self._in_speech = False
        self._pending_flushes: deque[float] = deque()  # send time of each unanswered flush
        self._idle = asyncio.Event()
        self._idle.set()

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
                ws = await asyncio.wait_for(http.ws_connect(self._parent.url), self._conn_options.timeout)
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                logger.info(f"streaming ASR connect failed: {e!r}")
                return "connect"
            async with ws:
                await ws.send_json({"type": "start", "sample_rate": SAMPLE_RATE})
                send = asyncio.create_task(self._send(ws), name="asr_stream_send")
                recv = asyncio.create_task(self._recv(ws), name="asr_stream_recv")
                try:
                    done, _ = await asyncio.wait({send, recv}, return_when=asyncio.FIRST_COMPLETED)
                    if send in done and send.exception() is None:
                        return None  # input ended and the last final came back (or timed out)
                    if recv not in done:
                        # A failed send usually means the server closed on us: its
                        # error message, if any, says why.
                        logger.info(f"streaming ASR send failed: {send.exception()!r}")
                        await asyncio.wait({recv}, timeout=1.0)
                    if recv.done() and not recv.cancelled() and recv.exception() is None:
                        return recv.result()
                    return "error"
                finally:
                    await aio.cancel_and_wait(send, recv)

    async def _send(self, ws: aiohttp.ClientWebSocketResponse[bool]) -> None:
        async for item in self._input_ch:
            if isinstance(item, rtc.AudioFrame):
                data = pcm16_mono(item)
                self._replay.add(data)
                await ws.send_bytes(data)
            else:
                self._replay.flushed()
                self._pending_flushes.append(time.monotonic())
                self._idle.clear()
                await ws.send_json({"type": "flush"})
        # end_input() flushes before closing: let that final arrive before leaving.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._idle.wait(), FINAL_WAIT_S)
        with contextlib.suppress(aiohttp.ClientError, ConnectionError):
            await ws.send_json({"type": "close"})

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
        return "error"  # the server closed the connection while we still had a call

    def _speech(self, kind: stt.SpeechEventType, text: str = "", **extra: Any) -> None:
        alternatives = (
            [stt.SpeechData(language=LanguageCode(self._language), text=text, **extra)] if kind in _TRANSCRIPTS else []
        )
        self._event_ch.send_nowait(stt.SpeechEvent(type=kind, alternatives=alternatives))

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

    async def _run_batch(self, replay: list[tuple[bytes, bool]]) -> None:
        """Serve the rest of this stream from the batch STT, starting with the replayed audio."""
        inner = self._parent.batch_stream(self._conn_options)

        async def forward_input() -> None:
            for audio, flushed in replay:
                if audio:
                    inner.push_frame(frame_from_pcm16(audio))
                if flushed:
                    inner.flush()
            async for item in self._input_ch:
                if isinstance(item, rtc.AudioFrame):
                    inner.push_frame(item)
                else:
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


def stream_url(asr_url: str, override: str = "") -> str:
    """The ``/v1/stream`` WebSocket URL: ``override`` if set, else derived from the ASR base URL."""
    if override.strip():
        return override.strip()
    base = asr_url.strip().rstrip("/")
    for http, ws in (("https://", "wss://"), ("http://", "ws://")):
        if base.startswith(http):
            base = ws + base[len(http) :]
            break
    return base + "/v1/stream"
