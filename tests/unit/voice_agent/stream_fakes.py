"""Shared fakes for the streaming STT tests: a batch STT stand-in and an event collector."""

from __future__ import annotations

import asyncio
from typing import Any

from livekit import rtc
from livekit.agents import stt, utils, vad


class FakeBatch(stt.STT):
    """Stands in for the batch STT: streaming, so no VAD is needed; finalizes on flush."""

    def __init__(self) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=False))
        self.audio = bytearray()

    async def _recognize_impl(self, buffer: Any, *, language: Any = None, conn_options: Any = None) -> Any:
        raise NotImplementedError

    def stream(self, *, language: Any = None, conn_options: Any = None) -> stt.RecognizeStream:
        return _FakeBatchStream(self, conn_options)


class _FakeBatchStream(stt.RecognizeStream):
    def __init__(self, parent: FakeBatch, conn_options: Any) -> None:
        super().__init__(stt=parent, conn_options=conn_options)
        self._parent = parent

    async def _run(self) -> None:
        async for item in self._input_ch:
            if isinstance(item, rtc.AudioFrame):
                self._parent.audio.extend(bytes(item.data))
            else:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[stt.SpeechData(language="en", text="from batch")],
                    )
                )


class NonStreamingBatch(stt.STT):
    """Like the production batch STT (openai.STT): no streaming, one recognize() per VAD segment."""

    def __init__(self) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self.buffers: list[bytes] = []

    async def _recognize_impl(self, buffer: Any, *, language: Any = None, conn_options: Any = None) -> Any:
        self.buffers.append(bytes(utils.merge_frames(buffer).data))
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language="en", text="from batch")],
        )


class FakeVad(vad.VAD):
    """Speech is any frame with a non-zero sample; a silent frame ends the segment.

    Honours the VADStream contract that a flush discards the accumulated speech,
    so a flush forwarded at the wrong moment loses the utterance, as with Silero.
    """

    def __init__(self) -> None:
        super().__init__(capabilities=vad.VADCapabilities(update_interval=0.1))

    def stream(self) -> vad.VADStream:
        return _FakeVadStream(self)


class _FakeVadStream(vad.VADStream):
    async def _main_task(self) -> None:
        speech: list[rtc.AudioFrame] = []
        async for item in self._input_ch:
            if not isinstance(item, rtc.AudioFrame):
                speech = []
                continue
            if any(bytes(item.data)):
                if not speech:
                    self._event_ch.send_nowait(_vad_event(vad.VADEventType.START_OF_SPEECH, []))
                speech.append(item)
            elif speech:
                self._event_ch.send_nowait(_vad_event(vad.VADEventType.END_OF_SPEECH, speech))
                speech = []


def _vad_event(kind: vad.VADEventType, frames: list[rtc.AudioFrame]) -> vad.VADEvent:
    return vad.VADEvent(
        type=kind, samples_index=0, timestamp=0.0, speech_duration=0.0, silence_duration=0.0, frames=frames
    )


async def collect_until(stream: stt.RecognizeStream, kind: str, timeout: float = 5) -> list[stt.SpeechEvent]:
    """Events up to and including the first of type ``kind``."""
    events: list[stt.SpeechEvent] = []

    async def collect() -> None:
        async for ev in stream:
            events.append(ev)
            if ev.type == kind:
                return

    await asyncio.wait_for(collect(), timeout=timeout)
    return events
