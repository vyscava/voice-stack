"""Shared fakes for the streaming STT tests: a batch STT stand-in and an event collector."""

from __future__ import annotations

import asyncio
from typing import Any

from livekit import rtc
from livekit.agents import stt


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
