"""Batch STT with a primary service and a fallback for the same audio.

Each utterance (one VAD segment, as in batch mode) goes to the primary ASR
first, typically a GPU service with a bigger model. If it fails or does not
answer within ``primary_timeout_s``, the SAME audio goes to the fallback at
once, so the caller never repeats themselves. After a failure the primary is
skipped for ``cooldown_s``: a dead GPU host then costs one timeout, not one per
turn. The first utterance after the cooldown tries it again.

livekit's ``stt.FallbackAdapter`` switches at the stream level instead, which
loses the utterance in flight when the primary fails mid-call.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from livekit.agents import stt
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer

from voice_agent import metrics

logger = logging.getLogger("voice_agent")

DEFAULT_PRIMARY_TIMEOUT_S = 3.0
DEFAULT_COOLDOWN_S = 30.0


class FallbackBatchSTT(stt.STT[Any]):
    def __init__(
        self,
        *,
        primary: stt.STT[Any],
        fallback: stt.STT[Any],
        primary_timeout_s: float = DEFAULT_PRIMARY_TIMEOUT_S,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._primary = primary
        self._fallback = fallback
        self.primary_timeout_s = primary_timeout_s
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._skip_primary_until = 0.0

    @property
    def primary(self) -> stt.STT[Any]:
        return self._primary

    @property
    def fallback(self) -> stt.STT[Any]:
        return self._fallback

    @property
    def model(self) -> str:
        return "asr-batch-fallback"

    @property
    def provider(self) -> str:
        return "voice-stack"

    async def _recognize_impl(
        self, buffer: AudioBuffer, *, language: NotGivenOr[str] = NOT_GIVEN, conn_options: APIConnectOptions
    ) -> stt.SpeechEvent:
        if self._clock() >= self._skip_primary_until:
            # One attempt, no retries: a retry is exactly the wait the fallback avoids.
            once = APIConnectOptions(max_retry=0, timeout=self.primary_timeout_s)
            try:
                event = await asyncio.wait_for(
                    self._primary.recognize(buffer, language=language, conn_options=once),
                    timeout=self.primary_timeout_s,
                )
            except Exception as exc:  # noqa: BLE001 - any failure of the primary means: use the fallback
                self._skip_primary_until = self._clock() + self.cooldown_s
                reason = "timeout" if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) else "error"
                metrics.STT_PRIMARY_FALLBACKS.labels(reason=reason).inc()
                logger.warning(
                    f"primary ASR failed ({reason}: {exc!r}); this turn and the next {self.cooldown_s:.0f} s "
                    "use the fallback ASR"
                )
            else:
                metrics.STT_REQUESTS.labels(service="primary").inc()
                return event
        event = await self._fallback.recognize(buffer, language=language, conn_options=conn_options)
        metrics.STT_REQUESTS.labels(service="fallback").inc()
        return event

    async def aclose(self) -> None:
        await self._primary.aclose()
        await self._fallback.aclose()
