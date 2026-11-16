"""Prometheus metrics for the voice agent.

The question these answer is "where did the time go?" A call that feels slow
can be slow in four different places, with four different fixes:

* ``eou``              waiting to be sure the caller stopped talking
* ``stt``              transcribing what they said (streaming: flush to final)
* ``brain_first_text`` the brain's first words
* ``tts_first_audio``  synthesis of the first audio

Jobs run as threads in this process (see worker.py), so these module-level
metrics are what /metrics serves.
"""

from __future__ import annotations

from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

STAGE_SECONDS = Histogram(
    "voice_agent_stage_seconds",
    "Latency of each stage of a turn.",
    ["stage"],
    buckets=(0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 20),
)
TURNS = Counter("voice_agent_turns_total", "Turns answered, by outcome.", ["outcome"])
BRAIN_FAILURES = Counter("voice_agent_brain_failures_total", "Brain replies replaced by an apology, by kind.", ["kind"])
BARGE_INS = Counter("voice_agent_barge_ins_total", "Times the caller started talking while the agent spoke.")
CALLS = Counter("voice_agent_calls_total", "Calls started.")
FILLERS = Counter("voice_agent_fillers_total", "Lines the agent said itself before the brain spoke, by kind.", ["kind"])
STT_STREAM_FALLBACKS = Counter(
    "voice_agent_stt_stream_fallbacks_total",
    "Calls that switched from streaming ASR to batch STT, by reason (busy, disabled, connect, stall, error).",
    ["reason"],
)
STT_PRIMARY_FALLBACKS = Counter(
    "voice_agent_stt_primary_fallbacks_total",
    "Utterances the primary batch ASR failed (timeout or error) and the fallback ASR transcribed instead.",
    ["reason"],
)
STT_REQUESTS = Counter(
    "voice_agent_stt_requests_total",
    "Batch utterances transcribed, by which ASR answered (primary, fallback). Only with AGENT_ASR_PRIMARY_URL set.",
    ["service"],
)
# Every label exists at 0 from the start: increase() needs a sample before the
# first event, or the first fallback after a (re)start is never counted and an
# alert on "2 in 30m" really fires on the third.
for _reason in ("timeout", "error"):
    STT_PRIMARY_FALLBACKS.labels(reason=_reason)
for _service in ("primary", "fallback"):
    STT_REQUESTS.labels(service=_service)

# LiveKit's session "error" events. A TTS failing every request leaves the
# caller in silence while the process looks healthy; this is how that shows.
ERRORS = Counter(
    "voice_agent_errors_total",
    "Errors LiveKit reported during calls, by component (tts, stt, brain, other) and recoverability.",
    ["component", "recoverable"],
)
_ERROR_COMPONENTS = {"tts_error": "tts", "stt_error": "stt", "llm_error": "brain"}

_STAGE_FIELDS = {
    "eou_metrics": ("end_of_utterance_delay", "eou"),
    "stt_metrics": ("duration", "stt"),
    "tts_metrics": ("ttfb", "tts_first_audio"),
}


def record_livekit_metrics(m: Any) -> None:
    """Feed one LiveKit ``metrics_collected`` payload into the stage histogram."""
    field_stage = _STAGE_FIELDS.get(getattr(m, "type", ""))
    if field_stage is None:
        return
    field, stage = field_stage
    value = getattr(m, field, None)
    if isinstance(value, int | float) and value > 0:
        STAGE_SECONDS.labels(stage=stage).observe(float(value))


def record_error(ev: Any) -> None:
    """Count one LiveKit ``error`` event (its ``error`` is a TTSError, STTError, LLMError or other)."""
    err = getattr(ev, "error", None)
    component = _ERROR_COMPONENTS.get(getattr(err, "type", ""), "other")
    recoverable = getattr(err, "recoverable", None)
    label = "unknown" if recoverable is None else str(bool(recoverable)).lower()
    ERRORS.labels(component=component, recoverable=label).inc()


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
