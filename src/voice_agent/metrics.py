"""Prometheus metrics for the voice agent.

The question these answer is "where did the time go?" A call that feels slow
can be slow in four different places, with four different fixes:

* ``eou``              waiting to be sure the caller stopped talking
* ``stt``              transcribing what they said
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


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
