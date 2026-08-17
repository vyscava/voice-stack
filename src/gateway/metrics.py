"""Prometheus metrics for the gateway.

Why this exists
---------------
Before this, the only signal any voice service emitted was whether its /health
endpoint answered. That distinguishes "process alive" from "process gone" and
nothing else -- so a gateway that is up and failing EVERY exchange looks
identical to a healthy one, and did.

The three-day TTS outage in August was invisible to Prometheus for its whole
duration because nothing was watched. Liveness probes (gitops/portainer!996)
fixed the crude half. This is the half that catches a service that is running
and useless.

Deliberately kept out of exchange.py, which stays free of I/O so its failure
policy can be tested without a metrics registry.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# Labelled by outcome so the interesting failures are separable. A spike in
# no_speech means silence or hallucinations are reaching the gateway; a spike in
# agent_error means the bus or the peer is gone. Those have different causes and
# different fixes, and a single "errors" counter would hide the difference.
EXCHANGES = Counter(
    "voice_gateway_exchanges_total",
    "Voice exchanges by outcome.",
    ["outcome"],
)

# The whole request, from audio in to audio out.
EXCHANGE_DURATION = Histogram(
    "voice_gateway_exchange_duration_seconds",
    "End-to-end duration of a voice exchange.",
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300),
)

# The AGENT's turnaround specifically, which is the dominant term. Measured at
# 19-20 s for a trivial question and minutes for anything needing tools, so the
# buckets are spread accordingly rather than clustered under a second.
AGENT_DURATION = Histogram(
    "voice_gateway_agent_duration_seconds",
    "Time spent waiting for the agent to reply.",
    buckets=(1, 5, 10, 20, 30, 60, 120, 300),
)

# Rejected BEFORE ASR by the speech gate. A rising rate means someone is
# sending silence -- a stuck button, a dead microphone, or a client bug -- and
# it is worth seeing even though each individual rejection is handled.
SILENCE_REJECTED = Counter(
    "voice_gateway_silence_rejected_total",
    "Clips rejected by the speech gate before reaching ASR.",
)


def render() -> tuple[bytes, str]:
    """The exposition payload and its content type."""
    return generate_latest(), CONTENT_TYPE_LATEST
