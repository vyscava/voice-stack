"""Prometheus metrics for streaming ASR (``/v1/stream``).

The number a caller feels is ``voice_asr_stream_flush_seconds``: from the
client's ``flush`` (end of utterance) to the ``final`` frame. Everything
before the flush overlaps with speech; this is the part that does not.

Kept out of the endpoint module so the handler reads as protocol, and so the
labels live in one place.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# Exactly one outcome per connection, decided at setup: "accepted" (got the
# slot and an engine) or the refusal code. Refusals later on an accepted
# connection go to REFUSALS, so these outcomes sum to the connection count.
CONNECTIONS = Counter(
    "voice_asr_stream_connections_total",
    "Streaming connections by outcome at setup.",
    ["outcome"],
)

REFUSALS = Counter(
    "voice_asr_stream_refusals_total",
    "Accepted connections closed for a protocol error.",
    ["code"],
)

ACTIVE = Gauge(
    "voice_asr_stream_active",
    "Streaming connections currently holding the stream slot.",
)

# Which rule produced the language of each final (see the endpoint docstring).
# "none" when no rule did.
FINALS = Counter(
    "voice_asr_stream_finals_total",
    "Final transcripts by language source.",
    ["language_source"],
)

# Only flushes that ended an utterance with audio; an empty flush would pull
# the histogram toward zero.
FLUSH_SECONDS = Histogram(
    "voice_asr_stream_flush_seconds",
    "Time from the client's flush to the final frame, for utterances with audio.",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
)

# One model step per fed frame batch. At 560 ms chunks a feed that regularly
# takes longer than the audio it carries means the stream falls behind.
FEED_SECONDS = Histogram(
    "voice_asr_stream_feed_seconds",
    "Compute time of one audio feed.",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0),
)

ERRORS = Counter(
    "voice_asr_stream_errors_total",
    "Streaming connections that failed after setup.",
)

ENGINE_LOADED = Gauge(
    "voice_asr_stream_engine_loaded",
    "1 once the streaming model is loaded in this process.",
)
