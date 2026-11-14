"""Prometheus metrics for streaming ASR (``/v1/stream``).

The number a caller feels is ``voice_asr_stream_flush_seconds``: from the
client's ``flush`` (end of utterance) to the ``final`` frame. Everything
before the flush overlaps with speech; this is the part that does not.

Kept out of the endpoint module so the handler reads as protocol, and so the
labels live in one place.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# One label value per way a connection can end before or instead of streaming.
# "accepted" counts connections that got the slot and an engine.
CONNECTIONS = Counter(
    "voice_asr_stream_connections_total",
    "Streaming connections by outcome at setup.",
    ["outcome"],
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

FLUSH_SECONDS = Histogram(
    "voice_asr_stream_flush_seconds",
    "Time from the client's flush to the final frame.",
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
