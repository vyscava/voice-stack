"""
Process-wide state for streaming ASR: the lazily-loaded engine and the stream slots.

Mirrors ``asr.engine_factory``: a double-checked, lock-guarded singleton built
from settings on first use, plus a ``set_stream_engine`` test hook so the
WebSocket handler can be exercised with a fake engine.
"""

from __future__ import annotations

from threading import Lock
from typing import Final

from asr.stream import metrics as stream_metrics
from asr.stream.base import StreamConfig, StreamEngine
from core.logging import logger_asr as logger
from core.settings import get_settings

_engine: StreamEngine | None = None
_LOCK: Final[Lock] = Lock()

# Concurrent streams per process, up to ASR_STREAM_MAX_SESSIONS. Each session
# decodes on its own shallow model copy (shared weights, see nemotron.py), so
# the limit is a CPU guard, not a correctness one. A counter under a plain
# threading.Lock never waits and is not bound to any event loop, so it behaves
# the same under uvicorn and under the test client's portal thread.
_SLOTS_LOCK: Final[Lock] = Lock()
_in_use = 0
_max_sessions: int | None = None  # test override; None reads the setting


def get_stream_config() -> StreamConfig:
    """The validated ASR_STREAM_* settings. Raises ValueError on an unsupported value."""
    return StreamConfig.from_settings(get_settings())


def get_stream_engine() -> StreamEngine:
    """Thread-safe lazy singleton. BLOCKING (the first call loads the model): call it from an executor."""
    global _engine
    if _engine is None:
        with _LOCK:
            if _engine is None:
                from asr.stream.nemotron import NemotronStreamEngine

                _engine = NemotronStreamEngine(get_stream_config())
                stream_metrics.ENGINE_LOADED.set(1)
                logger.info("Streaming ASR engine initialized")
    return _engine


def is_stream_engine_loaded() -> bool:
    """True once the streaming engine exists in this process (never loads it)."""
    return _engine is not None


def set_stream_engine(engine: StreamEngine | None) -> None:
    """Test helper: inject a fake engine (or None to force a reload)."""
    global _engine
    with _LOCK:
        _engine = engine


def _limit() -> int:
    if _max_sessions is not None:
        return _max_sessions
    return max(1, int(get_settings().ASR_STREAM_MAX_SESSIONS))


def set_max_sessions(limit: int | None) -> None:
    """Test helper: fix the number of concurrent streams (None: back to the setting)."""
    global _max_sessions
    _max_sessions = limit


def try_acquire_stream_slot() -> bool:
    """Claim one of the process's stream slots without waiting. True if claimed."""
    global _in_use
    with _SLOTS_LOCK:
        if _in_use >= _limit():
            return False
        _in_use += 1
        return True


def release_stream_slot() -> None:
    """Give a slot back. Call exactly once per successful try_acquire_stream_slot()."""
    global _in_use
    with _SLOTS_LOCK:
        if _in_use <= 0:
            raise RuntimeError("release_stream_slot() without a claimed slot")
        _in_use -= 1
