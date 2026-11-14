"""
Process-wide state for streaming ASR: the lazily-loaded engine and the single stream slot.

Mirrors ``asr.engine_factory``: a double-checked, lock-guarded singleton built
from settings on first use, plus a ``set_stream_engine`` test hook so the
WebSocket handler can be exercised with a fake engine.
"""

from __future__ import annotations

from threading import Lock
from typing import Final

from asr.stream.base import StreamConfig, StreamEngine
from core.logging import logger_asr as logger
from core.settings import get_settings

_engine: StreamEngine | None = None
_LOCK: Final[Lock] = Lock()

# One active stream per process. A threading.Lock taken with blocking=False is
# a try-acquire that never waits and is not bound to any event loop, so it
# behaves the same under uvicorn and under the test client's portal thread.
_SLOT: Final[Lock] = Lock()


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
                logger.info("Streaming ASR engine initialized")
    return _engine


def set_stream_engine(engine: StreamEngine | None) -> None:
    """Test helper: inject a fake engine (or None to force a reload)."""
    global _engine
    with _LOCK:
        _engine = engine


def try_acquire_stream_slot() -> bool:
    """Claim the process's single stream slot without waiting. True if claimed."""
    return _SLOT.acquire(blocking=False)


def release_stream_slot() -> None:
    """Give the slot back. Call exactly once per successful try_acquire_stream_slot()."""
    _SLOT.release()
