from __future__ import annotations

from threading import Lock
from typing import Final

from core.logging import logger_tts as logger
from core.settings import get_settings
from tts.engine.base import TTSBase

_engine: TTSBase | None = None
_LOCK: Final[Lock] = Lock()


def _create_engine_from_settings() -> TTSBase:
    """
    Build a new backend instance based on settings. Import concrete
    backends locally to avoid circular imports at module load time.
    """
    s = get_settings()
    engine = (getattr(s, "TTS_ENGINE", "") or "").lower()

    if engine in {"coqui", "xtts", "xtts-2"}:
        logger.info("TTS backend selected: Coqui")
        from tts.engine.coqui import TTSCoqui

        return TTSCoqui()

    # Default
    logger.info("TTS backend selected: coqui")
    from tts.engine.coqui import TTSCoqui

    return TTSCoqui()


def get_audio_engine() -> TTSBase:
    """
    Thread-safe, lazily-initialized singleton.
    Returns the same object for all callers (per *process*).
    """
    global _engine
    if _engine is None:
        # First check without lock (fast path), then double-check inside lock.
        with _LOCK:
            if _engine is None:
                _engine = _create_engine_from_settings()
    return _engine


def reset_audio_engine() -> None:
    """
    Recreate the singleton from current settings.
    Call during a quiet window; avoid doing this while requests are in-flight.
    """
    global _engine
    with _LOCK:
        _engine = _create_engine_from_settings()


def set_audio_engine(engine: TTSBase) -> None:
    """
    Test helper: inject a mocked/fake engine.
    """
    global _engine
    with _LOCK:
        _engine = engine
