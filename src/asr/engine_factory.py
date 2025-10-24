from __future__ import annotations

from threading import Lock
from typing import Final

from asr.engine.base import ASRBase
from core.logging import logger_asr as logger
from core.settings import get_settings

_engine: ASRBase | None = None
_LOCK: Final[Lock] = Lock()


def _create_engine_from_settings() -> ASRBase:
    """
    Build a new backend instance based on settings. Import concrete
    backends locally to avoid circular imports at module load time.
    """
    s = get_settings()
    engine = (getattr(s, "ASR_ENGINE", "") or "").lower()

    if engine in {"whisper", "whisper-torch"}:
        logger.info("ASR backend selected: OpenAI Whisper")
        from asr.engine.whisper import ASRWhisperTorch

        return ASRWhisperTorch()

    # Default
    logger.info("ASR backend selected: faster-whisper")
    from asr.engine.fasterwhisper import ASRFasterWhisper

    return ASRFasterWhisper()


def get_audio_engine() -> ASRBase:
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


def set_audio_engine(engine: ASRBase) -> None:
    """
    Test helper: inject a mocked/fake engine.
    """
    global _engine
    with _LOCK:
        _engine = engine
