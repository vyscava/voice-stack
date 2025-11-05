from __future__ import annotations

import asyncio
from threading import Lock
from typing import Final

from fastapi import HTTPException, status

from asr.engine.base import ASRBase
from core.logging import logger_asr as logger
from core.settings import get_settings

_engine: ASRBase | None = None
_LOCK: Final[Lock] = Lock()
_semaphore: asyncio.Semaphore | None = None


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

    Also initializes the concurrency control semaphore on first call.

    Returns:
        ASRBase: The ASR engine instance
    """
    global _engine, _semaphore

    if _engine is None:
        # First check without lock (fast path), then double-check inside lock.
        with _LOCK:
            if _engine is None:
                _engine = _create_engine_from_settings()

                # Initialize semaphore for concurrency control
                settings = get_settings()
                max_concurrent = getattr(settings, "ASR_MAX_CONCURRENT_REQUESTS", 2)
                _semaphore = asyncio.Semaphore(max_concurrent)

                logger.info(f"ASR engine initialized with max_concurrent={max_concurrent}")

    return _engine


async def acquire_engine() -> ASRBase:
    """
    Acquire the ASR engine with concurrency control.

    This function should be used by endpoints instead of calling get_audio_engine()
    directly. It enforces the MAX_CONCURRENT_REQUESTS limit by using a semaphore.

    If all concurrent slots are busy, this function raises an HTTP 429 error
    immediately (fail-fast) rather than queuing the request.

    Returns:
        ASRBase: The ASR engine instance

    Raises:
        HTTPException: 429 Too Many Requests if all slots are busy

    Example:
        ```python
        @router.post("/transcribe")
        async def transcribe(file: UploadFile):
            engine = await acquire_engine()
            try:
                return engine.transcribe_file(...)
            finally:
                release_engine()
        ```
    """
    global _semaphore

    # Ensure engine (and semaphore) are initialized
    if _semaphore is None:
        get_audio_engine()

    # Check if semaphore is locked (all slots busy)
    if _semaphore is not None and _semaphore.locked():
        logger.warning("All ASR inference slots busy - rejecting request with 429")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Server is processing maximum concurrent requests",
            headers={"Retry-After": "10"},  # Retry in 10 seconds
        )

    # Acquire a slot
    if _semaphore is not None:
        await _semaphore.acquire()
        logger.debug(f"ASR slot acquired, {_semaphore._value} slots remaining")

    return get_audio_engine()


def release_engine() -> None:
    """
    Release a concurrency slot after inference completes.

    This function MUST be called after acquire_engine() to release the slot,
    typically in a finally block to ensure it runs even if an error occurs.

    Example:
        ```python
        engine = await acquire_engine()
        try:
            result = engine.transcribe_file(...)
            return result
        finally:
            release_engine()
        ```
    """
    global _semaphore

    if _semaphore is not None:
        _semaphore.release()
        logger.debug(f"ASR slot released, {_semaphore._value} slots available")


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
