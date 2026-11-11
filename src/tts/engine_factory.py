from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass
from threading import Lock
from typing import Final

from fastapi import HTTPException, status

from core.logging import logger_tts as logger
from core.settings import get_settings
from tts.engine.base import TTSBase
from tts.exceptions import EngineUnavailableError

_engine: TTSBase | None = None
_LOCK: Final[Lock] = Lock()
_semaphore: asyncio.Semaphore | None = None


@dataclass(frozen=True)
class EngineSpec:
    """How to load one TTS engine, and what to tell the operator when it can't."""

    module: str
    class_name: str
    extra: str  # pyproject extra that installs the engine's dependencies
    image: str  # image tag that ships the engine


# Engines are imported lazily: each image ships only some of them, and importing
# an engine pulls its dependencies (torch + coqui for XTTS, onnxruntime for Kokoro).
ENGINES: Final[dict[str, EngineSpec]] = {
    "kokoro": EngineSpec("tts.engine.kokoro", "TTSKokoro", extra="tts-kokoro", image="voice-stack:kokoro"),
    "coqui": EngineSpec("tts.engine.coqui", "TTSCoqui", extra="tts-xtts", image="voice-stack:xtts"),
}
ALIASES: Final[dict[str, str]] = {"xtts": "coqui", "xtts-2": "coqui", "xtts_v2": "coqui"}
DEFAULT_ENGINE: Final[str] = "kokoro"


def resolve_engine_name(raw: str | None) -> str:
    """Canonical engine name for a TTS_ENGINE value. Empty means the default.

    An unknown name raises instead of falling back: the old factory booted Coqui
    for any value it did not recognise, so a typo silently ran the slow engine.
    """
    name = (raw or "").strip().lower() or DEFAULT_ENGINE
    name = ALIASES.get(name, name)
    if name not in ENGINES:
        known = sorted([*ENGINES, *ALIASES])
        raise EngineUnavailableError(
            message=f"Unknown TTS_ENGINE '{raw}'",
            details=f"Known engines: {', '.join(known)}",
        )
    return name


def _import_engine_class(name: str) -> type[TTSBase]:
    """Import the engine module for a canonical name, turning a missing dependency into a clear error."""
    spec = ENGINES[name]
    try:
        module = importlib.import_module(spec.module)
    except ImportError as e:
        raise EngineUnavailableError(
            message=f"TTS engine '{name}' is not installed in this image",
            details=(
                f"Missing dependency: {e.name or e}. Use the {spec.image} image, "
                f"or install the '{spec.extra}' extra."
            ),
            original_exception=e,
        ) from e
    engine_cls: type[TTSBase] = getattr(module, spec.class_name)
    return engine_cls


def _create_engine_from_settings() -> TTSBase:
    """Build the engine TTS_ENGINE selects. Never substitutes a different one."""
    name = resolve_engine_name(getattr(get_settings(), "TTS_ENGINE", ""))
    engine_cls = _import_engine_class(name)
    logger.info("TTS backend selected: %s", name)
    return engine_cls()


def get_audio_engine() -> TTSBase:
    """
    Thread-safe, lazily-initialized singleton.
    Returns the same object for all callers (per *process*).

    Also initializes the concurrency control semaphore on first call.

    Returns:
        TTSBase: The TTS engine instance
    """
    global _engine, _semaphore

    if _engine is None:
        # First check without lock (fast path), then double-check inside lock.
        with _LOCK:
            if _engine is None:
                _engine = _create_engine_from_settings()

                # Initialize semaphore for concurrency control
                settings = get_settings()
                max_concurrent = getattr(settings, "TTS_MAX_CONCURRENT_REQUESTS", 2)
                _semaphore = asyncio.Semaphore(max_concurrent)

                logger.info(f"TTS engine initialized with max_concurrent={max_concurrent}")

    return _engine


async def acquire_engine() -> TTSBase:
    """
    Acquire the TTS engine with concurrency control.

    This function should be used by endpoints instead of calling get_audio_engine()
    directly. It enforces the MAX_CONCURRENT_REQUESTS limit by using a semaphore.

    If all concurrent slots are busy, this function raises an HTTP 429 error
    immediately (fail-fast) rather than queuing the request.

    Returns:
        TTSBase: The TTS engine instance

    Raises:
        HTTPException: 429 Too Many Requests if all slots are busy

    Example:
        ```python
        @router.post("/speech")
        async def speech(request: SpeechRequest):
            engine = await acquire_engine()
            try:
                return engine.speech(...)
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
        logger.warning("All TTS synthesis slots busy - rejecting request with 429")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Server is processing maximum concurrent requests",
            headers={"Retry-After": "10"},  # Retry in 10 seconds
        )

    # Acquire a slot
    if _semaphore is not None:
        await _semaphore.acquire()
        logger.debug(f"TTS slot acquired, {_semaphore._value} slots remaining")

    return get_audio_engine()


def release_engine() -> None:
    """
    Release a concurrency slot after synthesis completes.

    This function MUST be called after acquire_engine() to release the slot,
    typically in a finally block to ensure it runs even if an error occurs.

    Example:
        ```python
        engine = await acquire_engine()
        try:
            result = engine.speech(...)
            return result
        finally:
            release_engine()
        ```
    """
    global _semaphore

    if _semaphore is not None:
        _semaphore.release()
        logger.debug(f"TTS slot released, {_semaphore._value} slots available")


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
