"""Turn policy: pure functions, so every branch is testable without LiveKit.

* ``voice_for_language`` picks the reply voice from the language ASR detected.
* ``with_filler`` says something short only when the brain is slow, because a
  filler in front of a fast answer only adds delay.
* ``guarded`` makes a dead or silent brain audible: a caller who hears nothing
  cannot tell a pause from a crash, and talks over it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import TypeVar

T = TypeVar("T")


def voice_for_language(lang: str | None, *, en: str, pt: str, current: str) -> str:
    code = (lang or "").strip().lower()
    if code.startswith("pt"):
        return pt
    if code.startswith("en"):
        return en
    return current


async def with_filler(chunks: AsyncIterator[str], *, after_s: float, filler: str) -> AsyncIterator[str]:
    """Yield ``filler`` first only if the first chunk takes longer than ``after_s``."""
    iterator = chunks.__aiter__()
    first = asyncio.ensure_future(iterator.__anext__())
    try:
        done, _ = await asyncio.wait({first}, timeout=after_s)
        if not done and filler:
            yield filler
        try:
            chunk = await first
        except StopAsyncIteration:
            return
        yield chunk
        async for chunk in iterator:
            yield chunk
    finally:
        if not first.done():
            first.cancel()


async def guarded(
    chunks: AsyncIterator[str],
    *,
    timeout_s: float,
    apology: str,
    on_failure: Callable[[str], None] | None = None,
) -> AsyncIterator[str]:
    """Turn a brain that times out or fails before its first words into ``apology``.

    After the first chunk has been spoken, an error ends the reply quietly:
    an apology in the middle of a sentence would be worse than stopping.
    ``on_failure`` receives "timeout", "error" or "empty" for metrics.
    """
    iterator = chunks.__aiter__()
    try:
        first = await asyncio.wait_for(iterator.__anext__(), timeout=timeout_s)
    except StopAsyncIteration:
        if on_failure:
            on_failure("empty")
        yield apology
        return
    except (asyncio.TimeoutError, TimeoutError):
        if on_failure:
            on_failure("timeout")
        yield apology
        return
    except Exception:
        if on_failure:
            on_failure("error")
        yield apology
        return
    yield first
    try:
        async for chunk in iterator:
            yield chunk
    except Exception:
        if on_failure:
            on_failure("error")
        return
