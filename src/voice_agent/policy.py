"""Turn policy: pure functions, so every branch is testable without LiveKit.

* ``voice_for_language`` picks the reply voice from the language ASR detected.
* ``with_fillers`` says short lines of its own while the brain has said
  nothing yet, each at its own deadline (see fillers.py).
* ``guarded`` makes a dead or silent brain audible: a caller who hears nothing
  cannot tell a pause from a crash, and talks over it.
* ``preemptive_generation_allowed`` keeps speculative replies away from a NATS
  peer, which would receive every one of them as a real message.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


def voice_for_language(lang: str | None, *, en: str, pt: str, current: str) -> str:
    code = (lang or "").strip().lower()
    if code.startswith("pt"):
        return pt
    if code.startswith("en"):
        return en
    return current


def preemptive_generation_allowed(enabled: bool, brain: str) -> bool:
    """Speculative brain calls only for the openai brain, and only when asked for."""
    return enabled and brain.strip().lower() == "openai"


Line = str | Callable[[], str | None] | None


async def with_fillers(chunks: AsyncIterator[str], *, stages: Sequence[tuple[float, Line]]) -> AsyncIterator[str]:
    """Yield each stage's line when its deadline passes before the first chunk.

    ``stages`` are ``(seconds after the call, line)``. A line may be a callable,
    called only when its deadline passes (so a rotating pick is not used up on
    a fast turn). A deadline <= 0, or a line that is or returns ``None``/empty,
    is skipped. Once the first chunk arrives no further line is spoken.
    """
    iterator = chunks.__aiter__()
    first = asyncio.ensure_future(iterator.__anext__())
    try:
        loop = asyncio.get_running_loop()
        started = loop.time()
        for at_s, line in sorted((st for st in stages if st[1] and st[0] > 0), key=lambda st: st[0]):
            done, _ = await asyncio.wait({first}, timeout=max(0.0, started + at_s - loop.time()))
            if done:
                break
            text = line() if callable(line) else line
            if text:
                yield text
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


async def with_filler(chunks: AsyncIterator[str], *, after_s: float, filler: str) -> AsyncIterator[str]:
    """Yield ``filler`` first only if the first chunk takes longer than ``after_s``."""
    async for chunk in with_fillers(chunks, stages=[(after_s, filler)]):
        yield chunk


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
