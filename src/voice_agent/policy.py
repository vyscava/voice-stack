"""Turn policy: pure functions, so every branch is testable without LiveKit.

* ``voice_for_language`` picks the reply voice from the language ASR detected.
* ``with_fillers`` says short lines of its own while the brain has said
  nothing yet, each at its own deadline (see fillers.py).
* ``guarded`` makes a dead or silent brain audible: a caller who hears nothing
  cannot tell a pause from a crash, and talks over it.
* ``with_progress`` says one short line when the brain goes quiet for a tool
  run after it has started speaking.

The brain's stream carries text (spoken) and non-text events (``ToolProgress``,
never spoken). Deadlines here are about SPEECH: an event passes straight
through and neither counts as the brain's first words nor cancels a filler.
* ``preemptive_generation_allowed`` keeps speculative replies away from a NATS
  peer, which would receive every one of them as a real message.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, TypeVar

T = TypeVar("T")


def _is_text(chunk: Any) -> bool:
    return isinstance(chunk, str)


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


async def with_fillers(chunks: AsyncIterator[Any], *, stages: Sequence[tuple[float, Line]]) -> AsyncIterator[Any]:
    """Yield each stage's line when its deadline passes before the first text.

    ``stages`` are ``(seconds after the call, line)``. A line may be a callable,
    called only when its deadline passes (so a rotating pick is not used up on
    a fast turn). A deadline <= 0, or a line that is or returns ``None``/empty,
    is skipped. Once the first text arrives no further line is spoken; events
    before it pass through without stopping the stages.
    """
    iterator = chunks.__aiter__()
    pending = sorted((st for st in stages if st[1] and st[0] > 0), key=lambda st: st[0])
    nxt: asyncio.Future[Any] | None = None
    try:
        loop = asyncio.get_running_loop()
        started = loop.time()
        while True:
            if nxt is None:
                nxt = asyncio.ensure_future(iterator.__anext__())
            timeout = max(0.0, started + pending[0][0] - loop.time()) if pending else None
            done, _ = await asyncio.wait({nxt}, timeout=timeout)
            if not done:
                _, line = pending.pop(0)
                text = line() if callable(line) else line
                if text:
                    yield text
                continue
            try:
                chunk = nxt.result()
            except StopAsyncIteration:
                return
            nxt = None
            yield chunk
            if _is_text(chunk):
                break
        async for chunk in iterator:
            yield chunk
    finally:
        if nxt is not None and not nxt.done():
            nxt.cancel()


async def with_filler(chunks: AsyncIterator[Any], *, after_s: float, filler: str) -> AsyncIterator[Any]:
    """Yield ``filler`` first only if the first chunk takes longer than ``after_s``."""
    async for chunk in with_fillers(chunks, stages=[(after_s, filler)]):
        yield chunk


async def guarded(
    chunks: AsyncIterator[Any],
    *,
    timeout_s: float,
    apology: str,
    on_failure: Callable[[str], None] | None = None,
) -> AsyncIterator[Any]:
    """Turn a brain that times out or fails before its first words into ``apology``.

    The deadline is on the first TEXT; events before it pass through. After
    the first words, an error ends the reply quietly: an apology in the middle
    of a sentence would be worse than stopping. ``on_failure`` receives
    "timeout", "error" or "empty" for metrics.
    """
    iterator = chunks.__aiter__()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    try:
        while True:
            first = await asyncio.wait_for(iterator.__anext__(), timeout=max(0.0, deadline - loop.time()))
            if _is_text(first):
                break
            yield first
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


async def with_progress(chunks: AsyncIterator[Any], *, after_s: float, line: Line) -> AsyncIterator[Any]:
    """After the brain has started speaking, cover a quiet tool run with one short line.

    When a tool starts (an event whose ``running`` is true) after the first
    text, and no text follows within ``after_s``, yield ``line`` once for that
    gap, and not again until the brain speaks (a chain of tools gets one line).
    Before the first text the ack and thinking line (``with_fillers``)
    cover the wait. ``after_s`` <= 0 turns it off.
    """
    if after_s <= 0 or not line:
        async for chunk in chunks:
            yield chunk
        return
    iterator = chunks.__aiter__()
    loop = asyncio.get_running_loop()
    spoke = False
    quiet_since: float | None = None  # a tool is running and nothing has been said since
    covered = False  # the line was said in this quiet stretch: not again until the brain speaks
    nxt: asyncio.Future[Any] | None = None
    try:
        while True:
            if nxt is None:
                nxt = asyncio.ensure_future(iterator.__anext__())
            timeout = None if quiet_since is None else max(0.0, quiet_since + after_s - loop.time())
            done, _ = await asyncio.wait({nxt}, timeout=timeout)
            if not done:
                quiet_since, covered = None, True
                text = line() if callable(line) else line
                if text:
                    yield text
                continue
            try:
                chunk = nxt.result()
            except StopAsyncIteration:
                return
            nxt = None
            if _is_text(chunk):
                if chunk.strip():
                    spoke, quiet_since, covered = True, None, False
            elif spoke and not covered and quiet_since is None and getattr(chunk, "running", False):
                quiet_since = loop.time()
            yield chunk
    finally:
        if nxt is not None and not nxt.done():
            nxt.cancel()
