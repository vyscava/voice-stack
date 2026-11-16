"""Small asyncio helpers shared by the agent's STT wrappers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


async def within(aw: Awaitable[T], timeout: float) -> T:
    """``asyncio.wait_for`` without its Python 3.10/3.11 flaw.

    There, a cancellation that lands as the awaited operation completes is
    swallowed: the task carries on and ``cancel_and_wait`` (livekit's call
    teardown) hangs on it forever. ``asyncio.wait`` never swallows it. On a
    timeout the inner task is cancelled, not awaited.
    """
    task = asyncio.ensure_future(aw)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
    except asyncio.CancelledError:
        task.cancel()
        raise
    if not done:
        task.cancel()
        raise asyncio.TimeoutError
    return task.result()
