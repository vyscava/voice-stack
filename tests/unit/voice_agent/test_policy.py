"""Turn policy: voice by language, filler only when slow, audible failures."""

from __future__ import annotations

import asyncio

import pytest

from voice_agent.policy import guarded, voice_for_language, with_filler


async def _collect(gen) -> list[str]:
    return [chunk async for chunk in gen]


async def _source(*items, delay=0.0, fail_after=None):
    for i, item in enumerate(items):
        if fail_after is not None and i == fail_after:
            raise RuntimeError("brain died")
        await asyncio.sleep(delay)
        yield item
    if fail_after is not None and fail_after >= len(items):
        raise RuntimeError("brain died")


def test_voice_for_language():
    kw = {"en": "EN", "pt": "PT", "current": "CUR"}
    assert voice_for_language("pt", **kw) == "PT"
    assert voice_for_language("pt-BR", **kw) == "PT"
    assert voice_for_language("en", **kw) == "EN"
    assert voice_for_language("fr", **kw) == "CUR"
    assert voice_for_language(None, **kw) == "CUR"


@pytest.mark.asyncio
async def test_no_filler_when_the_brain_is_fast():
    out = await _collect(with_filler(_source("Hi", " there"), after_s=0.2, filler="Hmm."))
    assert out == ["Hi", " there"]


@pytest.mark.asyncio
async def test_filler_first_when_the_brain_is_slow():
    out = await _collect(with_filler(_source("Hi", delay=0.3), after_s=0.05, filler="Hmm."))
    assert out == ["Hmm.", "Hi"]


@pytest.mark.asyncio
async def test_filler_source_that_ends_empty():
    out = await _collect(with_filler(_source(), after_s=0.05, filler="Hmm."))
    assert out == []


@pytest.mark.asyncio
async def test_guarded_apologises_on_timeout():
    failures: list[str] = []
    gen = guarded(_source("late", delay=0.5), timeout_s=0.05, apology="Sorry.", on_failure=failures.append)
    assert await _collect(gen) == ["Sorry."]
    assert failures == ["timeout"]


@pytest.mark.asyncio
async def test_guarded_apologises_on_error_before_first_words():
    failures: list[str] = []
    gen = guarded(_source("x", fail_after=0), timeout_s=1, apology="Sorry.", on_failure=failures.append)
    assert await _collect(gen) == ["Sorry."]
    assert failures == ["error"]


@pytest.mark.asyncio
async def test_guarded_apologises_on_an_empty_reply():
    failures: list[str] = []
    assert await _collect(guarded(_source(), timeout_s=1, apology="Sorry.", on_failure=failures.append)) == ["Sorry."]
    assert failures == ["empty"]


@pytest.mark.asyncio
async def test_guarded_stops_quietly_after_the_first_words():
    failures: list[str] = []
    gen = guarded(_source("One.", " Two.", fail_after=2), timeout_s=1, apology="Sorry.", on_failure=failures.append)
    assert await _collect(gen) == ["One.", " Two."]
    assert failures == ["error"]


@pytest.mark.asyncio
async def test_slow_dead_brain_gets_filler_then_apology():
    """The production order: with_filler(guarded(brain))."""
    gen = with_filler(guarded(_source("late", delay=0.5), timeout_s=0.1, apology="Sorry."), after_s=0.02, filler="Hmm.")
    assert await _collect(gen) == ["Hmm.", "Sorry."]


@pytest.mark.asyncio
async def test_barge_in_cancels_the_brain_and_leaves_nothing_pending():
    """The caller talks over the agent: LiveKit cancels the reply mid-await."""
    seen: list[str] = []

    async def slow_brain():
        try:
            yield "first"
            await asyncio.sleep(10)
            yield "never"
        except asyncio.CancelledError:
            seen.append("cancelled")
            raise

    async def consume():
        async for _ in with_filler(guarded(slow_brain(), timeout_s=5, apology="Sorry."), after_s=1, filler="Hmm."):
            pass

    task = asyncio.ensure_future(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert seen == ["cancelled"]
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert pending == []
