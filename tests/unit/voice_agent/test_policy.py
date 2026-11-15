"""Turn policy: voice by language, filler only when slow, audible failures."""

from __future__ import annotations

import asyncio

import pytest

from voice_agent.policy import guarded, preemptive_generation_allowed, voice_for_language, with_filler, with_fillers


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


def test_preemptive_generation_never_reaches_a_nats_peer():
    assert preemptive_generation_allowed(True, " OpenAI ") is True
    assert preemptive_generation_allowed(True, "nats") is False
    assert preemptive_generation_allowed(False, "openai") is False


@pytest.mark.asyncio
async def test_fillers_fast_brain_gets_none():
    out = await _collect(with_fillers(_source("Hi"), stages=[(0.2, "Okay."), (0.4, "Hmm.")]))
    assert out == ["Hi"]


@pytest.mark.asyncio
async def test_fillers_only_the_stages_that_expire_before_the_first_words():
    out = await _collect(with_fillers(_source("Hi", delay=0.15), stages=[(0.03, "Okay."), (0.5, "Hmm.")]))
    assert out == ["Okay.", "Hi"]


@pytest.mark.asyncio
async def test_fillers_slow_brain_gets_ack_then_thinking_line_in_order():
    out = await _collect(with_fillers(_source("Hi", delay=0.25), stages=[(0.1, "Hmm."), (0.03, "Okay.")]))
    assert out == ["Okay.", "Hmm.", "Hi"]


@pytest.mark.asyncio
async def test_fillers_deadlines_count_from_the_start_not_from_the_previous_line():
    """The second line comes at its own deadline, not ack + deadline."""
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    gen = with_fillers(_source("Hi", delay=0.6), stages=[(0.15, "Okay."), (0.25, "Hmm.")])
    seen = [(chunk, loop.time() - t0) async for chunk in gen]
    assert [c for c, _ in seen] == ["Okay.", "Hmm.", "Hi"]
    assert seen[1][1] < 0.35  # from the start: 0.25 s; from the previous line it would be 0.40 s


@pytest.mark.asyncio
async def test_fillers_disabled_or_empty_stages_are_skipped():
    out = await _collect(with_fillers(_source("Hi", delay=0.1), stages=[(0, "Okay."), (0.02, None), (0.03, "")]))
    assert out == ["Hi"]


@pytest.mark.asyncio
async def test_fillers_call_a_callable_line_only_when_its_deadline_passes():
    calls: list[str] = []

    def pick():
        calls.append("x")
        return "Okay."

    assert await _collect(with_fillers(_source("Hi"), stages=[(0.2, pick)])) == ["Hi"]
    assert calls == []
    assert await _collect(with_fillers(_source("Hi", delay=0.1), stages=[(0.02, pick)])) == ["Okay.", "Hi"]
    assert calls == ["x"]


@pytest.mark.asyncio
async def test_fillers_cancelled_mid_wait_leave_nothing_pending():
    seen: list[str] = []

    async def slow_brain():
        try:
            await asyncio.sleep(10)
            yield "never"
        except asyncio.CancelledError:
            seen.append("cancelled")
            raise

    async def consume():
        async for _ in with_fillers(slow_brain(), stages=[(0.01, "Okay."), (5, "Hmm.")]):
            pass

    task = asyncio.ensure_future(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert seen == ["cancelled"]
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert pending == []
