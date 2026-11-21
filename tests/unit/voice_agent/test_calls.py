"""Tests for the thread-safe outbound call registry."""

from datetime import datetime, timedelta, timezone

import pytest

from voice_agent.calls import CallInProgress, CallRegistry
from voice_agent.outbound import CallRequest

T0 = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
REQUEST = CallRequest("the backup finished", "agent-a")


def start(registry: CallRegistry, now: datetime = T0):
    return registry.start(REQUEST, now, ring_timeout_s=40, max_duration_s=1800)


def test_second_start_while_dialing_raises():
    registry = CallRegistry()
    start(registry)
    with pytest.raises(CallInProgress):
        start(registry)


def test_finished_call_allows_a_new_one():
    registry = CallRegistry()
    first = start(registry)
    registry.update(first.call_id, "no_answer", "sip_480", now=T0 + timedelta(seconds=40))
    second = start(registry, T0 + timedelta(minutes=31))
    assert second.status == "dialing"


def test_answered_call_stays_active_until_ended():
    registry = CallRegistry()
    first = start(registry)
    registry.update(first.call_id, "answered", now=T0 + timedelta(seconds=5))
    with pytest.raises(CallInProgress):
        start(registry)
    registry.mark_ended(first.call_id, T0 + timedelta(minutes=2))
    assert start(registry).status == "dialing"


def test_stale_dialing_expires_on_get():
    registry = CallRegistry()
    first = start(registry)
    record = registry.get(first.call_id, T0 + timedelta(seconds=101), 40, 1800)
    assert record is not None
    assert record.status == "failed"
    assert record.detail == "no_agent"
    assert record.ended_at == T0 + timedelta(seconds=101)


def test_start_after_stale_dialing_succeeds():
    registry = CallRegistry()
    start(registry)
    second = start(registry, T0 + timedelta(seconds=101))
    assert second.status == "dialing"


def test_registry_keeps_newest_50():
    registry = CallRegistry()
    for index in range(50):
        record = start(registry, T0 + timedelta(minutes=index))
        registry.update(record.call_id, "no_answer", now=T0 + timedelta(minutes=index, seconds=1))
    oldest = registry.attempt_times()[0]
    extra = start(registry, T0 + timedelta(hours=2))
    assert extra.status == "dialing"
    assert len(registry.attempt_times()) == 50
    assert oldest not in registry.attempt_times()


def test_get_returns_a_copy():
    registry = CallRegistry()
    first = start(registry)
    record = registry.get(first.call_id, T0, 40, 1800)
    assert record is not None
    record.status = "failed"
    record.reason = "changed"
    stored = registry.get(first.call_id, T0, 40, 1800)
    assert stored is not None
    assert stored.status == "dialing"
    assert stored.reason == "the backup finished"
