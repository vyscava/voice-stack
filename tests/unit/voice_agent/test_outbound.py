"""Tests for outbound request sanitization and attempt policy."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from voice_agent.outbound import (
    CallRequest,
    check_policy,
    in_quiet_hours,
    parse_quiet_hours,
    parse_request,
    sanitize_reason,
)

T0 = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)


def test_sanitize_strips_control_chars_and_collapses_whitespace():
    assert sanitize_reason("  hi\x00 there\n\nnow ") == "hi there now"


def test_sanitize_rejects_empty_and_long():
    for raw in ("", "   ", "x" * 301):
        with pytest.raises(ValueError):
            sanitize_reason(raw)
    assert sanitize_reason("x" * 300) == "x" * 300


def test_parse_request_rejects_bad_requester():
    with pytest.raises(ValueError):
        parse_request({"reason": "a", "requested_by": "Agent A!"})
    with pytest.raises(ValueError):
        parse_request({"reason": "a"})
    with pytest.raises(ValueError):
        parse_request([])
    assert parse_request({"reason": "the backup finished", "requested_by": "agent-a"}) == CallRequest(
        "the backup finished", "agent-a"
    )


@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [(23, 30, True), (7, 59, True), (8, 0, False), (21, 59, False), (22, 0, True)],
)
def test_quiet_hours_wraps_midnight(hour: int, minute: int, expected: bool):
    local = datetime(2026, 11, 20, hour, minute, tzinfo=ZoneInfo("America/New_York"))
    assert (
        in_quiet_hours(local.astimezone(timezone.utc), parse_quiet_hours("22:00-08:00"), "America/New_York") is expected
    )


def test_quiet_hours_on_dst_change_day():
    from zoneinfo import ZoneInfo

    zone = ZoneInfo("America/New_York")
    for hour, minute, expected in ((7, 30, True), (8, 30, False)):
        local = datetime(2026, 3, 8, hour, minute, tzinfo=zone)
        assert (
            in_quiet_hours(local.astimezone(timezone.utc), parse_quiet_hours("22:00-08:00"), "America/New_York")
            is expected
        )


def test_no_quiet_hours_when_empty():
    assert parse_quiet_hours("") is None
    assert not in_quiet_hours(T0, None, "America/New_York")


def test_bad_quiet_hours_raise():
    for raw in ("22-08", "25:00-08:00", "22:00"):
        with pytest.raises(ValueError):
            parse_quiet_hours(raw)


def test_policy_order_and_limits():
    from zoneinfo import ZoneInfo

    zone = ZoneInfo("America/New_York")
    quiet_now = datetime(2026, 11, 20, 23, 0, tzinfo=zone).astimezone(timezone.utc)
    assert (
        check_policy(
            quiet_now,
            [],
            quiet=parse_quiet_hours("22:00-08:00"),
            tz="America/New_York",
            max_per_day=3,
            min_interval_min=30,
        )
        == "quiet_hours"
    )

    noon = datetime(2026, 11, 20, 12, 0, tzinfo=zone).astimezone(timezone.utc)
    attempts = [noon.replace(hour=hour) for hour in (8, 9, 10)]
    assert (
        check_policy(noon, attempts, quiet=None, tz="America/New_York", max_per_day=3, min_interval_min=30)
        == "daily_limit"
    )
    assert (
        check_policy(
            noon, [noon - timedelta(minutes=10)], quiet=None, tz="America/New_York", max_per_day=3, min_interval_min=30
        )
        == "too_soon"
    )
    assert (
        check_policy(
            noon, [noon.replace(day=19, hour=11)], quiet=None, tz="America/New_York", max_per_day=3, min_interval_min=30
        )
        is None
    )
    assert check_policy(noon, [], quiet=None, tz="America/New_York", max_per_day=3, min_interval_min=30) is None
