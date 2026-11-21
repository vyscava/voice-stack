"""Pure validation and policy helpers for one configured outbound destination."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

_REQUESTER_RE = re.compile(r"^[a-z0-9_.-]{1,40}$")
_TIME_RANGE_RE = re.compile(r"^(\d{2}):(\d{2})-(\d{2}):(\d{2})$")
_NON_WHITESPACE_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class CallRequest:
    reason: str
    requested_by: str


def sanitize_reason(raw: str) -> str:
    """Remove controls, collapse whitespace, and enforce the spoken limit."""
    cleaned = _NON_WHITESPACE_CONTROL_RE.sub("", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        raise ValueError("reason must not be empty")
    if len(cleaned) > 300:
        raise ValueError("reason must be at most 300 characters")
    return cleaned


def parse_request(body: object) -> CallRequest:
    """Validate the JSON-shaped request without allowing destination selection."""
    if not isinstance(body, dict):
        raise ValueError("body must be a JSON object")

    raw_reason = body.get("reason")
    if not isinstance(raw_reason, str):
        raise ValueError("reason must be a string")
    reason = sanitize_reason(raw_reason)

    requested_by = body.get("requested_by")
    if not isinstance(requested_by, str):
        raise ValueError("requested_by must be a string")
    if not _REQUESTER_RE.fullmatch(requested_by):
        raise ValueError("requested_by must match ^[a-z0-9_.-]{1,40}$")
    return CallRequest(reason=reason, requested_by=requested_by)


def parse_quiet_hours(raw: str) -> tuple[time, time] | None:
    """Parse an inclusive-start, exclusive-end local clock window."""
    if raw == "":
        return None
    match = _TIME_RANGE_RE.fullmatch(raw)
    if match is None:
        raise ValueError("quiet hours must be HH:MM-HH:MM")
    start_hour, start_minute, end_hour, end_minute = (int(value) for value in match.groups())
    if start_hour > 23 or end_hour > 23 or start_minute > 59 or end_minute > 59:
        raise ValueError("quiet hours must contain valid 24-hour times")
    return time(start_hour, start_minute), time(end_hour, end_minute)


def in_quiet_hours(now: datetime, window: tuple[time, time] | None, tz: str) -> bool:
    """Return whether an aware UTC instant falls inside the local window."""
    if window is None:
        return False
    local_time = now.astimezone(ZoneInfo(tz)).timetz().replace(tzinfo=None)
    start, end = window
    if start < end:
        return start <= local_time < end
    if start > end:
        return local_time >= start or local_time < end
    return True


def check_policy(
    now: datetime,
    attempts: list[datetime],
    *,
    quiet: tuple[time, time] | None,
    tz: str,
    max_per_day: int,
    min_interval_min: int,
) -> str | None:
    """Return the first refusal reason, or None when the attempt is allowed."""
    if in_quiet_hours(now, quiet, tz):
        return "quiet_hours"

    now_utc = now.astimezone(timezone.utc)
    cutoff = now_utc - timedelta(hours=24)
    recent = [
        attempt.astimezone(timezone.utc)
        for attempt in attempts
        if cutoff <= attempt.astimezone(timezone.utc) <= now_utc
    ]
    if len(recent) >= max_per_day:
        return "daily_limit"
    if recent and now_utc - max(recent) < timedelta(minutes=min_interval_min):
        return "too_soon"
    return None
