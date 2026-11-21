"""Thread-safe in-memory state for outbound call attempts."""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

from voice_agent import metrics
from voice_agent.outbound import CallRequest

STATUSES = ("dialing", "answered", "no_answer", "busy", "failed")


@dataclass
class CallRecord:
    call_id: str
    reason: str
    requested_by: str
    status: str
    detail: str
    started_at: datetime
    ended_at: datetime | None


class CallInProgress(Exception):
    """Raised when another outbound attempt is still active."""


def _active(records: list[CallRecord]) -> bool:
    return any(
        record.status == "dialing" or (record.status == "answered" and record.ended_at is None) for record in records
    )


class CallRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[CallRecord] = []

    def _expire(self, now: datetime, ring_timeout_s: int, max_duration_s: int) -> None:
        """Expire abandoned jobs while the registry lock is held."""
        for record in self._records:
            if record.status == "dialing" and now - record.started_at > timedelta(seconds=ring_timeout_s + 60):
                record.status = "failed"
                record.detail = "no_agent"
                record.ended_at = now
                metrics.OUTBOUND_CALLS.labels("failed").inc()
            elif (
                record.status == "answered"
                and record.ended_at is None
                and now - record.started_at > timedelta(seconds=max_duration_s + 60)
            ):
                record.ended_at = now

    def has_active(self, now: datetime, ring_timeout_s: int, max_duration_s: int) -> bool:
        with self._lock:
            self._expire(now, ring_timeout_s, max_duration_s)
            return _active(self._records)

    def start(self, req: CallRequest, now: datetime, ring_timeout_s: int, max_duration_s: int) -> CallRecord:
        with self._lock:
            self._expire(now, ring_timeout_s, max_duration_s)
            if _active(self._records):
                raise CallInProgress
            record = CallRecord(
                call_id="call-out-" + secrets.token_hex(8),
                reason=req.reason,
                requested_by=req.requested_by,
                status="dialing",
                detail="",
                started_at=now,
                ended_at=None,
            )
            self._records.append(record)
            del self._records[:-50]
            return replace(record)

    def update(self, call_id: str, status: str, detail: str = "", now: datetime | None = None) -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown call status: {status}")
        with self._lock:
            for record in self._records:
                if record.call_id != call_id:
                    continue
                record.status = status
                record.detail = detail
                if status not in ("dialing", "answered"):
                    record.ended_at = now or datetime.now(timezone.utc)
                return

    def mark_ended(self, call_id: str, now: datetime) -> None:
        with self._lock:
            for record in self._records:
                if record.call_id == call_id and record.ended_at is None:
                    record.ended_at = now
                    return

    def get(self, call_id: str, now: datetime, ring_timeout_s: int, max_duration_s: int) -> CallRecord | None:
        with self._lock:
            self._expire(now, ring_timeout_s, max_duration_s)
            for record in self._records:
                if record.call_id == call_id:
                    return replace(record)
            return None

    def attempt_times(self) -> list[datetime]:
        with self._lock:
            return [record.started_at for record in self._records]


REGISTRY = CallRegistry()
