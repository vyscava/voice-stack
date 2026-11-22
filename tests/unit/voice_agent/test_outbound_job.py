"""Tests for outbound LiveKit SIP job helpers."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from core.settings import Settings
from voice_agent import calls
from voice_agent.outbound import CallRequest
from voice_agent.outbound_job import context_turn, outbound_from_metadata, run_outbound, wait_for_answer


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (
            '{"outbound":{"call_id":"c","reason":"r","requested_by":"agent-a"}}',
            {"call_id": "c", "reason": "r", "requested_by": "agent-a"},
        ),
        ('{"outbound":{"call_id":"c","reason":"r"}}', {"call_id": "c", "reason": "r", "requested_by": ""}),
        ("", None),
        ("{", None),
        ('{"outbound":{}}', None),
        ('{"x":1}', None),
        ('{"outbound":{"call_id":"c","reason":""}}', None),
    ],
)
def test_metadata_parsing(metadata, expected):
    assert outbound_from_metadata(metadata) == expected


def test_context_turn_is_tagged_and_carries_the_reason():
    text = context_turn({"call_id": "c", "reason": "the backup finished", "requested_by": "agent-a"})
    assert text.startswith("[phone-system] ")
    assert "the backup finished" in text


def settings(ring_timeout_s: int = 40) -> Settings:
    return Settings(
        LIVEKIT_URL="ws://livekit",
        LIVEKIT_API_KEY="key",
        LIVEKIT_API_SECRET="secret",
        AGENT_CALL_TO="callee",
        AGENT_CALL_TRUNK_NAME="asterisk-out",
        AGENT_CALL_RING_TIMEOUT_S=ring_timeout_s,
        AGENT_CALL_MAX_DURATION_S=1800,
    )


class Participant:
    def __init__(self, attributes: dict[str, str]) -> None:
        self.attributes = attributes


class Room:
    def __init__(self, participants: dict[str, Participant] | None = None) -> None:
        self.remote_participants = participants or {}
        self.listeners: dict[str, list[Any]] = {}

    def on(self, event: str, callback: Any) -> None:
        self.listeners.setdefault(event, []).append(callback)

    def off(self, event: str, callback: Any) -> None:
        callbacks = self.listeners.get(event, [])
        if callback in callbacks:
            callbacks.remove(callback)
        if not callbacks:
            self.listeners.pop(event, None)


class RoomApi:
    def __init__(self) -> None:
        self.deleted: list[Any] = []

    async def delete_room(self, request: Any) -> None:
        self.deleted.append(request)


class Ctx:
    def __init__(
        self,
        room: Room,
        participant: Participant | None = None,
        wait_error: Exception | None = None,
        room_name: str = "call-out-c",
    ):
        self.room = room
        self.room.name = room_name
        self._participant = participant
        self._wait_error = wait_error
        self.api = SimpleNamespace(room=RoomApi())
        self.callbacks: list[Any] = []
        self.shutdowns: list[str] = []

    async def wait_for_participant(self, **kwargs: Any) -> Participant | None:
        if self._wait_error is not None:
            raise self._wait_error
        assert kwargs.get("identity") == "callee"
        return self._participant

    def add_shutdown_callback(self, callback: Any) -> None:
        self.callbacks.append(callback)

    def shutdown(self, reason: str = "") -> None:
        self.shutdowns.append(reason)


class Session:
    def __init__(self) -> None:
        self.replies: list[str] = []

    async def generate_reply(self, **kwargs: Any) -> None:
        self.replies.append(kwargs["user_input"])


def now() -> datetime:
    return datetime.now(timezone.utc)


def test_wait_for_answer_active_immediately():
    participant = Participant({"sip.callStatus": "active"})
    room = Room({"callee": participant})
    ctx = Ctx(room, participant)
    assert asyncio.run(wait_for_answer(ctx, settings())) == ("answered", "")


def test_wait_for_answer_hangup_is_no_answer():
    participant = Participant({"sip.callStatus": "hangup"})
    room = Room({"callee": participant})
    ctx = Ctx(room, participant)
    assert asyncio.run(wait_for_answer(ctx, settings())) == ("no_answer", "hangup")


def test_wait_for_answer_observes_active_during_listener_registration():
    participant = Participant({"sip.callStatus": "ringing"})

    class ActivatingRoom(Room):
        def on(self, event: str, callback: Any) -> None:
            super().on(event, callback)
            if event == "participant_disconnected":
                participant.attributes["sip.callStatus"] = "active"

    room = ActivatingRoom({"callee": participant})
    ctx = Ctx(room, participant)
    assert asyncio.run(wait_for_answer(ctx, settings())) == ("answered", "")
    assert not room.listeners


@pytest.mark.asyncio
async def test_wait_for_answer_becomes_active_after_ringing():
    participant = Participant({"sip.callStatus": "ringing"})
    room = Room({"callee": participant})
    ctx = Ctx(room, participant)
    task = asyncio.create_task(wait_for_answer(ctx, settings()))
    for _ in range(1000):
        if room.listeners.get("participant_attributes_changed"):
            break
        await asyncio.sleep(0.001)
    participant.attributes["sip.callStatus"] = "active"
    for callback in list(room.listeners.get("participant_attributes_changed", [])):
        callback({"sip.callStatus": "active"}, participant)
    assert await task == ("answered", "")
    assert not room.listeners


@pytest.mark.asyncio
async def test_wait_for_answer_disconnect_is_no_answer():
    participant = Participant({"sip.callStatus": "ringing"})
    room = Room()
    ctx = Ctx(room, participant)
    task = asyncio.create_task(wait_for_answer(ctx, settings()))
    for _ in range(1000):
        if room.listeners.get("participant_disconnected"):
            break
        await asyncio.sleep(0.001)
    for callback in list(room.listeners.get("participant_disconnected", [])):
        callback(participant)
    assert await task == ("no_answer", "disconnected")
    assert not room.listeners


def test_wait_for_answer_timeout_while_ringing(monkeypatch):
    participant = Participant({"sip.callStatus": "ringing"})
    room = Room({"callee": participant})
    ctx = Ctx(room, participant)
    import voice_agent.outbound_job as job

    real_wait_for = job.asyncio.wait_for
    wait_calls = 0

    async def expire_event_wait(awaitable, timeout):
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            return await real_wait_for(awaitable, timeout)
        awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(job.asyncio, "wait_for", expire_event_wait)
    assert asyncio.run(wait_for_answer(ctx, settings())) == ("no_answer", "timeout")


def test_wait_for_answer_missing_participant_is_no_answer():
    room = Room()
    ctx = Ctx(room, wait_error=RuntimeError("participant not found"))
    assert asyncio.run(wait_for_answer(ctx, settings())) == ("no_answer", "disconnected")


@pytest.mark.asyncio
async def test_run_outbound_answered_opens_with_the_reason(monkeypatch):
    registry = calls.CallRegistry()
    record = registry.start(CallRequest("the backup finished", "agent-a"), now(), 40, 1800)
    monkeypatch.setattr(calls, "REGISTRY", registry)
    import voice_agent.outbound_job as job

    async def fake_wait(ctx_, settings_):
        return "answered", ""

    monkeypatch.setattr(job, "wait_for_answer", fake_wait)
    room = Room()
    ctx = Ctx(room)
    session = Session()
    await run_outbound(
        ctx,
        session,
        settings(),
        {"call_id": record.call_id, "reason": record.reason, "requested_by": record.requested_by},
    )
    assert "the backup finished" in session.replies[0]
    assert not ctx.api.room.deleted
    assert ctx.callbacks
    assert not ctx.shutdowns


@pytest.mark.asyncio
async def test_run_outbound_no_answer_shuts_down_job_and_deletes_room(monkeypatch):
    registry = calls.CallRegistry()
    record = registry.start(CallRequest("r", "agent-a"), now(), 40, 1800)
    monkeypatch.setattr(calls, "REGISTRY", registry)
    import voice_agent.outbound_job as job

    async def fake_wait(ctx_, settings_):
        return "no_answer", "timeout"

    monkeypatch.setattr(job, "wait_for_answer", fake_wait)
    ctx = Ctx(Room())
    await run_outbound(
        ctx, Session(), settings(), {"call_id": record.call_id, "reason": "r", "requested_by": "agent-a"}
    )
    stored = registry.get(record.call_id, record.started_at, 40, 1800)
    assert stored is not None and stored.status == "no_answer" and stored.detail == "timeout"
    assert ctx.api.room.deleted and ctx.shutdowns


@pytest.mark.asyncio
async def test_run_outbound_job_error_marks_failed(monkeypatch):
    registry = calls.CallRegistry()
    record = registry.start(CallRequest("r", "agent-a"), now(), 40, 1800)
    monkeypatch.setattr(calls, "REGISTRY", registry)
    import voice_agent.outbound_job as job

    async def broken(ctx_, settings_):
        raise RuntimeError("broken")

    monkeypatch.setattr(job, "wait_for_answer", broken)
    ctx = Ctx(Room())
    await run_outbound(
        ctx, Session(), settings(), {"call_id": record.call_id, "reason": "r", "requested_by": "agent-a"}
    )
    stored = registry.get(record.call_id, record.started_at, 40, 1800)
    assert stored is not None and stored.status == "failed" and stored.detail == "job_error"
    assert ctx.api.room.deleted and ctx.shutdowns
