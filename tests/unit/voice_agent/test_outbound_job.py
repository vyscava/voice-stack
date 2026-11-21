"""Tests for outbound LiveKit SIP job helpers."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from core.settings import Settings
from voice_agent import calls
from voice_agent.outbound import CallRequest
from voice_agent.outbound_job import context_turn, dial, outbound_from_metadata, run_outbound, status_for_sip_error


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


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (408, "no_answer"),
        (480, "no_answer"),
        (487, "no_answer"),
        (486, "busy"),
        (600, "busy"),
        (603, "busy"),
        (503, "failed"),
        (None, "failed"),
    ],
)
def test_sip_error_mapping(code, expected):
    assert status_for_sip_error(code) == expected


def test_context_turn_is_tagged_and_carries_the_reason():
    text = context_turn({"call_id": "c", "reason": "the backup finished", "requested_by": "agent-a"})
    assert text.startswith("[phone-system] ")
    assert "the backup finished" in text


def settings() -> Settings:
    return Settings(
        LIVEKIT_URL="ws://livekit",
        LIVEKIT_API_KEY="***",
        LIVEKIT_API_SECRET="***",
        AGENT_CALL_TO="callee",
        AGENT_CALL_TRUNK_NAME="asterisk-out",
        AGENT_CALL_RING_TIMEOUT_S=40,
        AGENT_CALL_MAX_DURATION_S=1800,
    )


class SipApi:
    def __init__(self, trunks=None, error=None):
        self.trunks = trunks or []
        self.error = error
        self.requests = []

    async def list_sip_outbound_trunk(self, request):
        return SimpleNamespace(items=self.trunks)

    async def create_sip_participant(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error


class RoomApi:
    def __init__(self):
        self.deleted = []

    async def delete_room(self, request):
        self.deleted.append(request)


class Ctx:
    def __init__(self, sip, room_name="call-out-c"):
        self.room = SimpleNamespace(name=room_name)
        self.api = SimpleNamespace(sip=sip, room=RoomApi())
        self.callbacks = []
        self.shutdowns = []

    def add_shutdown_callback(self, callback):
        self.callbacks.append(callback)

    def shutdown(self, reason=""):
        self.shutdowns.append(reason)


def now() -> datetime:
    return datetime.now(timezone.utc)


def test_dial_answered_uses_the_named_trunk_and_config_destination():
    sip = SipApi(
        [
            SimpleNamespace(name="other", sip_trunk_id="other-id"),
            SimpleNamespace(name="asterisk-out", sip_trunk_id="out-id"),
        ]
    )
    assert asyncio.run(dial(Ctx(sip), settings(), {"call_id": "c", "reason": "r", "requested_by": "agent-a"})) == (
        "answered",
        "",
    )
    request = sip.requests[0]
    assert request.sip_trunk_id == "out-id"
    assert request.sip_call_to == "callee"
    assert request.wait_until_answered is True
    assert request.ringing_timeout.seconds == 40


def test_dial_without_trunk_fails():
    assert asyncio.run(dial(Ctx(SipApi()), settings(), {"call_id": "c", "reason": "r"})) == ("failed", "no_trunk")


@pytest.mark.asyncio
async def test_run_outbound_answered_opens_with_the_reason(monkeypatch):
    registry = calls.CallRegistry()
    record = registry.start(CallRequest("the backup finished", "agent-a"), now(), 40, 1800)
    monkeypatch.setattr(calls, "REGISTRY", registry)
    import voice_agent.outbound_job as job

    async def fake_dial(ctx, settings, outbound):
        return "answered", ""

    monkeypatch.setattr(job, "dial", fake_dial)
    sip = SipApi([SimpleNamespace(name="asterisk-out", sip_trunk_id="out-id")])
    ctx = Ctx(sip)
    session = SimpleNamespace(replies=[])

    async def generate_reply(**kwargs):
        session.replies.append(kwargs["user_input"])

    session.generate_reply = generate_reply
    await run_outbound(
        ctx,
        session,
        settings(),
        {"call_id": record.call_id, "reason": record.reason, "requested_by": record.requested_by},
    )
    assert "the backup finished" in session.replies[0]
    assert not ctx.api.room.deleted
    assert ctx.callbacks


@pytest.mark.asyncio
async def test_run_outbound_failure_deletes_room_and_shuts_down(monkeypatch):
    registry = calls.CallRegistry()
    record = registry.start(CallRequest("r", "agent-a"), now(), 40, 1800)
    monkeypatch.setattr(calls, "REGISTRY", registry)
    import voice_agent.outbound_job as job

    async def fake_dial(ctx, settings, outbound):
        return "busy", "sip_486"

    monkeypatch.setattr(job, "dial", fake_dial)
    ctx = Ctx(SipApi())
    session = SimpleNamespace(generate_reply=None)
    await run_outbound(ctx, session, settings(), {"call_id": record.call_id, "reason": "r", "requested_by": "agent-a"})
    stored = registry.get(record.call_id, record.started_at, 40, 1800)
    assert stored is not None and stored.status == "busy"
    assert ctx.api.room.deleted and ctx.shutdowns


@pytest.mark.asyncio
async def test_run_outbound_job_error_marks_failed(monkeypatch):
    registry = calls.CallRegistry()
    record = registry.start(CallRequest("r", "agent-a"), now(), 40, 1800)
    monkeypatch.setattr(calls, "REGISTRY", registry)
    import voice_agent.outbound_job as job

    async def fake_dial(ctx, settings, outbound):
        raise RuntimeError("broken")

    monkeypatch.setattr(job, "dial", fake_dial)
    ctx = Ctx(SipApi())
    await run_outbound(
        ctx, SimpleNamespace(), settings(), {"call_id": record.call_id, "reason": "r", "requested_by": "agent-a"}
    )
    stored = registry.get(record.call_id, record.started_at, 40, 1800)
    assert stored is not None and stored.status == "failed" and stored.detail == "job_error"
    assert ctx.api.room.deleted and ctx.shutdowns
