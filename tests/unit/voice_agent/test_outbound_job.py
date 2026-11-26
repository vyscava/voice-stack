"""Tests for outbound LiveKit SIP job helpers."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from core.settings import Settings
from voice_agent import calls
from voice_agent.brains import ToolProgress, Turn
from voice_agent.outbound import CallRequest
from voice_agent.outbound_job import (
    DEFAULT_OPENING_FALLBACK,
    NAMED_OPENING_FALLBACK,
    context_turn,
    fallback_opening,
    generate_opening,
    outbound_from_metadata,
    run_outbound,
    wait_for_answer,
)


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


def settings(ring_timeout_s: int = 40, **overrides: Any) -> Settings:
    return Settings(
        LIVEKIT_URL="ws://livekit",
        LIVEKIT_API_KEY="key",
        LIVEKIT_API_SECRET="secret",
        AGENT_CALL_TO="callee",
        AGENT_CALL_TRUNK_NAME="asterisk-out",
        AGENT_CALL_RING_TIMEOUT_S=ring_timeout_s,
        AGENT_CALL_MAX_DURATION_S=1800,
        **overrides,
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


class ChatCtx:
    def __init__(self, items: list[Any] | None = None) -> None:
        self.items: list[Any] = list(items or [])

    def copy(self) -> "ChatCtx":
        return ChatCtx(self.items)

    def add_message(self, *, role: str, content: str) -> None:
        self.items.append(SimpleNamespace(role=role, text_content=content))


class Agent:
    def __init__(self) -> None:
        self._chat_ctx = ChatCtx()

    @property
    def chat_ctx(self) -> ChatCtx:
        return self._chat_ctx.copy()

    async def update_chat_ctx(self, chat_ctx: ChatCtx) -> None:
        self._chat_ctx = chat_ctx.copy()


class Handle:
    def __init__(self, on_playout: Any = None) -> None:
        self._on_playout = on_playout

    def __await__(self):
        async def wait() -> "Handle":
            if self._on_playout is not None:
                self._on_playout()
            return self

        return wait().__await__()


class Session:
    """Records what the job asked the session to say or reply.

    ``callee_says`` is appended to the agent's chat context while the opening
    plays, the way LiveKit keeps a user turn it heard during uninterruptible speech.
    """

    def __init__(self, callee_says: str | None = None) -> None:
        self.replies: list[dict[str, Any]] = []
        self.said: list[tuple[str, Any]] = []
        self.history_at_say: list[Any] = []
        self.current_agent = Agent()
        self._callee_says = callee_says

    async def generate_reply(self, **kwargs: Any) -> None:
        self.replies.append(kwargs)

    def say(self, text: str, **kwargs: Any) -> Handle:
        self.said.append((text, kwargs.get("allow_interruptions")))
        self.history_at_say = list(self.current_agent._chat_ctx.items)

        def playout() -> None:
            agent = self.current_agent
            if self._callee_says:
                agent._chat_ctx.add_message(role="user", content=self._callee_says)
            agent._chat_ctx.add_message(role="assistant", content=text)

        return Handle(playout)


class Brain:
    """A brain whose reply can be slow, fail, or include tool progress; records its calls."""

    def __init__(self, chunks: list[Any] | None = None, *, delay_s: float = 0.0, error: Exception | None = None):
        self.chunks = chunks if chunks is not None else ["Hi! ", "The backup ", "finished."]
        self.delay_s = delay_s
        self.error = error
        self.calls: list[list[Turn]] = []
        self.cancelled = False

    async def reply(self, history: list[Turn], *, call_id: str, turn_id: int, language: str | None):
        self.calls.append(history)
        try:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            if self.error is not None:
                raise self.error
            for chunk in self.chunks:
                yield chunk
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    async def aclose(self) -> None:
        pass


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
        brain=Brain(),
    )
    assert session.said == [("Hi! The backup finished.", False)]
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
        ctx,
        Session(),
        settings(),
        {"call_id": record.call_id, "reason": "r", "requested_by": "agent-a"},
        brain=Brain(),
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
        ctx,
        Session(),
        settings(),
        {"call_id": record.call_id, "reason": "r", "requested_by": "agent-a"},
        brain=Brain(),
    )
    stored = registry.get(record.call_id, record.started_at, 40, 1800)
    assert stored is not None and stored.status == "failed" and stored.detail == "job_error"
    assert ctx.api.room.deleted and ctx.shutdowns


# --- The opening: generated while the phone rings, spoken whole on answer ------------------

OUTBOUND = {"call_id": "call-out-c", "reason": "the backup finished", "requested_by": "agent-a"}


def _registry(monkeypatch, reason: str = "the backup finished") -> dict[str, str]:
    registry = calls.CallRegistry()
    record = registry.start(CallRequest(reason, "agent-a"), now(), 40, 1800)
    monkeypatch.setattr(calls, "REGISTRY", registry)
    return {"call_id": record.call_id, "reason": record.reason, "requested_by": record.requested_by}


def _answer_after(
    monkeypatch, status: str = "answered", detail: str = "", *, seen: list[int] | None = None, brain=None
):
    import voice_agent.outbound_job as job

    async def fake_wait(ctx_, settings_):
        await asyncio.sleep(0.01)  # ringing: the opening request is already on its way
        if seen is not None and brain is not None:
            seen.append(len(brain.calls))
        return status, detail

    monkeypatch.setattr(job, "wait_for_answer", fake_wait)


@pytest.mark.asyncio
async def test_generate_opening_asks_the_brain_with_the_context_turn_and_drops_tool_progress():
    brain = Brain(["Hi. ", ToolProgress(tool="t", label="l", status="running"), "Calling about the backup."])
    text = await generate_opening(brain, OUTBOUND)
    assert text == "Hi. Calling about the backup."
    assert brain.calls == [[Turn(role="user", text=context_turn(OUTBOUND))]]


def test_fallback_opening_uses_the_template_and_the_reason():
    assert fallback_opening(settings(), OUTBOUND) == DEFAULT_OPENING_FALLBACK.replace("{reason}", "the backup finished")
    custom = settings(AGENT_CALL_OPENING_FALLBACK="Hello, this is the assistant. {reason}.")
    assert fallback_opening(custom, OUTBOUND) == "Hello, this is the assistant. the backup finished."


def test_fallback_opening_names_the_caller_when_one_is_set():
    named = settings(AGENT_CALL_CALLER_NAME="  Ada ")
    assert fallback_opening(named, OUTBOUND) == "Hi, it's Ada, calling about: the backup finished"
    assert fallback_opening(settings(), OUTBOUND) == "Hi, I'm calling about: the backup finished"
    assert NAMED_OPENING_FALLBACK == "Hi, it's {name}, calling about: {reason}"


def test_an_explicit_fallback_template_wins_over_the_caller_name():
    both = settings(AGENT_CALL_CALLER_NAME="Ada", AGENT_CALL_OPENING_FALLBACK="{name} here. {reason}.")
    assert fallback_opening(both, OUTBOUND) == "Ada here. the backup finished."


def test_a_caller_name_with_braces_is_inserted_literally():
    odd = settings(AGENT_CALL_CALLER_NAME="{reason}")
    assert fallback_opening(odd, OUTBOUND) == "Hi, it's {reason}, calling about: the backup finished"


@pytest.mark.parametrize("template", ["", "{name}: {reason}"])
def test_placeholders_inside_the_reason_stay_literal(template):
    # The reason comes from another agent's request: it must never pull in the name.
    outbound = {**OUTBOUND, "reason": "x {name} {reason}"}
    named = settings(AGENT_CALL_CALLER_NAME="Ada", AGENT_CALL_OPENING_FALLBACK=template)
    assert "x {name} {reason}" in fallback_opening(named, outbound)
    assert fallback_opening(named, outbound).count("Ada") == 1


def test_an_empty_name_in_a_custom_template_leaves_no_stray_space():
    custom = settings(AGENT_CALL_OPENING_FALLBACK="{name} Hi, {reason}")
    assert fallback_opening(custom, OUTBOUND) == "Hi, the backup finished"


def test_fallback_opening_template_is_not_str_format():
    # Braces other than {reason} are left alone: a template is never str.format()ed.
    custom = settings(AGENT_CALL_OPENING_FALLBACK="{0} {reason} {missing}")
    assert fallback_opening(custom, OUTBOUND) == "{0} the backup finished {missing}"


@pytest.mark.asyncio
async def test_opening_is_requested_while_ringing_and_spoken_uninterruptible_on_answer(monkeypatch):
    outbound = _registry(monkeypatch)
    brain = Brain()
    seen: list[int] = []
    _answer_after(monkeypatch, seen=seen, brain=brain)
    session = Session()
    await run_outbound(Ctx(Room()), session, settings(), outbound, brain=brain)
    assert seen == [1]  # the brain was asked before the callee answered
    assert session.said == [("Hi! The backup finished.", False)]
    assert session.replies == []  # no generate_reply: no ack, no thinking line


@pytest.mark.asyncio
async def test_context_turn_is_in_the_chat_before_the_opening_is_spoken(monkeypatch):
    outbound = _registry(monkeypatch)
    _answer_after(monkeypatch)
    session = Session()
    await run_outbound(Ctx(Room()), session, settings(), outbound, brain=Brain())
    assert [(i.role, i.text_content) for i in session.history_at_say] == [("user", context_turn(outbound))]
    items = session.current_agent._chat_ctx.items
    assert [(i.role, i.text_content) for i in items] == [
        ("user", context_turn(outbound)),
        ("assistant", "Hi! The backup finished."),
    ]


@pytest.mark.asyncio
async def test_slow_brain_falls_back_after_the_wait_and_is_cancelled(monkeypatch):
    outbound = _registry(monkeypatch)
    _answer_after(monkeypatch)
    brain = Brain(delay_s=5.0)
    session = Session()
    cancelled_before_speaking: list[bool] = []
    say = session.say

    def recording_say(text: str, **kwargs: Any) -> Handle:
        cancelled_before_speaking.append(brain.cancelled)
        return say(text, **kwargs)

    session.say = recording_say  # type: ignore[method-assign]
    await run_outbound(Ctx(Room()), session, settings(AGENT_CALL_OPENING_WAIT_S=0.05), outbound, brain=brain)
    assert session.said == [(fallback_opening(settings(), outbound), False)]
    # The stale request is stopped before the fallback plays, not after the call's first turn.
    assert cancelled_before_speaking == [True]


@pytest.mark.asyncio
async def test_brain_ready_within_the_wait_after_answer_is_used(monkeypatch):
    outbound = _registry(monkeypatch)
    _answer_after(monkeypatch)
    brain = Brain(delay_s=0.05)  # still generating at answer, done inside the wait
    session = Session()
    await run_outbound(Ctx(Room()), session, settings(AGENT_CALL_OPENING_WAIT_S=1.0), outbound, brain=brain)
    assert session.said == [("Hi! The backup finished.", False)]


@pytest.mark.parametrize(
    "brain", [Brain(error=RuntimeError("down")), Brain(error=TimeoutError("peer")), Brain(chunks=["  ", ""])]
)
@pytest.mark.asyncio
async def test_failed_or_empty_brain_falls_back(monkeypatch, brain):
    outbound = _registry(monkeypatch)
    _answer_after(monkeypatch)
    session = Session()
    await run_outbound(Ctx(Room()), session, settings(), outbound, brain=brain)
    assert session.said == [(fallback_opening(settings(), outbound), False)]


@pytest.mark.asyncio
async def test_callee_speaking_during_the_opening_gets_a_reply_after_it(monkeypatch):
    outbound = _registry(monkeypatch)
    _answer_after(monkeypatch)
    session = Session(callee_says="hello?")
    await run_outbound(Ctx(Room()), session, settings(), outbound, brain=Brain())
    assert session.said == [("Hi! The backup finished.", False)]
    assert session.replies == [{}]  # one reply, to what is already in the chat (the callee's words)


@pytest.mark.asyncio
async def test_unanswered_call_discards_the_opening_without_speaking(monkeypatch):
    outbound = _registry(monkeypatch)
    _answer_after(monkeypatch, "no_answer", "timeout")
    brain = Brain(delay_s=5.0)
    session = Session()
    ctx = Ctx(Room())
    await run_outbound(ctx, session, settings(), outbound, brain=brain)
    assert session.said == [] and session.replies == []  # no TTS
    assert brain.cancelled
    assert ctx.shutdowns


@pytest.mark.asyncio
async def test_job_error_discards_the_opening(monkeypatch):
    outbound = _registry(monkeypatch)
    import voice_agent.outbound_job as job

    async def broken(ctx_, settings_):
        await asyncio.sleep(0.01)
        raise RuntimeError("broken")

    monkeypatch.setattr(job, "wait_for_answer", broken)
    brain = Brain(delay_s=5.0)
    session = Session()
    await run_outbound(Ctx(Room()), session, settings(), outbound, brain=brain)
    assert session.said == [] and brain.cancelled


@pytest.mark.asyncio
async def test_the_reason_and_the_opening_are_never_logged(monkeypatch, caplog):
    outbound = _registry(monkeypatch, reason="a private reason 7f3a")
    _answer_after(monkeypatch)
    caplog.set_level("DEBUG")
    await run_outbound(Ctx(Room()), Session(), settings(), outbound, brain=Brain(["Spoken opening 9c2e."]))
    assert caplog.records  # the job does log (call id and result)
    assert "7f3a" not in caplog.text and "9c2e" not in caplog.text
