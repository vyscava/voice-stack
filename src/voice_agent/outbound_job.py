"""Helpers for identifying and running an outbound LiveKit SIP call."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from livekit import api, rtc
from livekit.agents import AgentSession, JobContext

from core.settings import Settings
from voice_agent import calls, metrics
from voice_agent.brains import Brain, Turn

logger = logging.getLogger("voice_agent")

DEFAULT_OPENING_FALLBACK = "Hi, I'm calling about this: {reason}"


def outbound_from_metadata(metadata: object) -> dict[str, str] | None:
    if isinstance(metadata, str):
        try:
            value = json.loads(metadata)
        except (TypeError, ValueError):
            return None
    else:
        return None
    if not isinstance(value, dict):
        return None
    outbound = value.get("outbound")
    if not isinstance(outbound, dict):
        return None
    call_id = outbound.get("call_id")
    reason = outbound.get("reason")
    if not isinstance(call_id, str) or not call_id or not isinstance(reason, str) or not reason:
        return None
    requested_by = outbound.get("requested_by", "")
    if not isinstance(requested_by, str):
        requested_by = ""
    return {"call_id": call_id, "reason": reason, "requested_by": requested_by}


def context_turn(outbound: dict[str, str]) -> str:
    return (
        "[phone-system] You placed this call. The person you called just answered. Reason for the call: "
        f"{outbound['reason']}. Requested by: {outbound.get('requested_by', '')}. "
        "Greet them and say why you are calling in one or two short sentences."
    )


async def generate_opening(brain: Brain, outbound: dict[str, str]) -> str:
    """The brain's opening for this call, as one string.

    Started when the job joins the room, so it is generated while the phone
    rings. Tool progress is dropped: the opening is spoken whole, later.
    """
    history = [Turn(role="user", text=context_turn(outbound))]
    stream = brain.reply(history, call_id=outbound["call_id"], turn_id=0, language=None)
    parts = [chunk async for chunk in stream if isinstance(chunk, str)]
    return "".join(parts).strip()


def fallback_opening(settings: Settings, outbound: dict[str, str]) -> str:
    """A fixed opening built from the request's reason (a plain replace, never str.format)."""
    template = settings.AGENT_CALL_OPENING_FALLBACK.strip() or DEFAULT_OPENING_FALLBACK
    return template.replace("{reason}", outbound["reason"])


async def ready_opening(task: asyncio.Task[str], settings: Settings, outbound: dict[str, str]) -> tuple[str, str]:
    """The generated opening if it is ready within the wait after the answer, else the fallback."""
    try:
        text = await asyncio.wait_for(asyncio.shield(task), timeout=settings.AGENT_CALL_OPENING_WAIT_S)
    except Exception:
        if task.done():  # the brain itself failed (a brain's own TimeoutError lands here too)
            logger.warning("outbound opening generation failed; using the fallback")
        else:
            # Not ready in time: stop it now, so no stale request runs alongside the
            # callee's first turn.
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        text = ""
    if text:
        return text, "brain"
    return fallback_opening(settings, outbound), "fallback"


async def speak_opening(session: AgentSession[Any], text: str, outbound: dict[str, str]) -> None:
    """Say the opening whole, then answer anything the callee said while it played.

    The context turn goes into the chat first, so the brain knows on later turns
    why it called. The opening cannot be interrupted: a "hello?" over it is kept
    by LiveKit as a user message (the session keeps audio heard during
    uninterruptible speech) but gets no reply of its own, so it is answered here.
    """
    agent = session.current_agent
    chat_ctx = agent.chat_ctx.copy()
    chat_ctx.add_message(role="user", content=context_turn(outbound))
    await agent.update_chat_ctx(chat_ctx)
    users_before = _user_messages(agent)
    await session.say(text, allow_interruptions=False)
    # Let the session finish clearing the speech, so a turn that ended just as the
    # opening did is already in the chat when it is counted.
    await asyncio.sleep(0)
    if _user_messages(session.current_agent) > users_before:
        await session.generate_reply()


def _user_messages(agent: Any) -> int:
    return sum(1 for item in agent.chat_ctx.items if getattr(item, "role", None) == "user")


async def wait_for_answer(ctx: JobContext, settings: Settings) -> tuple[str, str]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + settings.AGENT_CALL_RING_TIMEOUT_S
    try:
        participant = await asyncio.wait_for(
            ctx.wait_for_participant(identity="callee", kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP),
            timeout=settings.AGENT_CALL_RING_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        return "no_answer", "timeout"
    except RuntimeError:
        return "no_answer", "disconnected"
    while True:
        changed = asyncio.Event()

        def on_attributes(
            changed_attributes: dict[str, str], changed_participant: rtc.Participant, event: asyncio.Event = changed
        ) -> None:
            if changed_participant is participant and "sip.callStatus" in changed_attributes:
                event.set()

        def on_disconnected(disconnected_participant: rtc.Participant, event: asyncio.Event = changed) -> None:
            if disconnected_participant is participant:
                event.set()

        ctx.room.on("participant_attributes_changed", on_attributes)
        ctx.room.on("participant_disconnected", on_disconnected)
        try:
            # Register before reading state so a transition cannot fall between the read and wait.
            status = participant.attributes.get("sip.callStatus", "")
            if status == "active":
                return "answered", ""
            if status in {"hangup", "disconnected", "failed", "terminated"}:
                return "no_answer", status
            if participant not in ctx.room.remote_participants.values():
                return "no_answer", "disconnected"
            remaining = deadline - loop.time()
            if remaining <= 0:
                return "no_answer", "timeout"
            try:
                await asyncio.wait_for(changed.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return "no_answer", "timeout"
        finally:
            ctx.room.off("participant_attributes_changed", on_attributes)
            ctx.room.off("participant_disconnected", on_disconnected)


async def run_outbound(
    ctx: JobContext, session: AgentSession[Any], settings: Settings, outbound: dict[str, str], *, brain: Brain
) -> None:
    call_id = outbound["call_id"]
    # Generate the opening while the phone rings, so it is ready when the callee
    # answers. Text only: nothing is synthesized until the call is answered.
    opening = asyncio.create_task(generate_opening(brain, outbound))

    try:
        status, detail = await wait_for_answer(ctx, settings)
        now = datetime.now(timezone.utc)
        calls.REGISTRY.update(call_id, status, detail, now=now)
        metrics.OUTBOUND_CALLS.labels(status).inc()
        logger.info("outbound call %s: %s %s", call_id, status, detail)
        if status == "answered":

            async def _ended() -> None:
                calls.REGISTRY.mark_ended(call_id, datetime.now(timezone.utc))

            ctx.add_shutdown_callback(_ended)
            text, source = await ready_opening(opening, settings, outbound)
            metrics.OUTBOUND_OPENINGS.labels(source).inc()
            logger.info("outbound call %s: opening from %s", call_id, source)
            await speak_opening(session, text, outbound)
            return
        await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
        ctx.shutdown(reason=f"outbound {status}")
    except Exception:
        calls.REGISTRY.update(call_id, "failed", "job_error")
        metrics.OUTBOUND_CALLS.labels("failed").inc()
        logger.exception("outbound call %s failed", call_id)
        try:
            await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
        except Exception:
            pass
        ctx.shutdown(reason="outbound job_error")
    finally:
        # Unanswered, failed or already spoken: never leave the brain request running.
        # Waiting for it closes the brain's stream now, and retrieves a failure so it
        # is not reported as unhandled.
        opening.cancel()
        await asyncio.gather(opening, return_exceptions=True)
