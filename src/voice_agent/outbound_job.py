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

logger = logging.getLogger("voice_agent")


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
    ctx: JobContext, session: AgentSession[Any], settings: Settings, outbound: dict[str, str]
) -> None:
    call_id = outbound["call_id"]

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
            await session.generate_reply(user_input=context_turn(outbound))
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
