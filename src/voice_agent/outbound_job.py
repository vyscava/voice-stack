"""Helpers for identifying and running an outbound LiveKit SIP call."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from google.protobuf.duration_pb2 import Duration
from livekit import api
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


def status_for_sip_error(code: int | None) -> str:
    if code in (408, 480, 487):
        return "no_answer"
    if code in (486, 600, 603):
        return "busy"
    return "failed"


def context_turn(outbound: dict[str, str]) -> str:
    return (
        "[phone-system] You placed this call. The person you called just answered. Reason for the call: "
        f"{outbound['reason']}. Requested by: {outbound.get('requested_by', '')}. "
        "Greet them and say why you are calling in one or two short sentences."
    )


async def dial(ctx: JobContext, settings: Settings, outbound: dict[str, str]) -> tuple[str, str]:
    del outbound
    trunks = (await ctx.api.sip.list_sip_outbound_trunk(api.ListSIPOutboundTrunkRequest())).items
    trunk = next((item for item in trunks if item.name == settings.AGENT_CALL_TRUNK_NAME), None)
    if trunk is None:
        return "failed", "no_trunk"
    try:
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=trunk.sip_trunk_id,
                sip_call_to=settings.AGENT_CALL_TO,
                room_name=ctx.room.name,
                participant_identity="callee",
                participant_name="callee",
                wait_until_answered=True,
                ringing_timeout=Duration(seconds=settings.AGENT_CALL_RING_TIMEOUT_S),
                max_call_duration=Duration(seconds=settings.AGENT_CALL_MAX_DURATION_S),
            )
        )
    except api.SipCallError as err:
        code = err.sip_status_code
        return status_for_sip_error(code), f"sip_{code}"
    except api.TwirpError:
        return "failed", "error"
    return "answered", ""


async def run_outbound(
    ctx: JobContext, session: AgentSession[Any], settings: Settings, outbound: dict[str, str]
) -> None:
    call_id = outbound["call_id"]

    async def _run() -> tuple[str, str]:
        return await dial(ctx, settings, outbound)

    try:
        status, detail = await _run()
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
