"""The LiveKit worker: one voice session per call.

Two things here are easy to get wrong and silently break every call:

* **A placeholder LLM is required.** LiveKit skips the reply step entirely when
  the session has no LLM, even when ``llm_node`` is overridden. Ours is never
  called; the brain answers instead.
* **ASR must detect the language.** The OpenAI STT plugin sends ``language=en``
  by default, which forces Portuguese speech into English nonsense.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator
from typing import Any

from livekit.agents import Agent, AgentSession, JobContext, JobExecutorType, WorkerOptions
from livekit.plugins import openai, silero

from core.settings import Settings, get_settings
from voice_agent import metrics
from voice_agent.brains import Brain, build_brain, history_from_chat_items
from voice_agent.policy import guarded, voice_for_language, with_filler

# Never contacted: llm_node is overridden. Port 9 is discard.
_PLACEHOLDER_LLM_URL = "http://127.0.0.1:9/v1"

logger = logging.getLogger("voice_agent")

_vad_lock = threading.Lock()
_vad: Any = None


def shared_vad() -> Any:
    """Silero VAD, loaded once per process and shared by every call."""
    global _vad
    with _vad_lock:
        if _vad is None:
            _vad = silero.VAD.load()
        return _vad


def turn_handling(settings: Settings) -> dict[str, Any]:
    return {
        "turn_detection": "vad",
        "endpointing": {
            "min_delay": settings.AGENT_MIN_ENDPOINTING_DELAY_S,
            "max_delay": settings.AGENT_MAX_ENDPOINTING_DELAY_S,
        },
        # VAD-based: adaptive interruption is a hosted feature.
        "interruption": {"mode": "vad"},
        # Background speech never goes silent; cap the turn so it still ends.
        "user_turn_limit": {"max_duration": settings.AGENT_MAX_USER_TURN_S},
        # Off: it calls the brain speculatively before the turn is final and
        # cancels if the transcript moves on. A NATS peer would receive the
        # duplicates, and every discarded call costs a real request.
        "preemptive_generation": {"enabled": False},
    }


def build_session(settings: Settings, *, vad: Any) -> AgentSession:
    return AgentSession(
        stt=openai.STT(
            base_url=settings.AGENT_ASR_URL.rstrip("/") + "/v1",
            api_key="unused",
            model="whisper-1",
            detect_language=True,
        ),
        tts=openai.TTS(
            base_url=settings.AGENT_TTS_URL.rstrip("/") + "/v1",
            api_key="unused",
            model="tts-1",
            voice=settings.AGENT_VOICE_EN,
            response_format="pcm",
        ),
        llm=openai.LLM(base_url=_PLACEHOLDER_LLM_URL, api_key="unused", model="placeholder"),
        vad=vad,
        turn_handling=turn_handling(settings),  # type: ignore[arg-type]
    )


class VoiceAgent(Agent):
    def __init__(self, *, brain: Brain, settings: Settings, call_id: str) -> None:
        super().__init__(instructions="")
        self._brain = brain
        self._settings = settings
        self._call_id = call_id
        self._turn_id = 0
        self.language: str | None = None

    async def llm_node(self, chat_ctx: Any, tools: Any, model_settings: Any) -> AsyncIterator[str]:  # type: ignore[override]
        self._turn_id += 1
        s = self._settings
        history = history_from_chat_items(chat_ctx.items, strip_prefixes=(s.AGENT_MSG_FILLER,))
        started = time.monotonic()
        outcome = {"value": "ok"}
        spoke = {"first": False}

        def failed(kind: str) -> None:
            # After the first words, a failure cuts the reply short: "partial".
            outcome["value"] = "partial" if spoke["first"] else kind
            metrics.BRAIN_FAILURES.labels(kind=kind).inc()

        reply = self._brain.reply(history, call_id=self._call_id, turn_id=self._turn_id, language=self.language)
        filler_pending = True
        async for chunk in with_filler(
            guarded(reply, timeout_s=s.AGENT_BRAIN_TIMEOUT_S, apology=s.AGENT_MSG_APOLOGY, on_failure=failed),
            after_s=s.AGENT_FILLER_AFTER_S,
            filler=s.AGENT_MSG_FILLER,
        ):
            if filler_pending and chunk == s.AGENT_MSG_FILLER and not spoke["first"]:
                filler_pending = False  # the filler is ours, not the brain's
            elif not spoke["first"]:
                spoke["first"] = True
                filler_pending = False
                if outcome["value"] == "ok":  # an apology is not brain latency
                    metrics.STAGE_SECONDS.labels(stage="brain_first_text").observe(time.monotonic() - started)
            yield chunk
        metrics.TURNS.labels(outcome=outcome["value"]).inc()


async def entrypoint(ctx: JobContext) -> None:
    settings = get_settings()
    await ctx.connect()
    call_id = ctx.room.name
    # One brain per call, closed with the call: its HTTP/NATS client is bound
    # to this call's event loop.
    brain = build_brain(settings)
    ctx.add_shutdown_callback(brain.aclose)
    agent = VoiceAgent(brain=brain, settings=settings, call_id=call_id)
    vad = await asyncio.to_thread(shared_vad)  # first load reads the model from disk
    session = build_session(settings, vad=vad)
    state = {"agent": "initializing", "voice": settings.AGENT_VOICE_EN}

    @session.on("user_input_transcribed")
    def _on_transcript(ev: Any) -> None:
        if not ev.is_final:
            return
        if ev.language:
            agent.language = ev.language
        voice = voice_for_language(
            ev.language, en=settings.AGENT_VOICE_EN, pt=settings.AGENT_VOICE_PT, current=state["voice"]
        )
        if voice != state["voice"]:
            state["voice"] = voice
            session.tts.update_options(voice=voice)

    @session.on("agent_state_changed")
    def _on_agent_state(ev: Any) -> None:
        state["agent"] = ev.new_state

    @session.on("user_state_changed")
    def _on_user_state(ev: Any) -> None:
        if ev.new_state == "speaking" and state["agent"] == "speaking":
            metrics.BARGE_INS.inc()

    @session.on("metrics_collected")
    def _on_metrics(ev: Any) -> None:
        metrics.record_livekit_metrics(ev.metrics)

    metrics.CALLS.inc()
    logger.info(f"voice call started: {call_id}")
    await session.start(room=ctx.room, agent=agent)
    if settings.AGENT_GREETING:
        await session.say(settings.AGENT_GREETING)


def worker_options(settings: Settings) -> WorkerOptions:
    # Threads, not processes: calls share one VAD model and one metrics registry.
    return WorkerOptions(
        entrypoint_fnc=entrypoint,
        job_executor_type=JobExecutorType.THREAD,
        ws_url=settings.LIVEKIT_URL,
        api_key=settings.LIVEKIT_API_KEY,
        api_secret=settings.LIVEKIT_API_SECRET,
        num_idle_processes=1,
    )
