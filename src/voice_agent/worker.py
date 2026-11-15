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

from livekit.agents import Agent, AgentSession, FlushSentinel, JobContext, JobExecutorType, WorkerOptions, stt
from livekit.agents.voice.background_audio import BackgroundAudioPlayer
from livekit.plugins import openai, silero

from core.settings import Settings, get_settings
from voice_agent import metrics
from voice_agent.brains import Brain, build_brain, history_from_chat_items
from voice_agent.fillers import Filler, FillerPicker, parse_phrases
from voice_agent.policy import guarded, preemptive_generation_allowed, voice_for_language, with_fillers
from voice_agent.stream_audio import stream_url
from voice_agent.stream_stt import AsrStreamSTT
from voice_agent.ticks import BackgroundTicker, NullTicker, Ticker

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
        # Off by default: it calls the brain speculatively before the turn is
        # final and cancels if the transcript moves on. Every discarded call
        # costs a real request, and a NATS peer would receive the duplicates,
        # so the nats brain never gets it whatever the setting says.
        "preemptive_generation": {
            "enabled": preemptive_generation_allowed(settings.AGENT_PREEMPTIVE_GENERATION, settings.AGENT_BRAIN)
        },
    }


def stt_mode(settings: Settings) -> str:
    mode = settings.AGENT_STT_MODE.strip().lower()
    if mode not in ("batch", "stream"):
        raise ValueError(f"unknown AGENT_STT_MODE {settings.AGENT_STT_MODE!r}: expected 'batch' or 'stream'")
    return mode


def build_stt(settings: Settings, *, vad: Any) -> stt.STT[Any]:
    """Batch (one request per VAD segment) or streaming with batch as its fallback.

    Built per call: a streaming fallback lasts for the rest of that call only.
    """
    batch = openai.STT(
        base_url=settings.AGENT_ASR_URL.rstrip("/") + "/v1",
        api_key="unused",
        model="whisper-1",
        detect_language=True,
    )
    if stt_mode(settings) == "batch":
        return batch
    url = stream_url(settings.AGENT_ASR_URL, settings.AGENT_ASR_STREAM_URL)
    return AsrStreamSTT(url=url, batch=batch, vad=vad, final_timeout_s=settings.AGENT_ASR_STREAM_FINAL_TIMEOUT_S)


def build_session(settings: Settings, *, vad: Any) -> AgentSession:
    return AgentSession(
        stt=build_stt(settings, vad=vad),
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


def build_fillers(settings: Settings) -> FillerPicker:
    return FillerPicker(
        acks={"en": parse_phrases(settings.AGENT_ACKS_EN), "pt": parse_phrases(settings.AGENT_ACKS_PT)},
        think={"en": settings.AGENT_MSG_FILLER, "pt": settings.AGENT_MSG_FILLER_PT},
    )


class VoiceAgent(Agent):
    def __init__(self, *, brain: Brain, settings: Settings, call_id: str, ticker: Ticker | None = None) -> None:
        super().__init__(instructions="")
        self._ticker: Ticker = ticker or NullTicker()
        self._brain = brain
        self._settings = settings
        self._call_id = call_id
        self._turn_id = 0
        self.language: str | None = None
        self._fillers = build_fillers(settings)

    async def llm_node(  # type: ignore[override]
        self, chat_ctx: Any, tools: Any, model_settings: Any
    ) -> AsyncIterator[str | FlushSentinel]:
        self._turn_id += 1
        s = self._settings
        history = history_from_chat_items(chat_ctx.items, strip_prefixes=self._fillers.phrases())
        started = time.monotonic()
        outcome = {"value": "ok"}
        spoke = {"first": False}

        def failed(kind: str) -> None:
            # After the first words, a failure cuts the reply short: "partial".
            outcome["value"] = "partial" if spoke["first"] else kind
            metrics.BRAIN_FAILURES.labels(kind=kind).inc()

        reply = self._brain.reply(history, call_id=self._call_id, turn_id=self._turn_id, language=self.language)
        lang = self.language
        stages = [
            (s.AGENT_ACK_AFTER_S, lambda: self._fillers.ack(lang)),
            (s.AGENT_FILLER_AFTER_S, lambda: self._fillers.think(lang)),
        ]
        try:
            async for chunk in with_fillers(
                guarded(reply, timeout_s=s.AGENT_BRAIN_TIMEOUT_S, apology=s.AGENT_MSG_APOLOGY, on_failure=failed),
                stages=stages,
            ):
                if isinstance(chunk, Filler):
                    metrics.FILLERS.labels(kind=chunk.kind).inc()
                    self._ticker.stop()  # never tick over our own words
                    yield chunk + " "  # ours, not the brain's
                    # End the TTS segment now. Without this LiveKit's sentence tokenizer
                    # holds a line under 20 characters until more text arrives, and the
                    # ack would only be heard glued to the brain's first sentence.
                    yield FlushSentinel()
                    self._ticker.start()  # "loading" ticks fill the wait after the line
                    continue
                if not spoke["first"]:
                    spoke["first"] = True
                    self._ticker.stop()
                    if outcome["value"] == "ok":  # an apology is not brain latency
                        metrics.STAGE_SECONDS.labels(stage="brain_first_text").observe(time.monotonic() - started)
                yield chunk
        finally:
            self._ticker.stop()  # barge-in or error: never leave the ticks running
        metrics.TURNS.labels(outcome=outcome["value"]).inc()


async def entrypoint(ctx: JobContext) -> None:
    settings = get_settings()
    await ctx.connect()
    call_id = ctx.room.name
    # One brain per call, closed with the call: its HTTP/NATS client is bound
    # to this call's event loop.
    brain = build_brain(settings)
    ctx.add_shutdown_callback(brain.aclose)
    ticker: Ticker | None = None
    if settings.AGENT_TICKS:
        player = BackgroundAudioPlayer()
        await player.start(room=ctx.room)
        ctx.add_shutdown_callback(player.aclose)
        ticker = BackgroundTicker(
            player,
            delay_s=settings.AGENT_TICKS_DELAY_S,
            volume=settings.AGENT_TICKS_VOLUME,
            interval_s=settings.AGENT_TICKS_INTERVAL_S,
        )
    agent = VoiceAgent(brain=brain, settings=settings, call_id=call_id, ticker=ticker)
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

    @session.on("error")
    def _on_error(ev: Any) -> None:
        metrics.record_error(ev)

    metrics.CALLS.inc()
    logger.info(f"voice call started: {call_id}")
    await session.start(room=ctx.room, agent=agent)
    if settings.AGENT_GREETING:
        await session.say(settings.AGENT_GREETING)


def worker_options(settings: Settings) -> WorkerOptions:
    stt_mode(settings)  # a typo fails at startup, not on every call
    # Threads, not processes: calls share one VAD model and one metrics registry.
    return WorkerOptions(
        entrypoint_fnc=entrypoint,
        job_executor_type=JobExecutorType.THREAD,
        ws_url=settings.LIVEKIT_URL,
        api_key=settings.LIVEKIT_API_KEY,
        api_secret=settings.LIVEKIT_API_SECRET,
        # Simultaneous arrivals: with no call running, LiveKit estimates each job's load as
        # load_threshold (0.7 in prod) / num_idle_processes and reserves that per job it
        # answers "available" to. With 1, the first reservation already reaches 0.7, so a
        # second call arriving in the same instant is refused ("no workers with sufficient
        # capacity") and never retried: that caller waited in an empty room (measured
        # 2026-11-15). N here admits N arrivals at once; it also keeps N runners warm.
        num_idle_processes=max(1, settings.AGENT_IDLE_RUNNERS),
    )
