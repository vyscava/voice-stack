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
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

from livekit import rtc
from livekit.agents import Agent, AgentSession, FlushSentinel, JobContext, JobExecutorType, WorkerOptions, stt
from livekit.agents.voice.background_audio import BackgroundAudioPlayer
from livekit.plugins import openai, silero

from core.settings import Settings, get_settings
from voice_agent import metrics
from voice_agent.brains import Brain, ToolProgress, build_brain, history_from_chat_items
from voice_agent.fallback_stt import FallbackBatchSTT
from voice_agent.fillers import Filler, FillerPicker, parse_phrases
from voice_agent.policy import guarded, preemptive_generation_allowed, voice_for_language, with_fillers, with_progress
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


def _openai_stt(base_url: str) -> openai.STT:
    return openai.STT(base_url=base_url.rstrip("/") + "/v1", api_key="unused", model="whisper-1", detect_language=True)


def build_batch_stt(settings: Settings) -> stt.STT[Any]:
    """AGENT_ASR_URL alone, or AGENT_ASR_PRIMARY_URL first with AGENT_ASR_URL as its fallback."""
    batch = _openai_stt(settings.AGENT_ASR_URL)
    primary_url = settings.AGENT_ASR_PRIMARY_URL.strip()
    if not primary_url:
        return batch
    return FallbackBatchSTT(
        primary=_openai_stt(primary_url),
        fallback=batch,
        primary_timeout_s=settings.AGENT_ASR_PRIMARY_TIMEOUT_S,
        cooldown_s=settings.AGENT_ASR_PRIMARY_COOLDOWN_S,
    )


def build_stt(settings: Settings, *, vad: Any) -> stt.STT[Any]:
    """Batch (one request per VAD segment) or streaming with batch as its fallback.

    Built per call: a streaming fallback lasts for the rest of that call only.
    """
    batch = build_batch_stt(settings)
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
        progress={"en": settings.AGENT_MSG_PROGRESS, "pt": settings.AGENT_MSG_PROGRESS_PT},
    )


def apology_for(settings: Settings, lang: str | None) -> str:
    if (lang or "").strip().lower().startswith("pt") and settings.AGENT_MSG_APOLOGY_PT:
        return settings.AGENT_MSG_APOLOGY_PT
    return settings.AGENT_MSG_APOLOGY


class VoiceAgent(Agent):
    def __init__(self, *, brain: Brain, settings: Settings, call_id: str, ticker: Ticker | None = None) -> None:
        super().__init__(instructions="")
        self._ticker: Ticker = ticker or NullTicker()
        # Set when the brain's first words are on their way to TTS: the ticks keep
        # going until that audio starts (tts_node), so the wait has no silent tail.
        self._ticks_until_audio = False
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
        self._ticks_until_audio = False
        self._ticker.stop()  # a new turn starts quiet
        completed = False
        tools: dict[str, bool | None] = {"preamble": None}  # None: no tool this turn
        # Ticks started by a progress line: the brain's next words hand them to
        # tts_node, which stops them when those words start sounding.
        tool_wait = {"ticking": False}
        brain = guarded(reply, timeout_s=s.AGENT_BRAIN_TIMEOUT_S, apology=apology_for(s, lang), on_failure=failed)
        brain = with_progress(brain, after_s=s.AGENT_PROGRESS_AFTER_S, line=lambda: self._fillers.progress(lang))
        try:
            async for chunk in with_fillers(brain, stages=stages):
                if isinstance(chunk, ToolProgress):
                    if chunk.running:
                        if tools["preamble"] is None:
                            tools["preamble"] = spoke["first"]
                        if spoke["first"]:
                            # Speak what the brain said before the tool NOW: LiveKit's
                            # sentence tokenizer would otherwise hold the last sentence
                            # until more text arrives, i.e. until the tool is done.
                            yield FlushSentinel()
                    continue
                if isinstance(chunk, Filler):
                    metrics.FILLERS.labels(kind=chunk.kind).inc()
                    self._ticker.stop()  # never tick over our own words
                    yield chunk + " "  # ours, not the brain's
                    # End the TTS segment now. Without this LiveKit's sentence tokenizer
                    # holds a line under 20 characters until more text arrives, and the
                    # ack would only be heard glued to the brain's first sentence.
                    yield FlushSentinel()
                    if not spoke["first"]:
                        # "loading" ticks fill the wait after the ack/thinking line.
                        if s.AGENT_TICKS:
                            self._ticker.start()
                    elif s.AGENT_TOOL_TICKS:
                        # A progress line: a tool is still running after the brain
                        # spoke (an agent call can take a minute). Tick through the
                        # rest of the wait, starting once the line has been heard.
                        self._ticker.start(delay_s=s.AGENT_TOOL_TICKS_DELAY_S)
                        tool_wait["ticking"] = True
                    continue
                if spoke["first"] and tools["preamble"] is not None and chunk.startswith("\n"):
                    # A new paragraph after a tool round: its own sentence, not glued
                    # to the preamble. Only in a tool turn, so a plain answer's
                    # paragraphs keep their prosody.
                    yield FlushSentinel()
                if not spoke["first"]:
                    spoke["first"] = True
                    # Keep ticking until the answer's audio starts (tts_node); the
                    # deadline covers a TTS that never produces any.
                    self._ticks_until_audio = True
                    self._ticker.stop_later(s.AGENT_TICKS_MAX_TAIL_S)
                    if outcome["value"] == "ok":  # an apology is not brain latency
                        metrics.STAGE_SECONDS.labels(stage="brain_first_text").observe(time.monotonic() - started)
                elif tool_wait["ticking"] and chunk.strip():
                    # The tool is done and the brain speaks again: tick until these
                    # words sound, as after the first words.
                    tool_wait["ticking"] = False
                    self._ticks_until_audio = True
                    self._ticker.stop_later(s.AGENT_TICKS_MAX_TAIL_S)
                yield chunk
            completed = True
        finally:
            # Barge-in, error, or tool-wait ticks no words will follow: never leave
            # the ticks running.
            if tool_wait["ticking"] or not (completed and self._ticks_until_audio):
                self._ticks_until_audio = False
                self._ticker.stop()
        metrics.TURNS.labels(outcome=outcome["value"]).inc()
        if tools["preamble"] is not None:
            metrics.TOOL_TURNS.labels(preamble="yes" if tools["preamble"] else "no").inc()

    async def tts_node(  # type: ignore[override]
        self, text: AsyncIterable[str], model_settings: Any
    ) -> AsyncIterator[rtc.AudioFrame]:
        """LiveKit's TTS, plus: stop the ticks when the brain's answer starts sounding."""
        try:
            async for frame in Agent.default.tts_node(self, text, model_settings):
                if self._ticks_until_audio:
                    self._ticks_until_audio = False
                    self._ticker.stop()
                yield frame
        finally:
            if self._ticks_until_audio:  # interrupted before the answer was heard
                self._ticks_until_audio = False
                self._ticker.stop()


async def entrypoint(ctx: JobContext) -> None:
    settings = get_settings()
    await ctx.connect()
    call_id = ctx.room.name
    # One brain per call, closed with the call: its HTTP/NATS client is bound
    # to this call's event loop.
    brain = build_brain(settings)
    ctx.add_shutdown_callback(brain.aclose)
    ticker: Ticker | None = None
    if settings.AGENT_TICKS or settings.AGENT_TOOL_TICKS:
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
