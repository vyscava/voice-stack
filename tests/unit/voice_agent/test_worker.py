"""Worker configuration that, if wrong, silently breaks every call."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from livekit.agents import FlushSentinel

from core.settings import Settings
from voice_agent import metrics, worker
from voice_agent.brains import Turn


def _settings(**kw) -> Settings:
    return Settings(AGENT_ASR_URL="http://asr:5001", AGENT_TTS_URL="http://tts:5002/", **kw)


@pytest.mark.asyncio
async def test_session_always_has_a_placeholder_llm():
    """LiveKit skips every reply when session.llm is None, even with llm_node overridden."""
    session = worker.build_session(_settings(), vad=None)
    assert session.llm is not None


@pytest.mark.asyncio
async def test_stt_detects_the_language_instead_of_assuming_english():
    session = worker.build_session(_settings(), vad=None)
    assert session.stt._opts.detect_language is True
    assert str(session.stt._client.base_url).rstrip("/") == "http://asr:5001/v1"


@pytest.mark.asyncio
async def test_tts_uses_pcm_and_the_english_voice_first():
    session = worker.build_session(_settings(AGENT_VOICE_EN="af_heart"), vad=None)
    assert session.tts._opts.voice == "af_heart"
    assert session.tts._opts.response_format == "pcm"
    assert str(session.tts._client.base_url).rstrip("/") == "http://tts:5002/v1"


def test_turn_handling_caps_the_turn_and_uses_vad_interruption():
    th = worker.turn_handling(
        _settings(AGENT_MAX_USER_TURN_S=12, AGENT_MIN_ENDPOINTING_DELAY_S=0.4, AGENT_MAX_ENDPOINTING_DELAY_S=2.5)
    )
    assert th["user_turn_limit"] == {"max_duration": 12}
    assert th["interruption"]["mode"] == "vad"
    assert th["endpointing"] == {"min_delay": 0.4, "max_delay": 2.5}


def test_worker_runs_calls_as_threads():
    from livekit.agents import JobExecutorType

    opts = worker.worker_options(_settings(LIVEKIT_URL="ws://lk:7880", LIVEKIT_API_KEY="k", LIVEKIT_API_SECRET="s"))
    assert opts.job_executor_type == JobExecutorType.THREAD
    assert opts.ws_url == "ws://lk:7880"


def _simultaneous_arrivals_admitted(num_idle_processes: int) -> int:
    """How many jobs LiveKit's own availability check admits at one instant, with no call running.

    Drives the installed livekit-agents AgentServer._is_available() and its reservation
    counter (private API: if they move, this fails loudly rather than passing).
    """
    import types

    from livekit.agents import JobExecutorType
    from livekit.agents import worker as lk_worker

    server = lk_worker.AgentServer(
        job_executor_type=JobExecutorType.THREAD,
        num_idle_processes=num_idle_processes,
        ws_url="ws://lk",
        api_key="k",
        api_secret="s",
    )
    server._devmode, server._draining, server._simulation = False, False, False
    server._worker_load, server._reserved_slots = 0.0, 0
    server._proc_pool = types.SimpleNamespace(processes=[])  # no call running
    admitted = 0
    for _ in range(4):
        if server._is_available():
            server._reserved_slots += 1
            admitted += 1
    return admitted


def test_two_simultaneous_calls_are_both_admitted_by_default():
    """One runner admitted one arrival at a time: the second of two simultaneous calls got no agent."""
    assert _simultaneous_arrivals_admitted(1) == 1  # the measured failure
    default = worker.worker_options(_settings()).num_idle_processes
    assert _simultaneous_arrivals_admitted(default) == 2


def test_the_idle_runner_setting_is_clamped_and_overridable():
    assert worker.worker_options(_settings()).num_idle_processes == 2
    assert worker.worker_options(_settings(AGENT_IDLE_RUNNERS=3)).num_idle_processes == 3
    assert worker.worker_options(_settings(AGENT_IDLE_RUNNERS=0)).num_idle_processes == 1


class ScriptedBrain:
    def __init__(self, *pieces, fail=False):
        self.pieces = pieces
        self.fail = fail
        self.seen: list = []

    async def reply(self, history, *, call_id, turn_id, language):
        self.seen.append((history, call_id, turn_id, language))
        if self.fail:
            raise RuntimeError("down")
        for p in self.pieces:
            yield p


def _chat_ctx(*pairs):
    return SimpleNamespace(items=[SimpleNamespace(role=r, text_content=t) for r, t in pairs])


@pytest.mark.asyncio
async def test_llm_node_passes_history_call_and_turn_to_the_brain():
    brain = ScriptedBrain("Olá!")
    agent = worker.VoiceAgent(brain=brain, settings=_settings(), call_id="call-x")
    agent.language = "pt"
    out = [c async for c in agent.llm_node(_chat_ctx(("user", "oi")), None, None)]
    out2 = [
        c async for c in agent.llm_node(_chat_ctx(("user", "oi"), ("assistant", "Olá!"), ("user", "e aí")), None, None)
    ]
    assert out == ["Olá!"] and out2 == ["Olá!"]
    assert brain.seen[0] == ([Turn("user", "oi")], "call-x", 1, "pt")
    assert brain.seen[1][2] == 2


@pytest.mark.asyncio
async def test_llm_node_speaks_the_apology_when_the_brain_fails():
    before = metrics.BRAIN_FAILURES.labels(kind="error")._value.get()
    agent = worker.VoiceAgent(
        brain=ScriptedBrain(fail=True), settings=_settings(AGENT_MSG_APOLOGY="Sorry!"), call_id="c"
    )
    out = [c async for c in agent.llm_node(_chat_ctx(("user", "hi")), None, None)]
    assert out == ["Sorry!"]
    assert metrics.BRAIN_FAILURES.labels(kind="error")._value.get() == before + 1


def test_stage_metrics_from_livekit_payloads():
    def count(stage):
        return metrics.STAGE_SECONDS.labels(stage=stage)._sum.get()

    before = {s: count(s) for s in ("eou", "stt", "tts_first_audio")}
    metrics.record_livekit_metrics(SimpleNamespace(type="eou_metrics", end_of_utterance_delay=0.5))
    metrics.record_livekit_metrics(SimpleNamespace(type="stt_metrics", duration=1.5))
    metrics.record_livekit_metrics(SimpleNamespace(type="tts_metrics", ttfb=0.25))
    metrics.record_livekit_metrics(SimpleNamespace(type="vad_metrics"))
    assert count("eou") - before["eou"] == pytest.approx(0.5)
    assert count("stt") - before["stt"] == pytest.approx(1.5)
    assert count("tts_first_audio") - before["tts_first_audio"] == pytest.approx(0.25)


def test_preemptive_generation_is_off():
    """Speculative brain calls would duplicate messages to a NATS peer."""
    assert worker.turn_handling(_settings())["preemptive_generation"] == {"enabled": False}


@pytest.mark.asyncio
async def test_filler_is_not_sent_back_to_the_brain():
    brain = ScriptedBrain("ok")
    agent = worker.VoiceAgent(brain=brain, settings=_settings(AGENT_MSG_FILLER="Hmm."), call_id="c")
    ctx = _chat_ctx(("user", "q1"), ("assistant", "Hmm. Answer one."), ("user", "q2"))
    _ = [c async for c in agent.llm_node(ctx, None, None)]
    assert brain.seen[0][0] == [Turn("user", "q1"), Turn("assistant", "Answer one."), Turn("user", "q2")]


@pytest.mark.asyncio
async def test_failure_after_first_words_is_partial_and_apology_is_not_latency():
    class DiesMidway:
        async def reply(self, history, *, call_id, turn_id, language):
            yield "One."
            raise RuntimeError("cut")

    def turns(outcome):
        return metrics.TURNS.labels(outcome=outcome)._value.get()

    def latency_count():
        return sum(
            s.value
            for m in metrics.STAGE_SECONDS.collect()
            for s in m.samples
            if s.name.endswith("_count") and s.labels.get("stage") == "brain_first_text"
        )

    partial_before = turns("partial")
    agent = worker.VoiceAgent(brain=DiesMidway(), settings=_settings(), call_id="c")
    assert [c async for c in agent.llm_node(_chat_ctx(("user", "hi")), None, None)] == ["One."]
    assert turns("partial") == partial_before + 1

    count_before = latency_count()
    dead = worker.VoiceAgent(
        brain=ScriptedBrain(fail=True), settings=_settings(AGENT_MSG_APOLOGY="Sorry!"), call_id="c"
    )
    assert [c async for c in dead.llm_node(_chat_ctx(("user", "hi")), None, None)] == ["Sorry!"]
    assert latency_count() == count_before, "an apology must not be recorded as brain latency"


@pytest.mark.asyncio
async def test_stt_mode_defaults_to_batch(monkeypatch):
    monkeypatch.delenv("AGENT_STT_MODE", raising=False)
    from livekit.plugins import openai

    assert isinstance(worker.build_stt(_settings(), vad=None), openai.STT)


@pytest.mark.asyncio
async def test_stt_mode_stream_from_env_derives_the_websocket_url(monkeypatch):
    from voice_agent.stream_stt import AsrStreamSTT

    monkeypatch.setenv("AGENT_STT_MODE", "stream")
    session = worker.build_session(_settings(), vad=None)
    assert isinstance(session.stt, AsrStreamSTT)
    assert session.stt.url == "ws://asr:5001/v1/stream"
    # The fallback is the same batch STT batch mode would use.
    assert session.stt.batch._opts.detect_language is True
    assert str(session.stt.batch._client.base_url).rstrip("/") == "http://asr:5001/v1"


@pytest.mark.asyncio
async def test_stt_stream_url_override(monkeypatch):
    monkeypatch.setenv("AGENT_STT_MODE", "stream")
    monkeypatch.setenv("AGENT_ASR_STREAM_URL", "wss://stream.example/v1/stream")
    monkeypatch.setenv("AGENT_ASR_STREAM_FINAL_TIMEOUT_S", "1.5")
    plugin = worker.build_stt(_settings(), vad=None)
    assert plugin.url == "wss://stream.example/v1/stream"
    assert plugin.final_timeout_s == 1.5


def test_unknown_stt_mode_fails_at_startup():
    with pytest.raises(ValueError, match="AGENT_STT_MODE"):
        worker.worker_options(_settings(AGENT_STT_MODE="streaming"))


@pytest.mark.parametrize(
    ("flag", "brain", "enabled"),
    [(True, "openai", True), (True, "nats", False), (False, "openai", False), (False, "nats", False)],
)
def test_preemptive_generation_only_for_the_openai_brain(flag, brain, enabled):
    th = worker.turn_handling(_settings(AGENT_PREEMPTIVE_GENERATION=flag, AGENT_BRAIN=brain))
    assert th["preemptive_generation"] == {"enabled": enabled}


class SlowBrain:
    def __init__(self, delay: float, text: str = "Answer."):
        self.delay, self.text = delay, text

    async def reply(self, history, *, call_id, turn_id, language):
        await asyncio.sleep(self.delay)
        yield self.text


def _filler_settings(**kw):
    base = {
        "AGENT_ACK_AFTER_S": 0.03,
        "AGENT_FILLER_AFTER_S": 0.12,
        "AGENT_ACKS_EN": "Okay.|Sure.",
        "AGENT_ACKS_PT": "Tá.",
    }
    base.update(kw)
    return _settings(**base)


def _fillers(kind):
    return metrics.FILLERS.labels(kind=kind)._value.get()


FLUSH = "<flush>"


def _shape(out):
    return [FLUSH if isinstance(c, FlushSentinel) else c for c in out]


def _brain_latency():
    for m in metrics.STAGE_SECONDS.collect():
        for smp in m.samples:
            if smp.labels.get("stage") == "brain_first_text":
                if smp.name.endswith("_sum"):
                    total = smp.value
                elif smp.name.endswith("_count"):
                    count = smp.value
    return count, total


@pytest.mark.asyncio
async def test_slow_brain_gets_an_ack_in_the_callers_language_and_it_is_not_brain_latency():
    agent = worker.VoiceAgent(brain=SlowBrain(0.08, "Resposta."), settings=_filler_settings(), call_id="c")
    agent.language = "pt"
    acks_before = _fillers("ack")
    count0, sum0 = _brain_latency()
    out = [c async for c in agent.llm_node(_chat_ctx(("user", "oi")), None, None)]
    assert _shape(out) == ["Tá. ", FLUSH, "Resposta."]
    assert _fillers("ack") == acks_before + 1
    count1, sum1 = _brain_latency()
    assert count1 == count0 + 1
    assert sum1 - sum0 >= 0.07, "brain latency must be the brain's words, not the ack"


@pytest.mark.asyncio
async def test_very_slow_brain_gets_ack_then_thinking_line_each_flushed():
    agent = worker.VoiceAgent(brain=SlowBrain(0.25), settings=_filler_settings(), call_id="c")
    out = [c async for c in agent.llm_node(_chat_ctx(("user", "hi")), None, None)]
    assert _shape(out) == ["Okay. ", FLUSH, "Hmm, let me think. ", FLUSH, "Answer."]


@pytest.mark.asyncio
async def test_a_fast_turn_does_not_use_up_an_ack():
    class FastThenSlow:
        def __init__(self):
            self.n = 0

        async def reply(self, history, *, call_id, turn_id, language):
            self.n += 1
            if self.n > 1:
                await asyncio.sleep(0.08)
            yield "A."

    agent = worker.VoiceAgent(brain=FastThenSlow(), settings=_filler_settings(), call_id="c")
    assert [c async for c in agent.llm_node(_chat_ctx(("user", "1")), None, None)] == ["A."]
    out = [c async for c in agent.llm_node(_chat_ctx(("user", "2")), None, None)]
    assert _shape(out) == ["Okay. ", FLUSH, "A."]


@pytest.mark.asyncio
async def test_dead_slow_brain_gets_fillers_then_the_apology():
    settings = _filler_settings(AGENT_BRAIN_TIMEOUT_S=0.2, AGENT_MSG_APOLOGY="Sorry!")
    agent = worker.VoiceAgent(brain=SlowBrain(1.0), settings=settings, call_id="c")
    out = [c async for c in agent.llm_node(_chat_ctx(("user", "hi")), None, None)]
    assert _shape(out) == ["Okay. ", FLUSH, "Hmm, let me think. ", FLUSH, "Sorry!"]


@pytest.mark.asyncio
async def test_fast_brain_gets_no_filler():
    agent = worker.VoiceAgent(brain=ScriptedBrain("Quick."), settings=_filler_settings(), call_id="c")
    assert [c async for c in agent.llm_node(_chat_ctx(("user", "hi")), None, None)] == ["Quick."]


@pytest.mark.asyncio
async def test_ack_and_thinking_line_are_stripped_from_the_history_sent_to_the_brain():
    brain = ScriptedBrain("ok")
    agent = worker.VoiceAgent(brain=brain, settings=_filler_settings(), call_id="c")
    ctx = _chat_ctx(("user", "q1"), ("assistant", "Sure. Hmm, let me think. Answer one."), ("user", "q2"))
    _ = [c async for c in agent.llm_node(ctx, None, None)]
    assert brain.seen[0][0] == [Turn("user", "q1"), Turn("assistant", "Answer one."), Turn("user", "q2")]


@pytest.mark.asyncio
async def test_a_filler_cut_off_mid_word_is_dropped_but_the_brains_own_okay_is_kept():
    brain = ScriptedBrain("ok")
    agent = worker.VoiceAgent(brain=brain, settings=_filler_settings(), call_id="c")
    ctx = _chat_ctx(
        ("user", "q1"),
        ("assistant", "Oka"),
        ("user", "q2"),
        ("assistant", "Okay. Hmm, let"),
        ("user", "q3"),
        ("assistant", "Okay, the answer is four."),
        ("user", "q4"),
    )
    _ = [c async for c in agent.llm_node(ctx, None, None)]
    assert brain.seen[0][0] == [
        Turn("user", "q1"),
        Turn("user", "q2"),
        Turn("user", "q3"),
        Turn("assistant", "Okay, the answer is four."),
        Turn("user", "q4"),
    ]


@pytest.mark.asyncio
async def test_acks_can_be_turned_off():
    agent = worker.VoiceAgent(brain=SlowBrain(0.08), settings=_filler_settings(AGENT_ACK_AFTER_S=0), call_id="c")
    assert [c async for c in agent.llm_node(_chat_ctx(("user", "hi")), None, None)] == ["Answer."]


class RecordingTicker:
    def __init__(self):
        self.events: list[str] = []

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")

    def stop_later(self, delay_s):
        self.events.append(f"stop_later:{delay_s}")


@pytest.mark.asyncio
async def test_ticks_run_from_the_ack_and_outlive_the_brains_text_until_its_audio():
    ticker = RecordingTicker()
    settings = _filler_settings(AGENT_TICKS_MAX_TAIL_S=1.5)
    agent = worker.VoiceAgent(brain=SlowBrain(0.08), settings=settings, call_id="c", ticker=ticker)
    seen = [(_shape([c])[0], list(ticker.events)) async for c in agent.llm_node(_chat_ctx(("user", "hi")), None, None)]
    # quiet at the turn start; ticks started after the ack's flush
    assert seen[1] == (FLUSH, ["stop", "stop"])
    # the answer's text does not stop them: only a deadline is armed
    assert seen[2] == ("Answer.", ["stop", "stop", "start", "stop_later:1.5"])
    assert ticker.events[-1] == "stop_later:1.5", "a normal end leaves the ticks for the answer's audio"


@pytest.mark.asyncio
async def test_the_answers_first_audio_frame_stops_the_ticks(monkeypatch):
    async def fake_tts(agent, text, model_settings):
        async for _ in text:
            pass
        yield "frame-1"
        yield "frame-2"

    monkeypatch.setattr(worker.Agent.default, "tts_node", fake_tts)

    async def words(*xs):
        for x in xs:
            yield x

    ticker = RecordingTicker()
    agent = worker.VoiceAgent(brain=SlowBrain(0.08), settings=_filler_settings(), call_id="c", ticker=ticker)
    _ = [c async for c in agent.llm_node(_chat_ctx(("user", "hi")), None, None)]
    n = len(ticker.events)
    tts = agent.tts_node(words("Answer."), None)
    assert await tts.__anext__() == "frame-1"
    assert ticker.events[n:] == ["stop"], "stopped as the first frame goes out, not at the end"
    assert [f async for f in tts] == ["frame-2"]
    assert ticker.events[n:] == ["stop"], "exactly one stop"


@pytest.mark.asyncio
async def test_filler_audio_does_not_stop_the_ticks(monkeypatch):
    async def fake_tts(agent, text, model_settings):
        yield "frame"

    monkeypatch.setattr(worker.Agent.default, "tts_node", fake_tts)
    ticker = RecordingTicker()
    agent = worker.VoiceAgent(brain=SlowBrain(5), settings=_filler_settings(), call_id="c", ticker=ticker)
    gen = agent.llm_node(_chat_ctx(("user", "hi")), None, None)
    assert await gen.__anext__() == "Okay. "
    await gen.__anext__()  # flush; ticks are told to start on the next pull
    n = len(ticker.events)
    _ = [f async for f in agent.tts_node(_one("Okay. "), None)]  # the ack's own audio
    assert ticker.events[n:] == []
    await gen.aclose()


async def _one(x):
    yield x


@pytest.mark.asyncio
async def test_tts_interrupted_before_the_answer_sounds_stops_the_ticks(monkeypatch):
    async def fake_tts(agent, text, model_settings):
        await asyncio.sleep(10)
        yield "never"

    monkeypatch.setattr(worker.Agent.default, "tts_node", fake_tts)
    ticker = RecordingTicker()
    agent = worker.VoiceAgent(brain=SlowBrain(0.08), settings=_filler_settings(), call_id="c", ticker=ticker)
    _ = [c async for c in agent.llm_node(_chat_ctx(("user", "hi")), None, None)]
    tts = agent.tts_node(_one("Answer."), None)
    task = asyncio.ensure_future(tts.__anext__())
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await tts.aclose()
    assert ticker.events[-1] == "stop"


@pytest.mark.asyncio
async def test_ticks_pause_for_the_thinking_line_and_resume():
    ticker = RecordingTicker()
    agent = worker.VoiceAgent(brain=SlowBrain(0.25), settings=_filler_settings(), call_id="c", ticker=ticker)
    _ = [c async for c in agent.llm_node(_chat_ctx(("user", "hi")), None, None)]
    assert ticker.events[:5] == ["stop", "stop", "start", "stop", "start"]
    assert ticker.events[-1].startswith("stop_later:")


@pytest.mark.asyncio
async def test_fast_brain_never_ticks():
    ticker = RecordingTicker()
    agent = worker.VoiceAgent(brain=ScriptedBrain("Quick."), settings=_filler_settings(), call_id="c", ticker=ticker)
    _ = [c async for c in agent.llm_node(_chat_ctx(("user", "hi")), None, None)]
    assert "start" not in ticker.events


@pytest.mark.asyncio
async def test_barge_in_stops_the_ticks():
    ticker = RecordingTicker()
    agent = worker.VoiceAgent(brain=SlowBrain(5), settings=_filler_settings(), call_id="c", ticker=ticker)
    gen = agent.llm_node(_chat_ctx(("user", "hi")), None, None)
    assert await gen.__anext__() == "Okay. "
    await gen.__anext__()  # the flush
    task = asyncio.ensure_future(gen.__anext__())  # ticks start; waiting on the brain
    await asyncio.sleep(0.02)
    assert ticker.events[-1] == "start"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await gen.aclose()
    assert ticker.events[-1] == "stop"


@pytest.mark.asyncio
async def test_batch_with_a_primary_url_tries_it_first_and_falls_back_to_asr_url():
    from voice_agent.fallback_stt import FallbackBatchSTT

    plugin = worker.build_stt(
        _settings(AGENT_ASR_PRIMARY_URL="http://gpu:5011/", AGENT_ASR_PRIMARY_TIMEOUT_S=2.0), vad=None
    )
    assert isinstance(plugin, FallbackBatchSTT)
    assert str(plugin.primary._client.base_url).rstrip("/") == "http://gpu:5011/v1"
    assert str(plugin.fallback._client.base_url).rstrip("/") == "http://asr:5001/v1"
    assert plugin.primary._opts.detect_language is True
    assert plugin.primary_timeout_s == 2.0


@pytest.mark.asyncio
async def test_a_blank_primary_url_keeps_the_single_batch_stt():
    from livekit.plugins import openai

    assert isinstance(worker.build_stt(_settings(AGENT_ASR_PRIMARY_URL="  "), vad=None), openai.STT)


@pytest.mark.asyncio
async def test_stream_mode_falls_back_to_the_primary_first_batch(monkeypatch):
    from voice_agent.fallback_stt import FallbackBatchSTT

    monkeypatch.setenv("AGENT_STT_MODE", "stream")
    plugin = worker.build_stt(_settings(AGENT_ASR_PRIMARY_URL="http://gpu:5011"), vad=None)
    assert isinstance(plugin.batch, FallbackBatchSTT)


class ToolBrain:
    """Speaks, runs a tool (progress events), then answers."""

    def __init__(self, *items):
        self.items = items

    async def reply(self, history, *, call_id, turn_id, language):
        for item in items_with_delays(self.items):
            delay, chunk = item
            await asyncio.sleep(delay)
            yield chunk


def items_with_delays(items):
    return [(i if isinstance(i, tuple) else (0, i)) for i in items]


def _tool(status="running"):
    from voice_agent.brains import ToolProgress

    return ToolProgress(tool="kb_search", label="KB", status=status)


def _tool_turns(preamble):
    return metrics.TOOL_TURNS.labels(preamble=preamble)._value.get()


@pytest.mark.asyncio
async def test_a_tool_start_flushes_the_preamble_and_events_never_reach_tts():
    yes = _tool_turns("yes")
    brain = ToolBrain("Let me check the KB.", _tool(), _tool("completed"), "\n\nFound it.")
    settings = _filler_settings(AGENT_ACK_AFTER_S=0, AGENT_FILLER_AFTER_S=0, AGENT_PROGRESS_AFTER_S=0)
    agent = worker.VoiceAgent(brain=brain, settings=settings, call_id="c")
    out = _shape([c async for c in agent.llm_node(_chat_ctx(("user", "kb?")), None, None)])
    assert out == ["Let me check the KB.", FLUSH, FLUSH, "\n\nFound it."]
    assert _tool_turns("yes") == yes + 1


@pytest.mark.asyncio
async def test_a_silent_tool_turn_counts_as_no_preamble():
    no = _tool_turns("no")
    settings = _filler_settings(AGENT_ACK_AFTER_S=0, AGENT_FILLER_AFTER_S=0)
    agent = worker.VoiceAgent(brain=ToolBrain(_tool(), "Found it."), settings=settings, call_id="c")
    out = _shape([c async for c in agent.llm_node(_chat_ctx(("user", "kb?")), None, None)])
    assert out == ["Found it."]
    assert _tool_turns("no") == no + 1


@pytest.mark.asyncio
async def test_a_quiet_tool_run_after_the_preamble_gets_a_progress_line():
    before = _fillers("progress")
    brain = ToolBrain("Vou verificar.", _tool(), (0.2, "Pronto."))
    settings = _filler_settings(AGENT_ACK_AFTER_S=0, AGENT_FILLER_AFTER_S=0, AGENT_PROGRESS_AFTER_S=0.05)
    agent = worker.VoiceAgent(brain=brain, settings=settings, call_id="c")
    agent.language = "pt"
    out = _shape([c async for c in agent.llm_node(_chat_ctx(("user", "oi")), None, None)])
    assert out == ["Vou verificar.", FLUSH, "Ainda estou verificando. ", FLUSH, "Pronto."]
    assert _fillers("progress") == before + 1


@pytest.mark.asyncio
async def test_a_portuguese_turn_gets_the_portuguese_apology():
    agent = worker.VoiceAgent(
        brain=ScriptedBrain(fail=True), settings=_settings(AGENT_MSG_APOLOGY_PT="Desculpa!"), call_id="c"
    )
    agent.language = "pt"
    out = [c async for c in agent.llm_node(_chat_ctx(("user", "oi")), None, None)]
    assert out == ["Desculpa!"]
    agent.language = "en"
    out = [c async for c in agent.llm_node(_chat_ctx(("user", "hi")), None, None)]
    assert out[0].startswith("Sorry")


@pytest.mark.asyncio
async def test_a_progress_line_does_not_restart_the_ticks():
    ticker = RecordingTicker()
    brain = ToolBrain("Vou verificar.", _tool(), (0.2, "Pronto."))
    settings = _filler_settings(AGENT_ACK_AFTER_S=0, AGENT_FILLER_AFTER_S=0, AGENT_PROGRESS_AFTER_S=0.05)
    agent = worker.VoiceAgent(brain=brain, settings=settings, call_id="c", ticker=ticker)
    agent.language = "pt"
    out = _shape([c async for c in agent.llm_node(_chat_ctx(("user", "oi")), None, None)])
    assert "Ainda estou verificando. " in out
    assert "start" not in ticker.events  # nothing would stop them over the answer


@pytest.mark.asyncio
async def test_a_plain_answers_paragraphs_are_not_flushed():
    settings = _filler_settings(AGENT_ACK_AFTER_S=0, AGENT_FILLER_AFTER_S=0)
    agent = worker.VoiceAgent(brain=ScriptedBrain("First.", "\n\nSecond."), settings=settings, call_id="c")
    out = _shape([c async for c in agent.llm_node(_chat_ctx(("user", "hi")), None, None)])
    assert out == ["First.", "\n\nSecond."]
