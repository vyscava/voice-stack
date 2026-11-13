"""Worker configuration that, if wrong, silently breaks every call."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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
