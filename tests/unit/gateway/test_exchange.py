"""Every outcome of a voice exchange, including the ones that only happen when something breaks.

The failure paths are the point of this file. A gateway that works when ASR,
the agent and TTS all cooperate is easy; what decides whether it is usable is
what a caller hears when one of them does not.

No network, no models, no message bus: ``run_exchange`` talks to three
protocols, so each collaborator is a few lines of fake here.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.exchange import (
    ExchangePolicy,
    Outcome,
    run_exchange,
)

AUDIO = b"RIFF....WAVEfake"
POLICY = ExchangePolicy(agent_timeout_s=0.05, min_transcript_chars=2)


class FakeTranscriber:
    def __init__(self, text: str = "what is the weather", exc: Exception | None = None) -> None:
        self.text, self.exc = text, exc
        self.calls: list[bytes] = []

    async def transcribe(self, audio: bytes, *, filename: str) -> str:
        self.calls.append(audio)
        if self.exc:
            raise self.exc
        return self.text


class FakeSpeaker:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc
        self.spoken: list[str] = []

    async def speak(self, text: str) -> bytes:
        self.spoken.append(text)
        if self.exc:
            raise self.exc
        return b"OggS" + text.encode()


class FakeAgent:
    def __init__(self, reply: str = "It is sunny.", exc: Exception | None = None, delay: float = 0.0) -> None:
        self.reply, self.exc, self.delay = reply, exc, delay
        self.asked: list[str] = []

    async def ask(self, text: str, *, timeout_s: float) -> str:
        self.asked.append(text)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return self.reply


async def _run(transcriber=None, speaker=None, agent=None, policy=POLICY):
    return await run_exchange(
        AUDIO,
        filename="utterance.wav",
        transcriber=transcriber or FakeTranscriber(),
        speaker=speaker or FakeSpeaker(),
        agent=agent or FakeAgent(),
        policy=policy,
    )


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_full_round_trip(self):
        agent, speaker = FakeAgent(reply="It is sunny."), FakeSpeaker()
        result = await _run(agent=agent, speaker=speaker)

        assert result.outcome is Outcome.OK
        assert result.transcript == "what is the weather"
        assert result.reply_text == "It is sunny."
        assert result.audio == b"OggSIt is sunny."
        assert not result.degraded

    @pytest.mark.asyncio
    async def test_agent_is_handed_the_RAW_transcript(self):
        """Stripping is the TRANSPORT's job, not the orchestrator's.

        The DM carries both a cleaned `text` and a verbatim `transcript`. If
        run_exchange pre-stripped, the raw form would be lost before the
        transport could record it, and `transcript` would silently become a
        second copy of `text`.
        """
        agent = FakeAgent()
        await _run(transcriber=FakeTranscriber("  hello there  "), agent=agent)
        assert agent.asked == ["  hello there  "], "the raw transcript must reach the transport intact"


class TestAudioNeverReachesTheAgent:
    """The invariant that keeps household audio off the message bus."""

    @pytest.mark.asyncio
    async def test_agent_receives_only_text(self):
        agent = FakeAgent()
        await _run(agent=agent)

        assert len(agent.asked) == 1
        sent = agent.asked[0]
        assert isinstance(sent, str)
        # The strong form: the audio's bytes appear nowhere in what was sent.
        assert AUDIO not in sent.encode()

    def test_agent_protocol_cannot_carry_bytes(self):
        """Structural, not conventional. `ask` has no parameter for audio.

        If someone adds one, this fails and they have to argue for it.
        """
        import inspect

        from gateway.exchange import Agent

        params = inspect.signature(Agent.ask).parameters
        assert set(params) == {"self", "text", "timeout_s"}

        # eval_str because the module uses `from __future__ import annotations`,
        # which would otherwise make this compare against the STRING "str" and
        # pass for any annotation at all, including bytes.
        hints = inspect.get_annotations(Agent.ask, eval_str=True)
        assert hints["text"] is str
        assert hints["return"] is str


class TestNoSpeech:
    """Noise must not become a question the agent answers."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("noise", ["", "   ", ".", "\n"])
    async def test_short_or_empty_transcript_is_not_forwarded(self, noise):
        agent = FakeAgent()
        result = await _run(transcriber=FakeTranscriber(noise), agent=agent)

        assert result.outcome is Outcome.NO_SPEECH
        assert agent.asked == [], "the agent was asked about noise"
        assert result.degraded

    @pytest.mark.asyncio
    async def test_caller_hears_a_prompt_to_repeat(self):
        speaker = FakeSpeaker()
        result = await _run(transcriber=FakeTranscriber(""), speaker=speaker)

        assert result.audio is not None, "silence would be indistinguishable from a crash"
        assert speaker.spoken == [POLICY.msg_no_speech]


class TestAgentFailures:
    @pytest.mark.asyncio
    async def test_timeout_is_spoken_not_hung(self):
        """The agent never answering must not become a request that never returns."""
        agent = FakeAgent(exc=TimeoutError("no reply"))
        speaker = FakeSpeaker()
        result = await _run(agent=agent, speaker=speaker)

        assert result.outcome is Outcome.AGENT_TIMEOUT
        assert result.audio is not None
        assert speaker.spoken == [POLICY.msg_agent_timeout]
        assert result.transcript == "what is the weather", "the transcript survives for the caller's context"

    @pytest.mark.asyncio
    async def test_asyncio_timeout_is_treated_as_a_timeout(self):
        result = await _run(agent=FakeAgent(exc=asyncio.TimeoutError()))
        assert result.outcome is Outcome.AGENT_TIMEOUT

    @pytest.mark.asyncio
    async def test_transport_error_is_spoken(self):
        result = await _run(agent=FakeAgent(exc=ConnectionRefusedError("bus down")))

        assert result.outcome is Outcome.AGENT_ERROR
        assert result.audio is not None
        assert "ConnectionRefusedError" in (result.detail or "")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("empty", ["", "   "])
    async def test_empty_reply_is_an_error_not_a_silence(self, empty):
        """Speaking an empty reply would look like the gateway itself broke."""
        result = await _run(agent=FakeAgent(reply=empty))

        assert result.outcome is Outcome.AGENT_ERROR
        assert result.detail == "agent returned an empty reply"


class TestAsrFailure:
    @pytest.mark.asyncio
    async def test_asr_error_does_not_reach_the_agent(self):
        agent = FakeAgent()
        result = await _run(transcriber=FakeTranscriber(exc=RuntimeError("model gone")), agent=agent)

        assert result.outcome is Outcome.ASR_FAILED
        assert agent.asked == []
        assert result.audio is not None
        assert "RuntimeError" in (result.detail or "")


class TestTtsFailure:
    @pytest.mark.asyncio
    async def test_reply_degrades_to_text_rather_than_being_lost(self):
        """The exchange succeeded in substance. Do not throw the answer away."""
        result = await _run(speaker=FakeSpeaker(exc=RuntimeError("cuda oom")))

        assert result.outcome is Outcome.TTS_FAILED
        assert result.reply_text == "It is sunny.", "the agent's answer must survive"
        assert result.audio is None
        assert result.degraded

    @pytest.mark.asyncio
    async def test_tts_failing_on_an_apology_does_not_raise(self):
        """Two failures at once must still produce a handled result.

        Without this, a TTS outage turns every ASR failure into a 500.
        """
        result = await _run(
            transcriber=FakeTranscriber(exc=RuntimeError("asr down")),
            speaker=FakeSpeaker(exc=RuntimeError("tts down")),
        )

        assert result.outcome is Outcome.ASR_FAILED
        assert result.audio is None


class TestNothingRaises:
    """run_exchange returns a result for every outcome and raises for none."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "transcriber,speaker,agent",
        [
            (FakeTranscriber(exc=OSError("x")), FakeSpeaker(), FakeAgent()),
            (FakeTranscriber(), FakeSpeaker(exc=OSError("x")), FakeAgent()),
            (FakeTranscriber(), FakeSpeaker(), FakeAgent(exc=OSError("x"))),
            (FakeTranscriber(exc=OSError("x")), FakeSpeaker(exc=OSError("y")), FakeAgent(exc=OSError("z"))),
        ],
    )
    async def test_every_collaborator_failing_still_returns(self, transcriber, speaker, agent):
        result = await _run(transcriber=transcriber, speaker=speaker, agent=agent)
        assert isinstance(result.outcome, Outcome)
        assert result.degraded


class TestDegradedFlag:
    @pytest.mark.asyncio
    async def test_only_ok_is_not_degraded(self):
        """A spoken apology is a successful HTTP response and a failed exchange.

        Conflating those is how a broken pipeline reads as healthy.
        """
        assert not (await _run()).degraded
        for bad in (
            FakeTranscriber(""),
            FakeTranscriber(exc=OSError("x")),
        ):
            assert (await _run(transcriber=bad)).degraded


class FakeGate:
    """A gate with both decisions forced, so each branch is reachable."""

    def __init__(self, speech: bool = True, hallucination: bool = False) -> None:
        self.speech, self.hallucination = speech, hallucination
        self.checked_audio: list[bytes] = []

    def looks_like_speech(self, audio: bytes) -> bool:
        self.checked_audio.append(audio)
        return self.speech

    def is_known_hallucination(self, transcript: str) -> bool:
        return self.hallucination


class TestSilenceNeverReachesTheAgent:
    """The failure this guard exists for, observed live on 2026-08-17.

    Whisper turned a near-empty clip into "Thank you for watching!", the gateway
    forwarded it as a real question, and the agent only declined to answer it
    because it happened to recognise the artifact.
    """

    @pytest.mark.asyncio
    async def test_silent_audio_is_rejected_before_asr(self):
        transcriber, agent = FakeTranscriber(), FakeAgent()
        result = await run_exchange(
            AUDIO,
            filename="u.wav",
            transcriber=transcriber,
            speaker=FakeSpeaker(),
            agent=agent,
            policy=POLICY,
            gate=FakeGate(speech=False),
        )

        assert result.outcome is Outcome.NO_SPEECH
        assert transcriber.calls == [], "ASR was called on a clip with no speech"
        assert agent.asked == [], "the agent was asked about silence"

    @pytest.mark.asyncio
    async def test_the_caller_still_hears_something(self):
        speaker = FakeSpeaker()
        result = await run_exchange(
            AUDIO,
            filename="u.wav",
            transcriber=FakeTranscriber(),
            speaker=speaker,
            agent=FakeAgent(),
            policy=POLICY,
            gate=FakeGate(speech=False),
        )
        assert result.audio is not None
        assert speaker.spoken == [POLICY.msg_no_speech]

    @pytest.mark.asyncio
    async def test_hallucinated_transcript_is_not_forwarded(self):
        """The backstop: audio passed the gate, Whisper invented text anyway."""
        agent = FakeAgent()
        result = await run_exchange(
            AUDIO,
            filename="u.wav",
            transcriber=FakeTranscriber("Thank you for watching!"),
            speaker=FakeSpeaker(),
            agent=agent,
            policy=POLICY,
            gate=FakeGate(speech=True, hallucination=True),
        )

        assert result.outcome is Outcome.NO_SPEECH
        assert agent.asked == [], "a known Whisper artifact was sent to the agent"
        assert result.transcript == "Thank you for watching!", "the transcript is kept for diagnosis"
        assert "silence artifact" in (result.detail or "")

    @pytest.mark.asyncio
    async def test_real_speech_still_gets_through_with_a_gate_present(self):
        """The guard must not become a wall."""
        agent = FakeAgent(reply="It is sunny.")
        result = await run_exchange(
            AUDIO,
            filename="u.wav",
            transcriber=FakeTranscriber("what is the weather"),
            speaker=FakeSpeaker(),
            agent=agent,
            policy=POLICY,
            gate=FakeGate(speech=True, hallucination=False),
        )

        assert result.outcome is Outcome.OK
        assert agent.asked == ["what is the weather"]

    @pytest.mark.asyncio
    async def test_no_gate_configured_keeps_prior_behaviour(self):
        """gate is optional, so existing callers are unaffected."""
        result = await _run()
        assert result.outcome is Outcome.OK
