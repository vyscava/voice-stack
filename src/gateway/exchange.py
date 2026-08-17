"""One voice exchange, end to end, and the policy for every way it can fail.

    audio in -> ASR -> agent -> TTS -> audio out

This module is deliberately free of HTTP, NATS and FastAPI. It talks to three
protocols, so every branch below is reachable in a unit test without a network,
a model, or a message bus. The failure policy is the valuable part of the
gateway and it should not require a running homelab to verify.

Two invariants worth stating out loud
-------------------------------------
**Audio never reaches the agent.** ``Agent.ask`` accepts ``str`` and returns
``str``. Not by convention: there is no parameter capable of carrying bytes.
Audio is produced and consumed on the same host, NATS is a message bus with a
~1 MB default payload rather than a media pipe, and raw household audio should
not exist as a routable message.

**Failure is always audible.** Every outcome tries to return speech, because a
caller who hears nothing cannot distinguish a crash from a pause and will simply
talk over it. The only silent outcome is TTS itself failing, which degrades to
text and says so.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class Outcome(str, Enum):
    """How an exchange ended. Every value is a normal, handled result."""

    OK = "ok"
    NO_SPEECH = "no_speech"
    ASR_FAILED = "asr_failed"
    AGENT_TIMEOUT = "agent_timeout"
    AGENT_ERROR = "agent_error"
    TTS_FAILED = "tts_failed"


class Transcriber(Protocol):
    async def transcribe(self, audio: bytes, *, filename: str) -> str: ...


class Speaker(Protocol):
    async def speak(self, text: str) -> bytes: ...


class Agent(Protocol):
    """The agent transport. Text in, text out -- audio cannot pass through it."""

    async def ask(self, text: str, *, timeout_s: float) -> str: ...


class SpeechGate(Protocol):
    """Decides whether a clip is worth sending to ASR at all.

    Injected rather than imported so this module stays free of subprocesses and
    stays testable without ffmpeg.
    """

    def looks_like_speech(self, audio: bytes) -> bool: ...

    def is_known_hallucination(self, transcript: str) -> bool: ...


@dataclass(frozen=True)
class ExchangePolicy:
    """The canned speech and thresholds, lifted out of settings for testability."""

    agent_timeout_s: float = 60.0
    min_transcript_chars: int = 2
    msg_no_speech: str = "Sorry, I did not catch that. Could you say it again?"
    msg_agent_timeout: str = "Sorry, that is taking too long. Please try again in a moment."
    msg_agent_error: str = "Sorry, I could not reach the assistant right now."


@dataclass
class ExchangeResult:
    outcome: Outcome
    transcript: str | None = None
    reply_text: str | None = None
    audio: bytes | None = None
    detail: str | None = None

    @property
    def degraded(self) -> bool:
        """True when the caller gets something, but not what was intended.

        ``OK`` is the only non-degraded outcome. A spoken apology is a
        successful HTTP response and a failed exchange; conflating the two is
        how a broken pipeline reads as healthy.
        """
        return self.outcome is not Outcome.OK


def _is_speech(transcript: str, policy: ExchangePolicy) -> bool:
    """Reject transcripts that are too short to be worth an agent round trip.

    Whisper emits punctuation-only or single-character output for silence,
    breathing and door noise. Forwarding that to an agent wastes a turn and
    invites a confidently wrong answer to a question nobody asked.
    """
    return len(transcript.strip()) >= policy.min_transcript_chars


async def _try_speak(speaker: Speaker, text: str) -> bytes | None:
    """Synthesise, or return None. Never raises.

    Used for the apology paths: if the exchange has already failed, a TTS
    failure on top must not replace a handled outcome with a 500.
    """
    try:
        return await speaker.speak(text)
    except Exception:
        return None


async def run_exchange(
    audio: bytes,
    *,
    filename: str,
    transcriber: Transcriber,
    speaker: Speaker,
    agent: Agent,
    policy: ExchangePolicy,
    gate: SpeechGate | None = None,
) -> ExchangeResult:
    """Run one exchange. Returns a result for every outcome; raises for none."""
    # 0. Is this audio at all? Whisper does NOT return "I heard nothing" -- it
    #    INVENTS text, and reliably invents the same things: 1.5 s of silence
    #    becomes "You", near-silence becomes "Thank you for watching!". Both
    #    observed here; the second reached a peer agent as a real question.
    #
    #    This cannot be caught downstream by transcript length, because a
    #    hallucination is confident and confidence has no length. The only
    #    place to stop it is before the model is called, since a model that is
    #    never called cannot invent anything.
    if gate is not None and not gate.looks_like_speech(audio):
        return ExchangeResult(
            outcome=Outcome.NO_SPEECH,
            reply_text=policy.msg_no_speech,
            audio=await _try_speak(speaker, policy.msg_no_speech),
            detail="audio did not contain speech (too short or too quiet)",
        )

    # 1. Hear.
    try:
        transcript = await transcriber.transcribe(audio, filename=filename)
    except Exception as exc:
        return ExchangeResult(
            outcome=Outcome.ASR_FAILED,
            reply_text=policy.msg_no_speech,
            audio=await _try_speak(speaker, policy.msg_no_speech),
            detail=f"{type(exc).__name__}: {exc}",
        )

    # 2. Decide whether that was speech at all. The agent is never asked about
    #    noise, so a cough cannot become a question.
    # Backstop: a clip that passed the audio gate can still transcribe to a
    # known artifact. Catches what has already been seen; the audio gate above
    # is what catches what has not.
    if gate is not None and transcript and gate.is_known_hallucination(transcript):
        return ExchangeResult(
            outcome=Outcome.NO_SPEECH,
            transcript=transcript,
            reply_text=policy.msg_no_speech,
            audio=await _try_speak(speaker, policy.msg_no_speech),
            detail=f"transcript is a known silence artifact: {transcript!r}",
        )

    if not _is_speech(transcript, policy):
        return ExchangeResult(
            outcome=Outcome.NO_SPEECH,
            transcript=transcript,
            reply_text=policy.msg_no_speech,
            audio=await _try_speak(speaker, policy.msg_no_speech),
        )

    # 3. Ask the agent. Text only, and bounded: an agent that never answers
    #    must not become a request that never returns.
    try:
        reply = await agent.ask(transcript.strip(), timeout_s=policy.agent_timeout_s)
    except (TimeoutError, asyncio.TimeoutError):
        return ExchangeResult(
            outcome=Outcome.AGENT_TIMEOUT,
            transcript=transcript,
            reply_text=policy.msg_agent_timeout,
            audio=await _try_speak(speaker, policy.msg_agent_timeout),
        )
    except Exception as exc:
        return ExchangeResult(
            outcome=Outcome.AGENT_ERROR,
            transcript=transcript,
            reply_text=policy.msg_agent_error,
            audio=await _try_speak(speaker, policy.msg_agent_error),
            detail=f"{type(exc).__name__}: {exc}",
        )

    # An agent that replies with nothing is an agent failure, not a silence to
    # pass on. Speaking empty audio would look like the gateway broke.
    if not reply or not reply.strip():
        return ExchangeResult(
            outcome=Outcome.AGENT_ERROR,
            transcript=transcript,
            reply_text=policy.msg_agent_error,
            audio=await _try_speak(speaker, policy.msg_agent_error),
            detail="agent returned an empty reply",
        )

    # 4. Speak the answer. If TTS fails the exchange still succeeded in
    #    substance, so hand back the text rather than throwing it away.
    try:
        spoken = await speaker.speak(reply)
    except Exception as exc:
        return ExchangeResult(
            outcome=Outcome.TTS_FAILED,
            transcript=transcript,
            reply_text=reply,
            audio=None,
            detail=f"{type(exc).__name__}: {exc}",
        )

    return ExchangeResult(outcome=Outcome.OK, transcript=transcript, reply_text=reply, audio=spoken)
