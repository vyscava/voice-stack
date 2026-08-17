"""Builds the gateway's collaborators from settings.

Kept apart from both the endpoint and the orchestration so that
``run_exchange`` stays free of configuration and stays testable with fakes.
"""

from __future__ import annotations

from core.settings import get_settings
from gateway import speech_gate
from gateway.agent import NatsAgent
from gateway.clients import AsrHttpClient, TtsHttpClient
from gateway.exchange import ExchangePolicy

settings = get_settings()

# One agent instance for the process: the NATS connection is reused across
# exchanges rather than dialled per utterance.
_agent: NatsAgent | None = None


def build_transcriber() -> AsrHttpClient:
    return AsrHttpClient(
        settings.GATEWAY_ASR_URL,
        timeout_s=settings.GATEWAY_ASR_TIMEOUT_S,
        language=settings.GATEWAY_LANGUAGE,
    )


def build_speaker() -> TtsHttpClient:
    return TtsHttpClient(
        settings.GATEWAY_TTS_URL,
        voice=settings.GATEWAY_VOICE,
        audio_format=settings.GATEWAY_AUDIO_FORMAT,
        timeout_s=settings.GATEWAY_TTS_TIMEOUT_S,
    )


def build_agent() -> NatsAgent:
    global _agent
    if _agent is None:
        _agent = NatsAgent(
            servers=settings.GATEWAY_NATS_URL,
            self_name=settings.GATEWAY_SELF_NAME,
            peer=settings.GATEWAY_AGENT,
            topic=settings.GATEWAY_NATS_TOPIC,
        )
    return _agent


class FfmpegSpeechGate:
    """The concrete gate. Thin adapter so exchange.py stays subprocess-free."""

    def looks_like_speech(self, audio: bytes) -> bool:
        return speech_gate.looks_like_speech(audio)

    def is_known_hallucination(self, transcript: str) -> bool:
        return speech_gate.is_known_hallucination(transcript)


def build_gate() -> FfmpegSpeechGate:
    return FfmpegSpeechGate()


def build_policy() -> ExchangePolicy:
    return ExchangePolicy(
        agent_timeout_s=settings.GATEWAY_AGENT_TIMEOUT_S,
        min_transcript_chars=settings.GATEWAY_MIN_TRANSCRIPT_CHARS,
        msg_no_speech=settings.GATEWAY_MSG_NO_SPEECH,
        msg_agent_timeout=settings.GATEWAY_MSG_AGENT_TIMEOUT,
        msg_agent_error=settings.GATEWAY_MSG_AGENT_ERROR,
    )


async def shutdown_agent() -> None:
    global _agent
    if _agent is not None:
        await _agent.close()
        _agent = None
