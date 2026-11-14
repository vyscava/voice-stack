"""Brains: whatever turns the conversation so far into the next thing to say.

A brain takes and yields TEXT only. Audio never reaches it: the call's audio
stays inside the LiveKit room and this project's ASR/TTS services.

Two implementations:

* ``OpenAICompatBrain`` streams from any OpenAI-compatible ``/chat/completions``
  endpoint (a local llama.cpp or vLLM server, a hosted API).
* ``NatsBrain`` asks a peer agent over NATS request/reply. Peers answer in one
  piece, so it yields the whole reply at once. The peer keeps the conversation,
  keyed by ``call_id``; only the latest user turn is sent.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from core.settings import Settings


@dataclass(frozen=True)
class Turn:
    role: Literal["user", "assistant"]
    text: str


class Brain(Protocol):
    def reply(self, history: list[Turn], *, call_id: str, turn_id: int, language: str | None) -> AsyncIterator[str]: ...

    async def aclose(self) -> None: ...


def history_from_chat_items(items: Iterable[Any], *, strip_prefixes: tuple[str, ...] = ()) -> list[Turn]:
    """The spoken conversation as plain turns.

    Keeps user and assistant messages that have text. When the caller
    interrupts, LiveKit truncates the assistant message to what was actually
    played, so the history matches what the caller heard.

    ``strip_prefixes`` removes lines the agent said on its own (the filler)
    from the start of assistant messages, so the brain does not learn to
    imitate them.
    """
    turns: list[Turn] = []
    for item in items:
        role = getattr(item, "role", None)
        text = getattr(item, "text_content", None)
        if role not in ("user", "assistant") or not isinstance(text, str):
            continue
        text = text.strip()
        if role == "assistant":
            for prefix in (p for p in strip_prefixes if p):
                if text.startswith(prefix):
                    text = text[len(prefix) :].strip()
        if text:
            turns.append(Turn(role=role, text=text))
    return turns


class OpenAICompatBrain:
    def __init__(
        self,
        *,
        client: Any,
        model: str,
        system_prompt: str,
        extra_body: dict[str, Any] | None = None,
        max_tokens: int = 300,
    ) -> None:
        self._client = client
        self._model = model
        self._system_prompt = system_prompt
        self._extra_body = extra_body or {}
        self._max_tokens = max_tokens

    async def reply(
        self, history: list[Turn], *, call_id: str, turn_id: int, language: str | None
    ) -> AsyncIterator[str]:
        messages = [{"role": "system", "content": self._system_prompt}]
        messages += [{"role": t.role, "content": t.text} for t in history]
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            max_tokens=self._max_tokens,
            extra_body=self._extra_body or None,
        )
        async for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            piece = choices[0].delta.content if choices else None
            if piece:
                yield piece

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()


class NatsBrain:
    def __init__(self, *, agent: Any, timeout_s: float) -> None:
        self._agent = agent
        self._timeout_s = timeout_s

    async def reply(
        self, history: list[Turn], *, call_id: str, turn_id: int, language: str | None
    ) -> AsyncIterator[str]:
        last_user = next((t.text for t in reversed(history) if t.role == "user"), "")
        if not last_user:
            return
        meta: dict[str, str | int] = {"call_id": call_id, "turn_id": turn_id}
        if language:
            meta["language"] = language
        text = await self._agent.ask(last_user, timeout_s=self._timeout_s, meta=meta)
        if text and text.strip():
            yield text

    async def aclose(self) -> None:
        await self._agent.close()


def parse_extra_body(raw: str) -> dict[str, Any]:
    """The AGENT_LLM_EXTRA_BODY setting as a dict. Fails at startup, not per call."""
    if not raw.strip():
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("AGENT_LLM_EXTRA_BODY must be a JSON object")
    return value


def build_brain(settings: Settings) -> Brain:
    kind = settings.AGENT_BRAIN.strip().lower()
    if kind == "openai":
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=settings.AGENT_LLM_URL, api_key=settings.AGENT_LLM_API_KEY)
        return OpenAICompatBrain(
            client=client,
            model=settings.AGENT_LLM_MODEL,
            system_prompt=settings.AGENT_SYSTEM_PROMPT,
            extra_body=parse_extra_body(settings.AGENT_LLM_EXTRA_BODY),
        )
    if kind == "nats":
        from gateway.agent import NatsAgent

        agent = NatsAgent(
            servers=settings.AGENT_NATS_URL,
            self_name=settings.AGENT_SELF_NAME,
            peer=settings.AGENT_PEER,
            topic=settings.AGENT_NATS_TOPIC,
            inbox_prefix=settings.AGENT_NATS_INBOX_PREFIX,
        )
        return NatsBrain(agent=agent, timeout_s=settings.AGENT_BRAIN_TIMEOUT_S)
    raise ValueError(f"unknown AGENT_BRAIN {settings.AGENT_BRAIN!r}: expected 'openai' or 'nats'")
