"""Brains: whatever turns the conversation so far into the next thing to say.

A brain takes and yields TEXT only. Audio never reaches it: the call's audio
stays inside the LiveKit room and this project's ASR/TTS services.

Two implementations:

* ``OpenAICompatBrain`` streams from any OpenAI-compatible ``/chat/completions``
  endpoint (a local llama.cpp or vLLM server, a hosted API). Besides text it
  yields ``ToolProgress`` for servers that report tool activity as named SSE
  events on the same stream (an agent server such as Hermes sends
  ``event: hermes.tool.progress``): not speech, but it tells the voice layer a
  tool is running, so the text said before it can be spoken now.
* ``NatsBrain`` asks a peer agent over NATS request/reply. Peers answer in one
  piece, so it yields the whole reply at once. The peer keeps the conversation,
  keyed by ``call_id``; only the latest user turn is sent.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from core.settings import Settings

logger = logging.getLogger("voice_agent")


@dataclass(frozen=True)
class ToolProgress:
    """The brain started (``running``) or finished (``completed``) a tool. Never spoken."""

    tool: str
    label: str
    status: str

    @property
    def running(self) -> bool:
        return self.status == "running"


BrainChunk = str | ToolProgress


@dataclass(frozen=True)
class Turn:
    role: Literal["user", "assistant"]
    text: str


class Brain(Protocol):
    def reply(
        self, history: list[Turn], *, call_id: str, turn_id: int, language: str | None
    ) -> AsyncIterator[BrainChunk]: ...

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
            # Repeat: a turn can open with an ack and then a thinking line.
            stripped = True
            while stripped:
                stripped = False
                for prefix in (p for p in strip_prefixes if p):
                    if text.startswith(prefix):
                        text = text[len(prefix) :].strip()
                        stripped = True
            # The caller cut a filler off mid-word ("Oka", "Hmm, let"): nothing the brain said.
            if text and any(p.startswith(text) for p in strip_prefixes if p):
                text = ""
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
        session_header: str = "",
    ) -> None:
        self._client = client
        self._session_header = session_header.strip()
        self._model = model
        self._system_prompt = system_prompt
        self._extra_body = extra_body or {}
        self._max_tokens = max_tokens

    async def reply(
        self, history: list[Turn], *, call_id: str, turn_id: int, language: str | None
    ) -> AsyncIterator[BrainChunk]:
        messages = [{"role": "system", "content": self._system_prompt}]
        messages += [{"role": t.role, "content": t.text} for t in history]
        # The raw stream, not the SDK's parsed one: the SDK drops named SSE events,
        # and tool progress arrives as exactly those. Leaving the block (a barge-in
        # or a discarded generation cancels this generator) closes the HTTP stream,
        # so the endpoint stops generating for nobody.
        async with self._client.chat.completions.with_streaming_response.create(
            model=self._model,
            messages=messages,
            stream=True,
            max_tokens=self._max_tokens,
            extra_body=self._extra_body or None,
            extra_headers=self._headers(call_id),
        ) as response:
            async for event, data in iter_sse(response.iter_lines()):
                if event is None or event == "message":
                    if data.strip() == "[DONE]":
                        return
                    piece = _delta_content(data)
                    if piece:
                        yield piece
                elif event.endswith(".tool.progress"):
                    progress = _tool_progress(data)
                    if progress is not None:
                        yield progress

    def _headers(self, call_id: str) -> dict[str, str] | None:
        """One session per call on endpoints that key conversations by a header.

        Without it a server that derives the session from the first user message
        (Hermes api_server does) merges two calls that open with the same words.
        """
        if not self._session_header or not call_id:
            return None
        # Room names from SIP dispatch can carry a phone number and symbols; keep the id
        # header- and filename-safe (servers store sessions under it) and short.
        safe = re.sub(r"[^A-Za-z0-9._-]", "-", call_id)[:64]
        return {self._session_header: f"voice-{safe}"}

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


async def iter_sse(lines: AsyncIterable[str]) -> AsyncIterator[tuple[str | None, str]]:
    """Server-sent events as ``(event name or None, data)``, per the SSE format.

    ``data:`` lines of one event are joined with newlines; ``:`` lines are
    comments (keep-alives); a blank line ends an event.
    """
    event: str | None = None
    data: list[str] = []
    async for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = None, []
        elif line.startswith(":"):
            continue
        else:
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "event":
                event = value
            elif field == "data":
                data.append(value)
    if data:
        yield event, "\n".join(data)


def _delta_content(data: str) -> str | None:
    try:
        choices = json.loads(data).get("choices") or []
        content = (choices[0].get("delta") or {}).get("content") if choices else None
    except (ValueError, AttributeError, IndexError):
        logger.warning(f"brain stream: skipped an unreadable chunk: {data[:120]!r}")
        return None
    return content if isinstance(content, str) else None


def _tool_progress(data: str) -> ToolProgress | None:
    try:
        obj = json.loads(data)
        tool, status = str(obj.get("tool") or ""), str(obj.get("status") or "")
        label = str(obj.get("label") or tool)
    except (ValueError, AttributeError):
        logger.warning(f"brain stream: skipped an unreadable tool event: {data[:120]!r}")
        return None
    return ToolProgress(tool=tool, label=label, status=status) if status else None


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
            session_header=settings.AGENT_LLM_SESSION_HEADER,
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
