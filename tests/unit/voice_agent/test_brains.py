"""Brains turn the conversation into the next thing to say, as text only."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.settings import Settings
from voice_agent.brains import (
    NatsBrain,
    OpenAICompatBrain,
    Turn,
    build_brain,
    history_from_chat_items,
    parse_extra_body,
)


async def _collect(gen) -> list[str]:
    return [chunk async for chunk in gen]


def _chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class FakeCompletions:
    def __init__(self, pieces):
        self.pieces = pieces
        self.kwargs: dict = {}

    async def create(self, **kwargs):
        self.kwargs = kwargs

        async def stream():
            for p in self.pieces:
                yield _chunk(p)
            yield SimpleNamespace(choices=[])  # usage-only chunk

        return stream()


def _client(pieces):
    completions = FakeCompletions(pieces)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


@pytest.mark.asyncio
async def test_openai_brain_streams_deltas_in_order():
    client, _ = _client(["Hello", None, " there", ""])
    brain = OpenAICompatBrain(client=client, model="m", system_prompt="be brief")
    out = await _collect(brain.reply([Turn("user", "hi")], call_id="c", turn_id=1, language="en"))
    assert out == ["Hello", " there"]


@pytest.mark.asyncio
async def test_openai_brain_sends_system_prompt_history_and_extra_body():
    client, completions = _client(["ok"])
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    brain = OpenAICompatBrain(client=client, model="kb", system_prompt="be brief", extra_body=extra)
    history = [Turn("user", "oi"), Turn("assistant", "olá"), Turn("user", "tudo bem?")]
    await _collect(brain.reply(history, call_id="c", turn_id=2, language="pt"))
    assert completions.kwargs["model"] == "kb"
    assert completions.kwargs["stream"] is True
    assert completions.kwargs["extra_body"] == extra
    assert completions.kwargs["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "oi"},
        {"role": "assistant", "content": "olá"},
        {"role": "user", "content": "tudo bem?"},
    ]


class FakeAgent:
    def __init__(self, reply="Sure."):
        self.reply = reply
        self.calls: list[tuple] = []

    async def ask(self, text, *, timeout_s, meta=None):
        self.calls.append((text, timeout_s, meta))
        return self.reply


@pytest.mark.asyncio
async def test_nats_brain_sends_last_user_turn_with_meta():
    agent = FakeAgent("Claro.")
    brain = NatsBrain(agent=agent, timeout_s=7)
    history = [Turn("user", "first"), Turn("assistant", "answer"), Turn("user", "second")]
    out = await _collect(brain.reply(history, call_id="call-1", turn_id=4, language="pt"))
    assert out == ["Claro."]
    assert agent.calls == [("second", 7, {"call_id": "call-1", "turn_id": 4, "language": "pt"})]


@pytest.mark.asyncio
async def test_nats_brain_yields_nothing_for_an_empty_reply():
    brain = NatsBrain(agent=FakeAgent("   "), timeout_s=1)
    assert await _collect(brain.reply([Turn("user", "x")], call_id="c", turn_id=1, language=None)) == []


def test_history_keeps_only_spoken_user_and_assistant_text():
    items = [
        SimpleNamespace(role="system", text_content="secret prompt"),
        SimpleNamespace(role="user", text_content="  hello  "),
        SimpleNamespace(role="assistant", text_content=""),
        SimpleNamespace(type="function_call"),
        SimpleNamespace(role="assistant", text_content="hi"),
    ]
    assert history_from_chat_items(items) == [Turn("user", "hello"), Turn("assistant", "hi")]


def test_extra_body_must_be_an_object():
    assert parse_extra_body("") == {}
    assert parse_extra_body('{"a": 1}') == {"a": 1}
    with pytest.raises(ValueError):
        parse_extra_body("[1, 2]")


def test_build_brain_selects_and_rejects():
    assert isinstance(build_brain(Settings(AGENT_BRAIN="openai")), OpenAICompatBrain)
    assert isinstance(build_brain(Settings(AGENT_BRAIN="NATS")), NatsBrain)
    with pytest.raises(ValueError, match="unknown AGENT_BRAIN"):
        build_brain(Settings(AGENT_BRAIN="telepathy"))


def test_history_strips_the_filler_from_assistant_turns():
    items = [
        SimpleNamespace(role="user", text_content="what time is it"),
        SimpleNamespace(role="assistant", text_content="Hmm, let me think. It is noon."),
        SimpleNamespace(role="assistant", text_content="Hmm, let me think."),
        SimpleNamespace(role="user", text_content="Hmm, let me think. is a user phrase too"),
    ]
    assert history_from_chat_items(items, strip_prefixes=("Hmm, let me think.",)) == [
        Turn("user", "what time is it"),
        Turn("assistant", "It is noon."),
        Turn("user", "Hmm, let me think. is a user phrase too"),
    ]


@pytest.mark.asyncio
async def test_brains_close_their_clients():
    closed: list[str] = []

    class Client:
        async def close(self):
            closed.append("http")

    class Agent:
        async def close(self):
            closed.append("nats")

    await OpenAICompatBrain(client=Client(), model="m", system_prompt="p").aclose()
    await NatsBrain(agent=Agent(), timeout_s=1).aclose()
    assert closed == ["http", "nats"]


def test_build_brain_passes_the_inbox_prefix_setting():
    """AGENT_NATS_INBOX_PREFIX reaches the NatsAgent."""
    brain = build_brain(Settings(AGENT_BRAIN="nats", AGENT_NATS_INBOX_PREFIX="_INBOX.voice-agent"))
    assert brain._agent._inbox_prefix == "_INBOX.voice-agent"
