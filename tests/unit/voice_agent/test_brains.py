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


def _sse_chunk(text):
    """One chat-completions stream chunk as the server sends it."""
    import json

    return ["data: " + json.dumps({"choices": [{"delta": {"content": text}}]}), ""]


def _sse_lines(pieces, *, events=()):
    """SSE lines for ``pieces`` (text deltas), with ``(index, event, data)`` named events spliced in."""
    import json

    lines: list[str] = [": keepalive", ""]
    for i, piece in enumerate(pieces):
        for at, name, data in events:
            if at == i:
                lines += [f"event: {name}", "data: " + json.dumps(data), ""]
        lines += _sse_chunk(piece)
    lines += ["data: " + json.dumps({"choices": []}), ""]  # usage-only chunk
    lines += ["data: [DONE]", ""]
    return lines


class FakeResponse:
    def __init__(self, lines, closed):
        self._lines = lines
        self._closed = closed

    async def iter_text(self):
        # Split across chunk boundaries on purpose: lines arrive in pieces on a real stream.
        body = "\n".join(self._lines) + "\n"
        for i in range(0, len(body), 7):
            yield body[i : i + 7]


class FakeStreamingCall:
    def __init__(self, lines, closed):
        self._response = FakeResponse(lines, closed)
        self._closed = closed

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        self._closed.append(True)
        return False


class FakeCompletions:
    """``client.chat.completions`` whose ``with_streaming_response.create`` serves raw SSE lines."""

    def __init__(self, pieces, *, events=()):
        self.pieces = pieces
        self.events = events
        self.kwargs: dict = {}
        self.closed: list[bool] = []
        self.with_streaming_response = self

    def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeStreamingCall(_sse_lines(self.pieces, events=self.events), self.closed)


def _client(pieces, *, events=()):
    completions = FakeCompletions(pieces, events=events)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


@pytest.mark.asyncio
async def test_openai_brain_streams_deltas_in_order():
    client, _ = _client(["Hello", None, " there", ""])  # null and empty deltas are skipped
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


async def test_a_cancelled_reply_closes_the_http_stream():
    """A barge-in stops the generator after the first piece: the stream must be closed, not left open."""
    client, completions = _client(["Hello", " there", " friend"])
    gen = OpenAICompatBrain(client=client, model="m", system_prompt="p").reply(
        [], call_id="c", turn_id=1, language=None
    )
    assert await gen.__anext__() == "Hello"
    assert completions.closed == []
    await gen.aclose()
    assert completions.closed == [True]


@pytest.mark.asyncio
async def test_session_header_carries_one_id_per_call():
    client, completions = _client(["ok"])
    brain = OpenAICompatBrain(client=client, model="m", system_prompt="p", session_header=" X-Hermes-Session-Id ")
    await _collect(brain.reply([Turn("user", "hi")], call_id="call-1", turn_id=1, language="en"))
    first = completions.kwargs["extra_headers"]
    await _collect(brain.reply([Turn("user", "again")], call_id="call-1", turn_id=2, language="en"))
    same_call = completions.kwargs["extra_headers"]
    await _collect(brain.reply([Turn("user", "hi")], call_id="call-2", turn_id=1, language="en"))
    other_call = completions.kwargs["extra_headers"]
    assert first == same_call == {"X-Hermes-Session-Id": "voice-call-1"}
    assert other_call == {"X-Hermes-Session-Id": "voice-call-2"}


@pytest.mark.asyncio
async def test_no_session_header_unless_configured():
    client, completions = _client(["ok"])
    brain = OpenAICompatBrain(client=client, model="m", system_prompt="p")
    await _collect(brain.reply([Turn("user", "hi")], call_id="call-1", turn_id=1, language="en"))
    assert completions.kwargs["extra_headers"] is None


def test_build_brain_passes_the_session_header_setting():
    s = Settings(AGENT_BRAIN="openai", AGENT_LLM_SESSION_HEADER="X-Hermes-Session-Id")
    brain = build_brain(s)
    assert brain._headers("c9") == {"X-Hermes-Session-Id": "voice-c9"}


def test_session_id_is_header_and_filename_safe_and_short():
    client, _ = _client(["ok"])
    brain = OpenAICompatBrain(client=client, model="m", system_prompt="p", session_header="X-S")
    assert brain._headers("call-_+13055550100_ab/../c\r\nd") == {"X-S": "voice-call-_-13055550100_ab-..-c--d"}
    assert len(brain._headers("x" * 500)["X-S"]) == len("voice-") + 64
    assert brain._headers("") is None


@pytest.mark.asyncio
async def test_tool_progress_events_arrive_in_order_between_the_text():
    from voice_agent.brains import ToolProgress

    running = {"tool": "kb_search", "emoji": "x", "label": "KB search", "toolCallId": "t1", "status": "running"}
    done = {**running, "status": "completed"}
    client, _ = _client(
        ["Let me check the KB.", "\n\nFound it."],
        events=[(1, "hermes.tool.progress", running), (1, "hermes.tool.progress", done)],
    )
    out = await _collect(
        OpenAICompatBrain(client=client, model="m", system_prompt="p").reply([], call_id="c", turn_id=1, language="en")
    )
    assert out == [
        "Let me check the KB.",
        ToolProgress(tool="kb_search", label="KB search", status="running"),
        ToolProgress(tool="kb_search", label="KB search", status="completed"),
        "\n\nFound it.",
    ]
    assert out[1].running and not out[2].running


@pytest.mark.asyncio
async def test_other_named_events_and_bad_payloads_are_skipped():
    client, _ = _client(
        ["one", " two"],
        events=[
            (1, "hermes.status", {"message": "waiting for provider"}),
            (1, "hermes.tool.progress", {"tool": "x"}),  # no status: not a progress event
        ],
    )
    out = await _collect(
        OpenAICompatBrain(client=client, model="m", system_prompt="p").reply([], call_id="c", turn_id=1, language="en")
    )
    assert out == ["one", " two"]


@pytest.mark.asyncio
async def test_iter_sse_follows_the_format():
    from voice_agent.brains import iter_sse

    async def lines(*xs):
        for x in xs:
            yield x

    got = [
        e
        async for e in iter_sse(
            lines(": ping", "", "event: a\r\n", "data: 1", "data: 2", "", "data:x", "", "data: tail")
        )
    ]
    assert got == [("a", "1\n2"), (None, "x"), (None, "tail")]


@pytest.mark.asyncio
async def test_nothing_after_done_is_spoken():
    client, completions = _client(["Hi."])
    lines = _sse_lines(["Hi."]) + _sse_chunk(" stray")
    completions.create = lambda **kw: FakeStreamingCall(lines, completions.closed)
    out = await _collect(
        OpenAICompatBrain(client=client, model="m", system_prompt="p").reply([], call_id="c", turn_id=1, language="en")
    )
    assert out == ["Hi."]


@pytest.mark.asyncio
async def test_sse_lines_split_only_on_cr_lf_crlf():
    from voice_agent.brains import sse_lines

    async def pieces(*xs):
        for x in xs:
            yield x

    got = [line async for line in sse_lines(pieces("a\u2028b\r", "\nc\rd\n", "\n", "e\x85f", "\r"))]
    assert got == ["a\u2028b", "c", "d", "", "e\x85f"]


@pytest.mark.asyncio
async def test_the_real_sdk_stream_keeps_unicode_line_separators_and_raises_on_http_errors():
    """Through the real AsyncOpenAI client and httpx, not a fake response."""
    import httpx
    from openai import APIStatusError, AsyncOpenAI

    body = (
        'data: {"choices":[{"delta":{"content":"a\u2028b"}}]}\n\n'
        'event: hermes.tool.progress\ndata: {"tool":"t","label":"T","status":"running"}\n\n'
        'data: {"choices":[{"delta":{"content":" done"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    status = {"code": 200}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status["code"], text=body, headers={"content-type": "text/event-stream"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncOpenAI(base_url="http://brain.test/v1", api_key="x", http_client=http, max_retries=0)
    brain = OpenAICompatBrain(client=client, model="m", system_prompt="p")
    out = await _collect(brain.reply([Turn("user", "hi")], call_id="c", turn_id=1, language="en"))
    assert [o for o in out if isinstance(o, str)] == ["a\u2028b", " done"]
    assert sum(1 for o in out if not isinstance(o, str)) == 1
    status["code"] = 500
    with pytest.raises(APIStatusError):
        await _collect(brain.reply([Turn("user", "hi")], call_id="c", turn_id=1, language="en"))
    await http.aclose()


@pytest.mark.asyncio
async def test_an_error_inside_the_stream_is_a_failure_not_a_silent_cut():
    from voice_agent.brains import BrainStreamError

    client, completions = _client(["Hi"])
    lines = _sse_chunk("Hi") + ['data: {"error": {"message": "upstream died"}}', ""]
    completions.create = lambda **kw: FakeStreamingCall(lines, completions.closed)
    gen = OpenAICompatBrain(client=client, model="m", system_prompt="p").reply(
        [], call_id="c", turn_id=1, language="en"
    )
    assert await gen.__anext__() == "Hi"
    with pytest.raises(BrainStreamError, match="upstream died"):
        await gen.__anext__()
