"""The transport details that fail silently if they are wrong.

Two of these are worth the file on their own:

* The NATS subject reads *from* dot dm dot *to*. Getting it backwards is
  rejected asynchronously by the server while the publish still reports
  success, so the message simply vanishes with no error anywhere. It has
  already happened once in this fleet.
* An agent's reply may arrive as a JSON envelope or as bare text depending on
  which peer answered. Mishandling it means reading JSON aloud to a person.
"""

from __future__ import annotations

import json

import httpx
import pytest

from gateway.agent import NatsAgent, _extract_reply
from gateway.clients import _extract_transcript


class TestSubjectGrammar:
    def test_subject_is_from_dot_dm_dot_to(self):
        agent = NatsAgent(servers="nats://x:4222", self_name="voice-gateway", peer="claude-code-channel")
        assert agent.subject == "agents.voice-gateway.dm.claude-code-channel.voice"

    def test_our_own_name_comes_first(self):
        """The ACL allows publishing only under agents.<self>.>.

        If these two ever swap, every utterance is silently dropped by the
        broker and nothing raises.
        """
        agent = NatsAgent(servers="nats://x:4222", self_name="me", peer="you")
        assert agent.subject.startswith("agents.me.")
        assert not agent.subject.startswith("agents.you.")

    def test_topic_is_configurable(self):
        agent = NatsAgent(servers="nats://x:4222", self_name="me", peer="you", topic="kitchen")
        assert agent.subject == "agents.me.dm.you.kitchen"


class TestReplyParsing:
    def test_standard_envelope(self):
        raw = json.dumps({"text": "It is sunny.", "from": "peer", "to": "me"}).encode()
        assert _extract_reply(raw) == "It is sunny."

    def test_bare_text(self):
        assert _extract_reply(b"It is sunny.") == "It is sunny."

    @pytest.mark.parametrize("key", ["text", "reply", "message", "body"])
    def test_alternative_field_names(self, key):
        """Peers differ. Accepting several is cheaper than changing every agent."""
        assert _extract_reply(json.dumps({key: "hello"}).encode()) == "hello"

    def test_malformed_json_falls_back_to_raw_text(self):
        assert _extract_reply(b'{"text": broken') == '{"text": broken'

    def test_json_without_any_text_field_is_empty_not_json(self):
        """Returning the raw JSON here would read the envelope ALOUD."""
        assert _extract_reply(json.dumps({"status": "ok", "code": 200}).encode()) == ""

    def test_json_with_blank_text_is_empty(self):
        assert _extract_reply(json.dumps({"text": "   "}).encode()) == ""

    def test_non_utf8_is_empty_rather_than_an_exception(self):
        assert _extract_reply(b"\xff\xfe\x00\x01") == ""

    def test_json_array_is_not_mistaken_for_an_envelope(self):
        assert _extract_reply(b'["a", "b"]') == '["a", "b"]'


class TestTranscriptParsing:
    def _response(self, *, content_type: str, body) -> httpx.Response:
        if isinstance(body, (dict, list)):
            return httpx.Response(200, json=body, headers={"content-type": content_type})
        return httpx.Response(200, text=body, headers={"content-type": content_type})

    def test_json_response(self):
        assert _extract_transcript(self._response(content_type="application/json", body={"text": "hi"})) == "hi"

    def test_plain_text_response(self):
        assert _extract_transcript(self._response(content_type="text/plain", body="hi")) == "hi"

    def test_json_without_text_field_is_empty(self):
        """Better an empty transcript, handled as no-speech, than a stray dict."""
        assert _extract_transcript(self._response(content_type="application/json", body={"oops": 1})) == ""

    def test_json_with_non_string_text_is_empty(self):
        assert _extract_transcript(self._response(content_type="application/json", body={"text": 42})) == ""

    def test_utf8_transcript_survives(self):
        body = {"text": "Está tudo bem?"}
        assert _extract_transcript(self._response(content_type="application/json", body=body)) == "Está tudo bem?"


class TestAgentCallIsBounded:
    """A hang is the one outcome the gateway must never produce.

    Bounding only the request leaves the CONNECT unbounded, and a client
    reconnecting to a dead server can sit there far longer than the configured
    agent timeout while the caller hears nothing at all.
    """

    @pytest.mark.asyncio
    async def test_a_hanging_connect_still_times_out(self, monkeypatch):
        import asyncio

        agent = NatsAgent(servers="nats://nowhere:4222", self_name="me", peer="you")

        async def never_returns() -> None:
            await asyncio.sleep(3600)

        monkeypatch.setattr(agent, "connect", never_returns)

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(agent.ask("hello", timeout_s=0.05), timeout=2.0)

    @pytest.mark.asyncio
    async def test_timeout_names_the_peer_and_the_budget(self):
        import asyncio

        agent = NatsAgent(servers="nats://nowhere:4222", self_name="me", peer="channel")

        async def never_returns() -> None:
            await asyncio.sleep(3600)

        agent.connect = never_returns  # type: ignore[method-assign]

        with pytest.raises(TimeoutError) as excinfo:
            await agent.ask("hello", timeout_s=0.05)

        message = str(excinfo.value)
        assert "channel" in message
        assert "0.05" in message
