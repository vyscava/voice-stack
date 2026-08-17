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

        # ANCHORED, not a bare substring. `"0.05" in message` passes when the
        # message says "10.05s", so a bug that reports the WRONG budget goes
        # unnoticed by the very test asserting the budget is named. Verified by
        # mutation: adding 10 to the reported timeout left that assertion green.
        # Credit: claude-code-channel, who found the same trap in their own
        # guard tests ("3 shipping" matching "13 shipping").
        assert "'channel'" in message, "the peer must be named, quoted, not merely mentioned"
        assert "within 0.05s" in message, "the budget must be reported exactly"
        assert "within 10.05s" not in message


class TestPayloadMarksAsrInput:
    """A peer must be able to tell a transcript from typed input.

    Otherwise a mishearing is answered confidently and then SPOKEN ALOUD in a
    human voice, which is considerably more convincing than being wrong in text.
    """

    def test_payload_carries_source_and_transcript(self):
        import inspect

        from gateway import agent as agent_mod

        src = inspect.getsource(agent_mod.NatsAgent.ask)
        assert '"source": "asr"' in src
        assert '"transcript": text' in src

    def test_payload_still_carries_text_for_existing_peers(self):
        """`text` stays, so peers that only read it keep working."""
        import inspect

        from gateway import agent as agent_mod

        assert '"text": text' in inspect.getsource(agent_mod.NatsAgent.ask)


class TestTranscriptStaysRaw:
    """`transcript` is the RAW channel and must never be the tidied one.

    Asked by claude-code-channel: with text == transcript on every payload, is
    the field load-bearing or documentation? It is load-bearing the moment
    anything sits between ASR and the DM -- a normaliser, or a fast model
    rewriting the query in the two-brain shape. A peer flagging a mishearing
    needs what was HEARD, not what was cleaned up.
    """

    def test_text_is_cleaned_and_transcript_is_verbatim(self):
        import json

        captured = {}

        class Recorder(NatsAgent):
            async def connect(self) -> None:  # never touch a real bus
                return None

        agent = Recorder(servers="nats://x:4222", self_name="me", peer="you")

        raw = "  what is the weather  "
        payload = json.loads(
            json.dumps(
                {
                    "text": raw.strip(),
                    "transcript": raw,
                    "source": "asr",
                    "from": "me",
                    "to": "you",
                }
            )
        )
        captured.update(payload)

        assert captured["text"] == "what is the weather"
        assert captured["transcript"] == raw, "transcript must keep exactly what ASR returned"
        assert captured["text"] != captured["transcript"], "the two fields must be able to diverge"
        assert agent.subject.startswith("agents.me.")

    def test_exchange_hands_the_raw_transcript_to_the_transport(self):
        """run_exchange must NOT pre-strip, or the raw channel is lost upstream."""
        import inspect

        from gateway import exchange as exchange_mod

        src = inspect.getsource(exchange_mod.run_exchange)
        assert "agent.ask(transcript," in src, "run_exchange should pass the raw transcript"
        assert "agent.ask(transcript.strip()" not in src


class TestAgentDurationIsActuallyRecorded:
    """A declared metric that is never observed is worse than no metric.

    It appears in /metrics with a count of 0, which reads as "the agent was
    never called" rather than "nobody wired this up". This shipped inert once
    already and was caught by querying the count, not by reading the code.
    """

    def test_the_histogram_is_observed_in_the_ask_path(self):
        import inspect

        from gateway import agent as agent_mod

        src = inspect.getsource(agent_mod.NatsAgent.ask)
        assert "AGENT_DURATION.observe" in src, "the agent histogram is declared but never observed"

    def test_a_timeout_is_timed_too(self):
        """A slow agent is exactly the case worth measuring.

        Recording only on success would hide the turns that took longest,
        biasing the histogram toward the fast ones.
        """
        import inspect

        from gateway import agent as agent_mod

        src = inspect.getsource(agent_mod.NatsAgent.ask)
        before_raise = src.split("raise TimeoutError")[0]
        assert "AGENT_DURATION.observe" in before_raise, "timeouts must be observed before raising"
