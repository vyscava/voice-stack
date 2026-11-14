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
        agent = NatsAgent(servers="nats://x:4222", self_name="voice-gateway", peer="assistant")
        assert agent.subject == "agents.voice-gateway.dm.assistant.voice"

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
        # Credit: a peer agent, who found the same trap in their own
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

    Asked by a peer agent: with text == transcript on every payload, is
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


class TestNatsTls:
    """messaging#7: under `allow_non_tls` nats-py with a tls:// URL stays plaintext
    unless the client asks for a handshake-first connection."""

    def test_tls_url_asks_for_handshake_first(self):
        from gateway.agent import nats_tls_kwargs

        assert nats_tls_kwargs("tls://u:p@nats.example.com:4222") == {"tls_handshake_first": True}
        assert nats_tls_kwargs("TLS://nats.example.com:4222") == {"tls_handshake_first": True}

    def test_nats_url_keeps_todays_behaviour(self):
        from gateway.agent import nats_tls_kwargs

        assert nats_tls_kwargs("nats://u:p@nats.example.com:4222") == {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("url", "expect_handshake_first"),
        [
            ("tls://voice-gateway:pw@nats.example.com:4222", True),
            ("nats://voice-gateway:pw@nats.example.com:4222", False),
        ],
    )
    async def test_connect_passes_the_option_through(self, monkeypatch, url, expect_handshake_first):
        import nats

        seen: dict = {}

        class FakeConn:
            is_connected = True

        async def fake_connect(servers, **kwargs):
            seen["servers"] = servers
            seen["kwargs"] = kwargs
            return FakeConn()

        monkeypatch.setattr(nats, "connect", fake_connect)
        await NatsAgent(servers=url, self_name="voice-gateway", peer="you").connect()

        assert seen["servers"] == url  # credentials stay in the URL, untouched
        assert ("tls_handshake_first" in seen["kwargs"]) is expect_handshake_first
        assert seen["kwargs"].get("tls_handshake_first", False) is expect_handshake_first

    @pytest.mark.parametrize("scheme", ["tls", "nats"])
    def test_nats_py_still_reads_url_credentials_for_either_scheme(self, scheme):
        """Pins the real library: userinfo is parsed for tls:// as for nats://,
        so the flip needs no separate user/password options."""
        from nats.aio.client import Client

        nc = Client()
        nc._setup_server_pool(f"{scheme}://voice-gateway:pw@nats.example.com:4222")
        uri = nc._server_pool[0].uri
        assert uri.scheme == scheme
        assert (uri.username, uri.password) == ("voice-gateway", "pw")
        assert uri.hostname == "nats.example.com" and uri.port == 4222


SECRET = "s3ntinel-PW-do-not-leak"


class TestRedactUrl:
    """The gateway URL embeds the NATS password; it must never reach a log or a response."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (
                f"nats://voice-gateway:{SECRET}@nats.example.com:4222",
                "nats://voice-gateway:***@nats.example.com:4222",
            ),
            (
                f"tls://voice-gateway:{SECRET}@nats.example.com:4222",
                "tls://voice-gateway:***@nats.example.com:4222",
            ),
            ("nats://nats.example.com:4222", "nats://nats.example.com:4222"),
            ("nats.example.com", "nats.example.com"),
            (f"voice-gateway:{SECRET}@nats-a:4222", "voice-gateway:***@nats-a:4222"),
            (f"nats://u:{SECRET}@a:4222, nats://u:{SECRET}@b:4222", "nats://u:***@a:4222,nats://u:***@b:4222"),
            (f"nats://u:{SECRET}@a:4222,nats://b:4222", "nats://u:***@a:4222,nats://b:4222"),
            # user-only: nats-py treats it as an auth TOKEN, so it is masked too
            (f"nats://{SECRET}@a:4222", "nats://***@a:4222"),
            # password containing @ and :
            (f"nats://u:{SECRET}@x:y@a:4222", "nats://u:***@a:4222"),
        ],
    )
    def test_cases(self, raw, expected):
        from gateway.agent import redact_url

        out = redact_url(raw)
        assert out == expected
        assert SECRET not in out and "x:y" not in out

    @pytest.mark.parametrize(
        "raw",
        [f"nats://u:{SECRET}@a:notaport", f"nats://u:{SECRET}@", f"nats://u:{SECRET}@[::1", ""],
    )
    def test_malformed_fails_safe(self, raw):
        from gateway.agent import redact_url

        out = redact_url(raw)
        assert SECRET not in out
        assert out == "***"

    @pytest.mark.asyncio
    async def test_connect_log_line_has_no_password(self, monkeypatch):
        import nats

        import gateway.agent as agent_mod

        lines: list[str] = []

        class Rec:
            def info(self, msg, *a, **k):
                lines.append(str(msg))

            debug = info

        class FakeConn:
            is_connected = True

        async def fake_connect(servers, **kwargs):
            return FakeConn()

        monkeypatch.setattr(nats, "connect", fake_connect)
        monkeypatch.setattr(agent_mod, "logger", Rec())
        url = f"tls://voice-gateway:{SECRET}@nats.example.com:4222"
        await NatsAgent(servers=url, self_name="voice-gateway", peer="you").connect()

        assert any("NATS connected" in line for line in lines)
        assert not any(SECRET in line for line in lines)
        assert any("voice-gateway:***@nats.example.com:4222" in line for line in lines)


class TestPerTurnMeta:
    """`meta` adds per-turn context for a live call (call id, turn number).

    It may add fields but never replace what was heard or where it came from.
    """

    @staticmethod
    async def _sent_payload(monkeypatch, **ask_kwargs) -> dict:
        import nats

        sent: dict = {}

        class Reply:
            data = b'{"text": "ok"}'

        class FakeConn:
            is_connected = True

            async def request(self, subject, payload, timeout):
                sent["subject"] = subject
                sent["body"] = json.loads(payload)
                return Reply()

        async def fake_connect(servers, **kwargs):
            return FakeConn()

        monkeypatch.setattr(nats, "connect", fake_connect)
        agent = NatsAgent(servers="nats://localhost:4222", self_name="voice-agent", peer="brain")
        assert await agent.ask(" hello there ", timeout_s=1, **ask_kwargs) == "ok"
        return sent["body"]

    @pytest.mark.asyncio
    async def test_meta_is_merged_into_the_payload(self, monkeypatch):
        body = await self._sent_payload(monkeypatch, meta={"call_id": "call-abc", "turn_id": 3, "language": "pt"})
        assert body["call_id"] == "call-abc"
        assert body["turn_id"] == 3
        assert body["language"] == "pt"
        assert body["text"] == "hello there"

    @pytest.mark.asyncio
    async def test_meta_cannot_override_what_was_heard(self, monkeypatch):
        body = await self._sent_payload(
            monkeypatch, meta={"text": "forged", "transcript": "forged", "source": "typed", "from": "x", "to": "y"}
        )
        assert body["text"] == "hello there"
        assert body["transcript"] == " hello there "
        assert body["source"] == "asr"
        assert (body["from"], body["to"]) == ("voice-agent", "brain")

    @pytest.mark.asyncio
    async def test_without_meta_the_payload_is_unchanged(self, monkeypatch):
        body = await self._sent_payload(monkeypatch)
        assert body == {
            "text": "hello there",
            "transcript": " hello there ",
            "source": "asr",
            "from": "voice-agent",
            "to": "brain",
        }


class TestNatsInboxPrefix:
    """A server may narrow each user's subscribe grant from
    ``_INBOX.>`` to ``_INBOX.<user>.>``. A reply inbox outside that prefix would
    never get its reply, so every request would time out."""

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("nats://voice-gateway:pw@nats.example.com:4222", {"inbox_prefix": "_INBOX.voice-gateway"}),
            ("tls://voice-agent:pw@nats.example.com:4222", {"inbox_prefix": "_INBOX.voice-agent"}),
            ("voice-gateway:pw@nats-a:4222", {"inbox_prefix": "_INBOX.voice-gateway"}),
            ("nats://u:p@x:y@a:4222", {"inbox_prefix": "_INBOX.u"}),
            ("nats://u1:pw@a:4222,nats://u2:pw@b:4222", {"inbox_prefix": "_INBOX.u1"}),
            ("nats://nats.example.com:4222", {}),
            ("", {}),
        ],
    )
    def test_prefix_comes_from_the_url_user(self, url, expected):
        from gateway.agent import nats_inbox_kwargs

        assert nats_inbox_kwargs(url) == expected

    def test_a_lone_userinfo_is_a_token_and_never_reaches_a_subject(self):
        from gateway.agent import nats_inbox_kwargs

        assert nats_inbox_kwargs(f"nats://{SECRET}@a:4222") == {}

    def test_explicit_prefix_wins_and_loses_a_trailing_dot(self):
        from gateway.agent import nats_inbox_kwargs

        url = "nats://voice-gateway:pw@a:4222"
        assert nats_inbox_kwargs(url, "_INBOX.custom") == {"inbox_prefix": "_INBOX.custom"}
        assert nats_inbox_kwargs(url, "_INBOX.custom.") == {"inbox_prefix": "_INBOX.custom"}
        assert nats_inbox_kwargs("nats://a:4222", "_INBOX.custom") == {"inbox_prefix": "_INBOX.custom"}

    @pytest.mark.asyncio
    async def test_connect_uses_the_url_user_not_self_name(self, monkeypatch):
        import nats

        seen: dict = {}

        class FakeConn:
            is_connected = True

        async def fake_connect(servers, **kwargs):
            seen["kwargs"] = kwargs
            return FakeConn()

        monkeypatch.setattr(nats, "connect", fake_connect)
        url = "nats://bus-user:pw@nats.example.com:4222"
        await NatsAgent(servers=url, self_name="voice-gateway", peer="you").connect()
        assert seen["kwargs"]["inbox_prefix"] == "_INBOX.bus-user"

    @pytest.mark.asyncio
    async def test_connect_without_a_user_keeps_the_default(self, monkeypatch):
        import nats

        seen: dict = {}

        class FakeConn:
            is_connected = True

        async def fake_connect(servers, **kwargs):
            seen["kwargs"] = kwargs
            return FakeConn()

        monkeypatch.setattr(nats, "connect", fake_connect)
        await NatsAgent(servers="nats://nats:4222", self_name="voice-gateway", peer="you").connect()
        assert "inbox_prefix" not in seen["kwargs"]

    def test_nats_py_builds_the_inbox_under_the_prefix(self):
        """Pins the real library: it appends the "." itself."""
        from nats.aio.client import Client

        from gateway.agent import nats_inbox_kwargs

        nc = Client()
        prefix = nats_inbox_kwargs("nats://voice-gateway:pw@a:4222")["inbox_prefix"]
        nc._inbox_prefix = bytearray(prefix.encode())  # what connect() does with the option
        inbox = nc.new_inbox()
        assert inbox.startswith("_INBOX.voice-gateway.")
        assert ".." not in inbox
