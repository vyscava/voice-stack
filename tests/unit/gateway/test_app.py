"""The gateway's HTTP surface, exercised against a running app.

These tests start the application through TestClient, which runs the lifespan.
That matters here: routes are registered during startup, so a test that only
imports the module would report an app with no voice endpoints at all and still
pass.

The collaborators are replaced with fakes, so nothing here needs ASR, TTS, NATS
or a model.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gateway.api.api_v1.endpoints import voice as voice_endpoint
from gateway.exchange import ExchangePolicy

POLICY = ExchangePolicy(agent_timeout_s=0.05, min_transcript_chars=2)


class FakeTranscriber:
    def __init__(self, text="hello there", exc=None):
        self.text, self.exc = text, exc

    async def transcribe(self, audio: bytes, *, filename: str) -> str:
        if self.exc:
            raise self.exc
        return self.text


class FakeSpeaker:
    def __init__(self, exc=None):
        self.exc = exc

    async def speak(self, text: str) -> bytes:
        if self.exc:
            raise self.exc
        return b"OggS" + text.encode()


class FakeAgent:
    def __init__(self, reply="Hi.", exc=None):
        self.reply, self.exc = reply, exc

    async def ask(self, text: str, *, timeout_s: float) -> str:
        if self.exc:
            raise self.exc
        return self.reply


@pytest.fixture
def client(monkeypatch):
    """A running gateway whose collaborators are fakes, patchable per test."""
    state = {
        "transcriber": FakeTranscriber(),
        "speaker": FakeSpeaker(),
        "agent": FakeAgent(),
    }
    monkeypatch.setattr(voice_endpoint, "build_transcriber", lambda: state["transcriber"])
    monkeypatch.setattr(voice_endpoint, "build_speaker", lambda: state["speaker"])
    monkeypatch.setattr(voice_endpoint, "build_agent", lambda: state["agent"])
    monkeypatch.setattr(voice_endpoint, "build_policy", lambda: POLICY)

    from gateway.app import app

    with TestClient(app) as c:
        c.state = state  # type: ignore[attr-defined]
        yield c


class TestHealth:
    @pytest.mark.parametrize("path", ["/health", "/healthz", "/healthcheck"])
    def test_liveness_endpoints(self, client, path):
        response = client.get(path)
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_does_not_depend_on_asr_tts_or_nats(self, client):
        """Liveness must stay green while dependencies are down.

        A dependency outage is something the gateway handles by speaking an
        error. If /health went red for it, an orchestrator would kill and
        restart a process that is working exactly as designed.
        """
        client.state["transcriber"] = FakeTranscriber(exc=OSError("asr down"))
        client.state["speaker"] = FakeSpeaker(exc=OSError("tts down"))
        client.state["agent"] = FakeAgent(exc=OSError("nats down"))

        assert client.get("/health").status_code == 200

    def test_detailed_health_reports_what_it_talks_to(self, client):
        body = client.get("/health/detailed").json()
        assert "asr_url" in body["routes"]
        assert "agent_subject" in body["routes"]
        assert body["routes"]["agent_subject"].startswith("agents.")


class TestExchangeEndpoint:
    def test_route_exists_after_startup(self, client):
        """Routes are registered in the lifespan, so this proves startup ran."""
        # Read the OpenAPI schema rather than walking app.routes: FastAPI wraps
        # an included router in an object with no .path, and whether sub-routes
        # are flattened varies by version. The schema is what the service
        # actually advertises.
        paths = client.get("/openapi.json").json()["paths"]
        assert "/v1/voice/exchange" in paths
        assert "/v1/voice/transcribe-only" in paths

    def test_happy_path_returns_audio(self, client):
        client.state["agent"] = FakeAgent(reply="It is sunny.")
        response = client.post("/v1/voice/exchange", files={"file": ("u.wav", b"RIFFfake", "audio/wav")})

        assert response.status_code == 200
        assert response.content == b"OggSIt is sunny."
        assert response.headers["X-Voice-Outcome"] == "ok"
        assert response.headers["X-Voice-Degraded"] == "false"

    def test_failure_is_still_200_with_audible_speech(self, client):
        """The caller is a person holding a microphone.

        An HTTP error code is not something they can hear, so failures come
        back as speech. The outcome header is how a machine tells the
        difference.
        """
        client.state["agent"] = FakeAgent(exc=TimeoutError("slow"))
        response = client.post("/v1/voice/exchange", files={"file": ("u.wav", b"RIFFfake", "audio/wav")})

        assert response.status_code == 200
        assert response.content, "a failed exchange must still make a sound"
        assert response.headers["X-Voice-Outcome"] == "agent_timeout"
        assert response.headers["X-Voice-Degraded"] == "true"

    def test_tts_failure_degrades_to_json_not_a_lost_answer(self, client):
        client.state["speaker"] = FakeSpeaker(exc=RuntimeError("oom"))
        client.state["agent"] = FakeAgent(reply="The answer is 42.")
        response = client.post("/v1/voice/exchange", files={"file": ("u.wav", b"RIFFfake", "audio/wav")})

        assert response.status_code == 200
        body = response.json()
        assert body["outcome"] == "tts_failed"
        assert body["reply_text"] == "The answer is 42.", "the agent's answer must survive"
        assert body["audio"] is None

    def test_non_latin1_transcript_does_not_break_the_response(self, client):
        """A Portuguese transcript in a header would raise on encoding.

        Header values must be latin-1 encodable. This is the bug that would
        only appear in production, on exactly the language this household
        speaks.
        """
        client.state["transcriber"] = FakeTranscriber(text="Está tudo bem, coração?")
        response = client.post("/v1/voice/exchange", files={"file": ("u.wav", b"RIFFfake", "audio/wav")})

        assert response.status_code == 200
        assert response.headers["X-Voice-Outcome"] == "ok"


class TestTranscribeOnly:
    def test_returns_text_without_asking_the_agent(self, client):
        client.state["transcriber"] = FakeTranscriber(text="microphone check")
        client.state["agent"] = FakeAgent(exc=AssertionError("the agent must not be called"))

        response = client.post("/v1/voice/transcribe-only", files={"file": ("u.wav", b"RIFFfake", "audio/wav")})
        assert response.status_code == 200
        assert response.json()["text"] == "microphone check"

    def test_asr_failure_is_reported_not_swallowed(self, client):
        client.state["transcriber"] = FakeTranscriber(exc=RuntimeError("model gone"))
        response = client.post("/v1/voice/transcribe-only", files={"file": ("u.wav", b"RIFFfake", "audio/wav")})
        assert response.status_code == 502


class TestWebClient:
    """The push-to-talk page is the only surface a person can speak into."""

    def test_root_serves_the_page(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        body = res.text
        assert "Hold to speak" in body
        assert "v1/voice/exchange" in body, "the page must post to the exchange endpoint"

    def test_page_ships_with_the_package(self):
        """Guards against the static file being dropped from the image.

        The app falls back to /docs when the file is missing, which is right at
        runtime and useless as a signal -- a vanished UI would look like a
        working service. This asserts the file is actually there.
        """
        from pathlib import Path

        import gateway.app as app_module

        index = Path(app_module.__file__).parent / "static" / "index.html"
        assert index.is_file(), f"web client missing at {index}"
        assert index.stat().st_size > 1000

    def test_page_is_self_contained(self):
        """No CDN, no external fetches.

        The gateway is LAN-only by policy (!53). A page that pulls a script
        from the public internet would both add a dependency and create egress
        from a network that is meant not to have any.
        """
        from pathlib import Path

        import gateway.app as app_module

        html = (Path(app_module.__file__).parent / "static" / "index.html").read_text()
        for needle in ("http://", "https://", "//cdn", "integrity="):
            assert needle not in html, f"page references something external: {needle}"

    def test_page_warns_about_insecure_context(self):
        """A browser silently withholds getUserMedia on a plain-http origin.

        Without this the page loads, the button appears, and the microphone
        never works -- with an error that names none of that.
        """
        from pathlib import Path

        import gateway.app as app_module

        html = (Path(app_module.__file__).parent / "static" / "index.html").read_text()
        assert "isSecureContext" in html
        assert "HTTPS" in html


class TestMetrics:
    """Liveness alone cannot see a service that is UP and failing everything.

    That is the gap these close: before them, a gateway failing every exchange
    was indistinguishable from a healthy one, and /health answered 200 either
    way.
    """

    def test_metrics_endpoint_serves_prometheus_exposition(self, client):
        res = client.get("/metrics")
        assert res.status_code == 200
        assert "text/plain" in res.headers["content-type"]
        assert "voice_gateway_exchanges_total" in res.text

    def test_a_successful_exchange_increments_the_ok_outcome(self, client):
        client.post("/v1/voice/exchange", files={"file": ("u.wav", b"RIFFfake", "audio/wav")})
        body = client.get("/metrics").text
        assert 'voice_gateway_exchanges_total{outcome="ok"}' in body

    def test_a_failed_exchange_is_counted_under_its_own_outcome(self, client):
        """Outcomes are separate labels, not one 'errors' bucket.

        An agent timeout and a silence rejection have different causes and
        different fixes; collapsing them would hide which is happening.
        """
        client.state["agent"] = FakeAgent(exc=TimeoutError("slow"))
        client.post("/v1/voice/exchange", files={"file": ("u.wav", b"RIFFfake", "audio/wav")})
        body = client.get("/metrics").text
        assert 'voice_gateway_exchanges_total{outcome="agent_timeout"}' in body

    def test_duration_is_observed(self, client):
        client.post("/v1/voice/exchange", files={"file": ("u.wav", b"RIFFfake", "audio/wav")})
        body = client.get("/metrics").text
        assert "voice_gateway_exchange_duration_seconds_count" in body
