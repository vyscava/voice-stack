"""The call page, call tokens, health and metrics."""

from __future__ import annotations

import time

import jwt as pyjwt
import pytest
from fastapi.testclient import TestClient

from core.settings import Settings
from voice_agent import web


@pytest.fixture()
def settings(monkeypatch):
    s = Settings(
        LIVEKIT_API_KEY="testkey",
        LIVEKIT_API_SECRET="x" * 40,  # a dummy, deliberately low-entropy so secret scanners pass it
        AGENT_PUBLIC_LIVEKIT_URL="wss://media.example.test",
        AGENT_TOKEN_TTL_MIN=30,
    )
    monkeypatch.setattr(web, "get_settings", lambda: s)
    return s


def test_token_is_scoped_to_one_new_room_and_expires(settings):
    client = TestClient(web.app)
    body = client.get("/token").json()
    claims = pyjwt.decode(body["token"], settings.LIVEKIT_API_SECRET, algorithms=["HS256"])
    assert claims["video"]["room"] == body["room"]
    assert claims["video"]["roomJoin"] is True
    assert not claims["video"].get("roomAdmin")
    assert body["room"].startswith("call-")
    assert claims["exp"] - time.time() <= 30 * 60 + 5


def test_each_token_gets_a_new_room(settings):
    client = TestClient(web.app)
    assert client.get("/token").json()["room"] != client.get("/token").json()["room"]


def test_token_without_keys_is_a_clear_503(monkeypatch):
    monkeypatch.setattr(web, "get_settings", lambda: Settings(LIVEKIT_API_KEY="", LIVEKIT_API_SECRET=""))
    r = TestClient(web.app).get("/token")
    assert r.status_code == 503
    assert "LIVEKIT_API_KEY" in r.json()["error"]


def test_page_uses_the_configured_livekit_url(settings):
    html = TestClient(web.app).get("/").text
    assert 'const CONFIGURED_URL = "wss://media.example.test";' in html
    assert "__LIVEKIT_URL__" not in html


def test_health_does_not_depend_on_livekit():
    r = TestClient(web.app).get("/health")
    assert r.status_code == 200 and r.text == "ok"


def test_metrics_expose_the_stage_histogram():
    from voice_agent import metrics

    metrics.STAGE_SECONDS.labels(stage="stt").observe(1.0)
    assert "voice_agent_stage_seconds" in TestClient(web.app).get("/metrics").text


def test_page_escapes_the_url_so_it_cannot_close_the_script(monkeypatch):
    evil = 'wss://x"</script><script>alert(1)</script>'
    monkeypatch.setattr(web, "get_settings", lambda: Settings(AGENT_PUBLIC_LIVEKIT_URL=evil))
    html = TestClient(web.app).get("/").text
    assert "<script>alert(1)" not in html
    assert "const CONFIGURED_URL = " in html


def test_room_names_are_long_enough_not_to_collide():
    assert len(web.new_room_name()) == len("call-") + 16
