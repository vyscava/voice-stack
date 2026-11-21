"""HTTP contract tests for outbound calls."""

from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from core.settings import Settings
from voice_agent import web


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_agent import calls

    monkeypatch.setattr(calls, "REGISTRY", calls.CallRegistry())


def _call_settings(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "LIVEKIT_API_KEY": "test-key",
        "LIVEKIT_API_SECRET": "s" * 40,
        "AGENT_CALL_TOKEN": "x" * 40,
        "AGENT_CALL_TO": "callee",
        "AGENT_CALL_TIMEZONE": "UTC",
    }
    values.update(overrides)
    configured = Settings(**values)
    monkeypatch.setattr(web, "get_settings", lambda: configured)
    monkeypatch.setattr(web, "_now", lambda: datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc))
    return configured


def _headers(settings: Settings) -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.AGENT_CALL_TOKEN}"}


def test_calls_disabled_without_token_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web, "get_settings", lambda: Settings(AGENT_CALL_TO="callee"))
    response = TestClient(web.app).post("/v1/calls", json={"reason": "r", "requested_by": "agent-a"})
    assert response.status_code == 404


def test_calls_wrong_token_is_401(monkeypatch: pytest.MonkeyPatch) -> None:
    _call_settings(monkeypatch)
    response = TestClient(web.app).post(
        "/v1/calls", headers={"Authorization": "Bearer wrong"}, json={"reason": "r", "requested_by": "agent-a"}
    )
    assert response.status_code == 401


def test_calls_missing_header_is_401(monkeypatch: pytest.MonkeyPatch) -> None:
    _call_settings(monkeypatch)
    assert TestClient(web.app).post("/v1/calls", json={"reason": "r", "requested_by": "agent-a"}).status_code == 401


def test_calls_invalid_body_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    configured = _call_settings(monkeypatch)
    client = TestClient(web.app)
    assert client.post("/v1/calls", headers=_headers(configured), json={"reason": ""}).status_code == 422
    assert client.post("/v1/calls", headers=_headers(configured), json=[]).status_code == 422


def test_call_accepted_creates_room_with_reason_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    configured = _call_settings(monkeypatch)
    calls_seen: list[tuple[Settings, Any]] = []

    async def fake_create(settings: Settings, record: Any) -> None:
        calls_seen.append((settings, record))

    monkeypatch.setattr(web, "create_outbound_room", fake_create)
    response = TestClient(web.app).post(
        "/v1/calls",
        headers=_headers(configured),
        json={"reason": "the backup finished", "requested_by": "agent-a"},
    )
    assert response.status_code == 202
    assert len(calls_seen) == 1
    assert calls_seen[0][1].reason == "the backup finished"


def test_second_call_is_409_not_429(monkeypatch: pytest.MonkeyPatch) -> None:
    configured = _call_settings(monkeypatch)

    async def fake_create(settings: Settings, record: Any) -> None:
        return None

    monkeypatch.setattr(web, "create_outbound_room", fake_create)
    client = TestClient(web.app)
    assert (
        client.post(
            "/v1/calls", headers=_headers(configured), json={"reason": "r", "requested_by": "agent-a"}
        ).status_code
        == 202
    )
    assert (
        client.post(
            "/v1/calls", headers=_headers(configured), json={"reason": "r", "requested_by": "agent-a"}
        ).status_code
        == 409
    )


def test_too_soon_after_a_finished_call_is_429(monkeypatch: pytest.MonkeyPatch) -> None:
    configured = _call_settings(monkeypatch)

    async def fake_create(settings: Settings, record: Any) -> None:
        return None

    monkeypatch.setattr(web, "create_outbound_room", fake_create)
    client = TestClient(web.app)
    first = client.post("/v1/calls", headers=_headers(configured), json={"reason": "r", "requested_by": "agent-a"})
    call_id = first.json()["call_id"]
    web.calls.REGISTRY.update(call_id, "no_answer", now=web._now())
    response = client.post("/v1/calls", headers=_headers(configured), json={"reason": "r", "requested_by": "agent-a"})
    assert response.status_code == 429
    assert response.json()["detail"] == "too_soon"


def test_quiet_hours_is_429_and_creates_no_room(monkeypatch: pytest.MonkeyPatch) -> None:
    configured = _call_settings(monkeypatch, AGENT_CALL_QUIET_HOURS="11:00-13:00")
    response = TestClient(web.app).post(
        "/v1/calls", headers=_headers(configured), json={"reason": "r", "requested_by": "agent-a"}
    )
    assert response.status_code == 429
    assert response.json()["detail"] == "quiet_hours"


def test_status_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    configured = _call_settings(monkeypatch)

    async def fake_create(settings: Settings, record: Any) -> None:
        return None

    monkeypatch.setattr(web, "create_outbound_room", fake_create)
    client = TestClient(web.app)
    created = client.post(
        "/v1/calls", headers=_headers(configured), json={"reason": "r", "requested_by": "agent-a"}
    ).json()
    response = client.get(f"/v1/calls/{created['call_id']}", headers=_headers(configured))
    assert response.status_code == 200
    assert response.json()["status"] == "dialing"
    assert response.json()["started_at"].endswith("+00:00")


def test_room_create_failure_is_503_and_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    configured = _call_settings(monkeypatch)

    async def fail(settings: Settings, record: Any) -> None:
        raise RuntimeError("down")

    monkeypatch.setattr(web, "create_outbound_room", fail)
    client = TestClient(web.app)
    response = client.post("/v1/calls", headers=_headers(configured), json={"reason": "r", "requested_by": "agent-a"})
    assert response.status_code == 503
    assert response.json()["error"] == "room_create_failed"
    call_id = web.calls.REGISTRY.attempt_times()
    assert len(call_id) == 1


def test_bad_quiet_hours_config_is_503_and_creates_no_room(monkeypatch: pytest.MonkeyPatch) -> None:
    configured = _call_settings(monkeypatch, AGENT_CALL_QUIET_HOURS="22-08")
    response = TestClient(web.app).post(
        "/v1/calls", headers=_headers(configured), json={"reason": "r", "requested_by": "agent-a"}
    )
    assert response.status_code == 503


def test_outbound_metric_counts_each_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    configured = _call_settings(monkeypatch)
    before = web.metrics.OUTBOUND_CALLS.labels("dialing")._value.get()

    async def fake_create(settings: Settings, record: Any) -> None:
        return None

    monkeypatch.setattr(web, "create_outbound_room", fake_create)
    response = TestClient(web.app).post(
        "/v1/calls", headers=_headers(configured), json={"reason": "r", "requested_by": "agent-a"}
    )
    assert response.status_code == 202
    assert web.metrics.OUTBOUND_CALLS.labels("dialing")._value.get() == before + 1
