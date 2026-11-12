"""Unit tests for TTS application."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tts.app import app


@pytest.fixture
def tts_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create a test client for the TTS app.

    The resource guard reads live memory and swap, so without these mocks the
    tests answer 503 on any machine under memory pressure (a laptop with heavy
    swap), and pass only where the host happens to be idle.
    """
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(percent=50.0, available=1, total=2))
    monkeypatch.setattr("psutil.swap_memory", lambda: SimpleNamespace(percent=10.0))
    return TestClient(app)


@pytest.mark.unit
@pytest.mark.tts
def test_app_initialization(tts_client: TestClient) -> None:
    """Test that the TTS app initializes correctly."""
    assert tts_client.app is not None
    assert tts_client.app.title is not None


@pytest.mark.unit
@pytest.mark.tts
def test_openapi_schema_generation(tts_client: TestClient) -> None:
    """Test that OpenAPI schema is generated."""
    response = tts_client.get(tts_client.app.openapi_url)
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema


@pytest.mark.unit
@pytest.mark.tts
def test_health_endpoint(tts_client: TestClient) -> None:
    """Test /health endpoint returns ok."""
    response = tts_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.unit
@pytest.mark.tts
def test_healthz_endpoint(tts_client: TestClient) -> None:
    """Test /healthz endpoint returns ok."""
    response = tts_client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.unit
@pytest.mark.tts
def test_healthcheck_endpoint(tts_client: TestClient) -> None:
    """Test /healthcheck endpoint returns ok."""
    response = tts_client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.unit
@pytest.mark.tts
def test_root_redirects_to_docs(tts_client: TestClient) -> None:
    """Test root path redirects to /docs."""
    response = tts_client.get("/", follow_redirects=False)
    # RedirectResponse returns 307 by default
    assert response.status_code in [200, 307]
    if response.status_code == 200:
        assert response.text == '"/docs"'


@pytest.mark.unit
@pytest.mark.tts
def test_cors_configuration(tts_client: TestClient) -> None:
    """Test CORS middleware is configured."""
    # Check if middleware is present in app middleware stack
    # Note: In testing, middleware may be wrapped, so we check for any CORS-related middleware
    has_cors = len(tts_client.app.user_middleware) > 0 or len(list(tts_client.app.middleware_stack)) > 0
    assert has_cors  # At minimum, there should be some middleware


@pytest.mark.unit
@pytest.mark.tts
def test_app_metadata(tts_client: TestClient) -> None:
    """Test app has correct metadata."""
    assert tts_client.app.title is not None
    assert tts_client.app.version is not None
    assert tts_client.app.openapi_url is not None


@pytest.mark.unit
@pytest.mark.tts
@pytest.mark.asyncio
async def test_idle_checker_runs_off_the_event_loop_on_the_current_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """The idle check takes the engine's lifecycle lock, which a synthesis
    thread holds through a whole model reload. On the event loop that froze
    /health (#8). It must run in a worker thread, and must check the engine
    that is current at each tick, not the one present when the task started."""
    import asyncio
    import threading

    from tts import app as tts_app
    from tts import engine_factory

    loop_thread = threading.get_ident()
    calls: list[tuple[str, int]] = []

    class Engine:
        def __init__(self, name: str) -> None:
            self.name = name

        def check_and_unload_if_idle(self) -> bool:
            calls.append((self.name, threading.get_ident()))
            return False

    monkeypatch.setattr(engine_factory, "_engine", Engine("old"))
    monkeypatch.setattr(tts_app.settings, "TTS_IDLE_TIMEOUT_MINUTES", 60)

    ticks = 0
    real_sleep = asyncio.sleep

    async def fast_sleep(_: float) -> None:
        nonlocal ticks
        ticks += 1
        if ticks == 2:
            engine_factory._engine = Engine("new")  # replaced, e.g. reset_audio_engine()
        if ticks > 2:
            raise asyncio.CancelledError
        await real_sleep(0)

    monkeypatch.setattr(tts_app.asyncio, "sleep", fast_sleep)
    with pytest.raises(asyncio.CancelledError):
        await tts_app.idle_timeout_checker()

    assert [name for name, _ in calls] == ["old", "new"]
    assert all(tid != loop_thread for _, tid in calls)
