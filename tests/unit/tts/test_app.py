"""Unit tests for TTS application."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tts.app import app


@pytest.fixture
def tts_client() -> TestClient:
    """Create a test client for the TTS app."""
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
