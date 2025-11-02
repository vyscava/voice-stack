"""Smoke tests for TTS integration testing infrastructure."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
@pytest.mark.tts
def test_tts_app_starts_with_lifespan(tts_integration_client: TestClient) -> None:
    """Test that TTS app initializes correctly with real lifespan."""
    assert tts_integration_client.app is not None
    assert tts_integration_client.app.title is not None


@pytest.mark.integration
@pytest.mark.tts
def test_tts_routes_are_registered(tts_integration_client: TestClient) -> None:
    """Test that TTS API routes are registered during lifespan startup."""
    routes = [route.path for route in tts_integration_client.app.routes]

    # Check for OpenAI-compatible routes (double /v1 prefix)
    assert any("/v1/v1/models" in route for route in routes)
    assert any("/v1/v1/audio/speech" in route for route in routes)


@pytest.mark.integration
@pytest.mark.tts
def test_tts_health_endpoint_works(tts_integration_client: TestClient) -> None:
    """Test TTS health endpoint returns ok."""
    response = tts_integration_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.integration
@pytest.mark.tts
def test_tts_openapi_schema_includes_all_routes(tts_integration_client: TestClient) -> None:
    """Test that OpenAPI schema is generated with all routes."""
    response = tts_integration_client.get(tts_integration_client.app.openapi_url)
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "paths" in schema

    # Verify key routes are in the schema
    # Note: Routes have double /v1 prefix (API_V1_STR + router prefix)
    assert "/v1/v1/models" in schema["paths"]
    assert "/v1/v1/audio/speech" in schema["paths"]
