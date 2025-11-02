"""Smoke tests for ASR integration testing infrastructure."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
@pytest.mark.asr
def test_asr_app_starts_with_lifespan(asr_integration_client: TestClient) -> None:
    """Test that ASR app initializes correctly with real lifespan."""
    assert asr_integration_client.app is not None
    assert asr_integration_client.app.title is not None


@pytest.mark.integration
@pytest.mark.asr
def test_asr_routes_are_registered(asr_integration_client: TestClient) -> None:
    """Test that ASR API routes are registered during lifespan startup."""
    routes = [route.path for route in asr_integration_client.app.routes]

    # Check for OpenAI-compatible routes
    assert any("/v1/audio/transcriptions" in route for route in routes)
    assert any("/v1/audio/translations" in route for route in routes)
    assert any("/v1/models" in route for route in routes)


@pytest.mark.integration
@pytest.mark.asr
def test_asr_health_endpoint_works(asr_integration_client: TestClient) -> None:
    """Test ASR health endpoint returns ok."""
    response = asr_integration_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.integration
@pytest.mark.asr
def test_asr_openapi_schema_includes_all_routes(asr_integration_client: TestClient) -> None:
    """Test that OpenAPI schema is generated with all routes."""
    response = asr_integration_client.get(asr_integration_client.app.openapi_url)
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "paths" in schema

    # Verify key routes are in the schema
    assert "/v1/audio/transcriptions" in schema["paths"]
    assert "/v1/audio/translations" in schema["paths"]
    assert "/v1/models" in schema["paths"]
