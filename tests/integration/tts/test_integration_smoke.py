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
    """Test that TTS API routes are registered during lifespan startup.

    Probes each expected route and asserts it is reachable (not 404). A method
    mismatch returns 405, which still confirms the path is registered. This is
    robust to FastAPI's internal route representation: since 0.139, included
    routers are attached lazily as ``_IncludedRouter`` objects (which have no
    ``.path``) and are not flattened into ``app.routes``, so iterating
    ``route.path`` over ``app.routes`` no longer surfaces included routes.
    """
    for path in ("/v1/models", "/v1/audio/speech"):
        status_code = tts_integration_client.get(path).status_code
        assert status_code != 404, f"expected route {path} to be registered, got 404"


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
    assert "/v1/models" in schema["paths"]
    assert "/v1/audio/speech" in schema["paths"]
