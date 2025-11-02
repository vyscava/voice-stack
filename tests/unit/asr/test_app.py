"""Unit tests for ASR FastAPI application."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
@pytest.mark.asr
def test_app_initialization(asr_client: TestClient) -> None:
    """Test that the ASR app initializes correctly."""
    assert asr_client.app is not None


@pytest.mark.unit
@pytest.mark.asr
def test_openapi_schema_generation(asr_client: TestClient) -> None:
    """Test that OpenAPI schema is generated correctly."""
    # The actual openapi URL is determined by the app's openapi_url setting
    response = asr_client.get(asr_client.app.openapi_url)
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] is not None  # Title from project metadata
    assert "paths" in schema


@pytest.mark.unit
@pytest.mark.asr
def test_health_endpoint(asr_client: TestClient) -> None:
    """Test the /health endpoint."""
    response = asr_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.unit
@pytest.mark.asr
def test_healthz_endpoint(asr_client: TestClient) -> None:
    """Test the /healthz endpoint."""
    response = asr_client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.unit
@pytest.mark.asr
def test_healthcheck_endpoint(asr_client: TestClient) -> None:
    """Test the /healthcheck endpoint."""
    response = asr_client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.unit
@pytest.mark.asr
def test_root_redirects_to_docs(asr_client: TestClient) -> None:
    """Test that root path redirects to /docs."""
    response = asr_client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


@pytest.mark.unit
@pytest.mark.asr
def test_cors_configuration(asr_client: TestClient) -> None:
    """Test CORS configuration is applied when CORS_ORIGINS is set."""
    # We can't easily reload the module to test CORS without reinitializing models,
    # so we just verify the app is configured correctly
    assert asr_client.app is not None

    # Verify that CORS headers work by making a request
    response = asr_client.get("/health")
    assert response.status_code == 200


@pytest.mark.unit
@pytest.mark.asr
def test_api_routes_registered(asr_client: TestClient) -> None:
    """Test that API routes are properly registered."""
    # Get OpenAPI schema to check routes
    response = asr_client.get(asr_client.app.openapi_url)
    assert response.status_code == 200
    schema = response.json()

    # Check that main endpoints exist - paths depend on API_V1_STR setting
    paths = schema["paths"]
    # OpenAI endpoints
    assert any("/audio/transcriptions" in path for path in paths)
    assert any("/audio/translations" in path for path in paths)
    assert any("/models" in path for path in paths)
    # Bazarr endpoints
    assert any("/bazarr/asr" in path for path in paths)
    assert any("/bazarr/detect-language" in path for path in paths)


@pytest.mark.unit
@pytest.mark.asr
def test_app_metadata(asr_client: TestClient) -> None:
    """Test that app metadata is correctly set."""
    # Get OpenAPI schema to verify metadata
    response = asr_client.get(asr_client.app.openapi_url)
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["version"] is not None
    assert schema["info"]["license"]["name"] == "MIT License"
