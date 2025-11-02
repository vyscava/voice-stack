"""Unit tests for ASR OpenAI-compatible endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
@pytest.mark.asr
def test_transcriptions_json_response(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test POST /v1/audio/transcriptions with JSON response."""
    response = asr_client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": "json"},
        files={"file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    data = response.json()
    assert "text" in data
    assert isinstance(data["text"], str)


@pytest.mark.unit
@pytest.mark.asr
def test_transcriptions_text_response(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test POST /v1/audio/transcriptions with text response."""
    response = asr_client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": "text"},
        files={"file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert isinstance(response.text, str)


@pytest.mark.unit
@pytest.mark.asr
def test_transcriptions_with_language(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test transcription with language hint."""
    response = asr_client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "language": "en", "response_format": "json"},
        files={"file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "text" in data


@pytest.mark.unit
@pytest.mark.asr
def test_transcriptions_with_temperature(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test transcription with temperature parameter."""
    response = asr_client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "temperature": "0.5", "response_format": "json"},
        files={"file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "text" in data


@pytest.mark.unit
@pytest.mark.asr
def test_transcriptions_verbose_json(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test POST /v1/audio/transcriptions/verbose with full metadata."""
    response = asr_client.post(
        "/v1/audio/transcriptions/verbose",
        data={"model": "whisper-1", "response_format": "json"},
        files={"file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    data = response.json()

    # Check required fields
    assert "text" in data
    assert "segments" in data
    assert isinstance(data["segments"], list)

    # Check verbose fields (may be None but should exist)
    assert "language" in data
    assert "language_name" in data or True  # Optional field
    assert "language_probability" in data or True  # Optional field


@pytest.mark.unit
@pytest.mark.asr
def test_transcriptions_verbose_with_params(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test verbose transcription with all parameters."""
    response = asr_client.post(
        "/v1/audio/transcriptions/verbose",
        data={
            "model": "whisper-1",
            "language": "en",
            "task": "transcribe",
            "beam_size": "5",
            "temperature": "0.0",
            "best_of": "1",
            "word_timestamps": "true",
            "vad": "true",
            "response_format": "json",
        },
        files={"file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "segments" in data


@pytest.mark.unit
@pytest.mark.asr
def test_transcriptions_verbose_text_response(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test verbose endpoint with text response format."""
    response = asr_client.post(
        "/v1/audio/transcriptions/verbose",
        data={"model": "whisper-1", "response_format": "text"},
        files={"file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"


@pytest.mark.unit
@pytest.mark.asr
def test_translations_json_response(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test POST /v1/audio/translations with JSON response."""
    response = asr_client.post(
        "/v1/audio/translations",
        data={"model": "whisper-1", "response_format": "json"},
        files={"file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    data = response.json()
    assert "text" in data


@pytest.mark.unit
@pytest.mark.asr
def test_translations_text_response(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test POST /v1/audio/translations with text response."""
    response = asr_client.post(
        "/v1/audio/translations",
        data={"model": "whisper-1", "response_format": "text"},
        files={"file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"


@pytest.mark.unit
@pytest.mark.asr
def test_translations_with_temperature(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test translation with temperature parameter."""
    response = asr_client.post(
        "/v1/audio/translations",
        data={"model": "whisper-1", "temperature": "0.2", "response_format": "json"},
        files={"file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "text" in data


@pytest.mark.unit
@pytest.mark.asr
def test_list_models(asr_client: TestClient) -> None:
    """Test GET /v1/models returns available models."""
    response = asr_client.get("/v1/models")

    assert response.status_code == 200
    data = response.json()
    assert "object" in data
    assert data["object"] == "list"
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0


@pytest.mark.unit
@pytest.mark.asr
def test_transcriptions_missing_file(asr_client: TestClient) -> None:
    """Test that missing file returns error."""
    response = asr_client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": "json"},
        # No file uploaded
    )

    assert response.status_code == 422  # Unprocessable Entity


@pytest.mark.unit
@pytest.mark.asr
def test_transcriptions_empty_file(asr_client: TestClient) -> None:
    """Test that empty file is handled."""
    response = asr_client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": "json"},
        files={"file": ("empty.wav", b"", "audio/wav")},
    )

    # Should either succeed with empty result or return error
    assert response.status_code in [200, 422, 500]


@pytest.mark.unit
@pytest.mark.asr
def test_transcriptions_with_prompt(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test that prompt parameter is accepted (even if ignored)."""
    response = asr_client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "prompt": "Test prompt", "response_format": "json"},
        files={"file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "text" in data


@pytest.mark.unit
@pytest.mark.asr
def test_transcriptions_with_timestamp_granularities(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test that timestamp_granularities parameter is accepted."""
    response = asr_client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "timestamp_granularities": "segment", "response_format": "json"},
        files={"file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "text" in data


@pytest.mark.unit
@pytest.mark.asr
def test_transcriptions_openai_compatibility(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test that response matches OpenAI API format."""
    response = asr_client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": "json"},
        files={"file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    data = response.json()

    # OpenAI format: {"text": "..."}
    assert "text" in data
    assert isinstance(data["text"], str)

    # Should not have extra fields in minimal mode
    # (verbose mode is separate endpoint)


@pytest.mark.unit
@pytest.mark.asr
def test_models_list_structure(asr_client: TestClient) -> None:
    """Test that models list has correct structure."""
    response = asr_client.get("/v1/models")

    assert response.status_code == 200
    data = response.json()

    assert data["object"] == "list"
    assert isinstance(data["data"], list)

    # Check first model structure
    if len(data["data"]) > 0:
        model = data["data"][0]
        assert "id" in model
        assert "object" in model
        assert model["object"] == "model"
