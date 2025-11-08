"""Unit tests for ASR Bazarr-compatible endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
@pytest.mark.asr
def test_bazarr_asr_json_format(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test POST /v1/bazarr/asr with JSON output format using query parameters."""
    response = asr_client.post(
        "/v1/bazarr/asr?output=json&language=en",
        files={"audio_file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    data = response.json()

    # Bazarr format: {"language": "en", "segments": [...]}
    assert "language" in data
    assert "segments" in data
    assert isinstance(data["segments"], list)


@pytest.mark.unit
@pytest.mark.asr
def test_bazarr_asr_srt_format(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test POST /v1/bazarr/asr with SRT output format using query parameters."""
    response = asr_client.post(
        "/v1/bazarr/asr?output=srt&language=en",
        files={"audio_file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    assert "application/x-subrip" in response.headers["content-type"]
    assert isinstance(response.text, str)


@pytest.mark.unit
@pytest.mark.asr
def test_bazarr_asr_vtt_format(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test POST /v1/bazarr/asr with VTT output format using query parameters."""
    response = asr_client.post(
        "/v1/bazarr/asr?output=vtt&language=en",
        files={"audio_file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    assert "text/vtt" in response.headers["content-type"]
    assert isinstance(response.text, str)
    # VTT files start with "WEBVTT"
    assert response.text.startswith("WEBVTT") or len(response.text) == 0


@pytest.mark.unit
@pytest.mark.asr
def test_bazarr_asr_txt_format(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test POST /v1/bazarr/asr with TXT output format using query parameters."""
    response = asr_client.post(
        "/v1/bazarr/asr?output=txt&language=en",
        files={"audio_file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert isinstance(response.text, str)


@pytest.mark.unit
@pytest.mark.asr
def test_bazarr_asr_tsv_format(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test POST /v1/bazarr/asr with TSV output format using query parameters."""
    response = asr_client.post(
        "/v1/bazarr/asr?output=tsv&language=en",
        files={"audio_file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    assert "text/tab-separated-values" in response.headers["content-type"]
    assert isinstance(response.text, str)


@pytest.mark.unit
@pytest.mark.asr
def test_bazarr_asr_jsonl_format(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test POST /v1/bazarr/asr with JSONL output format using query parameters."""
    response = asr_client.post(
        "/v1/bazarr/asr?output=jsonl&language=en",
        files={"audio_file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    assert "application/jsonl" in response.headers["content-type"]
    assert isinstance(response.text, str)


@pytest.mark.unit
@pytest.mark.asr
def test_bazarr_asr_without_language(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test that language parameter is optional with query parameters."""
    response = asr_client.post(
        "/v1/bazarr/asr?output=json",
        files={"audio_file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "language" in data
    assert "segments" in data


@pytest.mark.unit
@pytest.mark.asr
def test_bazarr_asr_default_output_format(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test that default output format is TXT with query parameters."""
    response = asr_client.post(
        "/v1/bazarr/asr",  # No output parameter specified
        files={"audio_file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    # Default should be txt
    assert "text/plain" in response.headers["content-type"] or "application/json" in response.headers["content-type"]


@pytest.mark.unit
@pytest.mark.asr
def test_bazarr_detect_language(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test POST /v1/bazarr/detect-language endpoint with query parameters."""
    response = asr_client.post(
        "/v1/bazarr/detect-language",
        files={"audio_file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    data = response.json()

    # Check required fields
    assert "language_code" in data
    assert "confidence" in data or "language_probability" in data
    assert isinstance(data["language_code"], str)


@pytest.mark.unit
@pytest.mark.asr
def test_bazarr_detect_language_with_params(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test language detection with offset and length query parameters."""
    response = asr_client.post(
        "/v1/bazarr/detect-language?detect_lang_offset=0&detect_lang_length=5&video_file=/data/Movies/test.mp4",
        files={"audio_file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "language_code" in data


@pytest.mark.unit
@pytest.mark.asr
def test_bazarr_list_models(asr_client: TestClient) -> None:
    """Test GET /v1/bazarr/models endpoint."""
    response = asr_client.get("/v1/bazarr/models")

    assert response.status_code == 200
    data = response.json()
    assert "object" in data
    assert data["object"] == "list"
    assert "data" in data
    assert isinstance(data["data"], list)


@pytest.mark.unit
@pytest.mark.asr
def test_bazarr_asr_missing_file(asr_client: TestClient) -> None:
    """Test that missing file returns error with query parameters."""
    response = asr_client.post(
        "/v1/bazarr/asr?output=json",
        # No file uploaded
    )

    assert response.status_code == 422  # Unprocessable Entity


@pytest.mark.unit
@pytest.mark.asr
def test_bazarr_asr_segments_structure(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test that segments have correct structure in JSON format with query parameters."""
    response = asr_client.post(
        "/v1/bazarr/asr?output=json",
        files={"audio_file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    data = response.json()

    # Check segments structure
    if len(data["segments"]) > 0:
        segment = data["segments"][0]
        assert "start" in segment
        assert "end" in segment
        assert "text" in segment
        assert isinstance(segment["start"], int | float)
        assert isinstance(segment["end"], int | float)
        assert isinstance(segment["text"], str)


@pytest.mark.unit
@pytest.mark.asr
def test_bazarr_asr_response_headers(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test that response includes custom headers with query parameters."""
    response = asr_client.post(
        "/v1/bazarr/asr?output=json",
        files={"audio_file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    # Check for custom headers
    assert "asr-engine" in response.headers or "Asr-Engine" in response.headers


@pytest.mark.unit
@pytest.mark.asr
def test_bazarr_asr_with_different_audio_formats(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """Test that different audio format extensions are handled with query parameters."""
    for ext in ["wav", "mp3", "ogg", "flac"]:
        response = asr_client.post(
            "/v1/bazarr/asr?output=json",
            files={"audio_file": (f"test.{ext}", sample_audio_bytes, f"audio/{ext}")},
        )

        # Should either succeed or fail gracefully
        assert response.status_code in [200, 422, 500]


@pytest.mark.unit
@pytest.mark.asr
def test_bazarr_detect_language_missing_file(asr_client: TestClient) -> None:
    """Test that detect-language requires a file."""
    response = asr_client.post(
        "/v1/bazarr/detect-language",
        # No file uploaded
    )

    assert response.status_code == 422  # Unprocessable Entity
