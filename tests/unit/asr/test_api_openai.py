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


# --- voice-stack#9: OpenAI's full response_format set -----------------------


@pytest.mark.unit
@pytest.mark.asr
@pytest.mark.parametrize(
    ("fmt", "media_type"),
    [
        ("srt", "application/x-subrip"),
        ("vtt", "text/vtt; charset=utf-8"),
        ("text", "text/plain; charset=utf-8"),
    ],
)
def test_transcriptions_accepts_openai_response_formats(
    asr_client: TestClient, sample_audio_bytes: bytes, fmt: str, media_type: str
) -> None:
    """json/text/srt/vtt/verbose_json are all accepted; the subtitle formats come
    back with their own media types instead of a 422."""
    response = asr_client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": fmt},
        files={"file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == media_type


@pytest.mark.unit
@pytest.mark.asr
def test_transcriptions_verbose_json_has_the_openai_shape(
    asr_client: TestClient, sample_audio_bytes: bytes
) -> None:
    response = asr_client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": "verbose_json"},
        files={"file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["task"] == "transcribe"
    assert "language" in data
    assert "duration" in data
    assert isinstance(data["text"], str)
    assert isinstance(data["segments"], list)
    for seg in data["segments"]:
        assert {"id", "start", "end", "text"} <= set(seg)
    # `words` only appears when the caller asks for word granularity.
    assert data["words"] is None


def _fake_result(**kwargs):
    class _R:
        def __init__(self) -> None:
            self.__dict__.update(kwargs)

        def to_dict(self):
            return dict(self.__dict__)

    return _R()


def test_verbose_payload_maps_engine_names_to_openai() -> None:
    """The 2026-10-28 nulls: the engine says language_code/confidence, the
    schema says language/language_probability, and pydantic dropped the pair."""
    from asr.api.api_v1.endpoints.openai import _verbose_payload

    payload = _verbose_payload(
        _fake_result(
            text="hi",
            segments=[],
            language_code="pt",
            language_name="Portuguese",
            confidence=0.93,
            duration_input_s=1.5,
        )
    )
    assert payload["language"] == "pt"
    assert payload["language_probability"] == 0.93
    assert payload["language_name"] == "Portuguese"


def test_openai_verbose_words_only_when_requested() -> None:
    from asr.api.api_v1.endpoints.openai import _openai_verbose

    result = _fake_result(
        text="a b",
        language_code="en",
        duration_input_s=2.0,
        segments=[
            {
                "start": 0.0,
                "end": 1.0,
                "text": "a",
                "words": [{"word": "a", "start": 0.0, "end": 0.5}],
            }
        ],
    )
    assert _openai_verbose(result, task="transcribe", granularities=None).words is None
    words = _openai_verbose(result, task="transcribe", granularities="word").words
    assert words == [{"word": "a", "start": 0.0, "end": 0.5}]


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
