"""voice-stack#9: OpenAI's full response_format set on the ASR endpoints."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from asr.api.api_v1.endpoints.openai import _openai_verbose, _verbose_payload


class _FakeResult:
    """Stand-in for TranscribeResult: attribute access plus to_dict()."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


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
    """The subtitle formats answer with their own media types instead of a 422."""
    response = asr_client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": fmt},
        files={"file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == media_type


@pytest.mark.unit
@pytest.mark.asr
def test_transcriptions_verbose_json_has_the_openai_shape(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    response = asr_client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": "verbose_json"},
        files={"file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["task"] == "transcribe"
    assert data["language"] == "en"
    assert data["duration"] is not None
    assert isinstance(data["text"], str)
    assert isinstance(data["segments"], list)
    for seg in data["segments"]:
        assert {"id", "start", "end", "text"} <= set(seg)
    # `words` only appears when the caller asks for word granularity.
    assert data["words"] is None


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
def test_translations_accepts_openai_response_formats(
    asr_client: TestClient, sample_audio_bytes: bytes, fmt: str, media_type: str
) -> None:
    """The translations route gets the same format set and media types."""
    response = asr_client.post(
        "/v1/audio/translations",
        data={"model": "whisper-1", "response_format": fmt},
        files={"file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == media_type


@pytest.mark.unit
@pytest.mark.asr
def test_translations_verbose_json_keeps_the_verbose_body(asr_client: TestClient, sample_audio_bytes: bytes) -> None:
    """`response_model=None` on translations: the body must not collapse to {"text"}."""
    response = asr_client.post(
        "/v1/audio/translations",
        data={"model": "whisper-1", "response_format": "verbose_json"},
        files={"file": ("test.wav", sample_audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["task"] == "translate"
    assert {"language", "duration", "text", "segments"} <= set(data)


@pytest.mark.unit
@pytest.mark.asr
def test_verbose_payload_maps_engine_names_to_openai() -> None:
    """The engine says language_code/confidence, the schema language/language_probability.

    Passing the raw dict let pydantic drop both, which is the null pair the
    2026-10-28 run reported next to a filled language_name.
    """
    payload = _verbose_payload(
        _FakeResult(
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


@pytest.mark.unit
@pytest.mark.asr
def test_openai_verbose_words_only_when_requested() -> None:
    result = _FakeResult(
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
