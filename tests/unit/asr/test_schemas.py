"""Unit tests for ASR Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from asr.schemas.audio_engine import Output, Task
from asr.schemas.openai import (
    OpenAITranscription,
    OpenAITranscriptionRequest,
    OpenAITranscriptionVerboseRequest,
    OpenAITranslationsRequest,
    ResponseFormat,
    Segment,
    TranscribeResponse,
    TranscribeVerboseResponse,
)


@pytest.mark.unit
@pytest.mark.asr
def test_response_format_enum() -> None:
    """Test ResponseFormat enum values."""
    assert ResponseFormat.JSON == "json"
    assert ResponseFormat.TEXT == "text"


@pytest.mark.unit
@pytest.mark.asr
def test_task_enum() -> None:
    """Test Task enum values."""
    assert Task.TRANSCRIBE == "transcribe"
    assert Task.TRANSLATE == "translate"


@pytest.mark.unit
@pytest.mark.asr
def test_output_enum() -> None:
    """Test Output enum values."""
    assert Output.JSON == "json"
    assert Output.TXT == "txt"
    assert Output.SRT == "srt"
    assert Output.VTT == "vtt"
    assert Output.TSV == "tsv"
    assert Output.JSONL == "jsonl"


@pytest.mark.unit
@pytest.mark.asr
def test_openai_transcription_request_defaults() -> None:
    """Test OpenAITranscriptionRequest default values."""
    request = OpenAITranscriptionRequest()

    assert request.model == "whisper-1"
    assert request.language is None
    assert request.temperature is None
    assert request.response_format == ResponseFormat.JSON
    assert request.prompt is None


@pytest.mark.unit
@pytest.mark.asr
def test_openai_transcription_request_with_values() -> None:
    """Test OpenAITranscriptionRequest with custom values."""
    request = OpenAITranscriptionRequest(
        model="whisper-1",
        language="en",
        temperature=0.5,
        response_format=ResponseFormat.TEXT,
        prompt="Test prompt",
    )

    assert request.model == "whisper-1"
    assert request.language == "en"
    assert request.temperature == 0.5
    assert request.response_format == ResponseFormat.TEXT
    assert request.prompt == "Test prompt"


@pytest.mark.unit
@pytest.mark.asr
def test_openai_transcription_request_empty_string_to_none() -> None:
    """Test that empty strings are coerced to None."""
    request = OpenAITranscriptionRequest(language="", prompt="")

    assert request.language is None
    assert request.prompt is None


@pytest.mark.unit
@pytest.mark.asr
def test_openai_transcription_verbose_request_defaults() -> None:
    """Test OpenAITranscriptionVerboseRequest default values."""
    request = OpenAITranscriptionVerboseRequest()

    assert request.model == "whisper-1"
    assert request.task is None
    assert request.beam_size is None
    assert request.best_of is None
    assert request.word_timestamps is None
    assert request.vad is None


@pytest.mark.unit
@pytest.mark.asr
def test_openai_transcription_verbose_request_with_values() -> None:
    """Test OpenAITranscriptionVerboseRequest with custom values."""
    request = OpenAITranscriptionVerboseRequest(
        model="whisper-1",
        language="en",
        task=Task.TRANSCRIBE,
        beam_size=5,
        temperature=0.0,
        best_of=1,
        word_timestamps=True,
        vad=True,
    )

    assert request.model == "whisper-1"
    assert request.language == "en"
    assert request.task == Task.TRANSCRIBE
    assert request.beam_size == 5
    assert request.temperature == 0.0
    assert request.best_of == 1
    assert request.word_timestamps is True
    assert request.vad is True


@pytest.mark.unit
@pytest.mark.asr
def test_openai_transcription_verbose_request_coercion() -> None:
    """Test type coercion in verbose request."""
    # Test string to int coercion
    request = OpenAITranscriptionVerboseRequest(beam_size="5", best_of="1")

    assert request.beam_size == 5
    assert request.best_of == 1

    # Test string to bool coercion
    request2 = OpenAITranscriptionVerboseRequest(word_timestamps="true", vad="1")

    assert request2.word_timestamps is True
    assert request2.vad is True


@pytest.mark.unit
@pytest.mark.asr
def test_openai_translations_request_defaults() -> None:
    """Test OpenAITranslationsRequest default values."""
    request = OpenAITranslationsRequest()

    assert request.model == "whisper-1"
    assert request.temperature is None
    assert request.response_format == ResponseFormat.JSON
    assert request.prompt is None


@pytest.mark.unit
@pytest.mark.asr
def test_openai_translations_request_with_values() -> None:
    """Test OpenAITranslationsRequest with custom values."""
    request = OpenAITranslationsRequest(
        model="whisper-1", temperature=0.2, response_format=ResponseFormat.TEXT, prompt="Test"
    )

    assert request.model == "whisper-1"
    assert request.temperature == 0.2
    assert request.response_format == ResponseFormat.TEXT
    assert request.prompt == "Test"


@pytest.mark.unit
@pytest.mark.asr
def test_segment_schema() -> None:
    """Test Segment schema."""
    segment = Segment(start=0.0, end=2.5, text="Hello world")

    assert segment.start == 0.0
    assert segment.end == 2.5
    assert segment.text == "Hello world"


@pytest.mark.unit
@pytest.mark.asr
def test_segment_validation() -> None:
    """Test Segment validation."""
    # Should require all fields
    with pytest.raises(ValidationError):
        Segment(start=0.0)  # Missing end and text


@pytest.mark.unit
@pytest.mark.asr
def test_transcribe_response() -> None:
    """Test TranscribeResponse schema."""
    response = TranscribeResponse(
        text="Hello world",
        segments=[{"start": 0.0, "end": 2.5, "text": "Hello world"}],
        language="en",
    )

    assert response.text == "Hello world"
    assert len(response.segments) == 1
    assert response.language == "en"


@pytest.mark.unit
@pytest.mark.asr
def test_transcribe_response_defaults() -> None:
    """Test TranscribeResponse with default values."""
    response = TranscribeResponse(text="Test")

    assert response.text == "Test"
    assert response.segments == []
    assert response.language is None


@pytest.mark.unit
@pytest.mark.asr
def test_transcribe_verbose_response() -> None:
    """Test TranscribeVerboseResponse schema."""
    response = TranscribeVerboseResponse(
        text="Hello world",
        segments=[],
        language="en",
        language_name="English",
        language_probability=0.98,
        duration_input_s=2.5,
        duration_after_vad_s=2.3,
        processing_ms=150,
        asr_ms=120,
        vad_used=True,
        model="base",
    )

    assert response.text == "Hello world"
    assert response.language == "en"
    assert response.language_name == "English"
    assert response.language_probability == 0.98
    assert response.duration_input_s == 2.5
    assert response.duration_after_vad_s == 2.3
    assert response.processing_ms == 150
    assert response.asr_ms == 120
    assert response.vad_used is True
    assert response.model == "base"


@pytest.mark.unit
@pytest.mark.asr
def test_transcribe_verbose_response_optional_fields() -> None:
    """Test that verbose response allows optional fields."""
    response = TranscribeVerboseResponse(text="Test", segments=[])

    assert response.text == "Test"
    assert response.language is None
    assert response.language_name is None
    assert response.language_probability is None


@pytest.mark.unit
@pytest.mark.asr
def test_openai_transcription_minimal() -> None:
    """Test OpenAITranscription minimal response."""
    response = OpenAITranscription(text="Hello world")

    assert response.text == "Hello world"


@pytest.mark.unit
@pytest.mark.asr
def test_openai_transcription_validation() -> None:
    """Test OpenAITranscription requires text field."""
    with pytest.raises(ValidationError):
        OpenAITranscription()  # Missing required text field


@pytest.mark.unit
@pytest.mark.asr
def test_as_form_method_transcription_request() -> None:
    """Test that OpenAITranscriptionRequest can be created from form data."""
    # The as_form method is designed to be used by FastAPI's dependency injection,
    # not called directly. Test the model creation instead.
    request = OpenAITranscriptionRequest(
        model="whisper-1",
        language="en",
        temperature=0.5,
        response_format=ResponseFormat.JSON,
        prompt=None,
        timestamp_granularities=None,
    )

    assert isinstance(request, OpenAITranscriptionRequest)
    assert request.model == "whisper-1"
    assert request.language == "en"
    assert request.temperature == 0.5


@pytest.mark.unit
@pytest.mark.asr
def test_as_form_method_verbose_request() -> None:
    """Test that OpenAITranscriptionVerboseRequest can be created from form data."""
    # The as_form method is designed to be used by FastAPI's dependency injection,
    # not called directly. Test the model creation instead.
    request = OpenAITranscriptionVerboseRequest(
        model="whisper-1",
        language="en",
        task=Task.TRANSCRIBE,
        beam_size=5,
        temperature=0.0,
        response_format=ResponseFormat.JSON,
        prompt=None,
        timestamp_granularities=None,
        best_of=None,
        word_timestamps=None,
        vad=None,
    )

    assert isinstance(request, OpenAITranscriptionVerboseRequest)
    assert request.language == "en"
    assert request.task == Task.TRANSCRIBE
    assert request.beam_size == 5


@pytest.mark.unit
@pytest.mark.asr
def test_as_form_method_translations_request() -> None:
    """Test that OpenAITranslationsRequest can be created from form data."""
    # The as_form method is designed to be used by FastAPI's dependency injection,
    # not called directly. Test the model creation instead.
    request = OpenAITranslationsRequest(
        model="whisper-1",
        temperature=0.2,
        response_format=ResponseFormat.TEXT,
        prompt=None,
    )

    assert isinstance(request, OpenAITranslationsRequest)
    assert request.model == "whisper-1"
    assert request.temperature == 0.2
    assert request.response_format == ResponseFormat.TEXT


@pytest.mark.unit
@pytest.mark.asr
def test_temperature_validation() -> None:
    """Test that temperature accepts float values."""
    request = OpenAITranscriptionRequest(temperature=0.7)
    assert request.temperature == 0.7

    request2 = OpenAITranscriptionRequest(temperature=0.0)
    assert request2.temperature == 0.0

    request3 = OpenAITranscriptionRequest(temperature=1.0)
    assert request3.temperature == 1.0


@pytest.mark.unit
@pytest.mark.asr
def test_bool_coercion_various_formats() -> None:
    """Test boolean coercion from various string formats."""
    # True values
    for val in ["true", "True", "1", "yes", "y", "on"]:
        request = OpenAITranscriptionVerboseRequest(vad=val)
        assert request.vad is True

    # False values
    for val in ["false", "False", "0", "no", "n", "off"]:
        request = OpenAITranscriptionVerboseRequest(vad=val)
        assert request.vad is False

    # Empty string to None
    request = OpenAITranscriptionVerboseRequest(vad="")
    assert request.vad is None


@pytest.mark.unit
@pytest.mark.asr
def test_schema_serialization() -> None:
    """Test that schemas can be serialized to dict/JSON."""
    response = TranscribeResponse(
        text="Test",
        segments=[Segment(start=0.0, end=1.0, text="Test")],
        language="en",
    )

    # Test model_dump (Pydantic v2)
    data = response.model_dump()
    assert isinstance(data, dict)
    assert data["text"] == "Test"
    assert len(data["segments"]) == 1

    # Test model_dump_json
    json_str = response.model_dump_json()
    assert isinstance(json_str, str)
    assert "Test" in json_str
