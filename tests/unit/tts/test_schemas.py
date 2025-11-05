"""Unit tests for TTS schemas module."""

from __future__ import annotations

import datetime

import pytest
from pydantic import ValidationError

from tts.schemas.audio_engine import (
    AudioFormat,
    ModelResponse,
    ModelsResponse,
    StreamFormat,
    VoiceResponse,
    VoicesResponse,
    return_date_as_unix_ts,
)
from tts.schemas.openai import OpenAICreateSpeechRequest


@pytest.mark.unit
class TestStreamFormat:
    """Tests for StreamFormat enum."""

    def test_stream_format_values(self) -> None:
        """Test StreamFormat enum values."""
        assert StreamFormat.SSE.value == "sse"
        assert StreamFormat.AUDIO.value == "audio"

    def test_stream_format_string_comparison(self) -> None:
        """Test StreamFormat can be compared with strings."""
        assert StreamFormat.SSE == "sse"
        assert StreamFormat.AUDIO == "audio"

    def test_stream_format_members(self) -> None:
        """Test StreamFormat has correct members."""
        members = list(StreamFormat)
        assert len(members) == 2
        assert StreamFormat.SSE in members
        assert StreamFormat.AUDIO in members


@pytest.mark.unit
class TestAudioFormat:
    """Tests for AudioFormat enum."""

    def test_audio_format_values(self) -> None:
        """Test AudioFormat enum values."""
        assert AudioFormat.MP3.value == "mp3"
        assert AudioFormat.OPUS.value == "opus"
        assert AudioFormat.AAC.value == "aac"
        assert AudioFormat.FLAC.value == "flac"
        assert AudioFormat.WAV.value == "wav"
        assert AudioFormat.PCM.value == "pcm"

    def test_audio_format_string_comparison(self) -> None:
        """Test AudioFormat can be compared with strings."""
        assert AudioFormat.MP3 == "mp3"
        assert AudioFormat.WAV == "wav"

    def test_audio_format_members_count(self) -> None:
        """Test AudioFormat has all expected members."""
        members = list(AudioFormat)
        assert len(members) == 6

    def test_audio_format_all_members(self) -> None:
        """Test all AudioFormat members are present."""
        members = {f.value for f in AudioFormat}
        expected = {"mp3", "opus", "aac", "flac", "wav", "pcm"}
        assert members == expected


@pytest.mark.unit
class TestReturnDateAsUnixTs:
    """Tests for return_date_as_unix_ts function."""

    def test_returns_integer(self) -> None:
        """Test function returns an integer."""
        result = return_date_as_unix_ts(year=2020, month=1, day=1)
        assert isinstance(result, int)

    def test_returns_positive_for_past_date(self) -> None:
        """Test function returns positive value for past dates."""
        result = return_date_as_unix_ts(year=2020, month=1, day=1)
        assert result > 0

    def test_older_date_returns_larger_value(self) -> None:
        """Test older dates return larger timestamp values."""
        old_date = return_date_as_unix_ts(year=2000, month=1, day=1)
        recent_date = return_date_as_unix_ts(year=2020, month=1, day=1)
        assert old_date > recent_date

    def test_same_date_returns_same_value(self) -> None:
        """Test calling with same date returns consistent value."""
        result1 = return_date_as_unix_ts(year=2020, month=1, day=1)
        result2 = return_date_as_unix_ts(year=2020, month=1, day=1)
        # Should be within 1 second (accounting for test execution time)
        assert abs(result1 - result2) <= 1

    def test_epoch_date_returns_current_timestamp(self) -> None:
        """Test date close to epoch."""
        result = return_date_as_unix_ts(year=1970, month=1, day=1)
        # Should be approximately current timestamp (seconds since epoch)
        now_ts = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
        assert abs(result - now_ts) < 100  # Within 100 seconds


@pytest.mark.unit
class TestModelResponse:
    """Tests for ModelResponse schema."""

    def test_model_response_creation_with_id(self) -> None:
        """Test creating ModelResponse with required id field."""
        response = ModelResponse(id="test-model-id")
        assert response.id == "test-model-id"

    def test_model_response_defaults(self) -> None:
        """Test ModelResponse has correct default values."""
        response = ModelResponse(id="test-model")
        assert response.object == "model"
        assert response.owned_by == "AllOfUs"
        assert isinstance(response.created, int)
        assert response.created > 0

    def test_model_response_custom_values(self) -> None:
        """Test ModelResponse with custom values."""
        response = ModelResponse(
            id="custom-model",
            object="custom-object",
            created=1234567890,
            owned_by="CustomOrg",
        )
        assert response.id == "custom-model"
        assert response.object == "custom-object"
        assert response.created == 1234567890
        assert response.owned_by == "CustomOrg"

    def test_model_response_serialization(self) -> None:
        """Test ModelResponse can be serialized to dict."""
        response = ModelResponse(id="test-model")
        data = response.model_dump()
        assert isinstance(data, dict)
        assert data["id"] == "test-model"
        assert data["object"] == "model"
        assert "created" in data
        assert "owned_by" in data

    def test_model_response_json_serialization(self) -> None:
        """Test ModelResponse can be serialized to JSON."""
        response = ModelResponse(id="test-model")
        json_str = response.model_dump_json()
        assert isinstance(json_str, str)
        assert "test-model" in json_str


@pytest.mark.unit
class TestModelsResponse:
    """Tests for ModelsResponse schema."""

    def test_models_response_creation(self) -> None:
        """Test creating ModelsResponse with data."""
        models = [
            ModelResponse(id="model-1"),
            ModelResponse(id="model-2"),
        ]
        response = ModelsResponse(data=models)
        assert len(response.data) == 2
        assert response.data[0].id == "model-1"
        assert response.data[1].id == "model-2"

    def test_models_response_default_object(self) -> None:
        """Test ModelsResponse has default object value."""
        response = ModelsResponse(data=[])
        assert response.object == "list"

    def test_models_response_custom_object(self) -> None:
        """Test ModelsResponse with custom object value."""
        response = ModelsResponse(data=[], object="custom-list")
        assert response.object == "custom-list"

    def test_models_response_empty_data(self) -> None:
        """Test ModelsResponse with empty data list."""
        response = ModelsResponse(data=[])
        assert response.data == []
        assert isinstance(response.data, list)

    def test_models_response_serialization(self) -> None:
        """Test ModelsResponse serialization."""
        models = [ModelResponse(id="model-1")]
        response = ModelsResponse(data=models)
        data = response.model_dump()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "model-1"


@pytest.mark.unit
class TestVoiceResponse:
    """Tests for VoiceResponse schema."""

    def test_voice_response_creation_with_required_fields(self) -> None:
        """Test creating VoiceResponse with required id and name fields."""
        response = VoiceResponse(id="test-voice-id", name="Test Voice")
        assert response.id == "test-voice-id"
        assert response.name == "Test Voice"

    def test_voice_response_defaults(self) -> None:
        """Test VoiceResponse has correct default object value."""
        response = VoiceResponse(id="test-voice", name="Test")
        assert response.object == "voice"

    def test_voice_response_custom_object(self) -> None:
        """Test VoiceResponse with custom object value."""
        response = VoiceResponse(
            id="custom-voice",
            name="Custom Voice",
            object="custom-voice-object",
        )
        assert response.id == "custom-voice"
        assert response.name == "Custom Voice"
        assert response.object == "custom-voice-object"

    def test_voice_response_serialization(self) -> None:
        """Test VoiceResponse can be serialized to dict."""
        response = VoiceResponse(id="voice-1", name="Voice One")
        data = response.model_dump()
        assert isinstance(data, dict)
        assert data["id"] == "voice-1"
        assert data["name"] == "Voice One"
        assert data["object"] == "voice"

    def test_voice_response_json_serialization(self) -> None:
        """Test VoiceResponse can be serialized to JSON."""
        response = VoiceResponse(id="test-voice", name="Test Voice")
        json_str = response.model_dump_json()
        assert isinstance(json_str, str)
        assert "test-voice" in json_str
        assert "Test Voice" in json_str


@pytest.mark.unit
class TestVoicesResponse:
    """Tests for VoicesResponse schema."""

    def test_voices_response_creation(self) -> None:
        """Test creating VoicesResponse with data."""
        voices = [
            VoiceResponse(id="voice-1", name="Voice One"),
            VoiceResponse(id="voice-2", name="Voice Two"),
        ]
        response = VoicesResponse(data=voices)
        assert len(response.data) == 2
        assert response.data[0].id == "voice-1"
        assert response.data[0].name == "Voice One"
        assert response.data[1].id == "voice-2"
        assert response.data[1].name == "Voice Two"

    def test_voices_response_default_object(self) -> None:
        """Test VoicesResponse has default object value."""
        response = VoicesResponse(data=[])
        assert response.object == "list"

    def test_voices_response_custom_object(self) -> None:
        """Test VoicesResponse with custom object value."""
        response = VoicesResponse(data=[], object="custom-list")
        assert response.object == "custom-list"

    def test_voices_response_empty_data(self) -> None:
        """Test VoicesResponse with empty data list."""
        response = VoicesResponse(data=[])
        assert response.data == []
        assert isinstance(response.data, list)

    def test_voices_response_serialization(self) -> None:
        """Test VoicesResponse serialization."""
        voices = [VoiceResponse(id="voice-1", name="Voice One")]
        response = VoicesResponse(data=voices)
        data = response.model_dump()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "voice-1"
        assert data["data"][0]["name"] == "Voice One"


@pytest.mark.unit
class TestOpenAICreateSpeechRequest:
    """Tests for OpenAICreateSpeechRequest schema."""

    def test_create_speech_request_required_fields(self) -> None:
        """Test creating request with only required fields."""
        request = OpenAICreateSpeechRequest(
            input="Hello world",
            voice="alloy",
        )
        assert request.input == "Hello world"
        assert request.voice == "alloy"

    def test_create_speech_request_defaults(self) -> None:
        """Test default values for optional fields."""
        request = OpenAICreateSpeechRequest(
            input="Test text",
            voice="shimmer",
        )
        assert request.model == "TTS (XTTS)"
        assert request.response_format == AudioFormat.MP3
        assert request.speed == 1.0
        assert request.stream_format == StreamFormat.AUDIO

    def test_create_speech_request_all_fields(self) -> None:
        """Test creating request with all fields specified."""
        request = OpenAICreateSpeechRequest(
            model="custom-model",
            input="Test input",
            voice="verse",
            response_format=AudioFormat.WAV,
            speed=1.5,
            stream_format=StreamFormat.SSE,
        )
        assert request.model == "custom-model"
        assert request.input == "Test input"
        assert request.voice == "verse"
        assert request.response_format == AudioFormat.WAV
        assert request.speed == 1.5
        assert request.stream_format == StreamFormat.SSE

    def test_create_speech_request_different_audio_formats(self) -> None:
        """Test request with different audio formats."""
        for format in AudioFormat:
            request = OpenAICreateSpeechRequest(
                input="Test",
                voice="alloy",
                response_format=format,
            )
            assert request.response_format == format

    def test_create_speech_request_different_stream_formats(self) -> None:
        """Test request with different stream formats."""
        for format in StreamFormat:
            request = OpenAICreateSpeechRequest(
                input="Test",
                voice="alloy",
                stream_format=format,
            )
            assert request.stream_format == format

    def test_create_speech_request_speed_boundaries(self) -> None:
        """Test speed field with boundary values."""
        # Minimum speed
        request_min = OpenAICreateSpeechRequest(
            input="Test",
            voice="alloy",
            speed=0.25,
        )
        assert request_min.speed == 0.25

        # Maximum speed
        request_max = OpenAICreateSpeechRequest(
            input="Test",
            voice="alloy",
            speed=4.0,
        )
        assert request_max.speed == 4.0

    def test_create_speech_request_speed_validation_below_min(self) -> None:
        """Test speed validation fails below minimum."""
        with pytest.raises(ValidationError):
            OpenAICreateSpeechRequest(
                input="Test",
                voice="alloy",
                speed=0.1,  # Below minimum 0.25
            )

    def test_create_speech_request_speed_validation_above_max(self) -> None:
        """Test speed validation fails above maximum."""
        with pytest.raises(ValidationError):
            OpenAICreateSpeechRequest(
                input="Test",
                voice="alloy",
                speed=5.0,  # Above maximum 4.0
            )

    def test_create_speech_request_none_speed(self) -> None:
        """Test request with None speed value."""
        request = OpenAICreateSpeechRequest(
            input="Test",
            voice="alloy",
            speed=None,
        )
        assert request.speed is None

    def test_create_speech_request_none_stream_format(self) -> None:
        """Test request with None stream_format."""
        request = OpenAICreateSpeechRequest(
            input="Test",
            voice="alloy",
            stream_format=None,
        )
        assert request.stream_format is None

    def test_create_speech_request_serialization(self) -> None:
        """Test request serialization to dict."""
        request = OpenAICreateSpeechRequest(
            input="Hello world",
            voice="alloy",
            response_format=AudioFormat.MP3,
        )
        data = request.model_dump()
        assert data["input"] == "Hello world"
        assert data["voice"] == "alloy"
        assert data["response_format"] == "mp3"

    def test_create_speech_request_different_voices(self) -> None:
        """Test request with different voice names."""
        voices = ["alloy", "verse", "shimmer", "echo", "fable", "onyx"]
        for voice in voices:
            request = OpenAICreateSpeechRequest(
                input="Test",
                voice=voice,
            )
            assert request.voice == voice


@pytest.mark.unit
class TestSchemasIntegration:
    """Integration tests for TTS schemas."""

    def test_models_response_with_multiple_models(self) -> None:
        """Test creating a complete models response."""
        models = [
            ModelResponse(id="xtts-v2", owned_by="Coqui"),
            ModelResponse(id="gpt-4o-mini-tts", owned_by="OpenAI"),
        ]
        response = ModelsResponse(data=models)

        assert len(response.data) == 2
        assert response.object == "list"
        assert response.data[0].id == "xtts-v2"
        assert response.data[1].id == "gpt-4o-mini-tts"

    def test_create_speech_request_realistic_scenario(self) -> None:
        """Test realistic speech request scenario."""
        request = OpenAICreateSpeechRequest(
            model="TTS (XTTS)",
            input="Welcome to our text-to-speech service. This is a test message.",
            voice="alloy",
            response_format=AudioFormat.MP3,
            speed=1.0,
            stream_format=StreamFormat.AUDIO,
        )

        assert request.model == "TTS (XTTS)"
        assert len(request.input) > 0
        assert request.response_format == AudioFormat.MP3
        assert request.stream_format == StreamFormat.AUDIO

    def test_enum_values_match_expected_api_values(self) -> None:
        """Test that enum values match expected API format."""
        # AudioFormat values should be lowercase strings
        for format in AudioFormat:
            assert format.value.islower()
            assert isinstance(format.value, str)

        # StreamFormat values should be lowercase strings
        for format in StreamFormat:
            assert format.value.islower()
            assert isinstance(format.value, str)
