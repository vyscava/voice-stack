"""Unit tests for TTS engine base module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tts.engine.base import (
    PropsConf,
    TTSBase,
    _disposition,
    _ext_for,
    _mime_for,
    _normalize_enum,
    languange_canonical_str,
    speech_effective_options,
)
from tts.schemas.audio_engine import AudioFormat, StreamFormat


@pytest.mark.unit
class TestLanguangeCanonicalStr:
    """Tests for languange_canonical_str function."""

    def test_none_returns_none(self) -> None:
        """Test None input returns None."""
        assert languange_canonical_str(None) is None

    def test_empty_string_returns_none(self) -> None:
        """Test empty string returns None."""
        assert languange_canonical_str("") is None

    def test_valid_language_code(self) -> None:
        """Test valid language codes return canonical form."""
        assert languange_canonical_str("en") == "en"
        assert languange_canonical_str("es") == "es"
        assert languange_canonical_str("pt") == "pt"

    def test_language_alias_returns_canonical(self) -> None:
        """Test language aliases return canonical form."""
        assert languange_canonical_str("en-us") == "en"
        assert languange_canonical_str("pt-br") == "pt"
        assert languange_canonical_str("zh") == "zh-cn"

    def test_case_insensitive(self) -> None:
        """Test function is case-insensitive."""
        assert languange_canonical_str("EN") == "en"
        assert languange_canonical_str("Es") == "es"

    def test_unknown_language_returns_none(self) -> None:
        """Test unknown language codes return None."""
        assert languange_canonical_str("xyz") is None
        assert languange_canonical_str("invalid") is None


@pytest.mark.unit
class TestMimeFor:
    """Tests for _mime_for function."""

    def test_mp3_mime_type(self) -> None:
        """Test MP3 format returns correct MIME type."""
        assert _mime_for("mp3") == "audio/mpeg"

    def test_opus_mime_type(self) -> None:
        """Test OPUS format returns correct MIME type."""
        assert _mime_for("opus") == "audio/ogg; codecs=opus"

    def test_aac_mime_type(self) -> None:
        """Test AAC format returns correct MIME type."""
        assert _mime_for("aac") == "audio/aac"

    def test_flac_mime_type(self) -> None:
        """Test FLAC format returns correct MIME type."""
        assert _mime_for("flac") == "audio/flac"

    def test_wav_mime_type(self) -> None:
        """Test WAV format returns correct MIME type."""
        assert _mime_for("wav") == "audio/wav"

    def test_pcm_mime_type_default(self) -> None:
        """Test PCM format returns correct MIME type with defaults."""
        result = _mime_for("pcm")
        assert "audio/L16" in result
        assert "rate=24000" in result
        assert "channels=1" in result

    def test_pcm_mime_type_custom_params(self) -> None:
        """Test PCM format with custom sample rate and channels."""
        result = _mime_for("pcm", sample_rate=48000, channels=2)
        assert "audio/L16" in result
        assert "rate=48000" in result
        assert "channels=2" in result

    def test_case_insensitive(self) -> None:
        """Test format is case-insensitive."""
        assert _mime_for("MP3") == "audio/mpeg"
        assert _mime_for("WaV") == "audio/wav"

    def test_unknown_format_returns_fallback(self) -> None:
        """Test unknown format returns fallback MIME type."""
        assert _mime_for("unknown") == "application/octet-stream"


@pytest.mark.unit
class TestExtFor:
    """Tests for _ext_for function."""

    def test_mp3_extension(self) -> None:
        """Test MP3 format returns correct extension."""
        assert _ext_for("mp3") == "mp3"

    def test_opus_extension(self) -> None:
        """Test OPUS format returns OGG extension."""
        assert _ext_for("opus") == "ogg"

    def test_aac_extension(self) -> None:
        """Test AAC format returns correct extension."""
        assert _ext_for("aac") == "aac"

    def test_flac_extension(self) -> None:
        """Test FLAC format returns correct extension."""
        assert _ext_for("flac") == "flac"

    def test_wav_extension(self) -> None:
        """Test WAV format returns correct extension."""
        assert _ext_for("wav") == "wav"

    def test_pcm_extension(self) -> None:
        """Test PCM format returns raw format extension."""
        assert _ext_for("pcm") == "s16le"

    def test_case_insensitive(self) -> None:
        """Test format is case-insensitive."""
        assert _ext_for("MP3") == "mp3"
        assert _ext_for("FLaC") == "flac"

    def test_unknown_format_returns_bin(self) -> None:
        """Test unknown format returns 'bin' extension."""
        assert _ext_for("unknown") == "bin"


@pytest.mark.unit
class TestDisposition:
    """Tests for _disposition function."""

    def test_creates_content_disposition_header(self) -> None:
        """Test function creates Content-Disposition header."""
        result = _disposition("test.mp3")
        assert "Content-Disposition" in result
        assert result["Content-Disposition"] == 'attachment; filename="test.mp3"'

    def test_preserves_filename(self) -> None:
        """Test filename is preserved in header."""
        result = _disposition("audio-file.wav")
        assert "audio-file.wav" in result["Content-Disposition"]

    def test_returns_dict(self) -> None:
        """Test function returns dictionary."""
        result = _disposition("file.mp3")
        assert isinstance(result, dict)


@pytest.mark.unit
class TestPropsConf:
    """Tests for PropsConf dataclass."""

    def test_propsconf_creation_with_input_only(self) -> None:
        """Test creating PropsConf with only required input field."""
        props = PropsConf(input="Test text")
        assert props.input == "Test text"

    def test_propsconf_defaults(self) -> None:
        """Test PropsConf has correct default values."""
        props = PropsConf(input="Test")
        assert props.voice == "default"
        assert props.response_format == "mp3"
        assert props.speed == 1.0
        assert props.stream_format == "audio"
        assert props.requested_language is None
        assert props.language_hint is None

    def test_propsconf_all_fields(self) -> None:
        """Test creating PropsConf with all fields."""
        props = PropsConf(
            input="Test input",
            voice="custom-voice",
            response_format="wav",
            speed=1.5,
            stream_format="sse",
            requested_language="en",
            language_hint="es",
        )
        assert props.input == "Test input"
        assert props.voice == "custom-voice"
        assert props.response_format == "wav"
        assert props.speed == 1.5
        assert props.stream_format == "sse"
        assert props.requested_language == "en"
        assert props.language_hint == "es"


@pytest.mark.unit
class TestNormalizeEnum:
    """Tests for _normalize_enum function."""

    def test_none_returns_default(self) -> None:
        """Test None input returns default value."""
        assert _normalize_enum(v=None, default="mp3") == "mp3"

    def test_string_value_returned(self) -> None:
        """Test string value is returned as-is."""
        assert _normalize_enum(v="wav", default="mp3") == "wav"

    def test_enum_value_extracted(self) -> None:
        """Test Enum value is extracted."""
        assert _normalize_enum(v=AudioFormat.MP3, default="wav") == "mp3"
        assert _normalize_enum(v=StreamFormat.SSE, default="audio") == "sse"

    def test_different_defaults(self) -> None:
        """Test with different default values."""
        assert _normalize_enum(v=None, default="flac") == "flac"
        assert _normalize_enum(v=None, default="sse") == "sse"


@pytest.mark.unit
class TestSpeechEffectiveOptions:
    """Tests for speech_effective_options function."""

    def test_all_none_returns_defaults(self) -> None:
        """Test all None inputs return default values."""
        props = speech_effective_options()
        assert props.input == "You need to provide an input!"
        assert props.voice == "default"
        assert props.response_format == "mp3"
        assert props.speed == 1.0
        assert props.stream_format == "audio"

    def test_input_provided(self) -> None:
        """Test with input provided."""
        props = speech_effective_options(input="Hello world")
        assert props.input == "Hello world"

    def test_voice_provided(self) -> None:
        """Test with voice provided."""
        props = speech_effective_options(voice="alloy")
        assert props.voice == "alloy"

    def test_response_format_string(self) -> None:
        """Test with response_format as string."""
        props = speech_effective_options(response_format="wav")
        assert props.response_format == "wav"

    def test_response_format_enum(self) -> None:
        """Test with response_format as enum."""
        props = speech_effective_options(response_format=AudioFormat.FLAC)
        assert props.response_format == "flac"

    def test_speed_provided(self) -> None:
        """Test with speed provided."""
        props = speech_effective_options(speed=1.5)
        assert props.speed == 1.5

    def test_stream_format_string(self) -> None:
        """Test with stream_format as string."""
        props = speech_effective_options(stream_format="sse")
        assert props.stream_format == "sse"

    def test_stream_format_enum(self) -> None:
        """Test with stream_format as enum."""
        props = speech_effective_options(stream_format=StreamFormat.SSE)
        assert props.stream_format == "sse"

    def test_requested_language(self) -> None:
        """Test with requested_language provided."""
        props = speech_effective_options(requested_language="en")
        assert props.requested_language == "en"

    def test_language_hint(self) -> None:
        """Test with language_hint provided."""
        props = speech_effective_options(language_hint="es")
        assert props.language_hint == "es"

    def test_all_parameters_provided(self) -> None:
        """Test with all parameters provided."""
        props = speech_effective_options(
            input="Test",
            voice="custom",
            response_format="flac",
            speed=2.0,
            stream_format="sse",
            requested_language="fr",
            language_hint="de",
        )
        assert props.input == "Test"
        assert props.voice == "custom"
        assert props.response_format == "flac"
        assert props.speed == 2.0
        assert props.stream_format == "sse"
        assert props.requested_language == "fr"
        assert props.language_hint == "de"

    def test_format_lowercased(self) -> None:
        """Test that response_format and stream_format are lowercased."""
        props = speech_effective_options(response_format="MP3", stream_format="SSE")
        assert props.response_format == "mp3"
        assert props.stream_format == "sse"


@pytest.mark.unit
class TestTTSBase:
    """Tests for TTSBase abstract class."""

    def test_ttsbase_initialization(self) -> None:
        """Test TTSBase initialization loads settings."""

        # Create a concrete implementation for testing
        class ConcreteTTS(TTSBase):
            def speech(self, **kwargs):
                pass

            def _unload_model(self):
                pass

            def list_voices(self):
                from tts.schemas.audio_engine import VoicesResponse

                return VoicesResponse(data=[])

        with patch("tts.engine.base.settings") as mock_settings:
            mock_settings.TTS_MODEL = "test-model"
            mock_settings.TTS_SAMPLE_RATE = 48000
            mock_settings.TTS_VOICES_DIR = "test-voices"
            mock_settings.TTS_MAX_CHARS = 200
            mock_settings.TTS_MIN_CHARS = 50
            mock_settings.TTS_RETRY_STEPS = 3
            mock_settings.TTS_AUTO_LANG = True
            mock_settings.TTS_DEFAULT_LANG = "en"
            mock_settings.TTS_LANG_HINT = None
            mock_settings.TTS_FORCE_LANG = None
            mock_settings.TTS_DEVICE = "cpu"

            tts = ConcreteTTS()

            assert tts.model_id == "test-model"
            assert tts.sample_rate == 48000
            assert tts.voices_dir == "test-voices"
            assert tts.max_chars == 200
            assert tts.min_chars == 50
            assert tts.retry_steps == 3
            assert tts.auto_language is True
            assert tts.default_languange == "en"
            assert tts.model_device == "cpu"

    def test_ttsbase_device_auto_selection(self) -> None:
        """Test TTSBase device auto selection."""

        class ConcreteTTS(TTSBase):
            def speech(self, **kwargs):
                pass

            def _unload_model(self):
                pass

            def list_voices(self):
                from tts.schemas.audio_engine import VoicesResponse

                return VoicesResponse(data=[])

        with (
            patch("tts.engine.base.settings") as mock_settings,
            patch("tts.engine.base.pick_torch_device") as mock_pick_device,
        ):
            mock_settings.TTS_DEVICE = "auto"
            mock_settings.TTS_MODEL = "model"
            mock_settings.TTS_SAMPLE_RATE = 24000
            mock_settings.TTS_VOICES_DIR = "voices"
            mock_settings.TTS_MAX_CHARS = 180
            mock_settings.TTS_MIN_CHARS = 70
            mock_settings.TTS_RETRY_STEPS = 2
            mock_settings.TTS_AUTO_LANG = True
            mock_settings.TTS_DEFAULT_LANG = "en"
            mock_settings.TTS_LANG_HINT = None
            mock_settings.TTS_FORCE_LANG = None

            mock_pick_device.return_value = "cuda"

            tts = ConcreteTTS()

            assert tts.model_device == "cuda"
            mock_pick_device.assert_called_once()

    def test_ttsbase_get_model(self) -> None:
        """Test get_model method returns ModelResponse."""

        class ConcreteTTS(TTSBase):
            def speech(self, **kwargs):
                pass

            def _unload_model(self):
                pass

            def list_voices(self):
                from tts.schemas.audio_engine import VoicesResponse

                return VoicesResponse(data=[])

        with patch("tts.engine.base.settings") as mock_settings:
            mock_settings.TTS_DEVICE = "cpu"
            mock_settings.TTS_MODEL = "model"
            mock_settings.TTS_SAMPLE_RATE = 24000
            mock_settings.TTS_VOICES_DIR = "voices"
            mock_settings.TTS_MAX_CHARS = 180
            mock_settings.TTS_MIN_CHARS = 70
            mock_settings.TTS_RETRY_STEPS = 2
            mock_settings.TTS_AUTO_LANG = True
            mock_settings.TTS_DEFAULT_LANG = "en"
            mock_settings.TTS_LANG_HINT = None
            mock_settings.TTS_FORCE_LANG = None

            tts = ConcreteTTS()
            model = tts.get_model("test-model")

            assert model.id == "test-model"
            assert model.object == "model"

    def test_ttsbase_list_models(self) -> None:
        """Test list_models method returns ModelsResponse."""

        class ConcreteTTS(TTSBase):
            def speech(self, **kwargs):
                pass

            def _unload_model(self):
                pass

            def list_voices(self):
                from tts.schemas.audio_engine import VoicesResponse

                return VoicesResponse(data=[])

        with patch("tts.engine.base.settings") as mock_settings:
            mock_settings.TTS_DEVICE = "cpu"
            mock_settings.TTS_MODEL = "model"
            mock_settings.TTS_SAMPLE_RATE = 24000
            mock_settings.TTS_VOICES_DIR = "voices"
            mock_settings.TTS_MAX_CHARS = 180
            mock_settings.TTS_MIN_CHARS = 70
            mock_settings.TTS_RETRY_STEPS = 2
            mock_settings.TTS_AUTO_LANG = True
            mock_settings.TTS_DEFAULT_LANG = "en"
            mock_settings.TTS_LANG_HINT = None
            mock_settings.TTS_FORCE_LANG = None

            tts = ConcreteTTS()
            models = tts.list_models()

            assert models.object == "list"
            assert len(models.data) == 10
            assert all(m.id.startswith("model_") for m in models.data)

    def test_ttsbase_list_voices_is_abstract(self) -> None:
        """Test that list_voices method must be implemented by subclasses."""

        # Attempting to create TTSBase without implementing list_voices should fail
        class IncompleteTTS(TTSBase):
            def speech(self, **kwargs):
                pass

        # Should not be able to instantiate an abstract class
        with pytest.raises(TypeError, match="abstract"):
            IncompleteTTS()

    def test_ttsbase_speech_is_abstract(self) -> None:
        """Test that speech method must be implemented by subclasses."""

        # Attempting to create TTSBase without implementing speech should fail
        class IncompleteTTS(TTSBase):
            def list_voices(self):
                pass

        # Should not be able to instantiate an abstract class
        with pytest.raises(TypeError, match="abstract"):
            IncompleteTTS()

    def test_ttsbase_helper_return_audio_file_bytes(self) -> None:
        """Test helper_return_audio_file with bytes input."""

        class ConcreteTTS(TTSBase):
            def speech(self, **kwargs):
                pass

            def _unload_model(self):
                pass

            def list_voices(self):
                from tts.schemas.audio_engine import VoicesResponse

                return VoicesResponse(data=[])

        with patch("tts.engine.base.settings") as mock_settings:
            mock_settings.TTS_DEVICE = "cpu"
            mock_settings.TTS_MODEL = "model"
            mock_settings.TTS_SAMPLE_RATE = 24000
            mock_settings.TTS_VOICES_DIR = "voices"
            mock_settings.TTS_MAX_CHARS = 180
            mock_settings.TTS_MIN_CHARS = 70
            mock_settings.TTS_RETRY_STEPS = 2
            mock_settings.TTS_AUTO_LANG = True
            mock_settings.TTS_DEFAULT_LANG = "en"
            mock_settings.TTS_LANG_HINT = None
            mock_settings.TTS_FORCE_LANG = None

            tts = ConcreteTTS()
            audio_data = b"fake audio data"

            from fastapi.responses import Response

            response = tts.helper_return_audio_file(audio=audio_data, response_format=AudioFormat.MP3)

            assert isinstance(response, Response)
            assert response.body == audio_data
            assert "audio/mpeg" in response.media_type
            assert "Content-Disposition" in response.headers

    def test_ttsbase_helper_return_sse_stream_bytes(self) -> None:
        """Test helper_return_sse_stream with bytes input."""

        class ConcreteTTS(TTSBase):
            def speech(self, **kwargs):
                pass

            def _unload_model(self):
                pass

            def list_voices(self):
                from tts.schemas.audio_engine import VoicesResponse

                return VoicesResponse(data=[])

        with patch("tts.engine.base.settings") as mock_settings:
            mock_settings.TTS_DEVICE = "cpu"
            mock_settings.TTS_MODEL = "model"
            mock_settings.TTS_SAMPLE_RATE = 24000
            mock_settings.TTS_VOICES_DIR = "voices"
            mock_settings.TTS_MAX_CHARS = 180
            mock_settings.TTS_MIN_CHARS = 70
            mock_settings.TTS_RETRY_STEPS = 2
            mock_settings.TTS_AUTO_LANG = True
            mock_settings.TTS_DEFAULT_LANG = "en"
            mock_settings.TTS_LANG_HINT = None
            mock_settings.TTS_FORCE_LANG = None

            tts = ConcreteTTS()
            audio_data = b"fake audio data"

            from fastapi.responses import StreamingResponse

            response = tts.helper_return_sse_stream(audio=audio_data)

            assert isinstance(response, StreamingResponse)
            assert response.media_type == "text/event-stream"
            assert "Cache-Control" in response.headers
            assert response.headers["Cache-Control"] == "no-cache"
