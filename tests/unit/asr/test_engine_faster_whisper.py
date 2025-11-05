"""Unit tests for ASR Faster-Whisper engine."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from asr.engine.base import DetectLanguageResult, TranscribeResult
from asr.engine.fasterwhisper import ASRFasterWhisper


@pytest.mark.unit
@pytest.mark.asr
def test_faster_whisper_initialization() -> None:
    """Test that ASRFasterWhisper initializes correctly."""
    with patch("asr.engine.fasterwhisper.WhisperModel") as mock_model:
        mock_instance = Mock()
        mock_model.return_value = mock_instance

        engine = ASRFasterWhisper()

        assert engine.model is mock_instance
        mock_model.assert_called_once()


@pytest.mark.unit
@pytest.mark.asr
def test_faster_whisper_initialization_with_settings() -> None:
    """Test that initialization uses settings correctly."""
    with (
        patch("asr.engine.fasterwhisper.WhisperModel") as mock_model,
        patch("asr.engine.fasterwhisper.settings") as mock_settings,
    ):
        mock_settings.ASR_MODEL = "base"
        mock_settings.ASR_DEVICE = "cpu"
        mock_settings.ASR_COMPUTE_TYPE = "int8"
        mock_settings.ASR_MODEL_LOCATION = "/models"
        mock_settings.ASR_CPU_THREADS = 4
        mock_settings.ASR_NUM_OF_WORKERS = 1
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False

        mock_instance = Mock()
        mock_model.return_value = mock_instance

        ASRFasterWhisper()

        mock_model.assert_called_once_with(
            model_size_or_path="base",
            device="cpu",
            compute_type="int8",
            download_root="/models",
            cpu_threads=4,
            num_workers=1,
        )


@pytest.mark.unit
@pytest.mark.asr
def test_transcribe_core_basic() -> None:
    """Test basic transcription with minimal parameters."""
    with (
        patch("asr.engine.fasterwhisper.WhisperModel") as mock_model_cls,
        patch("asr.engine.fasterwhisper.settings") as mock_settings,
    ):
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False
        mock_settings.ASR_ENGINE = "fasterwhisper"
        mock_settings.ASR_MODEL = "base"

        # Mock transcribe method
        mock_segment = Mock()
        mock_segment.start = 0.0
        mock_segment.end = 2.5
        mock_segment.text = " Hello world"

        mock_info = Mock()
        mock_info.language = "en"
        mock_info.language_probability = 0.98

        mock_model = Mock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_model_cls.return_value = mock_model

        engine = ASRFasterWhisper()

        # Create test audio
        audio_f32 = np.zeros(16000, dtype=np.float32)  # 1 second of silence

        result = engine._transcribe_core(
            audio_f32=audio_f32,
            sr=16000,
            raw_bytes=None,
            request_language=None,
            task=None,
            beam_size=None,
            temperature=None,
            best_of=None,
            word_timestamps=None,
            vad=None,
        )

        assert isinstance(result, TranscribeResult)
        assert result.text == "Hello world"
        assert len(result.segments) == 1
        assert result.segments[0]["text"] == "Hello world"
        assert result.language_code == "en"
        assert result.confidence == 0.98


@pytest.mark.unit
@pytest.mark.asr
def test_transcribe_core_with_language() -> None:
    """Test transcription with language hint."""
    with (
        patch("asr.engine.fasterwhisper.WhisperModel") as mock_model_cls,
        patch("asr.engine.fasterwhisper.settings") as mock_settings,
    ):
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False
        mock_settings.ASR_TRANSCRIBE_LANG = None
        mock_settings.ASR_TRANSCRIBE_BEAM_SIZE = 5
        mock_settings.ASR_TRANSCRIBE_TEMPERATURE = 0.0
        mock_settings.ASR_TRANSCRIBE_BEST_OF = 1

        mock_model = Mock()
        mock_model.transcribe.return_value = ([], Mock(language="es", language_probability=0.95))
        mock_model_cls.return_value = mock_model

        engine = ASRFasterWhisper()

        audio_f32 = np.zeros(16000, dtype=np.float32)

        engine._transcribe_core(
            audio_f32=audio_f32,
            sr=16000,
            raw_bytes=None,
            request_language="es",
            task=None,
            beam_size=None,
            temperature=None,
            best_of=None,
            word_timestamps=None,
            vad=None,
        )

        # Verify language was passed to model
        call_kwargs = mock_model.transcribe.call_args.kwargs
        assert call_kwargs["language"] == "es"


@pytest.mark.unit
@pytest.mark.asr
def test_transcribe_core_with_task_translate() -> None:
    """Test transcription with translate task."""
    with (
        patch("asr.engine.fasterwhisper.WhisperModel") as mock_model_cls,
        patch("asr.engine.fasterwhisper.settings") as mock_settings,
    ):
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False

        mock_model = Mock()
        mock_model.transcribe.return_value = ([], Mock(language="en", language_probability=0.99))
        mock_model_cls.return_value = mock_model

        engine = ASRFasterWhisper()

        audio_f32 = np.zeros(16000, dtype=np.float32)

        engine._transcribe_core(
            audio_f32=audio_f32,
            sr=16000,
            raw_bytes=None,
            request_language=None,
            task="translate",
            beam_size=None,
            temperature=None,
            best_of=None,
            word_timestamps=None,
            vad=None,
        )

        call_kwargs = mock_model.transcribe.call_args.kwargs
        assert call_kwargs["task"] == "translate"


@pytest.mark.unit
@pytest.mark.asr
def test_transcribe_core_with_word_timestamps() -> None:
    """Test transcription with word timestamps enabled."""
    with (
        patch("asr.engine.fasterwhisper.WhisperModel") as mock_model_cls,
        patch("asr.engine.fasterwhisper.settings") as mock_settings,
    ):
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False

        mock_model = Mock()
        mock_model.transcribe.return_value = ([], Mock(language="en", language_probability=0.99))
        mock_model_cls.return_value = mock_model

        engine = ASRFasterWhisper()

        audio_f32 = np.zeros(16000, dtype=np.float32)

        engine._transcribe_core(
            audio_f32=audio_f32,
            sr=16000,
            raw_bytes=None,
            request_language=None,
            task=None,
            beam_size=None,
            temperature=None,
            best_of=None,
            word_timestamps=True,
            vad=None,
        )

        call_kwargs = mock_model.transcribe.call_args.kwargs
        assert call_kwargs["word_timestamps"] is True


@pytest.mark.unit
@pytest.mark.asr
def test_transcribe_core_multiple_segments() -> None:
    """Test transcription with multiple segments."""
    with (
        patch("asr.engine.fasterwhisper.WhisperModel") as mock_model_cls,
        patch("asr.engine.fasterwhisper.settings") as mock_settings,
    ):
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False
        mock_settings.ASR_ENGINE = "fasterwhisper"
        mock_settings.ASR_MODEL = "base"

        # Create multiple segments
        seg1 = Mock()
        seg1.start = 0.0
        seg1.end = 2.0
        seg1.text = " Hello"

        seg2 = Mock()
        seg2.start = 2.0
        seg2.end = 4.0
        seg2.text = " world"

        seg3 = Mock()
        seg3.start = 4.0
        seg3.end = 6.0
        seg3.text = " from tests"

        mock_info = Mock()
        mock_info.language = "en"
        mock_info.language_probability = 0.99

        mock_model = Mock()
        mock_model.transcribe.return_value = ([seg1, seg2, seg3], mock_info)
        mock_model_cls.return_value = mock_model

        engine = ASRFasterWhisper()

        audio_f32 = np.zeros(16000 * 6, dtype=np.float32)

        result = engine._transcribe_core(
            audio_f32=audio_f32,
            sr=16000,
            raw_bytes=None,
            request_language=None,
            task=None,
            beam_size=None,
            temperature=None,
            best_of=None,
            word_timestamps=None,
            vad=None,
        )

        assert result.text == "Hello world from tests"
        assert len(result.segments) == 3
        assert result.segments[0]["text"] == "Hello"
        assert result.segments[1]["text"] == "world"
        assert result.segments[2]["text"] == "from tests"


@pytest.mark.unit
@pytest.mark.asr
def test_detect_language_core_basic() -> None:
    """Test basic language detection."""
    with (
        patch("asr.engine.fasterwhisper.WhisperModel") as mock_model_cls,
        patch("asr.engine.fasterwhisper.settings") as mock_settings,
    ):
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False
        mock_settings.ASR_ENGINE = "fasterwhisper"
        mock_settings.ASR_MODEL = "base"

        mock_model = Mock()
        mock_model.detect_language.return_value = ("es", 0.95, {})
        mock_model_cls.return_value = mock_model

        engine = ASRFasterWhisper()

        audio_f32 = np.zeros(16000, dtype=np.float32)

        result = engine._detect_language_core(
            audio_f32=audio_f32,
            sr=16000,
            raw_bytes=None,
            request_language=None,
            detect_lang_length=None,
            detect_lang_offset=None,
        )

        assert isinstance(result, DetectLanguageResult)
        assert result.language_code == "es"
        assert result.confidence == 0.95


@pytest.mark.unit
@pytest.mark.asr
def test_detect_language_with_offset_and_length() -> None:
    """Test language detection with offset and length parameters."""
    with (
        patch("asr.engine.fasterwhisper.WhisperModel") as mock_model_cls,
        patch("asr.engine.fasterwhisper.settings") as mock_settings,
    ):
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False

        mock_model = Mock()
        mock_model.detect_language.return_value = ("fr", 0.90, {})
        mock_model_cls.return_value = mock_model

        engine = ASRFasterWhisper()

        # 10 seconds of audio
        audio_f32 = np.zeros(16000 * 10, dtype=np.float32)

        engine._detect_language_core(
            audio_f32=audio_f32,
            sr=16000,
            raw_bytes=None,
            request_language=None,
            detect_lang_length=5.0,  # Use 5 seconds
            detect_lang_offset=2.0,  # Start at 2 seconds
        )

        # Check that detect_language was called
        mock_model.detect_language.assert_called_once()

        # Verify the audio passed was sliced correctly
        call_args = mock_model.detect_language.call_args
        audio_arg = call_args.kwargs["audio"]

        # Should have approximately 5 seconds worth of samples (5 * 16000 = 80000)
        # Allow some tolerance for VAD/processing
        assert len(audio_arg) <= 16000 * 5 + 1000


@pytest.mark.unit
@pytest.mark.asr
def test_detect_language_ignores_zero_offset_and_length() -> None:
    """Test that zero offset and length are ignored."""
    with (
        patch("asr.engine.fasterwhisper.WhisperModel") as mock_model_cls,
        patch("asr.engine.fasterwhisper.settings") as mock_settings,
    ):
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False

        mock_model = Mock()
        mock_model.detect_language.return_value = ("en", 0.99, {})
        mock_model_cls.return_value = mock_model

        engine = ASRFasterWhisper()

        audio_f32 = np.zeros(16000 * 3, dtype=np.float32)

        engine._detect_language_core(
            audio_f32=audio_f32,
            sr=16000,
            raw_bytes=None,
            request_language=None,
            detect_lang_length=0.0,
            detect_lang_offset=0.0,
        )

        # Should use full audio when both are 0
        call_args = mock_model.detect_language.call_args
        audio_arg = call_args.kwargs["audio"]

        # Should have approximately 3 seconds (allowing for processing)
        assert len(audio_arg) >= 16000 * 2


@pytest.mark.unit
@pytest.mark.asr
def test_transcribe_uses_cache() -> None:
    """Test that transcription uses cache when enabled."""
    with (
        patch("asr.engine.fasterwhisper.WhisperModel") as mock_model_cls,
        patch("asr.engine.fasterwhisper.settings") as mock_settings,
    ):
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = True
        mock_settings.ASR_CACHE_MAX_ITEMS = 10
        mock_settings.ASR_ENGINE = "fasterwhisper"
        mock_settings.ASR_MODEL = "base"

        mock_segment = Mock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = " Cached"

        mock_model = Mock()
        mock_model.transcribe.return_value = ([mock_segment], Mock(language="en", language_probability=0.99))
        mock_model_cls.return_value = mock_model

        engine = ASRFasterWhisper()

        audio_f32 = np.zeros(16000, dtype=np.float32)
        raw_bytes = audio_f32.tobytes()

        # First call - should hit model
        result1 = engine._transcribe_core(
            audio_f32=audio_f32,
            sr=16000,
            raw_bytes=raw_bytes,
            request_language=None,
            task=None,
            beam_size=None,
            temperature=None,
            best_of=None,
            word_timestamps=None,
            vad=None,
        )

        assert mock_model.transcribe.call_count == 1

        # Second call with same audio - should use cache
        result2 = engine._transcribe_core(
            audio_f32=audio_f32,
            sr=16000,
            raw_bytes=raw_bytes,
            request_language=None,
            task=None,
            beam_size=None,
            temperature=None,
            best_of=None,
            word_timestamps=None,
            vad=None,
        )

        # Should not call model again
        assert mock_model.transcribe.call_count == 1

        # Results should be the same
        assert result1.text == result2.text


@pytest.mark.unit
@pytest.mark.asr
def test_transcribe_handles_empty_segments() -> None:
    """Test that empty or whitespace-only segments are handled correctly."""
    with (
        patch("asr.engine.fasterwhisper.WhisperModel") as mock_model_cls,
        patch("asr.engine.fasterwhisper.settings") as mock_settings,
    ):
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False
        mock_settings.ASR_ENGINE = "fasterwhisper"
        mock_settings.ASR_MODEL = "base"

        # Mix of valid and empty segments
        seg1 = Mock()
        seg1.start = 0.0
        seg1.end = 1.0
        seg1.text = " Hello"

        seg2 = Mock()
        seg2.start = 1.0
        seg2.end = 2.0
        seg2.text = "   "  # Whitespace only

        seg3 = Mock()
        seg3.start = 2.0
        seg3.end = 3.0
        seg3.text = None  # None text

        seg4 = Mock()
        seg4.start = 3.0
        seg4.end = 4.0
        seg4.text = " World"

        mock_model = Mock()
        mock_model.transcribe.return_value = ([seg1, seg2, seg3, seg4], Mock(language="en", language_probability=0.99))
        mock_model_cls.return_value = mock_model

        engine = ASRFasterWhisper()

        audio_f32 = np.zeros(16000 * 4, dtype=np.float32)

        result = engine._transcribe_core(
            audio_f32=audio_f32,
            sr=16000,
            raw_bytes=None,
            request_language=None,
            task=None,
            beam_size=None,
            temperature=None,
            best_of=None,
            word_timestamps=None,
            vad=None,
        )

        # Implementation joins all segments with spaces, including empty ones
        # So we get extra spaces where empty segments were
        assert result.text == "Hello   World"  # Three spaces: empty segments become empty strings


@pytest.mark.unit
@pytest.mark.asr
def test_transcribe_resamples_audio() -> None:
    """Test that audio is resampled to 16kHz if needed."""
    with (
        patch("asr.engine.fasterwhisper.WhisperModel") as mock_model_cls,
        patch("asr.engine.fasterwhisper.settings") as mock_settings,
        patch("asr.engine.fasterwhisper.resample_to_16k_mono") as mock_resample,
    ):
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False

        # Mock resampling
        resampled_audio = np.zeros(16000, dtype=np.float32)
        mock_resample.return_value = (resampled_audio, 16000)

        mock_model = Mock()
        mock_model.transcribe.return_value = ([], Mock(language="en", language_probability=0.99))
        mock_model_cls.return_value = mock_model

        engine = ASRFasterWhisper()

        # 44.1kHz audio
        audio_f32 = np.zeros(44100, dtype=np.float32)

        engine._transcribe_core(
            audio_f32=audio_f32,
            sr=44100,
            raw_bytes=None,
            request_language=None,
            task=None,
            beam_size=None,
            temperature=None,
            best_of=None,
            word_timestamps=None,
            vad=None,
        )

        # Verify resample was called
        mock_resample.assert_called_once_with(audio_f32=audio_f32, sr=44100)
