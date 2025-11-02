"""Unit tests for ASR OpenAI Whisper (PyTorch) engine."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from asr.engine.base import DetectLanguageResult, TranscribeResult
from asr.engine.whisper import ASRWhisperTorch, _pick_device


@pytest.mark.unit
@pytest.mark.asr
def test_pick_device_prefers_mps() -> None:
    """Test that _pick_device prefers MPS on Apple Silicon."""
    with (
        patch("torch.backends.mps.is_available", return_value=True),
        patch("torch.cuda.is_available", return_value=False),
    ):
        assert _pick_device() == "mps"


@pytest.mark.unit
@pytest.mark.asr
def test_pick_device_falls_back_to_cuda() -> None:
    """Test that _pick_device falls back to CUDA."""
    with (
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=True),
    ):
        assert _pick_device() == "cuda"


@pytest.mark.unit
@pytest.mark.asr
def test_pick_device_falls_back_to_cpu() -> None:
    """Test that _pick_device falls back to CPU."""
    with (
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=False),
    ):
        assert _pick_device() == "cpu"


@pytest.mark.unit
@pytest.mark.asr
def test_pick_device_handles_mps_exception() -> None:
    """Test that _pick_device handles MPS exception gracefully."""
    with (
        patch("torch.backends.mps.is_available", side_effect=Exception("MPS not supported")),
        patch("torch.cuda.is_available", return_value=True),
    ):
        assert _pick_device() == "cuda"


@pytest.mark.unit
@pytest.mark.asr
def test_whisper_torch_initialization_cpu() -> None:
    """Test ASRWhisperTorch initialization on CPU."""
    with (
        patch("asr.engine.whisper.whisper.load_model") as mock_load,
        patch("asr.engine.whisper._pick_device", return_value="cpu"),
        patch("asr.engine.whisper.settings") as mock_settings,
    ):

        mock_settings.ASR_MODEL = "base"
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False

        mock_model = Mock()
        mock_load.return_value = mock_model

        engine = ASRWhisperTorch()

        assert engine.device == "cpu"
        assert engine.fp16 is False
        mock_load.assert_called_once_with("base", device="cpu")


@pytest.mark.unit
@pytest.mark.asr
def test_whisper_torch_initialization_cuda() -> None:
    """Test ASRWhisperTorch initialization on CUDA."""
    with (
        patch("asr.engine.whisper.whisper.load_model") as mock_load,
        patch("asr.engine.whisper._pick_device", return_value="cuda"),
        patch("asr.engine.whisper.settings") as mock_settings,
    ):

        mock_settings.ASR_MODEL = "base"
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False

        mock_model = Mock()
        mock_load.return_value = mock_model

        engine = ASRWhisperTorch()

        assert engine.device == "cuda"
        assert engine.fp16 is True  # fp16 on CUDA
        mock_load.assert_called_once_with("base", device="cuda")


@pytest.mark.unit
@pytest.mark.asr
def test_whisper_torch_initialization_mps() -> None:
    """Test ASRWhisperTorch initialization on MPS."""
    with (
        patch("asr.engine.whisper.whisper.load_model") as mock_load,
        patch("asr.engine.whisper._pick_device", return_value="mps"),
        patch("asr.engine.whisper.settings") as mock_settings,
    ):

        mock_settings.ASR_MODEL = "base"
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False

        mock_model = Mock()
        mock_load.return_value = mock_model

        engine = ASRWhisperTorch()

        assert engine.device == "mps"
        assert engine.fp16 is False  # fp32 on MPS
        mock_load.assert_called_once_with("base", device="mps")


@pytest.mark.unit
@pytest.mark.asr
def test_whisper_torch_mps_fallback_to_cpu() -> None:
    """Test that MPS falls back to CPU on NotImplementedError."""
    with (
        patch("asr.engine.whisper.whisper.load_model") as mock_load,
        patch("asr.engine.whisper._pick_device", return_value="mps"),
        patch("asr.engine.whisper.settings") as mock_settings,
    ):

        mock_settings.ASR_MODEL = "base"
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False

        # First call (MPS) raises NotImplementedError, second call (CPU) succeeds
        mock_model = Mock()
        mock_load.side_effect = [NotImplementedError("MPS kernel missing"), mock_model]

        engine = ASRWhisperTorch()

        # Should have fallen back to CPU
        assert engine.device == "cpu"
        assert engine.fp16 is False
        assert mock_load.call_count == 2
        mock_load.assert_any_call("base", device="mps")
        mock_load.assert_any_call("base", device="cpu")


@pytest.mark.unit
@pytest.mark.asr
def test_transcribe_core_basic() -> None:
    """Test basic transcription with minimal parameters."""
    with (
        patch("asr.engine.whisper.whisper.load_model") as mock_load,
        patch("asr.engine.whisper._pick_device", return_value="cpu"),
        patch("asr.engine.whisper.settings") as mock_settings,
    ):

        mock_settings.ASR_MODEL = "base"
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False
        mock_settings.ASR_ENGINE = "whisper"
        mock_settings.ASR_TRANSCRIBE_BEAM_SIZE = 5
        mock_settings.ASR_TRANSCRIBE_TEMPERATURE = 0.0
        mock_settings.ASR_TRANSCRIBE_BEST_OF = 1

        mock_model = Mock()
        mock_model.transcribe.return_value = {
            "text": "Hello world",
            "segments": [{"start": 0.0, "end": 2.5, "text": " Hello world"}],
            "language": "en",
        }
        mock_load.return_value = mock_model

        engine = ASRWhisperTorch()

        audio_f32 = np.zeros(16000, dtype=np.float32)

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
        assert result.language_code == "en"


@pytest.mark.unit
@pytest.mark.asr
def test_transcribe_core_with_language() -> None:
    """Test transcription with language hint."""
    with (
        patch("asr.engine.whisper.whisper.load_model") as mock_load,
        patch("asr.engine.whisper._pick_device", return_value="cpu"),
        patch("asr.engine.whisper.settings") as mock_settings,
    ):

        mock_settings.ASR_MODEL = "base"
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False
        mock_settings.ASR_TRANSCRIBE_LANG = None

        mock_model = Mock()
        mock_model.transcribe.return_value = {"text": "Hola mundo", "segments": [], "language": "es"}
        mock_load.return_value = mock_model

        engine = ASRWhisperTorch()

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

        # Verify language was passed
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "es"


@pytest.mark.unit
@pytest.mark.asr
def test_transcribe_core_translate_task() -> None:
    """Test transcription with translate task."""
    with (
        patch("asr.engine.whisper.whisper.load_model") as mock_load,
        patch("asr.engine.whisper._pick_device", return_value="cpu"),
        patch("asr.engine.whisper.settings") as mock_settings,
    ):

        mock_settings.ASR_MODEL = "base"
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False

        mock_model = Mock()
        mock_model.transcribe.return_value = {"text": "Hello world", "segments": [], "language": "en"}
        mock_load.return_value = mock_model

        engine = ASRWhisperTorch()

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

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["task"] == "translate"


@pytest.mark.unit
@pytest.mark.asr
def test_transcribe_core_with_beam_size() -> None:
    """Test transcription with custom beam size."""
    with (
        patch("asr.engine.whisper.whisper.load_model") as mock_load,
        patch("asr.engine.whisper._pick_device", return_value="cpu"),
        patch("asr.engine.whisper.settings") as mock_settings,
    ):

        mock_settings.ASR_MODEL = "base"
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False

        mock_model = Mock()
        mock_model.transcribe.return_value = {"text": "Test", "segments": [], "language": "en"}
        mock_load.return_value = mock_model

        engine = ASRWhisperTorch()

        audio_f32 = np.zeros(16000, dtype=np.float32)

        engine._transcribe_core(
            audio_f32=audio_f32,
            sr=16000,
            raw_bytes=None,
            request_language=None,
            task=None,
            beam_size=10,
            temperature=None,
            best_of=None,
            word_timestamps=None,
            vad=None,
        )

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["beam_size"] == 10


@pytest.mark.unit
@pytest.mark.asr
def test_transcribe_core_with_temperature() -> None:
    """Test transcription with custom temperature."""
    with (
        patch("asr.engine.whisper.whisper.load_model") as mock_load,
        patch("asr.engine.whisper._pick_device", return_value="cpu"),
        patch("asr.engine.whisper.settings") as mock_settings,
    ):

        mock_settings.ASR_MODEL = "base"
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False

        mock_model = Mock()
        mock_model.transcribe.return_value = {"text": "Test", "segments": [], "language": "en"}
        mock_load.return_value = mock_model

        engine = ASRWhisperTorch()

        audio_f32 = np.zeros(16000, dtype=np.float32)

        engine._transcribe_core(
            audio_f32=audio_f32,
            sr=16000,
            raw_bytes=None,
            request_language=None,
            task=None,
            beam_size=None,
            temperature=0.5,
            best_of=None,
            word_timestamps=None,
            vad=None,
        )

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["temperature"] == 0.5


@pytest.mark.unit
@pytest.mark.asr
def test_transcribe_core_multiple_segments() -> None:
    """Test transcription with multiple segments."""
    with (
        patch("asr.engine.whisper.whisper.load_model") as mock_load,
        patch("asr.engine.whisper._pick_device", return_value="cpu"),
        patch("asr.engine.whisper.settings") as mock_settings,
    ):

        mock_settings.ASR_MODEL = "base"
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False
        mock_settings.ASR_ENGINE = "whisper"

        mock_model = Mock()
        mock_model.transcribe.return_value = {
            "text": "Hello world from tests",
            "segments": [
                {"start": 0.0, "end": 2.0, "text": " Hello"},
                {"start": 2.0, "end": 4.0, "text": " world"},
                {"start": 4.0, "end": 6.0, "text": " from tests"},
            ],
            "language": "en",
        }
        mock_load.return_value = mock_model

        engine = ASRWhisperTorch()

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


@pytest.mark.unit
@pytest.mark.asr
def test_transcribe_handles_empty_segments() -> None:
    """Test that empty segments are filtered out."""
    with (
        patch("asr.engine.whisper.whisper.load_model") as mock_load,
        patch("asr.engine.whisper._pick_device", return_value="cpu"),
        patch("asr.engine.whisper.settings") as mock_settings,
    ):

        mock_settings.ASR_MODEL = "base"
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False
        mock_settings.ASR_ENGINE = "whisper"

        mock_model = Mock()
        mock_model.transcribe.return_value = {
            "text": "Hello World",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": " Hello"},
                {"start": 1.0, "end": 2.0, "text": "   "},  # Whitespace
                {"start": 2.0, "end": 3.0, "text": ""},  # Empty
                {"start": 3.0, "end": 4.0, "text": " World"},
            ],
            "language": "en",
        }
        mock_load.return_value = mock_model

        engine = ASRWhisperTorch()

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

        # Empty segments should be filtered
        assert result.text == "Hello World"
        assert len(result.segments) == 2


@pytest.mark.unit
@pytest.mark.asr
def test_detect_language_core_basic() -> None:
    """Test basic language detection."""
    with (
        patch("asr.engine.whisper.whisper.load_model") as mock_load,
        patch("asr.engine.whisper.whisper.log_mel_spectrogram") as mock_mel,
        patch("asr.engine.whisper._pick_device", return_value="cpu"),
        patch("asr.engine.whisper.settings") as mock_settings,
    ):

        mock_settings.ASR_MODEL = "base"
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False
        mock_settings.ASR_ENGINE = "whisper"

        mock_mel_data = Mock()
        mock_mel.return_value = mock_mel_data

        mock_model = Mock()
        mock_model.detect_language.return_value = ("es", 0.95)
        mock_load.return_value = mock_model

        engine = ASRWhisperTorch()

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
    """Test language detection with offset and length."""
    with (
        patch("asr.engine.whisper.whisper.load_model") as mock_load,
        patch("asr.engine.whisper.whisper.log_mel_spectrogram") as mock_mel,
        patch("asr.engine.whisper._pick_device", return_value="cpu"),
        patch("asr.engine.whisper.settings") as mock_settings,
    ):

        mock_settings.ASR_MODEL = "base"
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False

        mock_mel_data = Mock()
        mock_mel.return_value = mock_mel_data

        mock_model = Mock()
        mock_model.detect_language.return_value = ("fr", 0.90)
        mock_load.return_value = mock_model

        engine = ASRWhisperTorch()

        # 10 seconds of audio
        audio_f32 = np.zeros(16000 * 10, dtype=np.float32)

        engine._detect_language_core(
            audio_f32=audio_f32,
            sr=16000,
            raw_bytes=None,
            request_language=None,
            detect_lang_length=5.0,
            detect_lang_offset=2.0,
        )

        # Check that mel spectrogram was called
        mock_mel.assert_called_once()

        # Get the audio passed to mel spectrogram
        audio_arg = mock_mel.call_args[0][0]

        # Should be approximately 5 seconds (allowing for tolerance)
        assert len(audio_arg) <= 16000 * 5 + 1000


@pytest.mark.unit
@pytest.mark.asr
def test_transcribe_uses_cache() -> None:
    """Test that transcription uses cache when enabled."""
    with (
        patch("asr.engine.whisper.whisper.load_model") as mock_load,
        patch("asr.engine.whisper._pick_device", return_value="cpu"),
        patch("asr.engine.whisper.settings") as mock_settings,
    ):

        mock_settings.ASR_MODEL = "base"
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = True
        mock_settings.ASR_CACHE_MAX_ITEMS = 10
        mock_settings.ASR_ENGINE = "whisper"

        mock_model = Mock()
        mock_model.transcribe.return_value = {"text": "Cached result", "segments": [], "language": "en"}
        mock_load.return_value = mock_model

        engine = ASRWhisperTorch()

        audio_f32 = np.zeros(16000, dtype=np.float32)
        raw_bytes = audio_f32.tobytes()

        # First call
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

        # Second call with same audio should use cache
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
        assert result1.text == result2.text


@pytest.mark.unit
@pytest.mark.asr
def test_transcribe_fp16_on_cuda() -> None:
    """Test that fp16 is used on CUDA."""
    with (
        patch("asr.engine.whisper.whisper.load_model") as mock_load,
        patch("asr.engine.whisper._pick_device", return_value="cuda"),
        patch("asr.engine.whisper.settings") as mock_settings,
    ):

        mock_settings.ASR_MODEL = "base"
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False

        mock_model = Mock()
        mock_model.transcribe.return_value = {"text": "Test", "segments": [], "language": "en"}
        mock_load.return_value = mock_model

        engine = ASRWhisperTorch()

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
            word_timestamps=None,
            vad=None,
        )

        # Verify fp16=True was passed for CUDA
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["fp16"] is True


@pytest.mark.unit
@pytest.mark.asr
def test_transcribe_fp32_on_mps() -> None:
    """Test that fp32 is used on MPS."""
    with (
        patch("asr.engine.whisper.whisper.load_model") as mock_load,
        patch("asr.engine.whisper._pick_device", return_value="mps"),
        patch("asr.engine.whisper.settings") as mock_settings,
    ):

        mock_settings.ASR_MODEL = "base"
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False

        mock_model = Mock()
        mock_model.transcribe.return_value = {"text": "Test", "segments": [], "language": "en"}
        mock_load.return_value = mock_model

        engine = ASRWhisperTorch()

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
            word_timestamps=None,
            vad=None,
        )

        # Verify fp16=False was passed for MPS
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["fp16"] is False


@pytest.mark.unit
@pytest.mark.asr
def test_transcribe_resamples_audio() -> None:
    """Test that audio is resampled to 16kHz."""
    with (
        patch("asr.engine.whisper.whisper.load_model") as mock_load,
        patch("asr.engine.whisper._pick_device", return_value="cpu"),
        patch("asr.engine.whisper.resample_to_16k_mono") as mock_resample,
        patch("asr.engine.whisper.settings") as mock_settings,
    ):

        mock_settings.ASR_MODEL = "base"
        mock_settings.ASR_VAD_ENABLED = False
        mock_settings.ASR_CACHE_ENABLED = False

        resampled = np.zeros(16000, dtype=np.float32)
        mock_resample.return_value = (resampled, 16000)

        mock_model = Mock()
        mock_model.transcribe.return_value = {"text": "Test", "segments": [], "language": "en"}
        mock_load.return_value = mock_model

        engine = ASRWhisperTorch()

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
