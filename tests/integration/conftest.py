"""
Pytest configuration and shared fixtures for integration tests.

This file provides:
- Integration test fixtures with real FastAPI app lifespan
- Mock engines injected at the factory level
- Shared test data fixtures
"""

from __future__ import annotations

import io
import sys
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
from fastapi.testclient import TestClient

# =============================================================================
# Mock Coqui TTS Library - MUST be done before any TTS imports
# =============================================================================

# Mock the TTS library modules to prevent actual library loading during integration tests
# This avoids loading heavy ML models while still testing the full app lifespan
_tts_modules = [
    "TTS",
    "TTS.api",
    "TTS.config",
    "TTS.config.shared_configs",
    "TTS.tts",
    "TTS.tts.configs",
    "TTS.tts.configs.xtts_config",
    "TTS.tts.layers",
    "TTS.tts.layers.xtts",
    "TTS.tts.layers.xtts.tokenizer",
    "TTS.tts.models",
    "TTS.tts.models.xtts",
    "TTS.utils",
    "TTS.utils.audio",
    "TTS.utils.manage",
    "TTS.utils.synthesizer",
]

for module in _tts_modules:
    if module not in sys.modules:
        sys.modules[module] = MagicMock()

# =============================================================================
# Test Data Fixtures (reuse from unit tests)
# =============================================================================


@pytest.fixture
def sample_audio_bytes() -> bytes:
    """Return sample audio data (WAV format PCM16, 16kHz, mono)."""
    # Generate a simple WAV header + silent audio
    # WAV header for 16kHz, mono, PCM16, 1 second
    sample_rate = 16000
    num_samples = sample_rate  # 1 second
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align

    header = bytearray()
    # RIFF header
    header.extend(b"RIFF")
    header.extend((36 + data_size).to_bytes(4, "little"))
    header.extend(b"WAVE")
    # fmt chunk
    header.extend(b"fmt ")
    header.extend((16).to_bytes(4, "little"))  # chunk size
    header.extend((1).to_bytes(2, "little"))  # audio format (PCM)
    header.extend(num_channels.to_bytes(2, "little"))
    header.extend(sample_rate.to_bytes(4, "little"))
    header.extend(byte_rate.to_bytes(4, "little"))
    header.extend(block_align.to_bytes(2, "little"))
    header.extend(bits_per_sample.to_bytes(2, "little"))
    # data chunk
    header.extend(b"data")
    header.extend(data_size.to_bytes(4, "little"))

    # Silent audio data (all zeros)
    audio_data = bytes(data_size)

    return bytes(header) + audio_data


@pytest.fixture
def sample_text() -> str:
    """Return sample text for TTS testing."""
    return "Hello, this is a test of the text-to-speech system."


# =============================================================================
# Engine Mocks (reuse from unit tests)
# =============================================================================


@pytest.fixture
def mock_asr_result() -> dict[str, Any]:
    """Return mock ASR transcription result."""
    return {
        "text": "This is a test transcription.",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.5,
                "text": "This is a test transcription.",
                "tokens": [100, 200, 300],
                "temperature": 0.0,
                "avg_logprob": -0.5,
                "compression_ratio": 1.2,
                "no_speech_prob": 0.01,
            }
        ],
        "language": "en",
        "language_probability": 0.98,
        "duration": 2.5,
        "model": "base",
        "vad_used": True,
    }


@pytest.fixture
def mock_asr_engine(mock_asr_result: dict[str, Any]) -> Mock:
    """Return mock ASR engine with ASRBase methods."""
    from fastapi.responses import JSONResponse, PlainTextResponse

    from asr.engine.base import ASRBase, DetectLanguageResult, TranscribeResult
    from asr.schemas.audio_engine import Output

    # Create proper TranscribeResult object
    transcribe_result = TranscribeResult(
        text=mock_asr_result["text"],
        segments=[
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            }
            for seg in mock_asr_result["segments"]
        ],
        language_code="en",
        language_name="English",
        confidence=0.98,
        duration_input_s=2.5,
        duration_after_vad_s=2.5,
        processing_ms=100,
        asr_ms=80,
        vad_used=True,
        engine="faster-whisper",
        model="base",
        vad_segments=[],
    )

    # Create proper DetectLanguageResult object
    detect_lang_result = DetectLanguageResult(
        detected_language="English",
        language_code="en",
        language_name="English",
        confidence=0.98,
        duration_input_s=2.5,
        duration_after_vad_s=2.5,
        processing_ms=50,
        asr_ms=40,
        vad_used=True,
        engine="faster-whisper",
        model="base",
        vad_segments=[],
    )

    # Create a mock that inherits helper methods from ASRBase
    engine = Mock(spec=ASRBase)

    # Set up transcribe_file to return our result
    engine.transcribe_file.return_value = transcribe_result

    # Set up detect_language_file to return our result
    engine.detect_language_file.return_value = detect_lang_result

    # Mock list_models method - use proper format
    engine.list_models.return_value = {
        "object": "list",
        "data": [
            {
                "id": "whisper-1",
                "object": "model",
                "owned_by": "voice-stack",
                "supported_tasks": ["transcriptions", "translations"],
            },
            {
                "id": "whisper-1-base",
                "object": "model",
                "parent": "whisper-1",
                "owned_by": "voice-stack",
                "supported_tasks": ["transcriptions", "translations"],
                "active": True,
            },
        ],
    }

    # Mock helper_write_output to return proper responses based on output format
    def mock_helper_write_output(file, result, output, max_line_len=42):
        from fastapi.responses import StreamingResponse

        headers = {"Asr-Engine": "faster-whisper"}

        if output == Output.JSON:
            return JSONResponse(
                content={
                    "language": result.language_code,
                    "segments": result.segments,
                },
                headers=headers,
            )
        elif output == Output.SRT:
            content = result.to_srt(max_line_len=max_line_len)
            return StreamingResponse(
                content=iter([content]),
                media_type="application/x-subrip",
                headers=headers,
            )
        elif output == Output.VTT:
            content = result.to_vtt(max_line_len=max_line_len)
            return StreamingResponse(
                content=iter([content]),
                media_type="text/vtt; charset=utf-8",
                headers=headers,
            )
        elif output == Output.TSV:
            content = result.to_tsv()
            return StreamingResponse(
                content=iter([content]),
                media_type="text/tab-separated-values; charset=utf-8",
                headers=headers,
            )
        elif output == Output.JSONL:
            content = result.to_segments_jsonl()
            return StreamingResponse(
                content=iter([content]),
                media_type="application/jsonl; charset=utf-8",
                headers=headers,
            )
        else:  # TXT or default
            content = result.to_txt()
            return PlainTextResponse(
                content=content,
                headers=headers,
            )

    engine.helper_write_output.side_effect = mock_helper_write_output

    return engine


@pytest.fixture
def mock_tts_engine() -> Mock:
    """Return mock TTS engine."""
    engine = Mock()

    # Mock speech method
    audio_bytes = b"fake_audio_data_mp3"
    engine.speech.return_value = audio_bytes

    # Mock list_models method
    engine.list_models.return_value = {
        "object": "list",
        "data": [
            {"id": "xtts_v2", "object": "model"},
        ],
    }

    # Mock get_model method
    engine.get_model.return_value = {
        "id": "xtts_v2",
        "object": "model",
        "owned_by": "coqui",
    }

    return engine


# =============================================================================
# Integration Test Clients with Real App Lifespan
# =============================================================================


@pytest.fixture
def asr_integration_client(mock_asr_engine: Mock, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """
    Return FastAPI test client for ASR service with real app lifespan.

    This fixture:
    - Injects the mock engine at the factory level
    - Allows the app to go through its full lifespan initialization
    - Ensures routes are properly registered during startup
    """
    # Mock settings to use test values
    from unittest.mock import Mock as MockClass

    mock_settings = MockClass()
    mock_settings.PROJECT_NAME = "Voice Stack Test"
    mock_settings.API_V1_STR = "/v1"
    mock_settings.CORS_ORIGINS = ""
    mock_settings.LOG_LEVEL = "INFO"
    mock_settings.DEBUGPY_ENABLE = False

    # ASR settings
    mock_settings.ASR_DEVICE = "cpu"
    mock_settings.ASR_MODEL = "base"
    mock_settings.ASR_ENGINE = "faster-whisper"
    mock_settings.ASR_VAD_ENABLED = True
    mock_settings.ASR_LANGUAGE = None
    mock_settings.ASR_IDLE_TIMEOUT_MINUTES = 0  # Disable for tests
    mock_settings.ASR_MAX_CONCURRENT_REQUESTS = 2

    # Resource management settings
    mock_settings.MEMORY_THRESHOLD_PERCENT = 90
    mock_settings.SWAP_THRESHOLD_PERCENT = 80
    mock_settings.MAX_UPLOAD_SIZE_MB = 100

    monkeypatch.setattr("asr.app.settings", mock_settings)
    monkeypatch.setattr("core.settings.get_settings", lambda: mock_settings)

    # Inject mock engine at the factory level BEFORE importing the app
    # This ensures the engine is available when the app lifespan runs
    import asr.engine_factory

    asr.engine_factory.set_audio_engine(mock_asr_engine)

    # Now import the app - it will use our injected mock engine
    from asr.app import app

    # Create test client with real lifespan
    # raise_server_exceptions=False allows us to see proper error responses
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    # Cleanup: reset the engine after tests
    asr.engine_factory._engine = None


@pytest.fixture
def tts_integration_client(mock_tts_engine: Mock, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """
    Return FastAPI test client for TTS service with real app lifespan.

    This fixture:
    - Injects the mock engine at the factory level
    - Allows the app to go through its full lifespan initialization
    - Ensures routes are properly registered during startup
    """
    # Mock settings to use test values
    from unittest.mock import Mock as MockClass

    mock_settings = MockClass()
    mock_settings.PROJECT_NAME = "Voice Stack Test"
    mock_settings.API_V1_STR = "/v1"
    mock_settings.CORS_ORIGINS = ""
    mock_settings.LOG_LEVEL = "INFO"
    mock_settings.DEBUGPY_ENABLE = False

    # TTS settings
    mock_settings.TTS_DEVICE = "cpu"
    mock_settings.TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
    mock_settings.TTS_VOICE_DIR = "./voices"
    mock_settings.TTS_AUTO_LANG = True
    mock_settings.TTS_MAX_CHARS = 180
    mock_settings.TTS_SAMPLE_RATE = 24000
    mock_settings.TTS_IDLE_TIMEOUT_MINUTES = 0  # Disable for tests
    mock_settings.TTS_MAX_CONCURRENT_REQUESTS = 2

    # Resource management settings
    mock_settings.MEMORY_THRESHOLD_PERCENT = 90
    mock_settings.SWAP_THRESHOLD_PERCENT = 80
    mock_settings.MAX_UPLOAD_SIZE_MB = 100

    monkeypatch.setattr("tts.app.settings", mock_settings)
    monkeypatch.setattr("core.settings.get_settings", lambda: mock_settings)

    # Inject mock engine at the factory level BEFORE importing the app
    # This ensures the engine is available when the app lifespan runs
    import tts.engine_factory

    tts.engine_factory.set_audio_engine(mock_tts_engine)

    # Now import the app - it will use our injected mock engine
    from tts.app import app

    # Create test client with real lifespan
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    # Cleanup: reset the engine after tests
    tts.engine_factory._engine = None


# =============================================================================
# Utility Functions
# =============================================================================


@pytest.fixture
def create_upload_file() -> Any:
    """Factory fixture to create UploadFile objects for testing."""

    def _create(filename: str, content: bytes, content_type: str = "audio/wav"):
        from fastapi import UploadFile

        file_obj = io.BytesIO(content)
        return UploadFile(filename=filename, file=file_obj)

    return _create
