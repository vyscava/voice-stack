"""
Pytest configuration and shared fixtures for voice-stack tests.

This file provides:
- Shared fixtures for ASR and TTS services
- Mock objects for external dependencies
- Test utilities and helpers
"""

from __future__ import annotations

import io
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

# Disable debugpy for all tests by setting environment variable before any imports
os.environ["DEBUGPY_ENABLE"] = "false"


# =============================================================================
# Coqui TTS License Agreement Setup
# =============================================================================


def pytest_configure(config: Any) -> None:
    """
    Configure pytest with custom markers and create Coqui TTS license agreement.

    This runs before test collection to prevent TTS library from prompting
    for license agreement during imports.
    """
    # Configure custom markers
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "asr: ASR service tests")
    config.addinivalue_line("markers", "tts: TTS service tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "requires_gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "requires_models: Tests requiring model downloads")

    # Create Coqui TTS license agreement file to prevent interactive prompt
    # The TTS library checks for this file before prompting
    try:
        from pathlib import Path

        # Determine TTS model cache directory
        tts_cache = Path.home() / ".local" / "share" / "tts"

        # Common model directories that might trigger the prompt
        model_dirs = [
            tts_cache / "tts_models--multilingual--multi-dataset--xtts_v2",
            tts_cache,  # Also create in root for safety
        ]

        for model_dir in model_dirs:
            model_dir.mkdir(parents=True, exist_ok=True)
            tos_file = model_dir / "tos_agreed.txt"

            if not tos_file.exists():
                tos_file.write_text("1\n")  # "1" indicates agreement

    except Exception as e:
        # Don't fail test collection if TOS file creation fails
        # Tests will handle TTS initialization errors appropriately
        import warnings

        warnings.warn(f"Could not create Coqui TTS license agreement file: {e}", stacklevel=2)


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


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
# Settings Fixtures
# =============================================================================


@pytest.fixture
def mock_settings() -> Mock:
    """Return mock settings object."""
    settings = Mock()
    settings.PROJECT_NAME = "Voice Stack Test"
    settings.API_V1_STR = "/v1"
    settings.CORS_ORIGINS = ""
    settings.LOG_LEVEL = "INFO"
    settings.DEBUGPY_ENABLE = False

    # ASR settings
    settings.ASR_DEVICE = "cpu"
    settings.ASR_MODEL = "base"
    settings.ASR_ENGINE = "faster-whisper"
    settings.ASR_VAD_ENABLED = True
    settings.ASR_LANGUAGE = None
    settings.ASR_IDLE_TIMEOUT_MINUTES = 0  # Disable for tests
    settings.ASR_MAX_CONCURRENT_REQUESTS = 2

    # TTS settings
    settings.TTS_DEVICE = "cpu"
    settings.TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
    settings.TTS_VOICE_DIR = "./voices"
    settings.TTS_AUTO_LANG = True
    settings.TTS_MAX_CHARS = 180
    settings.TTS_SAMPLE_RATE = 24000
    settings.TTS_IDLE_TIMEOUT_MINUTES = 0  # Disable for tests
    settings.TTS_MAX_CONCURRENT_REQUESTS = 2

    # Resource management settings - set high for tests to avoid 503 errors
    settings.MEMORY_THRESHOLD_PERCENT = 99
    settings.SWAP_THRESHOLD_PERCENT = 99
    settings.MAX_UPLOAD_SIZE_MB = 100

    return settings


# =============================================================================
# Engine Mocks
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
# FastAPI Test Clients
# =============================================================================


@pytest.fixture
def asr_client(mock_asr_engine: Mock, mock_settings: Mock, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Return FastAPI test client for ASR service."""
    # Mock psutil to report low memory usage (prevents 503 errors in tests)
    mock_mem = Mock()
    mock_mem.percent = 50.0  # 50% memory usage
    mock_swap = Mock()
    mock_swap.percent = 10.0  # 10% swap usage
    monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)
    monkeypatch.setattr("psutil.swap_memory", lambda: mock_swap)

    # Mock the engine factory to return our mock engine
    monkeypatch.setattr("asr.engine_factory._engine", mock_asr_engine)
    monkeypatch.setattr("asr.engine_factory.get_audio_engine", lambda: mock_asr_engine)

    # Mock acquire_engine and release_engine for new concurrency control
    async def mock_acquire_engine():
        return mock_asr_engine

    def mock_release_engine():
        pass

    monkeypatch.setattr("asr.engine_factory.acquire_engine", mock_acquire_engine)
    monkeypatch.setattr("asr.engine_factory.release_engine", mock_release_engine)

    # Mock settings to use test values
    monkeypatch.setattr("asr.app.settings", mock_settings)
    monkeypatch.setattr("core.settings.get_settings", lambda: mock_settings)

    # Patch the acquire/release functions in the endpoint modules
    monkeypatch.setattr("asr.api.api_v1.endpoints.openai.acquire_engine", mock_acquire_engine)
    monkeypatch.setattr("asr.api.api_v1.endpoints.openai.release_engine", mock_release_engine)
    monkeypatch.setattr("asr.api.api_v1.endpoints.openai.get_audio_engine", lambda: mock_asr_engine)

    monkeypatch.setattr("asr.api.api_v1.endpoints.bazarr.acquire_engine", mock_acquire_engine)
    monkeypatch.setattr("asr.api.api_v1.endpoints.bazarr.release_engine", mock_release_engine)
    monkeypatch.setattr("asr.api.api_v1.endpoints.bazarr.get_audio_engine", lambda: mock_asr_engine)

    # Import after patching to ensure the mock is used
    from asr.app import app

    # The TestClient with raise_server_exceptions=False allows us to see proper error responses
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def tts_client(mock_tts_engine: Mock, mock_settings: Mock, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Return FastAPI test client for TTS service."""
    # Mock psutil to report low memory usage (prevents 503 errors in tests)
    mock_mem = Mock()
    mock_mem.percent = 50.0  # 50% memory usage
    mock_swap = Mock()
    mock_swap.percent = 10.0  # 10% swap usage
    monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)
    monkeypatch.setattr("psutil.swap_memory", lambda: mock_swap)

    # Mock the engine factory to return our mock engine
    monkeypatch.setattr("tts.engine_factory._engine", mock_tts_engine)
    monkeypatch.setattr("tts.engine_factory.get_audio_engine", lambda: mock_tts_engine)

    # Mock acquire_engine and release_engine for new concurrency control
    async def mock_acquire_engine():
        return mock_tts_engine

    def mock_release_engine():
        pass

    monkeypatch.setattr("tts.engine_factory.acquire_engine", mock_acquire_engine)
    monkeypatch.setattr("tts.engine_factory.release_engine", mock_release_engine)

    # Mock settings to use test values
    monkeypatch.setattr("tts.app.settings", mock_settings)
    monkeypatch.setattr("core.settings.get_settings", lambda: mock_settings)

    # Import after patching to ensure the mock is used
    from tts.app import app

    with TestClient(app) as client:
        yield client


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
