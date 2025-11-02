"""Shared fixtures for TTS unit tests."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, Mock

import pytest

# =============================================================================
# Mock Coqui TTS Library - MUST be done before any TTS imports
# =============================================================================

# Mock the TTS library modules to prevent actual library loading
# This avoids the TTS license prompt and prevents heavy model downloads during unit tests
# We need to mock all TTS submodules that might be imported
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


@pytest.fixture
def mock_tts_engine():
    """Mock TTS engine for testing."""
    engine = Mock()
    engine.speech.return_value = b"fake audio data"
    engine.get_model.return_value = Mock(id="test-model", object="model")
    engine.list_models.return_value = Mock(
        object="list",
        data=[Mock(id="model-1"), Mock(id="model-2")],
    )
    engine.helper_return_audio_file.return_value = Mock(
        body=b"fake audio",
        media_type="audio/mpeg",
    )
    engine.helper_return_sse_stream.return_value = Mock(
        media_type="text/event-stream",
    )
    return engine
