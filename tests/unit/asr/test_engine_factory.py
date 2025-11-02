"""Unit tests for ASR engine factory."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from asr.engine.base import ASRBase


@pytest.mark.unit
@pytest.mark.asr
def test_get_audio_engine_returns_singleton() -> None:
    """Test that get_audio_engine returns the same instance."""
    from asr import engine_factory

    # Reset engine to start fresh
    engine_factory._engine = None

    with patch("asr.engine_factory._create_engine_from_settings") as mock_create:
        mock_engine = Mock(spec=ASRBase)
        mock_create.return_value = mock_engine

        # First call creates engine
        engine1 = engine_factory.get_audio_engine()
        assert engine1 is mock_engine
        assert mock_create.call_count == 1

        # Second call returns same instance
        engine2 = engine_factory.get_audio_engine()
        assert engine2 is engine1
        assert mock_create.call_count == 1  # Not called again


@pytest.mark.unit
@pytest.mark.asr
def test_reset_audio_engine_recreates_instance() -> None:
    """Test that reset_audio_engine creates a new instance."""
    from asr import engine_factory

    # Set initial engine
    initial_engine = Mock(spec=ASRBase)
    engine_factory._engine = initial_engine

    with patch("asr.engine_factory._create_engine_from_settings") as mock_create:
        new_engine = Mock(spec=ASRBase)
        mock_create.return_value = new_engine

        # Reset should create new engine
        engine_factory.reset_audio_engine()
        assert engine_factory._engine is new_engine
        assert engine_factory._engine is not initial_engine
        assert mock_create.call_count == 1


@pytest.mark.unit
@pytest.mark.asr
def test_set_audio_engine_manual_override() -> None:
    """Test that set_audio_engine allows manual engine injection."""
    from asr import engine_factory

    # Create mock engine
    custom_engine = Mock(spec=ASRBase)

    # Set engine manually
    engine_factory.set_audio_engine(custom_engine)

    # Verify it's set
    assert engine_factory._engine is custom_engine

    # get_audio_engine should return the custom engine
    engine = engine_factory.get_audio_engine()
    assert engine is custom_engine


@pytest.mark.unit
@pytest.mark.asr
def test_create_engine_faster_whisper_default() -> None:
    """Test that faster-whisper is the default engine."""
    from asr import engine_factory

    with patch("asr.engine_factory.get_settings") as mock_get_settings:
        mock_settings = Mock()
        mock_settings.ASR_ENGINE = ""
        mock_get_settings.return_value = mock_settings

        with patch("asr.engine.fasterwhisper.ASRFasterWhisper") as mock_fw:
            mock_instance = Mock(spec=ASRBase)
            mock_fw.return_value = mock_instance

            engine = engine_factory._create_engine_from_settings()

            assert engine is mock_instance
            mock_fw.assert_called_once()


@pytest.mark.unit
@pytest.mark.asr
def test_create_engine_whisper_torch() -> None:
    """Test that whisper engine is created when specified."""
    from asr import engine_factory

    with patch("asr.engine_factory.get_settings") as mock_get_settings:
        mock_settings = Mock()
        mock_settings.ASR_ENGINE = "whisper"
        mock_get_settings.return_value = mock_settings

        with patch("asr.engine.whisper.ASRWhisperTorch") as mock_whisper:
            mock_instance = Mock(spec=ASRBase)
            mock_whisper.return_value = mock_instance

            engine = engine_factory._create_engine_from_settings()

            assert engine is mock_instance
            mock_whisper.assert_called_once()


@pytest.mark.unit
@pytest.mark.asr
def test_create_engine_whisper_torch_alias() -> None:
    """Test that whisper-torch alias works."""
    from asr import engine_factory

    with patch("asr.engine_factory.get_settings") as mock_get_settings:
        mock_settings = Mock()
        mock_settings.ASR_ENGINE = "whisper-torch"
        mock_get_settings.return_value = mock_settings

        with patch("asr.engine.whisper.ASRWhisperTorch") as mock_whisper:
            mock_instance = Mock(spec=ASRBase)
            mock_whisper.return_value = mock_instance

            engine = engine_factory._create_engine_from_settings()

            assert engine is mock_instance
            mock_whisper.assert_called_once()


@pytest.mark.unit
@pytest.mark.asr
def test_create_engine_case_insensitive() -> None:
    """Test that engine selection is case-insensitive."""
    from asr import engine_factory

    with patch("asr.engine_factory.get_settings") as mock_get_settings:
        mock_settings = Mock()
        mock_settings.ASR_ENGINE = "WHISPER"
        mock_get_settings.return_value = mock_settings

        with patch("asr.engine.whisper.ASRWhisperTorch") as mock_whisper:
            mock_instance = Mock(spec=ASRBase)
            mock_whisper.return_value = mock_instance

            engine = engine_factory._create_engine_from_settings()

            assert engine is mock_instance
            mock_whisper.assert_called_once()


@pytest.mark.unit
@pytest.mark.asr
def test_engine_factory_thread_safety() -> None:
    """Test that engine factory is thread-safe (uses lock)."""

    from asr import engine_factory

    # Verify that _LOCK exists and is a lock object
    assert hasattr(engine_factory, "_LOCK")
    # Check that it has the acquire and release methods (duck typing for lock)
    assert hasattr(engine_factory._LOCK, "acquire")
    assert hasattr(engine_factory._LOCK, "release")
    # Verify it's actually a lock by checking its type
    assert type(engine_factory._LOCK).__name__ == "lock"


@pytest.mark.unit
@pytest.mark.asr
def test_get_audio_engine_lazy_initialization() -> None:
    """Test that engine is only created when first requested."""
    from asr import engine_factory

    # Reset to None
    engine_factory._engine = None

    with patch("asr.engine_factory._create_engine_from_settings") as mock_create:
        mock_engine = Mock(spec=ASRBase)
        mock_create.return_value = mock_engine

        # Engine not created yet
        assert mock_create.call_count == 0

        # First access creates it
        engine = engine_factory.get_audio_engine()
        assert engine is mock_engine
        assert mock_create.call_count == 1
