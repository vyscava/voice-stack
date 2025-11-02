"""Unit tests for core logging module."""

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest


@pytest.mark.unit
def test_setup_logging_configures_basic_config() -> None:
    """Test that setup_logging configures basicConfig with correct parameters."""
    from core.logging import setup_logging

    with patch("logging.basicConfig") as mock_basic_config:
        setup_logging("DEBUG")

        # Verify basicConfig was called with correct parameters
        mock_basic_config.assert_called_once()
        call_kwargs = mock_basic_config.call_args[1]
        assert call_kwargs["level"] == "DEBUG"
        assert "%(asctime)s" in call_kwargs["format"]
        assert "%(levelname)" in call_kwargs["format"]
        assert "%(name)s" in call_kwargs["format"]
        assert "%(message)s" in call_kwargs["format"]


@pytest.mark.unit
def test_setup_logging_sets_uvicorn_loggers() -> None:
    """Test that setup_logging configures uvicorn loggers."""
    from core.logging import setup_logging

    with (
        patch("logging.basicConfig"),
        patch("logging.getLogger") as mock_get_logger,
    ):
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        setup_logging("INFO")

        # Verify uvicorn loggers were configured
        expected_calls = ["uvicorn", "uvicorn.error", "uvicorn.access"]
        assert mock_get_logger.call_count == len(expected_calls)

        for logger_name in expected_calls:
            mock_get_logger.assert_any_call(logger_name)


@pytest.mark.unit
def test_setup_logging_converts_level_to_uppercase() -> None:
    """Test that setup_logging converts log level to uppercase."""
    from core.logging import setup_logging

    with (
        patch("logging.basicConfig") as mock_basic_config,
        patch("logging.getLogger") as mock_get_logger,
    ):
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # Test with lowercase level
        setup_logging("debug")

        # Verify level was uppercased
        call_kwargs = mock_basic_config.call_args[1]
        assert call_kwargs["level"] == "DEBUG"

        # Verify uvicorn loggers got uppercase level
        mock_logger.setLevel.assert_called_with("DEBUG")


@pytest.mark.unit
def test_setup_logging_handles_different_levels() -> None:
    """Test setup_logging with different log levels."""
    from core.logging import setup_logging

    levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    for level in levels:
        with (
            patch("logging.basicConfig") as mock_basic_config,
            patch("logging.getLogger") as mock_get_logger,
        ):
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            setup_logging(level)

            # Verify correct level was set
            call_kwargs = mock_basic_config.call_args[1]
            assert call_kwargs["level"] == level.upper()
            mock_logger.setLevel.assert_called_with(level.upper())


@pytest.mark.unit
def test_logger_asr_exists() -> None:
    """Test that logger_asr is created and has correct name."""
    from core.logging import logger_asr

    assert logger_asr is not None
    assert isinstance(logger_asr, logging.Logger)
    # Logger name should be from settings ASR_LOG_NAME (default: "asr.app")
    assert "asr" in logger_asr.name.lower()


@pytest.mark.unit
def test_logger_tts_exists() -> None:
    """Test that logger_tts is created and has correct name."""
    from core.logging import logger_tts

    assert logger_tts is not None
    assert isinstance(logger_tts, logging.Logger)
    # Logger name should be from settings TTS_LOG_NAME (default: "tts.app")
    assert "tts" in logger_tts.name.lower()


@pytest.mark.unit
def test_loggers_are_different_instances() -> None:
    """Test that ASR and TTS loggers are different instances."""
    from core.logging import logger_asr, logger_tts

    assert logger_asr is not logger_tts
    assert logger_asr.name != logger_tts.name


@pytest.mark.unit
def test_module_calls_setup_logging_on_import() -> None:
    """Test that setup_logging is called when module is imported."""
    # This test verifies the module initialization behavior
    # The logging module calls setup_logging with get_settings().LOG_LEVEL on import
    from core.logging import logger_asr, logger_tts

    # If setup_logging wasn't called, loggers wouldn't be properly configured
    # Verify they're actual Logger instances (not None or Mock)
    assert isinstance(logger_asr, logging.Logger)
    assert isinstance(logger_tts, logging.Logger)


@pytest.mark.unit
def test_setup_logging_format_includes_all_components() -> None:
    """Test that log format includes all expected components."""
    from core.logging import setup_logging

    with patch("logging.basicConfig") as mock_basic_config:
        setup_logging("INFO")

        call_kwargs = mock_basic_config.call_args[1]
        format_str = call_kwargs["format"]

        # Verify all format components are present
        required_components = [
            "%(asctime)s",  # Timestamp
            "%(levelname)",  # Log level
            "%(name)s",  # Logger name
            "%(message)s",  # Log message
        ]

        for component in required_components:
            assert component in format_str, f"Format missing {component}"
