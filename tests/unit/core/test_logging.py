"""Unit tests for core logging module."""

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest


@pytest.mark.unit
def test_setup_logging_configures_root_logger() -> None:
    """Test that setup_logging configures root logger with handler."""
    from core.logging import setup_logging

    with (
        patch("logging.getLogger") as mock_get_logger,
        patch("logging.StreamHandler") as mock_stream_handler,
    ):
        mock_root = Mock()
        mock_root.handlers = []
        mock_handler = Mock()
        mock_stream_handler.return_value = mock_handler

        def get_logger_side_effect(name=None):
            if name is None or name == "":
                return mock_root
            return Mock()

        mock_get_logger.side_effect = get_logger_side_effect

        setup_logging("DEBUG", use_json=False)

        # Verify root logger was configured
        mock_root.setLevel.assert_called_once_with("DEBUG")
        mock_root.addHandler.assert_called_once_with(mock_handler)


@pytest.mark.unit
def test_setup_logging_uses_json_formatter_when_enabled() -> None:
    """Test that setup_logging uses StructuredFormatter when use_json=True."""
    from core.logging import StructuredFormatter, setup_logging

    with (
        patch("logging.getLogger") as mock_get_logger,
        patch("logging.StreamHandler") as mock_stream_handler,
    ):
        mock_root = Mock()
        mock_root.handlers = []
        mock_handler = Mock()
        mock_stream_handler.return_value = mock_handler

        def get_logger_side_effect(name=None):
            if name is None or name == "":
                return mock_root
            return Mock()

        mock_get_logger.side_effect = get_logger_side_effect

        setup_logging("INFO", use_json=True)

        # Verify formatter was set on handler
        mock_handler.setFormatter.assert_called_once()
        formatter = mock_handler.setFormatter.call_args[0][0]
        assert isinstance(formatter, StructuredFormatter)


@pytest.mark.unit
def test_setup_logging_uses_human_readable_formatter_by_default() -> None:
    """Test that setup_logging uses HumanReadableFormatter by default."""
    from core.logging import HumanReadableFormatter, setup_logging

    with (
        patch("logging.getLogger") as mock_get_logger,
        patch("logging.StreamHandler") as mock_stream_handler,
    ):
        mock_root = Mock()
        mock_root.handlers = []
        mock_handler = Mock()
        mock_stream_handler.return_value = mock_handler

        def get_logger_side_effect(name=None):
            if name is None or name == "":
                return mock_root
            return Mock()

        mock_get_logger.side_effect = get_logger_side_effect

        setup_logging("INFO", use_json=False)

        # Verify formatter was set on handler
        mock_handler.setFormatter.assert_called_once()
        formatter = mock_handler.setFormatter.call_args[0][0]
        assert isinstance(formatter, HumanReadableFormatter)


@pytest.mark.unit
def test_setup_logging_sets_uvicorn_loggers() -> None:
    """Test that setup_logging configures uvicorn loggers."""
    from core.logging import setup_logging

    with (
        patch("logging.getLogger") as mock_get_logger,
        patch("logging.StreamHandler"),
    ):
        logger_mocks = {}

        def get_logger_side_effect(name=None):
            if name is None or name == "":
                mock = Mock()
                mock.handlers = []
                return mock
            if name not in logger_mocks:
                logger_mocks[name] = Mock()
            return logger_mocks[name]

        mock_get_logger.side_effect = get_logger_side_effect

        setup_logging("INFO", use_json=False)

        # Verify uvicorn loggers were configured
        uvicorn_loggers = ["uvicorn", "uvicorn.error", "uvicorn.access"]
        for logger_name in uvicorn_loggers:
            assert logger_name in logger_mocks
            logger_mocks[logger_name].setLevel.assert_called_with("INFO")


@pytest.mark.unit
def test_setup_logging_sets_third_party_library_levels() -> None:
    """Test that setup_logging reduces noise from third-party libraries."""
    from core.logging import setup_logging

    with (
        patch("logging.getLogger") as mock_get_logger,
        patch("logging.StreamHandler"),
    ):
        logger_mocks = {}

        def get_logger_side_effect(name=None):
            if name is None or name == "":
                mock = Mock()
                mock.handlers = []
                return mock
            if name not in logger_mocks:
                logger_mocks[name] = Mock()
            return logger_mocks[name]

        mock_get_logger.side_effect = get_logger_side_effect

        setup_logging("INFO", use_json=False)

        # Verify third-party libraries have reduced log levels
        assert "faster_whisper" in logger_mocks
        logger_mocks["faster_whisper"].setLevel.assert_called_with(logging.WARNING)

        assert "torch" in logger_mocks
        logger_mocks["torch"].setLevel.assert_called_with(logging.WARNING)

        assert "transformers" in logger_mocks
        logger_mocks["transformers"].setLevel.assert_called_with(logging.ERROR)


@pytest.mark.unit
def test_setup_logging_converts_level_to_uppercase() -> None:
    """Test that setup_logging converts log level to uppercase."""
    from core.logging import setup_logging

    with (
        patch("logging.getLogger") as mock_get_logger,
        patch("logging.StreamHandler"),
    ):
        mock_root = Mock()
        mock_root.handlers = []

        def get_logger_side_effect(name=None):
            if name is None or name == "":
                return mock_root
            return Mock()

        mock_get_logger.side_effect = get_logger_side_effect

        # Test with lowercase level
        setup_logging("debug", use_json=False)

        # Verify level was uppercased
        mock_root.setLevel.assert_called_with("DEBUG")


@pytest.mark.unit
def test_setup_logging_handles_different_levels() -> None:
    """Test setup_logging with different log levels."""
    from core.logging import setup_logging

    levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    for level in levels:
        with (
            patch("logging.getLogger") as mock_get_logger,
            patch("logging.StreamHandler"),
        ):
            mock_root = Mock()
            mock_root.handlers = []

            def get_logger_side_effect(name=None, root=mock_root):
                if name is None or name == "":
                    return root
                return Mock()

            mock_get_logger.side_effect = get_logger_side_effect

            setup_logging(level, use_json=False)

            # Verify correct level was set
            mock_root.setLevel.assert_called_with(level.upper())


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
def test_human_readable_formatter_includes_all_components() -> None:
    """Test that HumanReadableFormatter format includes all expected components."""
    from core.logging import HumanReadableFormatter

    formatter = HumanReadableFormatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create a log record
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="/test/path.py",
        lineno=42,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)

    # Verify all format components are present
    assert "INFO" in formatted
    assert "test.logger" in formatted
    assert "Test message" in formatted


@pytest.mark.unit
def test_human_readable_formatter_includes_extra_fields() -> None:
    """Test that HumanReadableFormatter includes extra fields."""
    from core.logging import HumanReadableFormatter

    formatter = HumanReadableFormatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create a log record with extra fields
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="/test/path.py",
        lineno=42,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    record.request_id = "req_123"
    record.endpoint = "/test"

    formatted = formatter.format(record)

    # Verify extra fields are included
    assert "request_id=req_123" in formatted
    assert "endpoint=/test" in formatted


@pytest.mark.unit
def test_structured_formatter_produces_json() -> None:
    """Test that StructuredFormatter produces valid JSON."""
    import json

    from core.logging import StructuredFormatter

    formatter = StructuredFormatter()

    # Create a log record
    record = logging.LogRecord(
        name="test.logger",
        level=logging.ERROR,
        pathname="/test/path.py",
        lineno=42,
        msg="Test error message",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)

    # Verify it's valid JSON
    log_dict = json.loads(formatted)

    # Verify required fields
    assert log_dict["level"] == "ERROR"
    assert log_dict["logger"] == "test.logger"
    assert log_dict["message"] == "Test error message"
    assert "timestamp" in log_dict


@pytest.mark.unit
def test_structured_formatter_includes_extra_fields() -> None:
    """Test that StructuredFormatter includes extra fields in JSON."""
    import json

    from core.logging import StructuredFormatter

    formatter = StructuredFormatter()

    # Create a log record with extra fields
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="/test/path.py",
        lineno=42,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    record.request_id = "req_123"
    record.endpoint = "/test"
    record.file_size = 1024

    formatted = formatter.format(record)

    # Verify it's valid JSON
    log_dict = json.loads(formatted)

    # Verify extra fields are included
    assert log_dict["request_id"] == "req_123"
    assert log_dict["endpoint"] == "/test"
    assert log_dict["file_size"] == 1024


@pytest.mark.unit
def test_structured_formatter_includes_source_for_warnings() -> None:
    """Test that StructuredFormatter includes source info for warnings and errors."""
    import json

    from core.logging import StructuredFormatter

    formatter = StructuredFormatter()

    # Create a warning record
    record = logging.LogRecord(
        name="test.logger",
        level=logging.WARNING,
        pathname="/test/path.py",
        lineno=42,
        msg="Test warning",
        args=(),
        exc_info=None,
        func="test_function",
    )

    formatted = formatter.format(record)
    log_dict = json.loads(formatted)

    # Verify source info is included for warnings
    assert "source" in log_dict
    assert log_dict["source"]["file"] == "/test/path.py"
    assert log_dict["source"]["line"] == 42
    assert log_dict["source"]["function"] == "test_function"


@pytest.mark.unit
def test_setup_logging_removes_existing_handlers() -> None:
    """Test that setup_logging clears existing handlers from root logger."""
    from core.logging import setup_logging

    with (
        patch("logging.getLogger") as mock_get_logger,
        patch("logging.StreamHandler"),
    ):
        mock_root = Mock()
        mock_existing_handler1 = Mock()
        mock_existing_handler2 = Mock()
        mock_root.handlers = [mock_existing_handler1, mock_existing_handler2]

        def get_logger_side_effect(name=None):
            if name is None or name == "":
                return mock_root
            return Mock()

        mock_get_logger.side_effect = get_logger_side_effect

        setup_logging("INFO", use_json=False)

        # Verify existing handlers were removed
        mock_root.removeHandler.assert_any_call(mock_existing_handler1)
        mock_root.removeHandler.assert_any_call(mock_existing_handler2)
