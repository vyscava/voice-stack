"""Unit tests for core settings module."""

from __future__ import annotations

import pytest

from core.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def isolate_settings():
    """Isolate each test by clearing Settings cache."""
    # Clear the settings cache before each test
    from core.settings import get_settings

    get_settings.cache_clear()
    yield
    # Clear again after test
    get_settings.cache_clear()


@pytest.mark.unit
def test_settings_defaults() -> None:
    """Test that Settings model fields have correct default values defined."""
    # Check the model fields to verify defaults are properly defined
    # This tests the Settings class definition, not runtime configuration

    # General - check field defaults from model
    assert Settings.model_fields["PROJECT_NAME"].default == "Not Yet a Service to Share"
    assert Settings.model_fields["API_V1_STR"].default == ""
    assert Settings.model_fields["ENV"].default == "dev"
    assert Settings.model_fields["LOG_LEVEL"].default == "INFO"
    assert Settings.model_fields["HOST"].default == "0.0.0.0"

    # Ports
    assert Settings.model_fields["ASR_PORT"].default == 5001
    assert Settings.model_fields["TTS_PORT"].default == 5002

    # CORS
    assert Settings.model_fields["CORS_ORIGINS"].default == ""


@pytest.mark.unit
def test_settings_asr_defaults() -> None:
    """Test ASR-specific field default values."""
    # Check model field defaults
    assert Settings.model_fields["ASR_LOG_NAME"].default == "asr.app"
    assert Settings.model_fields["ASR_ENGINE"].default == "fasterwhisper"
    assert Settings.model_fields["ASR_DEVICE"].default == "cpu"
    assert Settings.model_fields["ASR_MODEL"].default == "base"
    assert Settings.model_fields["ASR_MODEL_LOCATION"].default is None
    assert Settings.model_fields["ASR_CPU_THREADS"].default == 8
    assert Settings.model_fields["ASR_NUM_OF_WORKERS"].default == 1
    assert Settings.model_fields["ASR_COMPUTE_TYPE"].default == "int8"
    assert Settings.model_fields["ASR_CACHE_ENABLED"].default is True
    assert Settings.model_fields["ASR_CACHE_MAX_ITEMS"].default == 64
    assert Settings.model_fields["ASR_VAD_ENABLED"].default is True
    assert Settings.model_fields["ASR_TRANSCRIBE_LANG"].default is None
    assert Settings.model_fields["ASR_TRANSCRIBE_BEAM_SIZE"].default == 5
    assert Settings.model_fields["ASR_TRANSCRIBE_TEMPERATURE"].default == 0.0
    assert Settings.model_fields["ASR_TRANSCRIBE_BEST_OF"].default == 1


@pytest.mark.unit
def test_settings_tts_defaults() -> None:
    """Test TTS-specific field default values."""
    # Check model field defaults
    assert Settings.model_fields["TTS_LOG_NAME"].default == "tts.app"
    assert Settings.model_fields["TTS_ENGINE"].default == "coqui"
    assert Settings.model_fields["TTS_DEVICE"].default == "cpu"
    assert Settings.model_fields["TTS_MODEL"].default == "tts_models/multilingual/multi-dataset/xtts_v2"
    assert Settings.model_fields["TTS_VOICES_DIR"].default == "voices"
    assert Settings.model_fields["TTS_SAMPLE_RATE"].default == 24000
    assert Settings.model_fields["TTS_MAX_CHARS"].default == 180
    assert Settings.model_fields["TTS_MIN_CHARS"].default == 70
    assert Settings.model_fields["TTS_RETRY_STEPS"].default == 2
    assert Settings.model_fields["TTS_DEFAULT_LANG"].default == "en"
    assert Settings.model_fields["TTS_AUTO_LANG"].default is True
    assert Settings.model_fields["TTS_LANG_HINT"].default is None
    assert Settings.model_fields["TTS_FORCE_LANG"].default is None


@pytest.mark.unit
def test_settings_debugpy_defaults() -> None:
    """Test Debugpy field default values."""
    # Check model field defaults
    assert Settings.model_fields["DEBUGPY_ENABLE"].default is False
    assert Settings.model_fields["DEBUGPY_HOST"].default == "0.0.0.0"
    assert Settings.model_fields["DEBUGPY_PORT"].default == 5678
    assert Settings.model_fields["DEBUGPY_WAIT_FOR_CLIENT"].default is False


@pytest.mark.unit
def test_settings_from_env_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that environment variables override defaults."""
    monkeypatch.setenv("PROJECT_NAME", "Test Project")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("ASR_PORT", "6001")
    monkeypatch.setenv("TTS_PORT", "6002")
    monkeypatch.setenv("ASR_ENGINE", "whisper")
    monkeypatch.setenv("TTS_DEVICE", "cuda")

    settings = Settings()

    assert settings.PROJECT_NAME == "Test Project"
    assert settings.LOG_LEVEL == "DEBUG"
    assert settings.ASR_PORT == 6001
    assert settings.TTS_PORT == 6002
    assert settings.ASR_ENGINE == "whisper"
    assert settings.TTS_DEVICE == "cuda"


@pytest.mark.unit
def test_settings_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that environment variables are case-insensitive."""
    monkeypatch.setenv("log_level", "WARNING")
    monkeypatch.setenv("asr_device", "cuda")

    settings = Settings()

    assert settings.LOG_LEVEL == "WARNING"
    assert settings.ASR_DEVICE == "cuda"


@pytest.mark.unit
def test_settings_boolean_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test boolean field parsing from environment."""
    # Test various boolean representations
    monkeypatch.setenv("ASR_CACHE_ENABLED", "false")
    monkeypatch.setenv("ASR_VAD_ENABLED", "0")
    monkeypatch.setenv("TTS_AUTO_LANG", "False")
    monkeypatch.setenv("DEBUGPY_ENABLE", "true")

    settings = Settings()

    assert settings.ASR_CACHE_ENABLED is False
    assert settings.ASR_VAD_ENABLED is False
    assert settings.TTS_AUTO_LANG is False
    assert settings.DEBUGPY_ENABLE is True


@pytest.mark.unit
def test_settings_integer_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test integer field parsing from environment."""
    monkeypatch.setenv("ASR_CPU_THREADS", "16")
    monkeypatch.setenv("ASR_CACHE_MAX_ITEMS", "128")
    monkeypatch.setenv("TTS_MAX_CHARS", "250")

    settings = Settings()

    assert settings.ASR_CPU_THREADS == 16
    assert settings.ASR_CACHE_MAX_ITEMS == 128
    assert settings.TTS_MAX_CHARS == 250


@pytest.mark.unit
def test_settings_float_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test float field parsing from environment."""
    monkeypatch.setenv("ASR_TRANSCRIBE_TEMPERATURE", "0.5")

    settings = Settings()

    assert settings.ASR_TRANSCRIBE_TEMPERATURE == 0.5


@pytest.mark.unit
def test_settings_optional_string_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test optional string fields."""
    monkeypatch.setenv("ASR_MODEL_LOCATION", "/models/whisper")
    monkeypatch.setenv("ASR_TRANSCRIBE_LANG", "en")
    monkeypatch.setenv("TTS_LANG_HINT", "es")

    settings = Settings()

    assert settings.ASR_MODEL_LOCATION == "/models/whisper"
    assert settings.ASR_TRANSCRIBE_LANG == "en"
    assert settings.TTS_LANG_HINT == "es"


@pytest.mark.unit
def test_settings_cors_origins_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test CORS_ORIGINS parsing."""
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000,https://example.com")

    settings = Settings()

    assert settings.CORS_ORIGINS == "http://localhost:3000,https://example.com"


@pytest.mark.unit
def test_get_settings_returns_settings() -> None:
    """Test that get_settings returns a Settings instance."""
    settings = get_settings()

    assert isinstance(settings, Settings)


@pytest.mark.unit
def test_get_settings_caching() -> None:
    """Test that get_settings caches the result."""
    settings1 = get_settings()
    settings2 = get_settings()

    # Should return the same instance
    assert settings1 is settings2


@pytest.mark.unit
def test_get_settings_cache_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test clearing the settings cache."""
    # Get initial settings
    settings1 = get_settings()

    # Clear the cache
    get_settings.cache_clear()

    # Change environment
    monkeypatch.setenv("LOG_LEVEL", "ERROR")

    # Get new settings (won't use cache)
    settings2 = get_settings()

    # Should be different instances
    assert settings1 is not settings2


@pytest.mark.unit
def test_settings_extra_fields_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that extra environment variables are ignored."""
    monkeypatch.setenv("UNKNOWN_FIELD", "some_value")
    monkeypatch.setenv("RANDOM_KEY", "random_value")

    # Should not raise an error
    settings = Settings()
    assert settings is not None


@pytest.mark.unit
def test_settings_env_values() -> None:
    """Test ENV field accepts valid values."""
    settings = Settings(ENV="prod")
    assert settings.ENV == "prod"

    settings = Settings(ENV="staging")
    assert settings.ENV == "staging"


@pytest.mark.unit
def test_settings_asr_engine_values() -> None:
    """Test ASR_ENGINE accepts valid values."""
    settings = Settings(ASR_ENGINE="whisper")
    assert settings.ASR_ENGINE == "whisper"

    settings = Settings(ASR_ENGINE="fasterwhisper")
    assert settings.ASR_ENGINE == "fasterwhisper"


@pytest.mark.unit
def test_settings_device_values() -> None:
    """Test device fields accept valid values."""
    settings = Settings(ASR_DEVICE="cuda")
    assert settings.ASR_DEVICE == "cuda"

    settings = Settings(TTS_DEVICE="mps")
    assert settings.TTS_DEVICE == "mps"


@pytest.mark.unit
def test_settings_compute_type_values() -> None:
    """Test ASR_COMPUTE_TYPE accepts valid values."""
    for compute_type in ["int8", "int8_float16", "float16", "float32"]:
        settings = Settings(ASR_COMPUTE_TYPE=compute_type)
        assert settings.ASR_COMPUTE_TYPE == compute_type


@pytest.mark.unit
def test_settings_model_config() -> None:
    """Test that settings model_config is properly configured."""
    settings = Settings()

    # Verify config exists
    assert hasattr(settings, "model_config")

    # Verify case insensitivity
    config = settings.model_config
    assert config.get("case_sensitive") is False

    # Verify extra fields are ignored
    assert config.get("extra") == "ignore"
