from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central application configuration.

    All values can be overridden via environment variables (or a .env file at project root).
    """

    # General project information
    PROJECT_NAME: str = "Not Yet a Service to Share"

    # API versioning
    API_V1_STR: str = ""

    # -------------------
    # App / HTTP server
    # -------------------
    ENV: str = Field(default="dev", description="Environment name (dev|staging|prod)")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    HOST: str = Field(default="0.0.0.0", description="Bind address")

    # ASR HTTP
    ASR_PORT: int = Field(default=5001)
    # TTS HTTP
    TTS_PORT: int = Field(default=5002)

    # CORS
    CORS_ORIGINS: str = Field(default="", description="Comma-separated origins")

    # -------------
    # ASR settings
    # -------------
    ASR_LOG_NAME: str = Field(default="asr.app", description="Python Logger name for the ASR Service.")
    ASR_ENGINE: str = Field(default="fasterwhisper", description="fasterwhisper|whisper")
    ASR_DEVICE: str = Field(default="cpu", description="cpu|cuda|auto")
    ASR_MODEL: str = Field(default="base", description="Faster-Whisper model.")
    ASR_MODEL_LOCATION: str | None = Field(default=None, description="Location where to store the downloaded models.")
    ASR_CPU_THREADS: int = Field(default=8, description="Whisper Threads Number (CPU Threads)")
    ASR_NUM_OF_WORKERS: int = Field(default=1, description="Number of ASR Workers (Max Concurrency on ASR)")
    ASR_COMPUTE_TYPE: str = Field(default="int8", description="int8|int8_float16|float16|float32")
    ASR_CACHE_ENABLED: bool = Field(default=True, description="Whether to use or not in memory cache")
    ASR_CACHE_MAX_ITEMS: int = Field(default=64, description="Number of transcription to keep in memory")
    ASR_VAD_ENABLED: bool = Field(default=True)
    ASR_TRANSCRIBE_LANG: str | None = Field(default=None, description="Force language to use while transcribing.")
    ASR_TRANSCRIBE_BEAM_SIZE: int = Field(
        default=5, description="Width of beam search (number of hypotheses tracked during decoding)"
    )
    ASR_TRANSCRIBE_TEMPERATURE: float = Field(
        default=0.0, description="Sampling temperature controlling output randomness"
    )
    ASR_TRANSCRIBE_BEST_OF: int = Field(
        default=1, description="Number of parallel sampling candidates when beam_size == 1"
    )

    # ASR Resource Management
    ASR_IDLE_TIMEOUT_MINUTES: int = Field(
        default=60, description="Unload ASR model after N minutes of inactivity (0 = disabled)"
    )
    ASR_MAX_CONCURRENT_REQUESTS: int = Field(
        default=2, description="Maximum number of concurrent ASR inference requests"
    )

    # -------------
    # TTS settings
    # -------------
    TTS_LOG_NAME: str = Field(default="tts.app", description="Python Logger name for the TTS Service.")
    TTS_ENGINE: str = Field(default="coqui", description="Only Coqui for now.")
    TTS_DEVICE: str = Field(default="cpu", description="cuda|cpu|mps")
    TTS_MODEL: str = Field(default="tts_models/multilingual/multi-dataset/xtts_v2")
    TTS_VOICES_DIR: str = Field(default="voices")
    TTS_SAMPLE_RATE: int = Field(default=24000)

    TTS_MAX_CHARS: int = Field(default=180)
    TTS_MIN_CHARS: int = Field(default=70)
    TTS_RETRY_STEPS: int = Field(default=2)

    TTS_DEFAULT_LANG: str = Field(default="en")
    TTS_AUTO_LANG: bool = Field(default=True)
    TTS_LANG_HINT: str | None = Field(default=None)
    TTS_FORCE_LANG: str | None = Field(default=None)

    # TTS Resource Management
    TTS_IDLE_TIMEOUT_MINUTES: int = Field(
        default=60, description="Unload TTS model after N minutes of inactivity (0 = disabled)"
    )
    TTS_MAX_CONCURRENT_REQUESTS: int = Field(
        default=2, description="Maximum number of concurrent TTS synthesis requests"
    )

    # -------------
    # Resource Safeguards (applies to both ASR and TTS)
    # -------------
    MAX_UPLOAD_SIZE_MB: int = Field(default=100, description="Maximum file upload size in MB")
    MEMORY_THRESHOLD_PERCENT: int = Field(
        default=90, description="Reject requests when RAM usage exceeds this percentage"
    )
    SWAP_THRESHOLD_PERCENT: int = Field(
        default=80, description="Reject requests when swap usage exceeds this percentage"
    )

    # Debugpy
    DEBUGPY_ENABLE: bool = Field(default=False)
    DEBUGPY_HOST: str = Field(default="0.0.0.0")
    DEBUGPY_PORT: int = Field(default=5678)
    DEBUGPY_WAIT_FOR_CLIENT: bool = Field(default=False)

    # -------------
    # Pydantic cfg
    # -------------
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), "../../.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached accessor so we don't parse .env multiple times.
    """
    return Settings()
