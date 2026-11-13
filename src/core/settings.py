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
    DEBUG: bool = Field(default=False, description="Enable debug mode (detailed error responses)")
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
    TTS_ENGINE: str = Field(
        default="kokoro",
        description="TTS engine to boot: kokoro (default, ONNX on CPU) or coqui (XTTS-v2; aliases xtts, xtts-2).",
    )
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

    # Coqui/XTTS engine (TTS_ENGINE=coqui): the speaker a request for "default" gets.
    TTS_XTTS_DEFAULT_VOICE: str = Field(
        default="Claribel Dervla", description="XTTS speaker used when a request asks for 'default'."
    )

    # Kokoro engine (TTS_ENGINE=kokoro). TTS_MODEL and TTS_DEVICE are Coqui settings and are ignored here.
    TTS_KOKORO_MODEL_PATH: str = Field(
        default="/app/models/kokoro/kokoro-v1.0.onnx", description="Path to the Kokoro ONNX model file."
    )
    TTS_KOKORO_VOICES_PATH: str = Field(
        default="/app/models/kokoro/voices-v1.0.bin", description="Path to the Kokoro voices archive."
    )
    TTS_KOKORO_DEFAULT_VOICE: str = Field(
        default="af_heart", description="Voice used when a request asks for 'default'."
    )
    TTS_KOKORO_THREADS: int = Field(
        default=0, description="ONNX Runtime intra-op threads for Kokoro (0 = runtime default, all cores)."
    )
    TTS_KOKORO_DEVICE: str = Field(
        default="cpu",
        description="cpu|cuda|auto: ONNX Runtime provider for Kokoro. cuda needs the onnxruntime-gpu (CUDA) image.",
    )

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
        default=90,
        description=(
            "Reject requests when RAM usage exceeds this percentage: of the container's"
            " cgroup memory limit when one is set, otherwise of host RAM"
        ),
    )
    SWAP_THRESHOLD_PERCENT: int = Field(
        default=80, description="Reject requests when swap usage exceeds this percentage"
    )

    # ----------------------------------------------------------------
    # Gateway (voice-gateway): turns an utterance into a spoken reply
    # ----------------------------------------------------------------
    # The gateway is a THIN ORCHESTRATOR. It owns no model and holds no state
    # beyond one in-flight exchange: it calls the ASR and TTS services over
    # HTTP and reaches the agent over NATS. That is why the URLs below are
    # configuration rather than an import.
    GATEWAY_LOG_NAME: str = Field(default="gateway.app", description="Python logger name for the gateway service.")
    GATEWAY_PORT: int = Field(default=5003)

    GATEWAY_ASR_URL: str = Field(default="http://localhost:5001", description="Base URL of the ASR service.")
    GATEWAY_TTS_URL: str = Field(default="http://localhost:5002", description="Base URL of the TTS service.")

    # Which agent an utterance is routed to, and as whom.
    GATEWAY_AGENT: str = Field(default="assistant", description="Peer agent that answers utterances.")
    GATEWAY_SELF_NAME: str = Field(
        default="voice-gateway",
        description="Our own name on the bus. NATS ACLs allow publishing only under agents.<self>.>, "
        "so a mismatch here is a silent publish rejection, not an error.",
    )
    GATEWAY_NATS_URL: str = Field(default="nats://localhost:4222", description="NATS server URL.")
    GATEWAY_NATS_TOPIC: str = Field(default="voice", description="Topic segment of the DM subject.")

    # An agent doing real work (tool calls, searches) is the slow part of an
    # exchange, not ASR. Measured ASR RTF is 0.006-0.020, so a 10 s utterance
    # transcribes in ~0.1 s.
    GATEWAY_AGENT_TIMEOUT_S: float = Field(
        default=60.0, description="How long to wait for the agent's reply before speaking a timeout message."
    )
    GATEWAY_ASR_TIMEOUT_S: float = Field(default=120.0)
    GATEWAY_TTS_TIMEOUT_S: float = Field(default=300.0)

    # The voice the gateway speaks in. Defaults to a STOCK speaker on purpose:
    # cloned voices of real people are private, and a default must never leak
    # one by accident.
    GATEWAY_VOICE: str = Field(
        default="default",
        description="Voice id used for spoken replies. 'default' is the TTS engine's own default voice, "
        "so switching TTS_ENGINE does not also require changing this.",
    )
    GATEWAY_AUDIO_FORMAT: str = Field(default="opus", description="Output format for spoken replies.")
    GATEWAY_LANGUAGE: str | None = Field(default=None, description="Force a language; None auto-detects.")

    # Spoken canned responses. Failure must be AUDIBLE: a caller who hears
    # nothing cannot tell a crash from a pause.
    GATEWAY_MSG_NO_SPEECH: str = Field(default="Sorry, I did not catch that. Could you say it again?")
    GATEWAY_MSG_AGENT_TIMEOUT: str = Field(default="Sorry, that is taking too long. Please try again in a moment.")
    GATEWAY_MSG_AGENT_ERROR: str = Field(default="Sorry, I could not reach the assistant right now.")

    # Below this many characters a transcript is treated as noise rather than
    # speech, and the agent is never asked about it.
    GATEWAY_MIN_TRANSCRIPT_CHARS: int = Field(default=2)

    # ----------------------------------------------------------------
    # Agent (SERVICE_MODE=agent): a live, interruptible voice call
    # ----------------------------------------------------------------
    # Like the gateway, it holds no speech model: ASR and TTS are this
    # project's own services over HTTP, and the "brain" is pluggable -- any
    # OpenAI-compatible chat endpoint, or an agent on a NATS bus. Media goes
    # through a LiveKit server; the brain only ever sees text.
    AGENT_WEB_PORT: int = Field(default=5004, description="Port of the built-in call page, token, health, metrics.")
    AGENT_ASR_URL: str = Field(default="http://localhost:5001", description="Base URL of the ASR service.")
    AGENT_TTS_URL: str = Field(default="http://localhost:5002", description="Base URL of the TTS service.")

    LIVEKIT_URL: str = Field(default="ws://localhost:7880", description="LiveKit server URL the worker connects to.")
    LIVEKIT_API_KEY: str = Field(default="", description="LiveKit API key (worker registration and call tokens).")
    LIVEKIT_API_SECRET: str = Field(default="", description="LiveKit API secret.")
    AGENT_PUBLIC_LIVEKIT_URL: str = Field(
        default="", description="wss:// URL the browser uses to reach LiveKit. Empty = same host as the page."
    )
    AGENT_CALLER_IDENTITY: str = Field(default="caller", description="Participant identity in call tokens.")
    AGENT_TOKEN_TTL_MIN: int = Field(default=60, description="Lifetime of a call token, in minutes.")

    # Which brain answers. openai = an OpenAI-compatible /chat/completions
    # endpoint; nats = a peer agent over NATS request/reply (text only).
    AGENT_BRAIN: str = Field(default="openai", description="openai | nats")
    AGENT_BRAIN_TIMEOUT_S: float = Field(
        default=20.0, description="How long to wait for the first words of a reply before apologising."
    )
    AGENT_LLM_URL: str = Field(default="http://localhost:8080/v1", description="OpenAI-compatible base URL.")
    AGENT_LLM_MODEL: str = Field(default="default", description="Model name sent to the OpenAI-compatible endpoint.")
    AGENT_LLM_API_KEY: str = Field(default="unused", description="API key for the OpenAI-compatible endpoint.")
    AGENT_LLM_EXTRA_BODY: str = Field(
        default="", description='JSON merged into each request, e.g. {"chat_template_kwargs":{"enable_thinking":false}}'
    )
    AGENT_SYSTEM_PROMPT: str = Field(
        default=(
            "You are a friendly voice assistant on a live call. Answer in one to three short, natural spoken "
            "sentences. No markdown, no lists, no emojis. Always reply in the same language the user just spoke. "
            "If you are interrupted, just continue the conversation naturally."
        ),
        description="System prompt for the openai brain.",
    )
    AGENT_NATS_URL: str = Field(default="nats://localhost:4222", description="NATS server URL for the nats brain.")
    AGENT_SELF_NAME: str = Field(default="voice-agent", description="Our identity on the bus.")
    AGENT_PEER: str = Field(default="assistant", description="Peer agent that answers, for the nats brain.")
    AGENT_NATS_TOPIC: str = Field(default="voice", description="Topic segment of the DM subject.")

    # Voices: stock speakers only. Switched per turn on the language ASR detects.
    AGENT_VOICE_EN: str = Field(default="af_heart", description="Voice for English replies.")
    AGENT_VOICE_PT: str = Field(default="pf_dora", description="Voice for Portuguese replies.")

    # Turn taking. A caller in a noisy room (TV, radio, a car) never goes
    # silent, so the turn length is capped or the agent would wait forever.
    AGENT_MIN_ENDPOINTING_DELAY_S: float = Field(default=0.5, description="Silence that ends a turn.")
    AGENT_MAX_ENDPOINTING_DELAY_S: float = Field(default=3.0, description="Upper bound on that wait.")
    AGENT_MAX_USER_TURN_S: float = Field(default=20.0, description="A turn ends after this long even without silence.")

    # Spoken lines. Failure must be audible.
    AGENT_GREETING: str = Field(default="Hi, I'm listening.", description="Spoken when a call starts. Empty = none.")
    AGENT_FILLER_AFTER_S: float = Field(
        default=1.2, description="Speak the filler only if the brain has said nothing after this long."
    )
    AGENT_MSG_FILLER: str = Field(default="Hmm, let me think.")
    AGENT_MSG_APOLOGY: str = Field(default="Sorry, I can't reach my brain right now. Please try again in a moment.")

    # Debug settings
    DEBUG_PRESERVE_FAILED_UPLOADS: bool = Field(
        default=False, description="Save failed audio uploads to /tmp for debugging"
    )
    DEBUG_PRESERVE_DIR: str = Field(default="/tmp/voice-stack-debug", description="Directory to save failed uploads")

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
