# Voice Stack Architecture

This document provides a detailed technical overview of the Voice Stack project architecture.

---

## System Overview

Voice Stack is a **microservices-based speech processing platform** consisting of two independent FastAPI services:

1. **ASR Service** (Port 5001): Automatic Speech Recognition with Whisper models
2. **TTS Service** (Port 5002): Text-to-Speech with Coqui XTTS-v2

Both services expose **OpenAI-compatible REST APIs** and can run independently on CPU or GPU (CUDA/Metal).

```
┌─────────────────────────────────────────────────────────────┐
│                        Voice Stack                          │
├─────────────────────────┬───────────────────────────────────┤
│   ASR Service :5001     │      TTS Service :5002            │
│                         │                                   │
│  ┌─────────────────┐    │    ┌─────────────────┐           │
│  │  FastAPI App    │    │    │  FastAPI App    │           │
│  │  + CORS         │    │    │  + CORS         │           │
│  └────────┬────────┘    │    └────────┬────────┘           │
│           │             │             │                    │
│  ┌────────▼────────┐    │    ┌────────▼────────┐           │
│  │ API Routes      │    │    │ API Routes      │           │
│  │ - OpenAI compat │    │    │ - OpenAI compat │           │
│  │ - Bazarr        │    │    │ - Voices        │           │
│  └────────┬────────┘    │    └────────┬────────┘           │
│           │             │             │                    │
│  ┌────────▼────────┐    │    ┌────────▼────────┐           │
│  │ Engine Factory  │    │    │ Engine Factory  │           │
│  └────────┬────────┘    │    └────────┬────────┘           │
│           │             │             │                    │
│  ┌────────▼────────┐    │    ┌────────▼────────┐           │
│  │ ASR Engines     │    │    │ TTS Engines     │           │
│  │ - Faster-Whisper│    │    │ - Coqui XTTS    │           │
│  │ - OpenAI Whisper│    │    │                 │           │
│  └────────┬────────┘    │    └────────┬────────┘           │
│           │             │             │                    │
│  ┌────────▼────────┐    │    ┌────────▼────────┐           │
│  │ Utils & Helpers │    │    │ Utils & Helpers │           │
│  │ - FFmpeg decode │    │    │ - Audio format  │           │
│  │ - VAD (Silero)  │    │    │ - Language det. │           │
│  │ - Resampling    │    │    │ - Text norm.    │           │
│  └─────────────────┘    │    └─────────────────┘           │
│                         │                                   │
└─────────────────────────┴───────────────────────────────────┘
                          │
              ┌───────────▼───────────┐
              │   Shared Core         │
              │   - Settings          │
              │   - Logging           │
              └───────────────────────┘
```

---

## Module Architecture

### 1. ASR Service (`src/asr/`)

#### Purpose
Transcribe audio files or streams into text using Whisper-based models.

#### Components

**`app.py`** — FastAPI Application Bootstrap
- Initializes FastAPI app with metadata from `pyproject.toml`
- Configures CORS middleware
- Registers API routes under `/v1` prefix
- Provides health check endpoints (`/health`, `/healthz`, `/healthcheck`)
- Optional debugpy integration for remote debugging

**`engine_factory.py`** — Engine Initialization & Management
- Singleton pattern for engine instances
- Lazy initialization on first API call
- Engine selection based on `ASR_ENGINE` setting:
  - `faster-whisper` (default): CTranslate2-based, faster inference
  - `whisper`: Original OpenAI implementation
- Device selection (CPU/CUDA) via `ASR_DEVICE`
- Model caching and reuse

**`api/api_v1/endpoints/`** — API Route Handlers

- **`openai.py`**: OpenAI-compatible transcription endpoints
  - `POST /v1/audio/transcriptions` — Simple transcription (minimal response)
  - `POST /v1/audio/transcriptions/verbose` — Full metadata response
  - `POST /v1/audio/translations` — Translate to English
  - `GET /v1/models` — List available models

- **`bazarr.py`**: Custom endpoints for Bazarr subtitle integration
  - `POST /v1/bazarr/asr` — Transcription with subtitle timing (supports JSON, SRT, VTT, TXT, TSV, JSONL)
  - `POST /v1/bazarr/detect-language` — Audio language detection (accepts query parameters: encode, detect_lang_length, detect_lang_offset, video_file)

**`engine/`** — ASR Engine Implementations

- **`base.py`**: Abstract base class defining the engine interface
  - `transcribe_file()`: Main transcription method
  - `list_models()`: Model enumeration
  - Result schemas with text, segments, language, timing

- **`fasterwhisper.py`**: Faster-Whisper engine (CTranslate2)
  - Model loading from HuggingFace
  - Optimized inference with CTranslate2
  - VAD integration (Silero)
  - Beam search configuration
  - Word-level timestamps

- **`whisper.py`**: OpenAI Whisper engine
  - Original PyTorch implementation
  - Full compatibility with research models
  - Language detection
  - Translation support

**`schemas/`** — Pydantic Models
- Request validation schemas
- Response formatting
- OpenAI API compatibility structures

#### Data Flow (ASR)

```
1. Client uploads audio file (multipart/form-data)
   ↓
2. FastAPI endpoint receives request
   ↓
3. File bytes extracted + parameters validated (Pydantic)
   ↓
4. Engine Factory provides singleton engine instance
   ↓
5. Audio preprocessing:
   - FFmpeg decodes audio to PCM16
   - Resampling to 16kHz mono (if needed)
   - VAD filters silence (optional)
   ↓
6. Model inference:
   - Audio features extracted
   - Beam search decoding
   - Language detection (if not specified)
   ↓
7. Post-processing:
   - Segment alignment
   - Word timestamps (if requested)
   - Text normalization
   ↓
8. Response formatting (JSON or plain text)
   ↓
9. Client receives transcription
```

---

### 2. TTS Service (`src/tts/`)

#### Purpose
Generate speech audio from text using neural TTS models with optional voice cloning.

#### Components

**`app.py`** — FastAPI Application Bootstrap
- Similar structure to ASR service
- Configures CORS and health endpoints
- Registers TTS-specific routes

**`engine_factory.py`** — TTS Engine Management
- Lazy loading of Coqui XTTS-v2 models
- Voice sample management
- Model caching strategy
- Device selection (CPU/CUDA)

**`api/api_v1/endpoints/openai.py`** — API Routes
- `POST /v1/audio/speech` — Generate speech from text
  - Streaming support (SSE) or complete file
  - Multiple output formats (MP3, OPUS, AAC, FLAC, WAV, PCM)
  - Speed control
- `GET /v1/models` — List available TTS models
- `GET /v1/models/{model_id}` — Get specific model info
- `GET /v1/audio/voices` — List available voice samples

**`engine/coqui.py`** — Coqui XTTS Engine
- XTTS-v2 model loading
- Voice cloning from reference audio
- Language detection from text
- Chunking for long texts
- Streaming audio generation
- Format conversion pipeline

**`config.py`** — TTS-specific Settings
- Voice directory management
- Model configuration
- Audio format defaults
- Language settings

#### Data Flow (TTS)

```
1. Client sends text + parameters (JSON)
   ↓
2. FastAPI endpoint validates request
   ↓
3. Engine Factory provides TTS engine
   ↓
4. Text preprocessing:
   - Language detection (if auto)
   - Sentence segmentation
   - Text normalization (punctuation, numbers)
   ↓
5. Voice sample loading:
   - Find reference .wav file
   - Validate audio quality
   - Extract voice embeddings
   ↓
6. TTS synthesis:
   - Generate audio per chunk
   - Apply speaker embeddings
   - Control speaking rate
   ↓
7. Audio processing:
   - Resampling to target sample rate
   - Format conversion (FFmpeg)
   - Streaming or buffering
   ↓
8. Response:
   - SSE stream (for streaming mode)
   - Complete audio file (for file mode)
   ↓
9. Client receives audio
```

---

### 3. Core Module (`src/core/`)

#### Purpose
Shared logic and configuration used by both ASR and TTS services.

#### Components

**`settings.py`** — Pydantic Settings
- Environment variable loading via `pydantic-settings`
- Type-safe configuration
- Default values for all settings
- Validation on startup

Key settings:
```python
PROJECT_NAME: str           # API title
API_V1_STR: str            # API route prefix ("/v1")
CORS_ORIGINS: str          # Comma-separated allowed origins
LOG_LEVEL: str             # DEBUG, INFO, WARNING, ERROR
DEBUGPY_*: bool/str/int    # Remote debugging config
```

**`logging.py`** — Structured Logging
- Separate loggers for ASR and TTS (`logger_asr`, `logger_tts`)
- Consistent log format across services
- Configurable log levels
- Error tracking and context

---

### 4. Utils Module (`src/utils/`)

#### Purpose
Reusable helper functions for audio processing, language detection, text manipulation, and task management.

#### Submodules

**`audio/`** — Audio Processing Utilities
- **`ffmpeg_helper.py`**: FFmpeg operations
  - Decode various formats to PCM16
  - Resample audio to target rate
  - Format conversion
  - CLI and Python API wrappers

- **`audio_helper.py`**: Audio manipulation
  - VAD (Voice Activity Detection) with Silero
  - Silence removal
  - Chunking for batch processing
  - PCM byte array operations
  - Torch device selection

**`language/`** — Language Detection & Management
- **`language_codes.py`**: ISO 639 language code mappings
- **`language_helper.py`**: Language detection from audio
- **`language_manager.py`**: Language preference handling
- Text-based language detection (langdetect library)
- Script detection heuristics (Latin, Cyrillic, CJK, Arabic)

**`text.py`** — Text Processing
- Sentence segmentation
- Text normalization (punctuation, whitespace)
- Word wrapping for TTS chunking
- Terminal punctuation enforcement

**`queue_manager.py`** — Task Queue (Future Use)
- Async task queue management
- Status tracking
- Priority scheduling
- Result caching

**`debugpy_helper.py`** — Remote Debugging
- Optional debugpy initialization
- Wait-for-client mode
- Configurable host/port

---

## Design Patterns

### 1. Engine Factory Pattern

Each service uses a factory pattern for engine management:

```python
# Singleton engine instance
_engine: Optional[EngineBase] = None

def get_audio_engine() -> EngineBase:
    global _engine
    if _engine is None:
        _engine = _create_engine_from_settings()
    return _engine

def reset_audio_engine() -> None:
    global _engine
    _engine = None
```

Benefits:
- Lazy initialization (models loaded only when needed)
- Resource efficiency (single model instance per service)
- Easy testing (reset engine between tests)
- Configuration flexibility

### 2. Abstract Base Class for Engines

Both ASR and TTS engines inherit from abstract base classes:

```python
class ASRBase(ABC):
    @abstractmethod
    def transcribe_file(self, file_bytes: bytes, **kwargs) -> TranscribeResult:
        pass

    @abstractmethod
    def list_models(self) -> dict[str, Any]:
        pass
```

Benefits:
- Swappable engine implementations
- Consistent interface across engines
- Type safety with protocols
- Easy to add new engines

### 3. Pydantic Schemas for Validation

All API inputs and outputs use Pydantic models:

```python
class OpenAITranscriptionRequest(BaseModel):
    file: UploadFile
    model: str = "base"
    language: Optional[str] = None
    temperature: Optional[float] = 0.0

    @classmethod
    def as_form(cls, ...):
        # Custom form parser for multipart data
        pass
```

Benefits:
- Automatic validation
- OpenAPI schema generation
- Type hints throughout codebase
- Error messages for invalid inputs

### 4. Lifespan Context Manager

Services use FastAPI's lifespan pattern:

```python
@asynccontextmanager
async def app_init(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Initializing...")
    app.include_router(api_router, prefix=settings.API_V1_STR)
    yield
    # Cleanup on shutdown
```

Benefits:
- Clean startup/shutdown hooks
- Resource management
- Async-friendly
- Testable initialization

---

## Configuration Strategy

### Environment-Based Config

All configuration via environment variables or `.env` file:

```bash
# Core
PROJECT_NAME="Voice Stack ASR"
API_V1_STR="/v1"
LOG_LEVEL="INFO"

# ASR
ASR_DEVICE="cuda"
ASR_MODEL="base"
ASR_ENGINE="faster-whisper"

# TTS
TTS_DEVICE="cuda"
TTS_MODEL="tts_models/multilingual/multi-dataset/xtts_v2"
TTS_VOICE_DIR="./voices"
```

### Multi-Environment Support

- **Development** (macOS): CPU-based, hot reload, debug logging
- **Production** (Linux): GPU-based, multi-worker, INFO logging
- **CI/CD**: Minimal deps, fast tests, no GPU

### Feature Flags

- `ASR_VAD_ENABLED`: Enable/disable voice activity detection
- `TTS_AUTO_LANG`: Automatic language detection for TTS
- `DEBUGPY_ENABLE`: Remote debugging support
- `CORS_ORIGINS`: Control API access

---

## Error Handling

### Layered Error Strategy

1. **Pydantic Validation**: Catch invalid inputs early
2. **Engine Exceptions**: Wrap model errors with context
3. **HTTP Exceptions**: FastAPI HTTPException with status codes
4. **Logging**: All errors logged with stack traces

```python
try:
    result = engine.transcribe_file(audio)
    return OpenAITranscription(**result.to_dict())
except ValidationError as e:
    raise HTTPException(400, detail=str(e))
except EngineError as e:
    logger.exception("Engine failed")
    raise HTTPException(500, detail="Transcription failed")
```

### Graceful Degradation

- VAD failures → continue without VAD
- Language detection failures → use default language
- Model loading failures → retry with fallback model
- Audio decode failures → clear error message

---

## Performance Considerations

### ASR Optimizations

1. **Model Selection**:
   - `tiny`: 39M params, ~32x real-time
   - `base`: 74M params, ~16x real-time
   - `large-v2`: 1.5B params, ~3x real-time

2. **Faster-Whisper Benefits**:
   - CTranslate2 optimizations
   - 8-bit quantization support
   - Lower memory usage
   - Better batch processing

3. **VAD Integration**:
   - Silero VAD removes silence
   - Reduces processing time 30-50%
   - Improves accuracy on noisy audio

### TTS Optimizations

1. **Chunking Strategy**:
   - Split text into ~180 character chunks
   - Generate chunks in parallel (future)
   - Stream chunks as generated

2. **Voice Caching**:
   - Preload voice embeddings
   - Cache per session
   - Lazy loading of unused voices

3. **Format Conversion**:
   - Direct PCM generation (no intermediate files)
   - FFmpeg piping for streaming
   - Optimal encoding settings per format

### Shared Optimizations

1. **Lazy Loading**: Models loaded on first request
2. **Singleton Engines**: One instance per service process
3. **Async I/O**: FastAPI async/await throughout
4. **Connection Pooling**: Reuse connections for model downloads

---

## Testing Strategy

### Unit Tests
- Engine implementations (mocked models)
- Utility functions (audio, text, language)
- Schema validation
- Configuration loading

### Integration Tests
- API endpoints (TestClient)
- End-to-end transcription
- End-to-end TTS generation
- Error scenarios

### Test Organization
```
tests/
├── test_asr_engine.py       # ASR engine tests
├── test_tts_engine.py       # TTS engine tests
├── test_audio_utils.py      # Audio utility tests
├── test_api_asr.py          # ASR API tests
├── test_api_tts.py          # TTS API tests
└── data/                    # Test fixtures
    ├── sample.wav
    └── sample.mp3
```

### CI/CD Pipeline

GitLab CI stages:
1. **Test**: Run pytest with coverage
2. **Secret Detection**: Scan for leaked secrets
3. **Build** (future): Create Docker images
4. **Deploy** (future): Push to registry

---

## Deployment Architecture

### Single-Host Deployment
```
┌────────────────────────────────────┐
│         Linux Server               │
│  ┌──────────────────────────────┐  │
│  │  Nginx Reverse Proxy         │  │
│  │  :80/:443                    │  │
│  └─────────┬────────────────────┘  │
│            │                       │
│  ┌─────────▼──────┐  ┌───────────┐│
│  │ ASR :5001     │  │ TTS :5002 ││
│  │ 4 workers     │  │ 2 workers ││
│  └───────────────┘  └───────────┘│
│                                   │
│  ┌──────────────────────────────┐│
│  │   NVIDIA GPU (CUDA)          ││
│  └──────────────────────────────┘│
└────────────────────────────────────┘
```

### Multi-Host Deployment
```
┌─────────────────────┐    ┌─────────────────────┐
│   Load Balancer     │    │   Load Balancer     │
│   ASR.example.com   │    │   TTS.example.com   │
└──────────┬──────────┘    └──────────┬──────────┘
           │                          │
    ┌──────▼──────┐            ┌──────▼──────┐
    │ ASR Node 1  │            │ TTS Node 1  │
    │ ASR Node 2  │            │ TTS Node 2  │
    │ ASR Node 3  │            │ TTS Node 3  │
    └─────────────┘            └─────────────┘
```

### Container Deployment (Future)

```yaml
# docker-compose.yml
services:
  asr:
    image: voice-stack-asr:latest
    ports: ["5001:5001"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  tts:
    image: voice-stack-tts:latest
    ports: ["5002:5002"]
    volumes:
      - ./voices:/app/voices
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Security Considerations

### Input Validation
- File size limits (max 25MB recommended)
- Format validation (whitelist: WAV, MP3, M4A, FLAC, OGG)
- Parameter bounds (temperature 0.0-1.0, speed 0.25-4.0)
- Text length limits for TTS

### CORS Configuration
- Explicit origin whitelisting
- No wildcard `*` in production
- Credentials support optional

### Secrets Management
- Never commit `.env` files
- Use environment variables in production
- API keys not required (local services)
- Future: integrate with vault/secrets manager

### Resource Limits
- Request timeout (30s default)
- Memory limits per worker
- Concurrent request limits
- Rate limiting (future)

---

## Monitoring & Observability

### Current Logging
- Structured logs with timestamps
- Request/response logging
- Error stack traces
- Performance metrics (inference time)

### Future Enhancements
- Prometheus metrics endpoint
- Grafana dashboards
- Distributed tracing (OpenTelemetry)
- Health check enhancements (model status)

---

## Future Roadmap

### Short-term
- [ ] Comprehensive test coverage (>80%)
- [ ] Docker/Docker Compose setup
- [ ] Voice sample management API
- [ ] Batch processing endpoints

### Medium-term
- [ ] WebSocket support for streaming ASR
- [ ] Speaker diarization (who spoke when)
- [ ] Multi-speaker TTS
- [ ] Prometheus metrics

### Long-term
- [ ] Kubernetes manifests
- [ ] Auto-scaling based on queue depth
- [ ] Multi-region deployment
- [ ] Model fine-tuning pipeline

---

## Development Guidelines

See `.clinerules/` directory for detailed coding standards:

1. **[01-coding.md](.clinerules/01-coding.md)**: Python style, typing, imports
2. **[02-documentation.md](.clinerules/02-documentation.md)**: Docstring format, README structure
3. **[03-infra-available.md](.clinerules/03-infra-available.md)**: Homelab integration options
4. **[04-repo-mapping.md](.clinerules/04-repo-mapping.md)**: Code base indexing strategy

### Key Principles

- **Type everything**: Use PEP 585 builtins (`list[str]`), PEP 604 unions (`str | None`)
- **Test everything**: Add tests for new features
- **Document everything**: Google-style docstrings, clear READMEs
- **Format everything**: Black + Ruff before commits

---

## Troubleshooting Guide

### Common Issues

**1. CUDA Out of Memory**
```bash
# Use smaller model
ASR_MODEL=tiny

# Or force CPU
ASR_DEVICE=cpu
```

**2. FFmpeg Not Found**
```bash
# macOS
brew install ffmpeg

# Debian/Ubuntu
apt install ffmpeg
```

**3. Model Download Fails**
```bash
# Check internet connection
# Check HuggingFace status
# Try manual download and place in cache
```

**4. Port Already in Use**
```bash
# Change port in .env
ASR_PORT=5011
TTS_PORT=5012
```

**5. Voice Clone Quality Poor**
```bash
# Use better quality reference
# 16kHz or higher
# 6-10 seconds long
# Clear speech, no background noise
```

---

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [Hatch Documentation](https://hatch.pypa.io/)
