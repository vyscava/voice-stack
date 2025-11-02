# Voice-Stack Testing Plan

This document outlines the comprehensive testing strategy for the voice-stack project.

---

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── __init__.py
├── unit/                          # Unit tests (fast, isolated)
│   ├── __init__.py
│   ├── asr/                       # ASR service unit tests
│   │   ├── __init__.py
│   │   ├── test_app.py           # FastAPI app tests
│   │   ├── test_engine_factory.py # Engine factory tests
│   │   ├── test_engine_faster_whisper.py
│   │   ├── test_engine_whisper.py
│   │   ├── test_api_openai.py    # OpenAI endpoints
│   │   ├── test_api_bazarr.py    # Bazarr endpoints
│   │   └── test_schemas.py       # Pydantic schemas
│   ├── tts/                       # TTS service unit tests
│   │   ├── __init__.py
│   │   ├── test_app.py
│   │   ├── test_engine_factory.py
│   │   ├── test_engine_coqui.py
│   │   ├── test_api_openai.py
│   │   └── test_schemas.py
│   ├── core/                      # Core module tests
│   │   ├── __init__.py
│   │   ├── test_settings.py
│   │   └── test_logging.py
│   └── utils/                     # Utils tests
│       ├── __init__.py
│       ├── test_audio_helper.py
│       ├── test_ffmpeg_helper.py
│       ├── test_language_helper.py
│       └── test_text.py
├── integration/                   # Integration tests (slower)
│   ├── __init__.py
│   ├── test_asr_e2e.py           # End-to-end ASR tests
│   └── test_tts_e2e.py           # End-to-end TTS tests
└── data/                          # Test data files
    └── sample_audio.wav
```

---

## Test Categories

### Unit Tests (`tests/unit/`)

**Purpose**: Test individual components in isolation with mocked dependencies.

**Characteristics**:
- Fast (<10ms per test)
- No external dependencies
- Use mocked engines, settings, and external services
- Test single functions/methods
- High coverage of edge cases

**Coverage Target**: 80%+ for unit tests

### Integration Tests (`tests/integration/`)

**Purpose**: Test complete workflows with real or near-real components.

**Characteristics**:
- Slower (100ms-1s per test)
- May require actual model files (marked with `@pytest.mark.requires_models`)
- Test API → Engine → Output pipelines
- Verify OpenAI API compatibility

**Coverage Target**: Key happy paths covered

---

## Testing Strategy by Module

### ASR Service Tests

#### 1. `test_app.py` — FastAPI Application
- ✓ App initialization
- ✓ CORS configuration
- ✓ Health check endpoints (`/health`, `/healthz`, `/healthcheck`)
- ✓ OpenAPI schema generation
- ✓ API route registration

#### 2. `test_engine_factory.py` — Engine Management
- ✓ `get_audio_engine()` returns singleton
- ✓ `reset_audio_engine()` clears singleton
- ✓ `set_audio_engine()` allows manual override
- ✓ Engine selection based on settings (faster-whisper vs whisper)
- ✓ Device selection (CPU vs CUDA)

#### 3. `test_engine_faster_whisper.py` — Faster-Whisper Engine
- ✓ Model loading and caching
- ✓ `transcribe_file()` with various parameters
- ✓ VAD integration
- ✓ Language detection
- ✓ Word timestamps
- ✓ Error handling (invalid audio, missing models)

#### 4. `test_engine_whisper.py` — OpenAI Whisper Engine
- ✓ Model loading
- ✓ `transcribe_file()` basic functionality
- ✓ Translation mode
- ✓ Temperature parameter
- ✓ Error handling

#### 5. `test_api_openai.py` — OpenAI-Compatible Endpoints
- ✓ `POST /v1/audio/transcriptions` (JSON response)
- ✓ `POST /v1/audio/transcriptions` (text response)
- ✓ `POST /v1/audio/transcriptions/verbose` (full metadata)
- ✓ `POST /v1/audio/translations`
- ✓ `GET /v1/models`
- ✓ Invalid file uploads
- ✓ Parameter validation

#### 6. `test_api_bazarr.py` — Bazarr Endpoints
- ✓ `POST /bazarr/asr` (JSON format)
- ✓ `POST /bazarr/asr` (SRT format)
- ✓ `POST /bazarr/asr` (VTT format)
- ✓ `POST /bazarr/detect-language`
- ✓ `GET /bazarr/models`
- ✓ Output format variations

#### 7. `test_schemas.py` — Pydantic Schemas
- ✓ Schema validation
- ✓ Form data parsing (`as_form` methods)
- ✓ Default values
- ✓ Type coercion
- ✓ Validation errors

### TTS Service Tests

#### 1. `test_app.py` — FastAPI Application
- ✓ App initialization
- ✓ Health endpoints
- ✓ Route registration

#### 2. `test_engine_factory.py` — Engine Management
- ✓ Singleton pattern
- ✓ Engine initialization
- ✓ Model loading

#### 3. `test_engine_coqui.py` — Coqui XTTS Engine
- ✓ Model loading
- ✓ `speech()` method
- ✓ Voice sample loading
- ✓ Language detection from text
- ✓ Text chunking
- ✓ Streaming vs file modes
- ✓ Format conversion
- ✓ Speed control

#### 4. `test_api_openai.py` — OpenAI-Compatible Endpoints
- ✓ `POST /v1/audio/speech` (file mode)
- ✓ `POST /v1/audio/speech` (streaming mode)
- ✓ `GET /v1/models`
- ✓ `GET /v1/models/{model_id}`
- ✓ `GET /v1/audio/voices`
- ✓ Format variations (MP3, WAV, OPUS, etc.)
- ✓ Invalid voice names
- ✓ Parameter validation

#### 5. `test_schemas.py` — Pydantic Schemas
- ✓ Request validation
- ✓ Response formatting
- ✓ Default values

### Core Module Tests

#### 1. `test_settings.py` — Configuration
- ✓ Default values
- ✓ Environment variable loading
- ✓ Type validation
- ✓ Invalid values handling

#### 2. `test_logging.py` — Logging
- ✓ Logger initialization
- ✓ Log level configuration
- ✓ Structured logging format

### Utils Tests

#### 1. `test_audio_helper.py` — Audio Processing
- ✓ VAD (Voice Activity Detection)
- ✓ Resampling
- ✓ PCM conversion
- ✓ Chunking

#### 2. `test_ffmpeg_helper.py` — FFmpeg Operations
- ✓ Audio decoding (various formats)
- ✓ Format conversion
- ✓ Sample rate conversion
- ✓ Error handling (corrupt files)

#### 3. `test_language_helper.py` — Language Detection
- ✓ Language code normalization
- ✓ Text-based detection
- ✓ Script detection (Latin, Cyrillic, etc.)

#### 4. `test_text.py` — Text Processing
- ✓ Sentence segmentation
- ✓ Text normalization
- ✓ Word wrapping
- ✓ Punctuation handling

---

## Running Tests

### All Tests
```bash
hatch run test
```

### Specific Service
```bash
hatch run test-asr      # ASR tests only
hatch run test-tts      # TTS tests only
hatch run test-core     # Core tests only
hatch run test-utils    # Utils tests only
```

### With Coverage
```bash
hatch run cov           # All tests with coverage
hatch run cov-asr       # ASR coverage
hatch run cov-tts       # TTS coverage
```

### Fast Tests Only
```bash
hatch run test-fast     # Skip slow tests
```

### By Markers
```bash
pytest -m unit                  # Unit tests only
pytest -m integration           # Integration tests only
pytest -m "not slow"            # Skip slow tests
pytest -m "not requires_models" # Skip tests requiring models
pytest -m "not requires_gpu"    # Skip GPU tests
```

---

## GitLab CI Integration

### Coverage Reporting

The test suite generates coverage reports in multiple formats:
- **Terminal**: Real-time feedback during CI runs
- **HTML**: `htmlcov/index.html` for detailed local review
- **XML**: `coverage.xml` for GitLab CI integration

### GitLab CI Configuration

Add to `.gitlab-ci.yml`:

```yaml
test:
  stage: test
  image: python:3.11
  before_script:
    - pip install hatch
    - hatch env create
  script:
    - hatch run cov
  coverage: '/(?i)total.*? (100(?:\.0+)?\%|[1-9]?\d(?:\.\d+)?\%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
    paths:
      - htmlcov/
    expire_in: 30 days
```

### Coverage Badges

GitLab will automatically generate a coverage badge from the `coverage.xml` file.

---

## Test Writing Guidelines

### 1. Use Descriptive Test Names

```python
def test_transcribe_file_returns_text():
    """Test that transcribe_file returns text string."""
    pass

def test_transcribe_file_with_invalid_audio_raises_error():
    """Test that invalid audio raises appropriate error."""
    pass
```

### 2. Follow AAA Pattern

```python
def test_example():
    # Arrange
    engine = Mock()
    request = {"file": sample_audio}

    # Act
    result = engine.transcribe(request)

    # Assert
    assert result["text"] == "expected text"
```

### 3. Use Fixtures for Setup

```python
@pytest.fixture
def configured_engine():
    engine = ASREngine(model="base", device="cpu")
    return engine

def test_with_fixture(configured_engine):
    result = configured_engine.transcribe(...)
    assert result is not None
```

### 4. Mark Tests Appropriately

```python
@pytest.mark.unit
@pytest.mark.asr
def test_asr_unit():
    pass

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_models
def test_full_pipeline():
    pass
```

### 5. Mock External Dependencies

```python
@pytest.mark.unit
def test_with_mocked_model(mocker):
    mock_model = mocker.patch("whisper.load_model")
    mock_model.return_value = Mock()

    engine = WhisperEngine()
    # Test continues with mocked model
```

### 6. Test Error Cases

```python
def test_transcribe_with_empty_file_raises_error():
    with pytest.raises(ValueError, match="Empty audio file"):
        engine.transcribe(b"")
```

---

## Continuous Integration Workflow

1. **Pre-commit**: Format and lint checks
2. **Test Stage**: Run all tests with coverage
3. **Coverage Report**: Upload coverage to GitLab
4. **Badge Update**: Automatic coverage badge
5. **Artifact Storage**: Save HTML coverage reports

---

## Coverage Goals

| Module | Target Coverage | Current |
|--------|----------------|---------|
| ASR API | 90% | TBD |
| ASR Engines | 80% | TBD |
| TTS API | 90% | TBD |
| TTS Engines | 80% | TBD |
| Core | 85% | TBD |
| Utils | 85% | TBD |
| **Overall** | **80%** | **TBD** |

---

## Future Enhancements

- [ ] Performance benchmarks (pytest-benchmark)
- [ ] Property-based testing (hypothesis)
- [ ] Mutation testing (mutmut)
- [ ] Visual regression tests for output formats
- [ ] Load testing for concurrent requests
- [ ] Fuzzing for input validation

---

## Troubleshooting

### Tests Failing Due to Missing Models
```bash
# Skip model-dependent tests
pytest -m "not requires_models"
```

### Tests Failing on CPU-only Machines
```bash
# Skip GPU tests
pytest -m "not requires_gpu"
```

### Slow Test Suite
```bash
# Run only fast unit tests
pytest -m unit -m "not slow"
```

### Coverage Too Low
1. Identify uncovered lines: `hatch run cov-report`
2. View HTML report: `open htmlcov/index.html`
3. Add targeted tests for uncovered code
