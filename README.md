
# 🗣️ Voice Stack — ASR + TTS Microservices

Production-ready **speech services** with **Automatic Speech Recognition (ASR)** and **Text-to-Speech (TTS)** featuring OpenAI-compatible APIs.

Designed for **local development** on macOS and **production deployment** on Linux (Debian 12 + NVIDIA GPU).

---

## 🚀 Key Features

### ASR (Automatic Speech Recognition)
- **OpenAI-compatible API**: Drop-in replacement for OpenAI's Whisper API
- **Multiple engines**: Faster-Whisper and OpenAI Whisper
- **Advanced features**:
  - Voice Activity Detection (VAD) with Silero
  - Language auto-detection
  - Translation to English
  - Word-level timestamps
  - Verbose output mode with segments
- **Supported formats**: WAV, MP3, M4A, FLAC, OGG (via FFmpeg)

### TTS (Text-to-Speech)
- **OpenAI-compatible API**: Compatible with OpenAI's TTS endpoint
- **Coqui XTTS-v2**: Multi-lingual neural TTS
- **Voice cloning**: Load custom `.wav` voice samples
- **Streaming support**: Real-time audio generation with SSE
- **Multiple formats**: MP3, OPUS, AAC, FLAC, WAV, PCM

### Infrastructure
- **FastAPI**: High-performance async web framework
- **Cross-platform**: macOS (Metal/CPU) and Linux (CUDA/CPU)
- **Health checks**: Standard `/health`, `/healthz`, `/healthcheck` endpoints
- **CORS**: Configurable cross-origin support
- **Remote debugging**: Built-in debugpy support
- **Hatch + Nox**: Modern Python toolchain for dev, test, and build
- **CI/CD**: GitLab pipelines with testing and secret detection

---

## 🧰 Prerequisites

### macOS (Development)
- **Python 3.11+** (3.10 minimum)
- **Hatch** (>=1.13) — `pip install hatch`
- **Nox** (>=2024.4) — `pip install nox`
- **pre-commit** (>=3.5) — `pip install pre-commit`
- **FFmpeg** — `brew install ffmpeg`

### Debian 12 (Production)
- **Python 3.11+**
- **FFmpeg** — `apt install ffmpeg`
- **CUDA 12.x + cuDNN** (for GPU acceleration)
- **NVIDIA drivers** (525+ recommended)
- Optional: **Docker** for containerized deployment

---

## 🧑‍💻 Development Setup

1. **Clone the repository**
   ```bash
   git clone https://gitlab.vitorgarbim.me/data-platforms/voice-stack.git
   cd voice-stack
   ```

2. **Install tooling (if not already)**
   ```bash
   pip install --upgrade hatch nox pre-commit
   ```

3. **Create and activate the dev environment**
   ```bash
   # 1. Install dependencies and create the virtual environment
   hatch env create

   # 2. Activate the environment
   hatch shell
   ```

4. **Run ASR and TTS in development mode**
   ```bash
   # In separate terminals:
   hatch run run_asr    # Starts on http://0.0.0.0:5001
   hatch run run_tts    # Starts on http://0.0.0.0:5002
   ```

   Features:
   - **Auto-reload** on code changes in `/src`
   - **Interactive docs** at `/docs` (Swagger UI)
   - **Health checks** at `/health`, `/healthz`, `/healthcheck`

   For remote debugging:
   ```bash
   hatch run run_asr_dbg    # Debugpy on 0.0.0.0:5678
   hatch run run_tts_dbg    # Debugpy on 0.0.0.0:5678
   ```

5. **Install pre-commit hooks (recommended)**
   ```bash
   pre-commit install
   ```
   Commits now run `nox -s fmt` (auto-format + import sort) and `nox -s lint` (Black check + Ruff).

6. **Format, lint, and type-check code**
   ```bash
   hatch run fmt          # Auto-format with Black + Ruff
   hatch run lint         # Check style and types
   hatch run typecheck    # Run mypy
   ```

7. **Run tests**
   ```bash
   hatch run test         # Quick test run
   hatch run cov          # With coverage report
   ```

8. **Deactivate environment**
   ```bash
   exit
   ```

---

## 🚀 Production Deployment (Systemd Services)

For production deployment on Debian/Ubuntu Linux, Voice Stack provides automated systemd service installation scripts.

### Quick Installation

```bash
# Clone repository to your desired location
cd /opt  # or any location you prefer
git clone https://gitlab.vitorgarbim.me/data-platforms/voice-stack.git
cd voice-stack

# Install system dependencies
sudo ./scripts/install_system_deps.sh

# Install ASR service
sudo ./scripts/install-asr.sh

# Install TTS service
sudo ./scripts/install-tts.sh
```

### Service Management

```bash
# Start/Stop/Restart services
sudo systemctl start voice-stack-asr
sudo systemctl stop voice-stack-asr
sudo systemctl restart voice-stack-asr

# View logs
sudo journalctl -u voice-stack-asr -f

# Check service status
sudo systemctl status voice-stack-asr
```

### Updating Services

After pulling code changes:

```bash
cd /path/to/voice-stack
git pull --rebase
sudo ./scripts/update-asr.sh
sudo ./scripts/update-tts.sh
```

### Full Documentation

For complete installation, configuration, and troubleshooting guides:

📖 **[Service Installation Guide](scripts/SERVICE_INSTALLATION.md)**

Topics covered:
- Installation (system-wide or user-specific)
- Configuration (.env setup)
- GPU/CUDA configuration
- Service management and monitoring
- Updating and uninstalling services
- Troubleshooting common issues
- Advanced topics (Nginx proxy, custom paths, etc.)

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file or export variables:

```bash
# Service Configuration
PROJECT_NAME="Voice Stack ASR"           # API title
API_V1_STR="/v1"                        # API prefix
CORS_ORIGINS="http://localhost:3000"    # Comma-separated origins

# ASR Settings
ASR_DEVICE="cpu"                        # "cuda" or "cpu"
ASR_MODEL="base"                        # Whisper model: tiny/base/small/medium/large
ASR_ENGINE="faster-whisper"             # "faster-whisper" or "whisper"
ASR_VAD_ENABLED="true"                  # Enable Voice Activity Detection
ASR_LANGUAGE=""                         # Force language (empty = auto-detect)

# TTS Settings
TTS_DEVICE="cpu"                        # "cuda" or "cpu"
TTS_MODEL="tts_models/multilingual/multi-dataset/xtts_v2"
TTS_VOICE_DIR="./voices"                # Directory for voice samples
TTS_AUTO_LANG="true"                    # Auto language detection
TTS_MAX_CHARS="180"                     # Max chars per TTS chunk
TTS_SAMPLE_RATE="24000"                 # Output sample rate

# Logging
LOG_LEVEL="INFO"                        # DEBUG, INFO, WARNING, ERROR

# Remote Debugging (optional)
DEBUGPY_ENABLE="false"
DEBUGPY_HOST="0.0.0.0"
DEBUGPY_PORT="5678"
DEBUGPY_WAIT_FOR_CLIENT="false"
```

### Voice Samples for TTS

Place `.wav` files in the `voices/` directory:
```
voices/
  ├── speaker_en.wav    # English voice sample
  ├── speaker_pt.wav    # Portuguese voice sample
  └── speaker_es.wav    # Spanish voice sample
```

Reference voices in API calls using the filename (without extension).

---

## 🧪 Build & Packaging

### Build the project
```bash
hatch build
```
This creates a wheel and source distribution under `dist/`.

### Run tests in isolation
```bash
nox -s tests
```
This executes the full suite across the supported Python versions. For targeted runs (matching the CI split) you can use:
```bash
hatch run test-asr         # Unit tests for ASR
hatch run test-tts         # Unit tests for TTS
hatch run test-core        # Core-only tests
hatch run test-utils       # Utility tests
hatch run test-integration # Integration tests
```

### Format and lint with Nox
```bash
nox -s fmt lint
```
`fmt` mutates files using Black and Ruff fixes for import order; `lint` performs the non-mutating checks.

### Nox sessions at a glance
- `nox -s fmt` — auto-format via Black and Ruff import fixes
- `nox -s lint` — format check + Ruff lint (read-only)
- `nox -s typecheck` — run MyPy with the project configuration
- `nox -s tests` — pytest suite under each configured Python interpreter
- `nox -s ci` — convenience bundle (Black check, Ruff, MyPy, pytest)

### Pre-commit hooks (recap)
Hooks are installed in the development setup (step 5). Re-run `pre-commit install` if the repo’s hook configuration changes.

---

## 📁 Project Structure

```
voice-stack/
├── pyproject.toml              # Hatch project config + dependencies
├── noxfile.py                  # Nox sessions for CI/test automation
├── .gitlab-ci.yml              # GitLab CI/CD pipeline
├── .clinerules/                # AI agent coding guidelines
│   ├── 01-coding.md
│   ├── 02-documentation.md
│   ├── 03-infra-available.md
│   └── 04-repo-mapping.md
├── scripts/
│   ├── build_repo_map.py       # Generate code base index
│   └── repo_tokens.py          # Token analysis tool
├── src/
│   ├── asr/                    # ASR microservice
│   │   ├── app.py             # FastAPI application
│   │   ├── engine_factory.py  # Engine initialization
│   │   ├── api/api_v1/        # API routes
│   │   │   └── endpoints/
│   │   │       ├── openai.py  # OpenAI-compatible endpoints
│   │   │       └── bazarr.py  # Bazarr subtitle integration
│   │   ├── engine/            # ASR engines
│   │   │   ├── base.py
│   │   │   ├── whisper.py
│   │   │   └── fasterwhisper.py
│   │   └── schemas/           # Pydantic models
│   ├── tts/                    # TTS microservice
│   │   ├── app.py
│   │   ├── engine_factory.py
│   │   ├── api/api_v1/
│   │   │   └── endpoints/
│   │   │       └── openai.py  # OpenAI-compatible endpoints
│   │   ├── engine/
│   │   │   ├── base.py
│   │   │   └── coqui.py       # Coqui XTTS engine
│   │   └── schemas/
│   ├── core/                   # Shared core logic
│   │   ├── settings.py        # Pydantic settings
│   │   └── logging.py         # Structured logging
│   └── utils/                  # Shared utilities
│       ├── audio/             # FFmpeg, resampling, VAD
│       ├── language/          # Language detection
│       ├── queue_manager.py   # Task queueing
│       └── text.py            # Text normalization
├── tests/                      # Test suite (pytest)
├── voices/                     # TTS voice samples (.wav)
├── .docs/                      # Generated documentation
│   ├── README.md
│   └── repo_map.json          # Code base index
└── README.md
```

## 📚 Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [Whisper Cheat Sheet](docs/WHISPER_CHEAT_SHEET.md)
- [Generated Code Index](.docs/README.md)

---

## 🧩 Common Commands

| Task | Command |
|------|----------|
| **Development** |
| Activate environment | `hatch shell` |
| Start ASR service | `hatch run run_asr` |
| Start TTS service | `hatch run run_tts` |
| Start with debugger | `hatch run run_asr_dbg` / `run_tts_dbg` |
| **Code Quality** |
| Format code | `hatch run fmt` |
| Lint code | `hatch run lint` |
| Type check | `hatch run typecheck` |
| Run tests | `hatch run test` |
| Coverage report | `hatch run cov` |
| **Build & Deploy** |
| Build package | `hatch build` |
| Generate code map | `python scripts/build_repo_map.py` |
| **Nox (CI/Test)** |
| Run all tests | `nox` |
| Specific session | `nox -s tests` / `nox -s lint` |

---

## 🔌 API Endpoints

### ASR Service (Port 5001)

**OpenAI-compatible:**
- `POST /v1/audio/transcriptions` — Transcribe audio (minimal response)
- `POST /v1/audio/transcriptions/verbose` — Transcribe with full metadata
- `POST /v1/audio/translations` — Translate to English
- `GET /v1/models` — List available models

**Bazarr Integration:**
- `POST /bazarr/asr` — Transcribe for subtitle generation (supports JSON, SRT, VTT, TXT, TSV, JSONL)
- `POST /bazarr/detect-language` — Detect language from audio
- `GET /bazarr/models` — List available models

**Health:**
- `GET /health`, `/healthz`, `/healthcheck` — Health check endpoints

**Example (cURL):**
```bash
# OpenAI-compatible transcription
curl -X POST "http://localhost:5001/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.mp3" \
  -F "model=base" \
  -F "language=en"

# Bazarr subtitle generation (JSON format)
curl -X POST "http://localhost:5001/bazarr/asr" \
  -F "file=@audio.mp3" \
  -F "language=en" \
  -F "output=json"

# Bazarr subtitle generation (SRT format)
curl -X POST "http://localhost:5001/bazarr/asr" \
  -F "file=@audio.mp3" \
  -F "language=en" \
  -F "output=srt" \
  --output subtitles.srt

# Bazarr language detection
curl -X POST "http://localhost:5001/bazarr/detect-language" \
  -F "file=@audio.mp3"
```

### TTS Service (Port 5002)

**OpenAI-compatible:**
- `POST /v1/audio/speech` — Generate speech from text
- `GET /v1/models` — List available models
- `GET /v1/models/{model_id}` — Get specific model info
- `GET /v1/audio/voices` — List available voices

**Example (cURL):**
```bash
curl -X POST "http://localhost:5002/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, world!",
    "voice": "speaker_en",
    "response_format": "mp3",
    "speed": 1.0
  }' \
  --output speech.mp3
```

## 🔧 Advanced Usage

### ASR with OpenAI SDK

```python
from openai import OpenAI

# Point to local ASR service
client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:5001/v1"
)

# Transcribe audio
with open("audio.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="base",
        file=f
    )
print(transcript.text)
```

### ASR with Bazarr Integration

```python
import requests

# Transcribe for subtitles (JSON format with segments)
url = "http://localhost:5001/bazarr/asr"
files = {"file": open("movie.mkv", "rb")}
data = {
    "language": "en",
    "output": "json"  # Options: json, srt, vtt, txt, tsv, jsonl
}

response = requests.post(url, files=files, data=data)
subtitle_data = response.json()

print(f"Detected language: {subtitle_data['language']}")
for segment in subtitle_data['segments']:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")

# Detect language only
url = "http://localhost:5001/bazarr/detect-language"
files = {"file": open("audio.mp3", "rb")}

response = requests.post(url, files=files)
lang_info = response.json()
print(f"Language: {lang_info['language_code']} (confidence: {lang_info['confidence']})")
```

### TTS with OpenAI SDK

```python
from openai import OpenAI

# TTS with local service
client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:5002/v1"
)

response = client.audio.speech.create(
    model="xtts_v2",
    voice="speaker_en",
    input="Hello from Voice Stack!"
)
response.stream_to_file("output.mp3")
```

### Docker Deployment

```bash
# Build services
docker build -t voice-stack-asr -f Dockerfile.asr .
docker build -t voice-stack-tts -f Dockerfile.tts .

# Run with GPU support
docker run -d --gpus all \
  -p 5001:5001 \
  -e ASR_DEVICE=cuda \
  voice-stack-asr

docker run -d --gpus all \
  -p 5002:5002 \
  -e TTS_DEVICE=cuda \
  -v ./voices:/app/voices \
  voice-stack-tts
```

## 🧪 Testing

Run the test suite:
```bash
# All tests with coverage
hatch run cov

# Specific test file
hatch run test tests/test_asr.py

# With Nox (isolated environments)
nox -s tests
```

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce batch size in settings
- Use smaller Whisper model (tiny/base instead of large)
- Enable model offloading: `ASR_DEVICE=cpu`

### Audio Format Not Supported
- Ensure FFmpeg is installed: `ffmpeg -version`
- Check supported formats: WAV, MP3, M4A, FLAC, OGG

### Voice Clone Quality Issues
- Use high-quality `.wav` samples (16kHz+, mono)
- Sample should be 6-10 seconds long
- Clear speech without background noise

### Remote Debugging Not Working
```bash
# Enable debugpy in .env
DEBUGPY_ENABLE=true
DEBUGPY_HOST=0.0.0.0
DEBUGPY_PORT=5678
DEBUGPY_WAIT_FOR_CLIENT=true

# Run service
hatch run run_asr_dbg

# Connect from VS Code (see .vscode/launch.json)
```

## 📚 Documentation

- **[ARCHITECTURE.md](./ARCHITECTURE.md)** — System design and architecture
- **[WHISPER_CHEAT_SHEET.md](./WHISPER_CHEAT_SHEET.md)** — Whisper model guide
- **[SERVICE_INSTALLATION.md](./scripts/SERVICE_INSTALLATION.md)** — Production deployment with systemd

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Follow coding standards: `hatch run fmt && hatch run lint`
4. Write tests: `hatch run test`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📊 Performance

**ASR Benchmarks** (on NVIDIA RTX 4060 Ti):
- Faster-Whisper (base): ~10x real-time
- Faster-Whisper (large-v2): ~3x real-time
- OpenAI Whisper (base): ~5x real-time

**TTS Benchmarks**:
- XTTS-v2: ~0.8-1.2s latency for first chunk
- Streaming: Near real-time with chunking

## 🛣️ Roadmap

- [ ] Docker Compose setup
- [ ] WebSocket streaming for ASR
- [ ] Speaker diarization
- [ ] Multi-language voice cloning
- [ ] Prometheus metrics
- [ ] Kubernetes deployment manifests
- [ ] Voice sample management API
- [ ] Batch processing endpoints

## 🧠 Technical Notes

- **Cross-platform**: Develop on macOS (Metal/CPU) → Deploy on Linux (CUDA)
- **Voice samples**: Place in `/voices/` with naming convention `{speaker}_{lang}.wav`
- **Production**: Use Gunicorn/Uvicorn with multiple workers
- **Scaling**: Deploy ASR and TTS as separate services behind load balancer

---

## 📜 License

MIT License © 2025 Vitor Bordini Garbim
