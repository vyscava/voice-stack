# Voice Stack

A self-hosted speech stack for homelabs — ASR (speech-to-text) and TTS (text-to-speech) as a pair of FastAPI microservices with OpenAI-compatible APIs. One unified Docker image runs either service based on a single environment variable. Drop it next to Bazarr for automatic subtitles, plug it into Open WebUI for voice chat, or use it as a local replacement for cloud speech APIs.

**Key features:**

- OpenAI-compatible `/v1/audio/transcriptions` and `/v1/audio/speech` endpoints
- Bazarr-native subtitle endpoints (SRT, VTT, JSON, JSONL)
- Single Docker image, ~6 GB, switches between ASR and TTS via `SERVICE_MODE`
- NVIDIA GPU acceleration with automatic CUDA detection
- Voice cloning via Coqui XTTS-v2 (drop a .wav file, get a new voice)
- Built-in concurrency limits, memory guards, and idle model unloading
- Portainer-ready compose files for homelab deployment

---

## Quick Start (Docker)

The fastest way to get running. Requires Docker and optionally an NVIDIA GPU.

### Build

```bash
git clone https://github.com/vyscava/voice-stack.git
cd voice-stack
docker build -t voice-stack:latest .
```

### Run Both Services

```bash
# ASR on port 5001
docker run -d --name voice-stack-asr \
  --gpus all \
  -p 5001:5001 \
  -e SERVICE_MODE=asr \
  -e ASR_DEVICE=cuda \
  -e ASR_MODEL=base \
  -v asr-models:/app/models \
  voice-stack:latest

# TTS on port 5002
docker run -d --name voice-stack-tts \
  --gpus all \
  -p 5002:5002 \
  -e SERVICE_MODE=tts \
  -e TTS_DEVICE=cuda \
  -v tts-models:/app/models \
  voice-stack:latest
```

No GPU? Replace `--gpus all` with nothing and set `ASR_DEVICE=cpu` / `TTS_DEVICE=cpu`.

### Verify

```bash
curl http://localhost:5001/health   # {"status":"ok"}
curl http://localhost:5002/health   # {"status":"ok"}
```

### Try It

```bash
# Transcribe audio
curl -X POST http://localhost:5001/v1/audio/transcriptions \
  -F "file=@sample.mp3" -F "model=base"

# Generate speech
curl -X POST http://localhost:5002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello from Voice Stack", "voice": "default", "response_format": "mp3"}' \
  --output speech.mp3
```

---

## Deploy with Docker Compose

For running both services together locally:

```bash
docker-compose up -d        # Start ASR + TTS
docker-compose up -d tts    # Start TTS only
docker-compose logs -f      # View logs
docker-compose down         # Stop
```

The default `docker-compose.yml` runs on CPU. To enable GPU, uncomment the `runtime: nvidia` and device lines in the compose file.

---

## Deploy with Portainer

The `compose/` directory contains production-ready stack files designed for Portainer:

| File | Service | Port |
|------|---------|------|
| `compose/asr.yml` | ASR (Faster-Whisper) | 5001 |
| `compose/tts.yml` | TTS (Coqui XTTS-v2) | 5002 |

These use `${VAR}` syntax so you can configure everything through Portainer's stack environment variables — no config files on the host needed.

**To deploy:**

1. Push your built image to a registry (or use a local one)
2. In Portainer, create a new stack from this Git repository
3. Set the compose path to `compose/asr.yml` or `compose/tts.yml`
4. Fill in the environment variables (see [Configuration](#configuration) below)
5. Deploy

Both stacks include GPU reservations, health checks, log rotation, and memory limits out of the box.

---

## Install as Systemd Service (Bare Metal)

If you prefer running directly on the host without Docker:

```bash
cd /opt
git clone https://github.com/vyscava/voice-stack.git
cd voice-stack
sudo ./scripts/install_system_deps.sh

# Install one or both
sudo ./scripts/install-asr.sh
sudo ./scripts/install-tts.sh
```

Each script creates an isolated Python venv, installs PyTorch with CUDA detection, and registers a systemd service. See `scripts/SERVICE_INSTALLATION.md` for full details.

```bash
sudo systemctl status voice-stack-asr
sudo journalctl -u voice-stack-tts -f
```

---

## Configuration

All configuration is via environment variables. Templates with full documentation:

- **ASR**: [`scripts/.env.production.asr`](scripts/.env.production.asr)
- **TTS**: [`scripts/.env.production.tts`](scripts/.env.production.tts)

### Essential Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_MODE` | _(required)_ | `asr` or `tts` — selects which service the container runs |
| `ASR_DEVICE` / `TTS_DEVICE` | `cpu` | Set to `cuda` for GPU acceleration |
| `ASR_MODEL` | `base` | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `ASR_ENGINE` | `fasterwhisper` | `fasterwhisper` (recommended) or `whisper` |
| `TTS_MODEL` | `tts_models/multilingual/multi-dataset/xtts_v2` | Coqui model ID |
| `ASR_PORT` / `TTS_PORT` | `5001` / `5002` | Service ports |
| `LOG_LEVEL` | `info` | `debug`, `info`, `warning`, `error` |
| `ASR_VAD_ENABLED` | `true` | Voice Activity Detection (30-50% faster transcription) |
| `TTS_VOICES_DIR` | `/app/voices` | Directory with .wav samples for voice cloning |

### Resource / Concurrency

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_MAX_CONCURRENT_REQUESTS` | `2` | Max simultaneous transcriptions |
| `TTS_MAX_CONCURRENT_REQUESTS` | `2` | Max simultaneous speech generations |
| `ASR_IDLE_TIMEOUT_MINUTES` | `60` | Unload model after inactivity (0 = never) |
| `MAX_UPLOAD_SIZE_MB` | `100` | Max audio file upload size |

---

## API Reference

Both services expose OpenAI-compatible endpoints, plus extras for Bazarr and monitoring.

### ASR Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/audio/transcriptions` | Transcribe audio (OpenAI-compatible) |
| `POST` | `/v1/audio/transcriptions/verbose` | Transcribe with segment metadata |
| `POST` | `/v1/audio/translations` | Translate audio to English |
| `POST` | `/v1/bazarr/asr` | Subtitle output (SRT, VTT, JSON, TXT, TSV, JSONL) |
| `POST` | `/v1/bazarr/detect-language` | Detect spoken language |
| `GET` | `/v1/models` | List available models |

### TTS Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/audio/speech` | Generate speech (MP3, OPUS, AAC, FLAC, WAV, PCM) |
| `GET` | `/v1/audio/voices` | List available voices |
| `GET` | `/v1/models` | List available models |

### Health Endpoints (Both Services)

| Endpoint | Description |
|----------|-------------|
| `/health` | Simple liveness check |
| `/healthz`, `/healthcheck` | Kubernetes / Docker compatible probes |
| `/health/detailed` | Memory, swap, GPU, concurrency slots, model state |

---

## Voice Cloning (TTS)

Add `.wav` files to the voices directory and they become available as voice options:

```bash
# Add a voice sample (6-12 seconds, clear audio, single speaker)
cp my-voice.wav voices/

# Use it
curl -X POST http://localhost:5002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello in my voice", "voice": "my-voice"}'
```

Mount the voices directory in Docker: `-v ./voices:/app/voices:ro`

---

## Resource Requirements

| Setup | CPU | RAM | Disk | Notes |
|-------|-----|-----|------|-------|
| ASR (CPU, base model) | 4+ cores | 4 GB | 2 GB | Good for short clips |
| ASR (GPU, large-v3) | 2+ cores | 4 GB + 4 GB VRAM | 6 GB | Best accuracy |
| TTS (CPU) | 4+ cores | 8 GB | 4 GB | Slow but works |
| TTS (GPU) | 2+ cores | 4 GB + 4 GB VRAM | 4 GB | Recommended for real-time |
| Both (GPU) | 4+ cores | 8 GB + 8 GB VRAM | 10 GB | Full stack on one machine |

Models are downloaded on first request and cached in Docker volumes.

---

## Development

```bash
git clone https://github.com/vyscava/voice-stack.git
cd voice-stack
pip install hatch nox pre-commit
hatch env create
pre-commit install

# Run with hot reload
hatch run run_asr    # ASR on :5001
hatch run run_tts    # TTS on :5002

# Quality checks
hatch run fmt        # Format (Black + Ruff)
hatch run lint       # Lint
hatch run test       # All tests
hatch run cov        # Coverage report
```

Swagger UI available at `/docs` on each running service.

---

## Repository Structure

```
voice-stack/
├── src/
│   ├── asr/               # ASR microservice (Faster-Whisper, OpenAI Whisper)
│   ├── tts/               # TTS microservice (Coqui XTTS-v2)
│   ├── core/              # Shared settings, logging, middleware
│   └── utils/             # Audio processing, language detection, text helpers
├── compose/               # Portainer-ready production stacks (asr.yml, tts.yml)
├── scripts/               # Installers, systemd helpers, build automation
├── docs/                  # Architecture, concurrency, load testing, cheat sheets
├── voices/                # Voice samples for TTS cloning
├── tests/                 # Unit + integration test suites
├── Dockerfile             # Unified production image (~6 GB)
├── Dockerfile.ci          # CI image with dev dependencies
├── docker-compose.yml     # Local development compose
├── pyproject.toml         # Hatch project configuration
└── .gitlab-ci.yml         # CI/CD pipeline
```

---

## Further Reading

- [`DOCKER_DEPLOYMENT.md`](DOCKER_DEPLOYMENT.md) — Docker builds, CI pipeline, release process, GPU setup
- [`scripts/SERVICE_INSTALLATION.md`](scripts/SERVICE_INSTALLATION.md) — Systemd installation, updates, troubleshooting
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — Component design and request flows
- [`docs/CONCURRENCY_FEATURES.md`](docs/CONCURRENCY_FEATURES.md) — Resource guards, semaphores, idle unloading
- [`docs/LOAD_TESTING.md`](docs/LOAD_TESTING.md) — Performance benchmarks
- [`docs/WHISPER_CHEAT_SHEET.md`](docs/WHISPER_CHEAT_SHEET.md) — Model size vs accuracy tradeoffs

---

## Acknowledgements

- [SubGen](https://github.com/McCloudS/subgen) — Bazarr FastAPI project whose subtitle routing design influenced the Bazarr endpoints here
- [whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice) — Inspiration for OpenAI-compatible ASR workflows

---

## License

MIT License &copy; 2025 Vitor Bordini Garbim. TTS components also require acceptance of the [Coqui Public Model License](https://coqui.ai/cpml); see `scripts/accept_coqui_license.sh` for the automated acceptance flow.
