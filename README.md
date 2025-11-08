# Voice Stack

Voice Stack is the speech stack I built for my own homelab so I could keep ASR and TTS workloads close to the data I care about—mostly Bazarr in my media server and my OpenWebUI containers so I can literally talk to my AI. It now doubles as a practical portfolio piece: a pair of FastAPI microservices, tuned for Debian 12 on bare metal with NVIDIA GPUs, but still easy to iterate on from a laptop. This README is the canonical entry point—installation, container builds, development workflow, and deep links to every supporting document all live here.

---

## Table of Contents

- [Install as Debian Service (systemd)](#install-as-debian-service-systemd)
- [Build the Production Container Image](#build-the-production-container-image)
- [Development Environment and Workflow](#development-environment-and-workflow)
- [Configuration Overview](#configuration-overview)
- [Environment Variables (Services and Containers)](#environment-variables-services-and-containers)
- [API Surface (OpenAI-Compatible Highlights)](#api-surface-openai-compatible-highlights)
- [Reference Materials](#reference-materials)
- [Repository Structure](#repository-structure)
- [Licensing](#licensing)

---

## Install as Debian Service (systemd)

The project ships with scripted installers that provision isolated Hatch environments, accept the Coqui license, configure `.env`, and register long-running services. Use them whenever you want hands-free setup on Debian 11+/Ubuntu 20.04+.

### Requirements

- Python 3.10+ with `pip`
- `git`, `ffmpeg`, `libsndfile1`, `libportaudio2`, `build-essential`
- Optional but recommended: CUDA 12.1+, cuDNN, and NVIDIA drivers ≥ 525 for GPU workloads

Install the prerequisites with the helper script:

```bash
cd /opt
git clone https://github.com/vyscava/voice-stack.git
cd voice-stack
sudo ./scripts/install_system_deps.sh
```

### Install One or Both Services

```bash
# Automatic ASR installation (system service)
sudo ./scripts/install-asr.sh

# Automatic TTS installation (system service)
sudo ./scripts/install-tts.sh

# To install as user services instead, omit sudo.
```

Each script will:

1. Bootstrap Hatch if needed and create `.venv-asr`/`.venv-tts`
2. Install PyTorch (CUDA-aware when GPUs are present)
3. Render `.env` from the template and prompt for overrides when missing
4. Register and start the systemd unit (`voice-stack-asr` or `voice-stack-tts`)

### Manage, Inspect, Update

```bash
# Health
curl http://localhost:5001/health    # ASR
curl http://localhost:5002/health    # TTS

# Service lifecycle
sudo systemctl status voice-stack-asr
sudo systemctl restart voice-stack-tts

# Logs
sudo journalctl -u voice-stack-asr -f

# Updating after pulling new code
git pull --rebase
sudo ./scripts/update-asr.sh
sudo ./scripts/update-tts.sh
```

Troubleshooting, environment variable details, and uninstall steps are fully documented in `scripts/SERVICE_INSTALLATION.md`. Refer to that guide whenever you need to override the unit files, relocate model caches, or run behind a reverse proxy.

---

## Build the Production Container Image

The repository maintains a single production image (`Dockerfile`) that switches between ASR and TTS via `SERVICE_MODE`. CUDA runtimes are detected automatically when the container is launched with `--gpus`.

### Build

```bash
docker build -t voice-stack:latest .
```

### Run with CUDA

```bash
# ASR on GPU
docker run --rm -d \
  --gpus all \
  -p 5001:5001 \
  -e SERVICE_MODE=asr \
  -e ASR_DEVICE=cuda \
  voice-stack:latest

# TTS on GPU, mounting custom voices
docker run --rm -d \
  --gpus all \
  -p 5002:5002 \
  -e SERVICE_MODE=tts \
  -e TTS_DEVICE=cuda \
  -v $(pwd)/voices:/app/voices \
  voice-stack:latest
```

### Quick Verification

```bash
# Minimal ASR transcription
curl -X POST http://localhost:5001/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.mp3" \
  -F "model=base"

# Minimal TTS generation
curl -X POST http://localhost:5002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello from Voice Stack", "voice": "speaker_en", "response_format": "mp3"}' \
  --output speech.mp3
```

More involved deployment patterns—Compose files, CI images, release automation, and troubleshooting—are captured in `DOCKER_DEPLOYMENT.md`.

---

## Development Environment and Workflow

Voice Stack was designed for repeatable local development with Hatch managing environments and Nox mirroring CI. The flow below keeps both ASR and TTS services runnable with hot reload while staying aligned with the automation that publishes releases.

### Tooling Prerequisites

| Platform | Requirements |
| --- | --- |
| macOS (dev) | Python 3.11+, `brew install ffmpeg`, `pip install --upgrade hatch nox pre-commit`, optional `uv` |
| Debian/Ubuntu (dev/prod) | Python 3.11+, `apt install ffmpeg libsndfile1 libportaudio2`, CUDA 12.1+ for GPU |

### Bootstrap Steps

```bash
git clone https://github.com/vyscava/voice-stack.git
cd voice-stack
python -m pip install --upgrade hatch nox pre-commit
hatch env create          # creates the default dev environment
pre-commit install
```

Run the services in separate shells:

```bash
hatch run run_asr
hatch run run_tts
```

Both commands expose Swagger UI under `/docs` and reload automatically when code under `src/` changes.

### Hatch Command Cheat Sheet

| Purpose | Command |
| --- | --- |
| Launch dev shell | `hatch shell` |
| Start ASR / TTS | `hatch run run_asr` / `hatch run run_tts` |
| Start with debugpy | `hatch run run_asr_dbg` / `hatch run run_tts_dbg` |
| Format code | `hatch run fmt` |
| Ruff + lint checks | `hatch run lint` |
| MyPy type checking | `hatch run typecheck` |
| Tests (unit mix) | `hatch run test` |
| Coverage suite | `hatch run cov` |

### Nox Sessions

`nox` mirrors the CI layout and is the recommended way to validate changes against multiple Python interpreters:

- `nox -s fmt` — formatting (Black + Ruff import fixes)
- `nox -s lint` — Ruff linting plus style enforcement
- `nox -s typecheck` — MyPy using `pyproject.toml` settings
- `nox -s tests` — full pytest matrix across configured Python versions
- `nox -s ci` — convenience bundle for local CI parity

### Recommended Development Loop

1. Create or update feature branches normally (`git checkout -b feature/...`).
2. Keep `.env` synced with `example.env` or the service installers if you need custom defaults.
3. Run `hatch run fmt && hatch run lint` before committing; pre-commit hooks enforce the same pair.
4. Execute the relevant `hatch run test-*` command (ASR, TTS, core, utils, integration) or just `nox -s tests` for wider coverage.
5. When touching Docker or deployment files, rebuild locally (`docker-compose build`) so Compose + GPU flags remain valid.

### Merge Request Workflow

- After staging or committing work, run `pre-commit run` so the configured Nox-backed hooks reformat and lint the files touched in your change set.
- If you need a repo-wide sweep—for example before a major release—use `pre-commit run --all-files` to execute the same Nox checkers across the entire tree.
- Push only after these hooks pass locally; the GitLab pipeline runs the exact same tooling, so keeping them green avoids MR churn.

---

## Configuration Overview

All runtime configuration is expressed through environment variables; both the installers and Docker entrypoint honor `.env` values. The most commonly tuned fields are:

| Category | Variables |
| --- | --- |
| General API | `PROJECT_NAME`, `API_V1_STR`, `CORS_ORIGINS`, `LOG_LEVEL` |
| ASR | `ASR_DEVICE`, `ASR_MODEL`, `ASR_ENGINE`, `ASR_VAD_ENABLED`, `ASR_LANGUAGE` |
| TTS | `TTS_DEVICE`, `TTS_MODEL`, `TTS_VOICE_DIR`, `TTS_AUTO_LANG`, `TTS_SAMPLE_RATE`, `TTS_MAX_CHARS` |
| Debug | `DEBUGPY_ENABLE`, `DEBUGPY_HOST`, `DEBUGPY_PORT`, `DEBUGPY_WAIT_FOR_CLIENT` |

The `.env` generated by the install scripts includes inline comments. For hand-crafted environments, consult `scripts/SERVICE_INSTALLATION.md#configuration`.

---

## Environment Variables (Services and Containers)

Both the systemd installers and Docker workflows ultimately rely on the same variable surface. Use [`scripts/.env.production.asr`](scripts/.env.production.asr) and [`scripts/.env.production.tts`](scripts/.env.production.tts) as the authoritative references when promoting changes across environments.

### Service Installers (`.env` next to the code)

| Scope | Key Variables | Notes |
| --- | --- | --- |
| Server/runtime | `ENV`, `HOST`, `ASR_PORT`/`TTS_PORT`, `LOG_LEVEL`, `CORS_ORIGINS` | Mirrors FastAPI settings regardless of service mode |
| Engines | `ASR_ENGINE`, `ASR_MODEL`, `ASR_DEVICE`, `ASR_COMPUTE_TYPE`, `ASR_CPU_THREADS`, `ASR_NUM_OF_WORKERS` | CPU/GPU tuning plus model cache paths |
| TTS specifics | `TTS_ENGINE`, `TTS_MODEL`, `TTS_DEVICE`, `TTS_VOICES_DIR`, `TTS_SAMPLE_RATE`, `TTS_MAX_CHARS`, `TTS_MIN_CHARS` | Voice cloning, chunking, and audio format controls |
| Language controls | `ASR_LANGUAGE`, `TTS_DEFAULT_LANG`, `TTS_AUTO_LANG`, `TTS_LANG_HINT`, `TTS_FORCE_LANG` | Force or hint language detection |
| Diagnostics | `DEBUGPY_ENABLE`, `DEBUGPY_HOST`, `DEBUGPY_PORT`, `DEBUGPY_WAIT_FOR_CLIENT` | Switch off for production services |

Install scripts will prompt for overrides when `.env` is missing, but you can also copy the production templates directly and keep per-service overrides in the same file.

### Docker/Compose (`SERVICE_MODE` driven)

The container image reads the same `.env` values, but Compose provides sensible defaults in `docker-compose.yml`:

| Container | Key Variables | Description |
| --- | --- | --- |
| Shared | `SERVICE_MODE` (`asr` or `tts`), `ENV`, `LOG_LEVEL`, `HOST`, `CORS_ORIGINS` | Determines which FastAPI app boots |
| ASR | `ASR_PORT`, `ASR_ENGINE`, `ASR_DEVICE`, `ASR_MODEL`, `ASR_COMPUTE_TYPE`, `ASR_BEAM_SIZE`, `ASR_MAX_WORKERS` | Swap `ASR_DEVICE` to `cuda` when running with `--gpus all` |
| TTS | `TTS_PORT`, `TTS_ENGINE`, `TTS_DEVICE`, `TTS_MODEL`, `TTS_SAMPLE_RATE`, `TTS_MAX_CHARS`, `TTS_MIN_CHARS`, `TTS_RETRY_STEPS`, `TTS_DEFAULT_LANG`, `TTS_AUTO_LANG`, `TTS_VOICES_DIR` | Mount `./voices` into `/app/voices` for custom clones |

You can mount a single `.env` into the container (`- ./.env:/app/.env:ro`) to avoid duplicating values. Anything not explicitly set in Compose falls back to the defaults described in the production templates.

---

## API Surface (OpenAI + Bazarr Compatible)

Both services maintain OpenAI-compatible routes for drop-in SDK use, while the ASR side exposes native Bazarr endpoints that I rely on in my media server stack. Health instrumentation also includes `/health/detailed` for resource and model telemetry.

| Service | Endpoint | Notes |
| --- | --- | --- |
| ASR (OpenAI) | `POST /v1/audio/transcriptions` | Multipart uploads for Whisper transcription |
| ASR (OpenAI) | `POST /v1/audio/transcriptions/verbose` | Segment-level metadata output |
| ASR (OpenAI) | `POST /v1/audio/translations` | Translate to English |
| ASR (Bazarr) | `POST /bazarr/asr` | Subtitle-friendly output formats (JSON, SRT, VTT, TXT, TSV, JSONL) |
| ASR (Bazarr) | `POST /bazarr/detect-language` | Language detection with tunable offsets |
| TTS (OpenAI) | `POST /v1/audio/speech` | Multi-format (MP3, OPUS, AAC, FLAC, WAV, PCM) generation |
| TTS (OpenAI) | `GET /v1/audio/voices` | Enumerate cloned voice samples |
| Both | `GET /v1/models` | Discover loaded models |
| Both | `/health`, `/healthz`, `/healthcheck` | Liveness probes |
| Both | `/health/detailed` | Exposes memory/swap usage, active concurrency slots, and model load state (see `src/asr/app.py` and `src/tts/app.py`) |

See `tests/` for cURL, SDK, and integration test coverage whenever you need concrete payloads.

---

## Reference Materials

- `scripts/SERVICE_INSTALLATION.md` — full systemd guide (installation, updates, troubleshooting, advanced networking)
- `DOCKER_DEPLOYMENT.md` — Docker/Compose builds, CI images, release automation, and operational tips
- `docs/ARCHITECTURE.md` — component breakdowns, request flows, and diagrams; use it when explaining the platform
- `docs/CONCURRENCY_FEATURES.md` — thread-safety and queueing notes for high-throughput workloads
- `docs/LOAD_TESTING.md` — methodology and reference numbers for stress tests
- `docs/WHISPER_CHEAT_SHEET.md` — practical parameters for each Whisper model size
- `.docs/README.md` — generated repository map linking symbols to files (useful when navigating large updates)

---

## Acknowledgements

- Shout out to [SubGen](https://github.com/McCloudS/subgen), the Bazarr FastAPI project whose subtitle-friendly routing and schema design heavily influenced the Bazarr endpoints in this stack.
- Shout out to [whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice) for the early inspiration around OpenAI-compatible ASR workflows and deployment ergonomics.

---

## Repository Structure

```
voice-stack/
├── src/
│   ├── asr/               # ASR microservice
│   ├── tts/               # TTS microservice
│   ├── core/              # Shared settings/logging
│   └── utils/             # Audio, language, queues, text helpers
├── scripts/               # Installers, system helpers, tooling automation
├── docs/                  # Architecture, concurrency, load testing, cheat sheets
├── voices/                # Sample voice seeds for TTS cloning
├── tests/                 # Pytest suites (unit + integration)
├── Dockerfile             # Unified production image
├── Dockerfile.ci          # CI automation image
├── docker-compose.yml     # Local multi-service orchestration
├── pyproject.toml         # Hatch project file
└── .gitlab-ci.yml         # Pipeline configuration
```

---

## Licensing

Voice Stack is released under the MIT License © 2025 Vitor Bordini Garbim. Building or distributing the TTS components also implies acceptance of the Coqui Public Model License; refer to `scripts/accept_coqui_license.sh` for the automated acceptance flow bundled with the project.
