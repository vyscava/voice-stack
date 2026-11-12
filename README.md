# Voice Stack

A self-hosted speech stack for homelabs — ASR (speech-to-text) and TTS (text-to-speech) as FastAPI microservices with OpenAI-compatible APIs, plus an optional gateway that turns the pair into a spoken conversation with an agent. One unified Docker image runs either service based on a single environment variable. Drop it next to Bazarr for automatic subtitles, plug it into Open WebUI for voice chat, or use it as a local replacement for cloud speech APIs.

**Key features:**

- OpenAI-compatible `/v1/audio/transcriptions` and `/v1/audio/speech` endpoints
- Bazarr-native subtitle endpoints (SRT, VTT, JSON, JSONL)
- Two TTS engines, picked at boot with `TTS_ENGINE`: **Kokoro** (default, fast on CPU) and **Coqui XTTS-v2** (voice cloning)
- One image runs ASR, TTS or the gateway via `SERVICE_MODE`; a slim CPU image (`:kokoro`) and a full image (`:xtts` / `:latest`)
- GPU images for NVIDIA (CUDA: ASR, XTTS and Kokoro) and AMD (ROCm: ASR and XTTS), built from `docker/`
- Voice cloning via Coqui XTTS-v2 (drop a .wav file, get a new voice)
- Built-in concurrency limits, memory guards, and idle model unloading
- A worked production deployment: per-service compose stacks deployed by CI

---

## Quick Start (Docker)

The fastest way to get running. Requires Docker; a GPU is optional.

### Build

Pick the image for your hardware. All four serve the same API.

```bash
git clone https://github.com/vyscava/voice-stack.git
cd voice-stack
docker build -t voice-stack:latest .                          # CPU, full: ASR + both TTS engines
docker build --target kokoro -t voice-stack:kokoro .          # CPU, slim: ASR + Kokoro + gateway
docker build -f docker/Dockerfile.cuda -t voice-stack:cuda .  # NVIDIA GPU (driver >= 580)
docker build -f docker/Dockerfile.rocm -t voice-stack:rocm .  # AMD GPU (ROCm 7)
```

| Image | Dockerfile | ASR (faster-whisper) | XTTS | Kokoro |
|---|---|---|---|---|
| `:kokoro` | `Dockerfile --target kokoro` | CPU | not included | CPU |
| `:latest` / `:xtts` | `Dockerfile` | CPU | CPU | CPU |
| `:cuda` | `docker/Dockerfile.cuda` | GPU | GPU | GPU (`TTS_KOKORO_DEVICE=cuda`) |
| `:rocm` | `docker/Dockerfile.rocm` | GPU | GPU | CPU, by design (no onnxruntime ROCm build) |

The GPU images default to the GPU (`ASR_DEVICE=cuda`, `TTS_DEVICE=cuda`; ROCm uses the
same `cuda` device name). There is no published GPU image: build the one you need as
above. See
[`DOCKER_DEPLOYMENT.md`](DOCKER_DEPLOYMENT.md#gpu-support) for host setup, GPU targets
and what has been verified on real GPUs.

### Run Both Services

```bash
# ASR on port 5001
docker run -d --name voice-stack-asr \
  -p 5001:5001 \
  -e SERVICE_MODE=asr \
  -e ASR_MODEL=base \
  -e ASR_MODEL_LOCATION=/app/models \
  -v asr-models:/app/models \
  voice-stack:latest

# TTS on port 5002
docker run -d --name voice-stack-tts \
  -p 5002:5002 \
  -e SERVICE_MODE=tts \
  -v tts-models:/home/voicestack/.local/share/tts \
  voice-stack:latest
```

On a GPU, run the matching image and give the container the GPU:

```bash
# NVIDIA (NVIDIA Container Toolkit installed)
docker run -d --gpus all -p 5001:5001 -e SERVICE_MODE=asr \
  -e ASR_MODEL_LOCATION=/app/models -v asr-models:/app/models voice-stack:cuda

# AMD
docker run -d --device /dev/kfd --device /dev/dri --group-add video \
  --group-add "$(getent group render | cut -d: -f3)" -p 5001:5001 -e SERVICE_MODE=asr \
  -e ASR_MODEL_LOCATION=/app/models -v asr-models:/app/models voice-stack:rocm
```

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
  -d '{"input": "Hello from Voice Stack", "voice": "Claribel Dervla", "response_format": "mp3"}' \
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

The default `docker-compose.yml` runs on CPU. For a GPU, add the vendor override, which
builds the GPU image and passes the device through:

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d --build   # NVIDIA
docker compose -f docker-compose.yml -f docker-compose.rocm.yml up -d --build   # AMD (set RENDER_GID in .env)
```

---

## Deploy as separate stacks

`compose/deploy/` holds a real production deployment: one compose stack
per service (`asr.yml`, `tts.yml`, `gateway.yml`), each pinned to an image by
`${VOICE_STACK_IMAGE}` and deployed by CI over an SSH docker context
(`.gitlab/ci/deploy.yml`). Any compose-aware tool (`docker compose -p`,
Portainer, a CI job) can run them the same way: set `VOICE_STACK_IMAGE` to your
own image and deploy ASR and TTS before the gateway, which joins their
networks.

They carry deployment-specific values (a private registry, tuning for a 4-core
host); see [compose/README.md](compose/README.md) for what to change. For a
single-host start, use the root `docker-compose.yml` instead.

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

All configuration is via environment variables.

For Docker Compose, copy the example file and edit it — `docker compose` reads
`.env` from the project directory automatically:

```bash
cp .env.example .env
```

[`.env.example`](.env.example) documents every setting with its real default.
Bare-metal/systemd installs use the per-service templates instead:

- **ASR**: [`scripts/.env.production.asr`](scripts/.env.production.asr)
- **TTS**: [`scripts/.env.production.tts`](scripts/.env.production.tts)

### Essential Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_MODE` | _(required)_ | `asr` or `tts` — selects which service the container runs |
| `ASR_DEVICE` / `TTS_DEVICE` | `cpu` (`cuda` in the GPU images) | `cuda` runs on the GPU, NVIDIA or AMD; needs a GPU image |
| `TTS_KOKORO_DEVICE` | `cpu` (`cuda` in the CUDA image) | `cpu`, `cuda` or `auto` for Kokoro's ONNX Runtime; `cuda` fails the start if the GPU is not used |
| `ASR_MODEL` | `base` | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `ASR_ENGINE` | `fasterwhisper` | `fasterwhisper` (recommended) or `whisper` |
| `TTS_ENGINE` | `kokoro` | `kokoro` or `coqui` (aliases `xtts`, `xtts-2`); see [TTS Engines](#tts-engines) |
| `TTS_MODEL` | `tts_models/multilingual/multi-dataset/xtts_v2` | Coqui model ID (XTTS only) |
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
| `POST` | `/v1/audio/transcriptions` | Transcribe audio (OpenAI-compatible; `response_format`: `json`, `text`, `srt`, `vtt`, `verbose_json`) |
| `POST` | `/v1/audio/transcriptions/verbose` | Transcribe with segment metadata |
| `POST` | `/v1/audio/translations` | Translate audio to English (same `response_format` set as transcriptions) |
| `POST` | `/bazarr/asr` | Subtitle output (SRT, VTT, JSON, TXT, TSV, JSONL) |
| `POST` | `/bazarr/detect-language` | Detect spoken language |
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

## Voice Gateway (optional)

A third mode, `SERVICE_MODE=gateway`, that turns one utterance into a spoken reply:

```
audio in ─▶ ASR ─▶ agent (over NATS) ─▶ TTS ─▶ audio out
```

It **adds no model and no AI of its own.** It calls the ASR and TTS services over HTTP and asks an
existing agent on a [NATS](https://nats.io/) bus for the answer. That keeps it a small process that
starts in milliseconds, holds no GPU memory, and can be restarted without evicting a 1.8 GB model
from anyone's cache.

If you have no agent bus, you do not need this mode. The ASR and TTS services are independent of it.

### Run it

```bash
docker run -d --name voice-gateway -p 5003:5003 \
  -e SERVICE_MODE=gateway \
  -e GATEWAY_ASR_URL=http://voice-stack-asr:5001 \
  -e GATEWAY_TTS_URL=http://voice-stack-tts:5002 \
  -e GATEWAY_NATS_URL=nats://nats:4222 \
  -e GATEWAY_SELF_NAME=voice-gateway \
  -e GATEWAY_AGENT=my-agent \
  voice-stack:latest
```

With Compose it is behind a profile, because it needs a bus the compose file cannot provide:

```bash
docker compose --profile gateway up -d
```

### Use it

```bash
curl -X POST http://localhost:5003/v1/voice/exchange \
  -F 'file=@question.wav' -D headers.txt -o reply.opus
```

```bash
# Check the microphone path without spending an agent turn
curl -X POST http://localhost:5003/v1/voice/transcribe-only -F 'file=@question.wav'
```

### It always answers, and it says how it went

The caller is usually a person holding a microphone, and **an HTTP status code is not something they
can hear.** So a failed exchange still returns `200` with *speech* explaining the problem, and
reports what actually happened in headers:

| Header | Meaning |
|---|---|
| `X-Voice-Outcome` | `ok`, `no_speech`, `asr_failed`, `agent_timeout`, `agent_error`, `tts_failed` |
| `X-Voice-Degraded` | `false` only for `ok` |
| `X-Voice-Transcript` | what was heard, escaped |

| Situation | What the caller gets |
|---|---|
| agent does not reply in time | a spoken "that is taking too long" |
| the audio was noise, not speech | a spoken "say that again", and **the agent is never asked** |
| ASR fails | a spoken prompt to repeat |
| TTS fails | JSON containing the agent's answer as text, so the answer is not lost |
| everything fails at once | a handled result, never a stack trace |

The one case that returns JSON instead of audio is `tts_failed`, because there is no way to speak a
message when speech itself is what broke.

### Two things it deliberately will not do

**Audio never crosses the message bus.** Only the transcript and the reply text do. NATS has a ~1 MB
default payload and is a message bus rather than a media pipe, and raw household audio should not
exist as a routable message. This is structural, not a convention: the agent transport accepts and
returns `str`, with no parameter capable of carrying bytes.

**It defaults to a stock speaker.** If you clone a real person's voice, that clone is theirs. A good
one is mistakable for them by the people who know them, and a watermark is forensic rather than
audible — it can help establish misuse afterwards and prevents nothing at the moment it happens. Keep
cloned voices on networks you control, and set `GATEWAY_VOICE` deliberately rather than by accident.

### Configuration

| Variable | Default | Notes |
|---|---|---|
| `GATEWAY_PORT` | `5003` | |
| `GATEWAY_ASR_URL` | `http://localhost:5001` | |
| `GATEWAY_TTS_URL` | `http://localhost:5002` | |
| `GATEWAY_NATS_URL` | `nats://localhost:4222` | |
| `GATEWAY_SELF_NAME` | `voice-gateway` | Must match this service's identity on the bus **exactly** |
| `GATEWAY_AGENT` | `assistant` | The peer that answers |
| `GATEWAY_NATS_TOPIC` | `voice` | |
| `GATEWAY_AGENT_TIMEOUT_S` | `60` | Bounds connect **and** request, so an exchange cannot hang |
| `GATEWAY_VOICE` | `default` | The engine's own default voice (a stock one, on purpose), so switching `TTS_ENGINE` needs no change here |
| `GATEWAY_AUDIO_FORMAT` | `opus` | |
| `GATEWAY_MIN_TRANSCRIPT_CHARS` | `2` | Below this, treated as noise and not sent to the agent |

The subject it publishes on is `agents.<GATEWAY_SELF_NAME>.dm.<GATEWAY_AGENT>.<GATEWAY_NATS_TOPIC>`,
which reads *from* `.dm.` *to*. If your bus restricts publishing to `agents.<self>.>`, getting
`GATEWAY_SELF_NAME` wrong is rejected **asynchronously**: the send still looks like it worked and the
message simply never arrives.

---

## TTS Engines

The engine is chosen once, at boot, with `TTS_ENGINE`. The API (`/v1/audio/speech`,
`/v1/audio/voices`) is the same for both.

| | Kokoro (`kokoro`, default) | Coqui XTTS-v2 (`coqui`) |
|---|---|---|
| Speed on CPU | faster than real time | seconds per sentence |
| Voice cloning | no | yes, from `.wav` samples |
| Languages | en-US, en-GB, pt-BR, es, fr, it, hi | 16 incl. pt, ja, zh |
| Image | `:kokoro` (slim), `:xtts` / `:latest`, `:cuda`, `:rocm` | `:xtts` / `:latest`, `:cuda`, `:rocm` |
| GPU | `:cuda` (`TTS_KOKORO_DEVICE=cuda`); CPU in `:rocm` | `:cuda` and `:rocm` (`TTS_DEVICE=cuda`) |
| `default` voice | `TTS_KOKORO_DEFAULT_VOICE` (`af_heart`) | `TTS_XTTS_DEFAULT_VOICE` (`Claribel Dervla`) |

Kokoro voices are bound to a language (`pf_dora`, `pm_alex` and `pm_santa` are Brazilian
Portuguese; `GET /v1/audio/voices` lists them all), so the language comes from
`TTS_FORCE_LANG`, then the request, then the voice itself. The model files are baked into
both images; set `TTS_KOKORO_MODEL_PATH=/app/models/kokoro/kokoro-v1.0.int8.onnx` for the
int8 model, about twice as fast with some quality cost.

An unknown `TTS_ENGINE`, or an engine the image does not ship, fails the start with the
image to use instead. Nothing silently falls back to another engine.

Kokoro phonemizes with espeak-ng, which copies its shared library into a temporary
directory and loads it from there, so `/tmp` must be writable **and allow exec**. A
read-only root filesystem or a `noexec` tmpfs on `/tmp` makes every synthesis fail with
`failed to map segment from shared object`; mount `--tmpfs /tmp:exec` if you lock the
container down.

---

## Voice Cloning (TTS)

Requires `TTS_ENGINE=coqui` and the full image.


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
| TTS Kokoro (CPU) | 2+ cores | 1 GB | 1 GB | Faster than real time |
| TTS XTTS (CPU) | 4+ cores | 8 GB | 4 GB | Slow but works |
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
│   ├── tts/               # TTS microservice (Kokoro, Coqui XTTS-v2)
│   ├── core/              # Shared settings, logging, middleware
│   └── utils/             # Audio processing, language detection, text helpers
├── compose/deploy/        # Production stacks deployed by CI (asr.yml, tts.yml, gateway.yml)
├── scripts/               # Installers, systemd helpers, build automation
├── docs/                  # Architecture, concurrency, load testing, cheat sheets
├── voices/                # Voice samples for TTS cloning
├── tests/                 # Unit + integration test suites
├── Dockerfile             # CPU images: full (default target) and slim (--target kokoro)
├── docker/                # GPU images: Dockerfile.cuda (NVIDIA), Dockerfile.rocm (AMD)
├── Dockerfile.ci          # CI image with dev dependencies
├── docker-compose.yml     # Local compose (CPU)
├── docker-compose.{cuda,rocm}.yml  # GPU overrides for docker-compose.yml
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
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache 2.0) and [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) (MIT) — the default TTS engine

---

## License

MIT License &copy; 2025 Vitor Bordini Garbim. TTS components also require acceptance of the [Coqui Public Model License](https://coqui.ai/cpml); see `scripts/accept_coqui_license.sh` for the automated acceptance flow.
