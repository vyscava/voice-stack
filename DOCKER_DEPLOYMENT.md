# Voice Stack Docker Deployment Guide

Complete guide for Docker-based development, CI/CD, and production deployment.

---

## Table of Contents

1. [Overview](#overview)
2. [Docker Images](#docker-images)
3. [Local Development](#local-development)
4. [CI/CD Pipeline](#cicd-pipeline)
5. [Production Deployment](#production-deployment)
6. [Release Process](#release-process)
7. [Troubleshooting](#troubleshooting)

---

## Overview

Voice Stack uses a **Docker-first approach** for reliable, reproducible builds and deployments:

- **CI Image** ([Dockerfile.ci](Dockerfile.ci)): Contains all dev dependencies for running tests and quality checks
- **Production Image** ([Dockerfile](Dockerfile)): Optimized unified image that runs either ASR or TTS based on environment variables
- **Smart Entrypoint**: Single image, multiple services - configure via `SERVICE_MODE`

### Key Benefits

✅ **Reliable**: No more cache inconsistencies - dependencies baked into images
✅ **Fast**: Pre-built images mean tests start instantly
✅ **Consistent**: Same environment locally, in CI, and production
✅ **Simple**: One unified image for both ASR and TTS services
✅ **Non-Interactive**: Coqui TTS license pre-accepted during build - no prompts in containers

### Coqui TTS License Handling

Both Docker images automatically pre-accept the Coqui TTS license during the build process using [scripts/accept_coqui_license.sh](scripts/accept_coqui_license.sh). This prevents interactive prompts when containers start, which is essential for:

- **CI/CD pipelines** - Automated testing without manual intervention
- **Docker containers** - No TTY available for interactive input
- **Production deployments** - Services start automatically without prompts

The license is accepted by creating a `tos_agreed.txt` file in the TTS cache directory (`~/.local/share/tts/`), following the same mechanism used in our test suite.

**Note**: By building these Docker images, you acknowledge acceptance of the [Coqui Public Model License](https://coqui.ai/cpml).

---

## Docker Images

### CI Image (Dockerfile.ci)

**Purpose**: Running tests, linting, formatting in CI/CD pipeline

**Contains**:
- All dev dependencies (pytest, black, ruff, mypy)
- ASR dependencies (faster-whisper with its bundled Silero ONNX VAD; openai-whisper)
- TTS dependencies (coqui-tts)
- PyTorch, the `+cpu` build (forced from the PyTorch CPU index)

**Build**:
```bash
docker build -f Dockerfile.ci -t voice-stack-ci:latest .
```

**Usage** (not typically run directly - used by GitLab CI):
```bash
docker run --rm -v $(pwd):/workspace voice-stack-ci:latest hatch run test
```

### Production Image (Dockerfile)

**Purpose**: Running ASR or TTS services in production

**Contains**:
- Production dependencies only (no dev tools)
- Both ASR and TTS dependencies (unified image)
- Optimized multi-stage build (smaller size)
- Smart entrypoint for service selection

**Build**:
```bash
docker build -t voice-stack:latest .
```

**Run ASR**:
```bash
docker run -e SERVICE_MODE=asr -p 5001:5001 voice-stack:latest
```

**Run TTS**:
```bash
docker run -e SERVICE_MODE=tts -p 5002:5002 voice-stack:latest
```

The same file builds the slim CPU image with `--target kokoro` (ASR + Kokoro +
gateway, no torch), which is what `compose/deploy/` runs in production.

### GPU Images (docker/Dockerfile.cuda, docker/Dockerfile.rocm)

**Purpose**: The full stack on NVIDIA (CUDA) or AMD (ROCm) GPUs. Build them
yourself with the commands below. See [GPU Support](#gpu-support).

```bash
docker build -f docker/Dockerfile.cuda -t voice-stack:cuda .
docker build -f docker/Dockerfile.rocm -t voice-stack:rocm .
```

---

## Local Development

### Using Docker Compose (Recommended)

The easiest way to test both services locally:

```bash
# Start both services
docker-compose up -d

# View logs
docker-compose logs -f

# Test ASR service
curl http://localhost:5001/health

# Test TTS service
curl http://localhost:5002/health

# Stop services
docker-compose down
```

### Start Individual Services

**ASR only**:
```bash
docker-compose up -d asr
```

**TTS only**:
```bash
docker-compose up -d tts
```

### Rebuild After Changes

```bash
# Rebuild images
docker-compose build

# Restart services
docker-compose up -d
```

### Using Hatch (Traditional Method)

If you prefer local development without Docker:

```bash
# Install dependencies
hatch env create

# Run ASR locally
hatch run run_asr

# Run TTS locally
hatch run run_tts
```

**Note**: Docker approach ensures consistency with CI/production environments.

---

## CI/CD Pipeline

### Pipeline Stages

```
build-images → quality → test → coverage → release → secret-detection
```

### How It Works

1. **Build Images Stage**
   - Triggers only when dependencies change (`pyproject.toml`, `Dockerfile`, scripts)
   - Builds CI image and production image
   - Pushes to GitLab Container Registry
   - Tagged with commit SHA and `:latest`

2. **Quality Stage**
   - Uses pre-built CI image
   - Runs `hatch run fmt` (black, ruff)
   - Runs `hatch run lint`
   - Fails if code is not formatted or has lint errors

3. **Test Stage**
   - Runs in parallel: `unit:asr`, `unit:tts`, `unit:core`, `unit:utils`, `integration`
   - Each job uses pre-built CI image (fast startup!)
   - Generates JUnit XML reports

4. **Coverage Stage**
   - Aggregates test results
   - Generates coverage report
   - Uploads to GitLab (visible in merge requests)

5. **Release Stage**
   - **Automatic** (main branch): Promotes `:latest` tag
   - **Manual** (git tags): Creates versioned release (e.g., `v1.0.0`)

6. **Secret Detection Stage**
   - Scans for leaked credentials
   - Uses GitLab's built-in template

### Image Rebuilding Logic

Images are **only rebuilt** when relevant files change:

**CI Image rebuilds when**:
- `pyproject.toml` changes
- `Dockerfile.ci` changes
- `scripts/install_system_deps.sh` changes
- `scripts/assert_runtime.py` changes

**Production Image rebuilds when**:
- `pyproject.toml` changes
- `Dockerfile` changes
- `scripts/assert_runtime.py` changes
- `scripts/entrypoint.sh` changes
- Any file in `src/` changes

**All other commits**: Use cached images (fast pipeline!)

### First Pipeline Run

On the first run (or when images don't exist):
1. Build stage might take 5-10 minutes (installing PyTorch, ML models)
2. Subsequent runs use cached images and finish in ~2-3 minutes

### CI Image Not Available?

If the CI image doesn't exist yet, the pipeline will fail at quality/test stages. Solutions:

**Option 1**: Manually trigger build stage
```bash
# Make a trivial change to pyproject.toml to trigger rebuild
touch pyproject.toml
git add pyproject.toml
git commit -m "Trigger CI image rebuild"
git push
```

**Option 2**: Force rebuild by deleting image from registry
- Go to GitLab → Packages & Registries → Container Registry
- Delete the CI image
- Push a commit that changes `pyproject.toml`

---

## Production Deployment

### Using Docker Compose (Simple Deployment)

1. **Copy docker-compose.yml to your server**:
```bash
scp docker-compose.yml user@server:/opt/voice-stack/
```

2. **Pull latest image**:
```bash
docker pull your-registry/voice-stack:latest
```

3. **Start services**:
```bash
docker-compose up -d
```

4. **Check health**:
```bash
curl http://localhost:5001/health  # ASR
curl http://localhost:5002/health  # TTS
```

### Using Docker Run (Manual)

These use the CPU image; for a GPU use `voice-stack:cuda` / `:rocm` and
the device flags in [GPU Support](#gpu-support).

**ASR Service**:
```bash
docker run -d \
  --name voice-stack-asr \
  --restart unless-stopped \
  -e SERVICE_MODE=asr \
  -e ASR_MODEL=base \
  -p 5001:5001 \
  -v asr-models:/app/models \
  your-registry/voice-stack:latest
```

**TTS Service**:
```bash
docker run -d \
  --name voice-stack-tts \
  --restart unless-stopped \
  -e SERVICE_MODE=tts \
  -e TTS_MODEL=tts_models/multilingual/multi-dataset/xtts_v2 \
  -p 5002:5002 \
  -v tts-models:/app/models \
  -v ./voices:/app/voices:ro \
  your-registry/voice-stack:latest
```

### Separate stacks (production)

`compose/deploy/` holds one compose stack per service (`asr.yml`, `tts.yml`,
`gateway.yml`). Each takes its image from `${VOICE_STACK_IMAGE}` and keeps its
non-secret settings literal in the file; the gateway's bus identity and NATS URL
come from the environment. CI deploys them (`.gitlab/ci/deploy.yml`, through the
shared compose deployer): a merge to main pins the slim image that pipeline
built, runs `compose up -d` per stack over an SSH docker context, and asserts
the running digest and health. ASR and TTS deploy before the gateway, which
joins their networks.

To run them by hand: `VOICE_STACK_IMAGE=<your image> docker compose -p
voice-stack-asr -f compose/deploy/asr.yml up -d`, the same for `tts`, then the
gateway with its variables set. See `compose/README.md` for the
deployment-specific values to change.

### GPU Support

The root `Dockerfile` images are CPU-only (torch `+cpu`, onnxruntime CPU). GPU
images are separate Dockerfiles under `docker/`, built from the repository root:

| Image | Build | Base | GPU stack |
|---|---|---|---|
| `voice-stack:cuda` | `docker build -f docker/Dockerfile.cuda -t voice-stack:cuda .` | `python:3.11-slim` + CUDA from pip wheels | torch 2.15.0+cu132, CTranslate2 (cuBLAS 12), onnxruntime-gpu |
| `voice-stack:rocm` | `docker build -f docker/Dockerfile.rocm -t voice-stack:rocm .` | `python:3.11-slim` + ROCm 7.14 from AMD's pip packages | torch 2.15.0+rocm7.14, CTranslate2 ROCm wheel |

Both use the same final-release Python 3.11 as the CPU images. The old full image
ran on `nvidia/cuda` Ubuntu 22.04, whose Python 3.11 is 3.11.0rc1: that broke
torch 2.13 at import and took TTS down for three days. `scripts/assert_runtime.py`
refuses a pre-release interpreter.

**What runs where:**

| Component | `:cuda` | `:rocm` |
|---|---|---|
| ASR, faster-whisper / CTranslate2 (`ASR_DEVICE=cuda`) | GPU | GPU |
| ASR, openai-whisper (`ASR_ENGINE=whisper`) | GPU | GPU |
| TTS, XTTS v2 (`TTS_ENGINE=coqui`, `TTS_DEVICE=cuda`) | GPU | GPU |
| TTS, Kokoro (`TTS_ENGINE=kokoro`) | GPU with `TTS_KOKORO_DEVICE=cuda` (the image default) | **CPU, by design** (see below) |
| Silero VAD (ONNX) | CPU | CPU |

The GPU images default to `ASR_DEVICE=cuda`, `ASR_COMPUTE_TYPE=float16` and
`TTS_DEVICE=cuda` (ROCm uses the same `cuda` device name in torch and
CTranslate2). `TTS_KOKORO_DEVICE=cuda` fails the start if onnxruntime did not
actually put the session on the GPU, instead of silently running on the CPU;
`auto` takes the GPU when available and logs which it got.

**Kokoro on AMD runs on the CPU, by design.** onnxruntime dropped its ROCm
execution provider after 1.22, and AMD's replacement (MIGraphX) needs a
system-wide MIGraphX install beside the image's pip ROCm, with a newest wheel
(`onnxruntime-migraphx` 1.27) that trails the pinned onnxruntime 1.31. Kokoro is
small enough not to need the GPU: 0.56 s per sentence on a Strix Halo CPU,
faster than real time. This may change if a MIGraphX wheel matching the
onnxruntime pin appears.

**What is verified, and what is not.** CI builds both images on every merge
request that changes their inputs (`.gitlab/ci/gpu-images.yml`). The build runs
`scripts/assert_runtime.py --variant cuda|rocm`, which checks without a GPU that
torch, torchaudio and torchcodec are the GPU builds, cuDNN loads (CUDA),
CTranslate2 is the cuBLAS / HIP build and the libraries it loads resolve,
onnxruntime-gpu's CUDA provider library loads (CUDA), and the XTTS, Whisper and
Kokoro imports work. The images are then pip-audited and trivy-scanned. A
manual CI job can push the image that passed as `${DOCKER_REGISTRY}/voice-stack:cuda`
/ `:rocm` to the registry the pipeline is configured with; there is no public
registry, so build the images yourself as above. The CI runners have no GPU.
Inference has been verified by hand: `:cuda` on an RTX 5090 (all engines on
the GPU), and `:rocm` built with `ROCM_GPU_TARGETS=gfx1151` on an AMD Radeon
8060S (faster-whisper and XTTS on the GPU; XTTS slower than real time there).
RDNA4 (gfx1201) has not been tested.

#### NVIDIA (`:cuda`)

Host: an NVIDIA driver >= 580 (CUDA 13) and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker run --gpus all -e SERVICE_MODE=asr -p 5001:5001 \
  -e ASR_MODEL_LOCATION=/app/models -v asr-models:/app/models voice-stack:cuda
docker run --gpus all -e SERVICE_MODE=tts -p 5002:5002 \
  -v tts-models:/home/voicestack/.local/share/tts voice-stack:cuda
```

With compose: `docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d --build`.

#### AMD (`:rocm`)

Host: the `amdgpu` kernel driver (the ROCm user-space libraries are inside the
image, so the host needs no ROCm install). The container needs `/dev/kfd`,
`/dev/dri` and membership of the host's `video` and `render` groups:

```bash
docker run --device /dev/kfd --device /dev/dri \
  --group-add video --group-add "$(getent group render | cut -d: -f3)" \
  -e SERVICE_MODE=asr -p 5001:5001 \
  -e ASR_MODEL_LOCATION=/app/models -v asr-models:/app/models voice-stack:rocm
```

With compose: set `RENDER_GID` in `.env`, then
`docker compose -f docker-compose.yml -f docker-compose.rocm.yml up -d --build`.

**GPU targets.** ROCm's math libraries ship kernels per GPU architecture, one
package each, so the image carries only the targets in the `ROCM_GPU_TARGETS`
build argument (default `gfx1030 gfx1100 gfx1101 gfx1102 gfx1200 gfx1201`:
RDNA2, RDNA3 and RDNA4 cards such as the RX 9070 XT and Radeon AI PRO R9700).
`rocminfo | grep gfx` names yours; add it with
`--build-arg ROCM_GPU_TARGETS="gfx1100 gfx942"` (or `ROCM_GPU_TARGETS` in `.env`
for compose). A card outside the list fails at its first GPU operation.

### Environment Variables

See configuration templates:
- ASR: [scripts/.env.production.asr](scripts/.env.production.asr)
- TTS: [scripts/.env.production.tts](scripts/.env.production.tts)

Common variables:
```bash
# Service selection (REQUIRED)
SERVICE_MODE=asr  # or 'tts'

# Server config
HOST=0.0.0.0
ASR_PORT=5001
TTS_PORT=5002
LOG_LEVEL=INFO

# Device ('cuda' needs a GPU image; it is also the device name on ROCm)
ASR_DEVICE=cpu         # or 'cuda'
TTS_DEVICE=cpu         # or 'cuda'
TTS_KOKORO_DEVICE=cpu  # or 'cuda' / 'auto' (CUDA image only)

# Models
ASR_MODEL=base
TTS_MODEL=tts_models/multilingual/multi-dataset/xtts_v2
```

### Persistent Volumes

**Models** (recommended):
```bash
# Create named volumes
docker volume create asr-models
docker volume create tts-models

# Use in docker run
-v asr-models:/app/models
```

**Voice Samples** (TTS only):
```bash
# Mount local directory
-v ./voices:/app/voices:ro
```

---

## Release Process

### Automatic Release (main branch)

Every push to `main` that passes tests automatically:
1. Runs full test suite
2. Promotes production image to `:latest` tag
3. Available immediately: `your-registry/voice-stack:latest`

### Versioned Release (git tags)

For production releases with version numbers:

1. **Update version** in [pyproject.toml](pyproject.toml):
```toml
[project]
version = "1.2.3"
```

2. **Commit and tag**:
```bash
git add pyproject.toml
git commit -m "Release v1.2.3"
git tag v1.2.3
git push origin main --tags
```

3. **GitLab CI automatically**:
   - Builds and tests the release
   - Tags image as `v1.2.3` and `:latest`
   - Pushes to registry

4. **Pull specific version**:
```bash
docker pull your-registry/voice-stack:v1.2.3
```

### Semantic Versioning

We follow [SemVer](https://semver.org):
- `v1.0.0` - Major release (breaking changes)
- `v1.1.0` - Minor release (new features, backwards compatible)
- `v1.1.1` - Patch release (bug fixes)

---

## Troubleshooting

### Pipeline Issues

**Problem**: Quality/test jobs fail with "image not found"
**Solution**: CI image hasn't been built yet. Make a change to `pyproject.toml` to trigger rebuild.

**Problem**: Build stage takes 10+ minutes
**Solution**: This is normal for first build (PyTorch installation). Subsequent builds use cache.

**Problem**: Tests pass locally but fail in CI
**Solution**:
- Run tests using CI image: `docker run --rm -v $(pwd):/workspace voice-stack-ci:latest hatch run test`
- Check for environment-specific issues

### Docker Issues

**Problem**: Service starts but immediately exits
**Solution**: Check logs with `docker logs <container>`. Common causes:
- Missing `SERVICE_MODE` environment variable
- Port already in use
- Permission issues with volumes

**Problem**: Out of memory during build
**Solution**: Increase Docker memory limit (Docker Desktop → Settings → Resources)

**Problem**: GPU not detected in container
**Solution**:
- Verify nvidia-container-toolkit is installed
- Check `nvidia-smi` works on host
- Use `--gpus all` flag or `runtime: nvidia`

### Performance Issues

**Problem**: Slow model download on first run
**Solution**: Models are downloaded on first request. Persist `/app/models` volume to avoid re-downloading.

**Problem**: High memory usage
**Solution**: Adjust resource limits in [docker-compose.yml](docker-compose.yml):
```yaml
deploy:
  resources:
    limits:
      memory: 4G
```

### Common Commands

```bash
# View container logs
docker logs -f voice-stack-asr

# Execute command in running container
docker exec -it voice-stack-asr bash

# Inspect container
docker inspect voice-stack-asr

# Check resource usage
docker stats

# Remove all containers and volumes (CAUTION: deletes data)
docker-compose down -v

# Rebuild from scratch (no cache)
docker-compose build --no-cache
```

---

## Additional Resources

- **GitLab CI/CD Docs**: https://docs.gitlab.com/ee/ci/
- **Docker Compose Reference**: https://docs.docker.com/compose/
- **NVIDIA Container Toolkit**: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

---

## Summary

**Key Takeaways**:
1. ✅ Docker images ensure consistent, reliable builds
2. ✅ CI image auto-rebuilds when dependencies change
3. ✅ Production image is unified (ASR + TTS) with smart entrypoint
4. ✅ Release process: merge to main = `:latest`, git tag = versioned release
5. ✅ Use `docker-compose up` for easy local testing

**Questions?** Check [issues](https://gitlab.com/your-group/voice-stack/-/issues) or contact the team.
