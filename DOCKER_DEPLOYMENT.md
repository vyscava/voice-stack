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
- ASR dependencies (faster-whisper, silero-vad)
- TTS dependencies (coqui-tts)
- PyTorch (CPU or CUDA based on auto-detection)

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
- `scripts/install_torch.sh` changes

**Production Image rebuilds when**:
- `pyproject.toml` changes
- `Dockerfile` changes
- `scripts/install_torch.sh` changes
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
docker pull registry.gitlab.com/your-group/voice-stack:latest
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

**ASR Service**:
```bash
docker run -d \
  --name voice-stack-asr \
  --restart unless-stopped \
  -e SERVICE_MODE=asr \
  -e ASR_DEVICE=cuda \
  -e ASR_MODEL=base \
  -p 5001:5001 \
  -v asr-models:/app/models \
  registry.gitlab.com/your-group/voice-stack:latest
```

**TTS Service**:
```bash
docker run -d \
  --name voice-stack-tts \
  --restart unless-stopped \
  -e SERVICE_MODE=tts \
  -e TTS_DEVICE=cuda \
  -e TTS_MODEL=tts_models/multilingual/multi-dataset/xtts_v2 \
  -p 5002:5002 \
  -v tts-models:/app/models \
  -v ./voices:/app/voices:ro \
  registry.gitlab.com/your-group/voice-stack:latest
```

### GPU Support

To use NVIDIA GPU with Docker:

1. **Install NVIDIA Container Toolkit**:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

2. **Update docker-compose.yml**:
```yaml
services:
  asr:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - ASR_DEVICE=cuda
```

3. **Or use `--gpus all` flag**:
```bash
docker run --gpus all -e ASR_DEVICE=cuda ...
```

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

# Device
ASR_DEVICE=cpu    # or 'cuda'
TTS_DEVICE=cpu    # or 'cuda'

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
3. Available immediately: `registry.gitlab.com/your-group/voice-stack:latest`

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
docker pull registry.gitlab.com/your-group/voice-stack:v1.2.3
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
