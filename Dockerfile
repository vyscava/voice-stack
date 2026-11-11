# ============================================================================
# Voice Stack Production Docker Image (Unified ASR + TTS)
# ============================================================================
# Multi-stage build for optimized production image
# Supports both ASR and TTS services in a single image
#
# Two images from one file (pick with --target):
#
#   full    (default, last stage) ASR + both TTS engines (Kokoro and XTTS).
#           docker build -t voice-stack:xtts .
#   kokoro  slim CPU image on python:3.11-slim: ASR + Kokoro + gateway, no
#           CUDA, no coqui/transformers.
#           docker build --target kokoro -t voice-stack:kokoro .
#
# Both images bake in the Kokoro model (checksum-verified), so TTS_ENGINE=kokoro
# works offline in either. Pick the engine at run time with TTS_ENGINE.
#
# Run ASR:
#   docker run -e SERVICE_MODE=asr -p 5001:5001 voice-stack:latest
#
# Run TTS:
#   docker run -e SERVICE_MODE=tts -p 5002:5002 voice-stack:latest
#
# TORCH: pinned by version in pyproject.toml AND by build variant via the CPU
# index below -- PyPI serves `torch==2.9.1` as the CUDA wheel, and these services
# run on CPU. scripts/assert_runtime.py then proves what actually landed. Do not
# add a `pip install --upgrade torch` step; see the comment in pyproject.toml.
# ============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install all dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder-base

# Install build-time system dependencies (not needed at runtime)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3-dev \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy dependency specification and README (required by pyproject.toml)
COPY pyproject.toml README.md ./

# Copy the scripts the build needs
COPY scripts/accept_coqui_license.sh scripts/assert_runtime.py ./scripts/
RUN chmod +x scripts/accept_coqui_license.sh

# Single unified venv: torch FIRST from the CPU index, then everything else.
#
# Order matters. `pip install -e '.[server,asr,tts]'` on its own resolves
# torch==2.9.1 from PyPI, which is the CUDA build (~4 GB of nvidia-*-cu12 wheels
# this CPU-only service never uses). Installing the CPU wheels first means the
# editable install finds 2.9.1+cpu already satisfying ==2.9.1 and leaves it.
# pip and setuptools are upgraded first: the venv's bundled copies (pip 24.0,
# setuptools 79) carry advisories pip-audit fails on, and both ship in the
# images because the venv is copied whole.
RUN python -m venv /build/.venv && \
    /build/.venv/bin/pip install --no-cache-dir --upgrade pip setuptools && \
    /build/.venv/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.9.1 torchaudio==2.9.1

# Shared venv cleanup, run by both builders after their install.
RUN printf '%s\n' '#!/bin/sh' 'set -e' \
      "sed -i 's|#!/build/.venv/bin/python|#!/app/.venv/bin/python|g' /build/.venv/bin/*" \
      "find /build/.venv -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true" \
      "find /build/.venv -type d -name 'tests' -exec rm -rf {} + 2>/dev/null || true" \
      "find /build/.venv -name '*.py[co]' -delete 2>/dev/null || true" \
      > /usr/local/bin/finish-venv && chmod +x /usr/local/bin/finish-venv

# -----------------------------------------------------------------------------
# Builder for the slim image: ASR + Kokoro + gateway. torch stays (CPU wheel)
# because the ASR path needs it for silero-vad and openai-whisper; coqui-tts
# and transformers do not come in.
# -----------------------------------------------------------------------------
FROM builder-base AS builder-kokoro
RUN /build/.venv/bin/pip install --no-cache-dir -e ".[server,asr,tts-kokoro,gateway]" && finish-venv

# -----------------------------------------------------------------------------
# Builder for the full image: everything, both TTS engines.
# -----------------------------------------------------------------------------
FROM builder-base AS builder-full
RUN /build/.venv/bin/pip install --no-cache-dir -e ".[server,asr,tts,gateway]"

# Pre-accept Coqui TTS license to prevent interactive prompts
RUN bash scripts/accept_coqui_license.sh && finish-venv

# -----------------------------------------------------------------------------
# Kokoro model files, downloaded once and verified. Upstream publishes no
# checksums; these were measured from the release assets on 2026-11-10, so a
# changed asset fails the build instead of shipping silently. Both the fp32 and
# int8 models are included: fp32 is the default (quality), int8 is ~2x faster
# and selected with TTS_KOKORO_MODEL_PATH.
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS models-kokoro
ARG KOKORO_RELEASE=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
RUN apt-get update -y && apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /models/kokoro
RUN curl -fsSL -o kokoro-v1.0.onnx "$KOKORO_RELEASE/kokoro-v1.0.onnx" && \
    curl -fsSL -o kokoro-v1.0.int8.onnx "$KOKORO_RELEASE/kokoro-v1.0.int8.onnx" && \
    curl -fsSL -o voices-v1.0.bin "$KOKORO_RELEASE/voices-v1.0.bin" && \
    printf '%s\n' \
      '7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5  kokoro-v1.0.onnx' \
      '6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb  kokoro-v1.0.int8.onnx' \
      'bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d  voices-v1.0.bin' \
      | sha256sum -c -

# -----------------------------------------------------------------------------
# Slim CPU image: ASR + Kokoro + gateway. No CUDA base, no coqui.
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS kokoro

LABEL description="Voice Stack slim CPU image: ASR + Kokoro TTS + gateway"
LABEL version="0.1.0"

# APT_REFRESH busts the cached apt layer (CI passes the UTC date, or a manual
# pipeline's APT_REFRESH_FORCE), and the
# upgrade pulls the base image's security fixes. Without both, a layer cached
# months ago keeps shipping packages trivy already has fixes for.
ARG APT_REFRESH=unset
RUN echo "apt refresh: ${APT_REFRESH}" && apt-get update -y && \
    apt-get upgrade -y --no-install-recommends && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 -s /bin/bash voicestack
WORKDIR /app

COPY --from=builder-kokoro --chown=voicestack:voicestack /build/.venv /app/.venv
COPY --chown=voicestack:voicestack src/ /app/src/
COPY --chown=voicestack:voicestack scripts/.env.production.* /app/config/
COPY --chown=voicestack:voicestack scripts/assert_runtime.py /app/scripts/
COPY --from=models-kokoro --chown=voicestack:voicestack /models/kokoro /app/models/kokoro

# The venv was built under /build with python:3.11-slim's interpreter, which is
# the same /usr/local/bin/python3.11 here, so its symlinks resolve unchanged.
RUN /app/.venv/bin/python /app/scripts/assert_runtime.py --variant kokoro

COPY --chown=voicestack:voicestack scripts/entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh && \
    mkdir -p /app/voices && chown voicestack:voicestack /app/voices /app/models

USER voicestack

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    ENV=production \
    LOG_LEVEL=info \
    HOST=0.0.0.0 \
    TTS_ENGINE=kokoro

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${ASR_PORT:-5001}/health || curl -f http://localhost:${TTS_PORT:-5002}/health || exit 1

EXPOSE 5001 5002
ENTRYPOINT ["/app/entrypoint.sh"]

# -----------------------------------------------------------------------------
# Stage 2: Production - Minimal Runtime
# -----------------------------------------------------------------------------
# Using nvidia/cuda base image which includes libcublas, libcublas.so.12, and cuDNN
# This fixes the "Library libcublas.so.12 is not found" error for ASR GPU inference
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS full

LABEL maintainer="vyscava@gmail.com"
LABEL description="Voice Stack unified ASR + TTS service"
LABEL version="0.1.0"

# Install only runtime system dependencies (no build tools)
# Note: CUDA and cuDNN are already included in the base image
# APT_REFRESH and the upgrade: see the kokoro stage. The CUDA base image lags
# Ubuntu's security updates, so the upgrade is what clears its fixed CVEs.
ARG APT_REFRESH=unset
RUN echo "apt refresh: ${APT_REFRESH}" && apt-get update -y && \
    apt-get upgrade -y --no-install-recommends && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libportaudio2 \
        curl \
        ca-certificates \
        python3.11 \
        python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash voicestack

# Set working directory
WORKDIR /app

# Copy Python environment from builder (--chown sets ownership, no extra chown needed)
COPY --from=builder-full --chown=voicestack:voicestack /build/.venv /app/.venv

# Copy application source
COPY --chown=voicestack:voicestack src/ /app/src/

# Copy scripts and config templates (for reference)
COPY --chown=voicestack:voicestack scripts/.env.production.* /app/config/
COPY --chown=voicestack:voicestack scripts/assert_runtime.py /app/scripts/
COPY --from=models-kokoro --chown=voicestack:voicestack /models/kokoro /app/models/kokoro

# Fail the BUILD if the image that SHIPS cannot run the service.
#
# This deliberately runs in the production stage, not the builder. torchcodec
# dlopen()s libtorchcodec against the system FFmpeg, and the builder
# (python:3.11-slim) has no FFmpeg -- so asserting there fails on an image that
# would have worked, and proves nothing about the artifact anyone runs.
# Asserting here validates the real thing: this venv, this base image, this
# FFmpeg. See scripts/assert_runtime.py for what is checked and why.
RUN /app/.venv/bin/python /app/scripts/assert_runtime.py --variant full

# Copy entrypoint script
COPY --chown=voicestack:voicestack scripts/entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Create directories for runtime data and pre-accept Coqui license
# NOTE: Do NOT chown -R /app here — it duplicates the entire .venv layer
RUN mkdir -p /app/voices /app/models && \
    chown voicestack:voicestack /app/voices /app/models && \
    mkdir -p /home/voicestack/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2 && \
    echo "1" > /home/voicestack/.local/share/tts/tos_agreed.txt && \
    echo "1" > /home/voicestack/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/tos_agreed.txt && \
    chown -R voicestack:voicestack /home/voicestack/.local

# Switch to non-root user
USER voicestack

# Set environment variables
# LOG_LEVEL is lowercase on purpose: uvicorn rejects 'INFO' and exits. The
# entrypoint also normalises it, so a caller passing INFO still works.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src:$PYTHONPATH" \
    # Default to production environment
    ENV=production \
    LOG_LEVEL=info \
    HOST=0.0.0.0

# Health check (will be service-specific via entrypoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${ASR_PORT:-5001}/health || curl -f http://localhost:${TTS_PORT:-5002}/health || exit 1

# Expose both ports (only the active service port will be used)
EXPOSE 5001 5002

# Use entrypoint script to determine which service to run
ENTRYPOINT ["/app/entrypoint.sh"]
