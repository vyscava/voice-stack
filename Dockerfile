# ============================================================================
# Voice Stack Production Docker Image (Unified ASR + TTS)
# ============================================================================
# Multi-stage build for optimized production image
# Supports both ASR and TTS services in a single image
#
# Usage:
#   docker build -t voice-stack:latest .
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
FROM python:3.11-slim AS builder

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
RUN python -m venv /build/.venv && \
    /build/.venv/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.9.1 torchaudio==2.9.1 && \
    /build/.venv/bin/pip install --no-cache-dir -e ".[server,asr,tts]"

# Fail the BUILD if the environment is not one this service can run on.
# See scripts/assert_runtime.py for what is checked and why each check exists.
RUN /build/.venv/bin/python scripts/assert_runtime.py

# Pre-accept Coqui TTS license to prevent interactive prompts
RUN bash scripts/accept_coqui_license.sh

# Fix shebangs to match production path
RUN sed -i 's|#!/build/.venv/bin/python|#!/app/.venv/bin/python|g' /build/.venv/bin/* && \
    # Clean up unnecessary files to reduce image size
    find /build/.venv -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true && \
    find /build/.venv -type d -name 'tests' -exec rm -rf {} + 2>/dev/null || true && \
    find /build/.venv -name '*.pyc' -delete 2>/dev/null || true && \
    find /build/.venv -name '*.pyo' -delete 2>/dev/null || true

# -----------------------------------------------------------------------------
# Stage 2: Production - Minimal Runtime
# -----------------------------------------------------------------------------
# Using nvidia/cuda base image which includes libcublas, libcublas.so.12, and cuDNN
# This fixes the "Library libcublas.so.12 is not found" error for ASR GPU inference
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS production

LABEL maintainer="vyscava@gmail.com"
LABEL description="Voice Stack unified ASR + TTS service"
LABEL version="0.1.0"

# Install only runtime system dependencies (no build tools)
# Note: CUDA and cuDNN are already included in the base image
RUN apt-get update -y && \
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
COPY --from=builder --chown=voicestack:voicestack /build/.venv /app/.venv

# Copy application source
COPY --chown=voicestack:voicestack src/ /app/src/

# Copy scripts and config templates (for reference)
COPY --chown=voicestack:voicestack scripts/.env.production.* /app/config/
COPY --chown=voicestack:voicestack scripts/assert_runtime.py /app/scripts/

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
