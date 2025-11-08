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
# Environment Variables:
#   SERVICE_MODE: "asr" or "tts" (required)
#   All other vars from .env.production.asr or .env.production.tts
# ============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base - System Dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS base

# Install system dependencies using our script approach
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libportaudio2 \
        python3-dev \
        build-essential \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Stage 2: Builder - Python Dependencies
# -----------------------------------------------------------------------------
FROM base AS builder

WORKDIR /build

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy dependency specification and README (required by pyproject.toml)
COPY pyproject.toml README.md ./

# Install hatch
RUN pip install --no-cache-dir hatch

# Copy scripts for PyTorch installation and license acceptance
COPY scripts/install_torch.sh scripts/accept_coqui_license.sh ./scripts/
RUN chmod +x scripts/*.sh

# Create production environment with BOTH ASR and TTS
# This creates a unified venv with all production dependencies
RUN hatch env create default

# Install PyTorch (auto-detects CPU/CUDA)
RUN bash scripts/install_torch.sh

# Pre-accept Coqui TTS license to prevent interactive prompts
# This creates the tos_agreed.txt file that Coqui checks for
RUN bash scripts/accept_coqui_license.sh

# -----------------------------------------------------------------------------
# Stage 3: Production - Minimal Runtime
# -----------------------------------------------------------------------------
FROM base AS production

LABEL maintainer="vyscava@gmail.com"
LABEL description="Voice Stack unified ASR + TTS service"
LABEL version="0.1.0"

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash voicestack

# Set working directory
WORKDIR /app

# Copy Python environment from builder
COPY --from=builder --chown=voicestack:voicestack /build/.venv /app/.venv

# Copy application source
COPY --chown=voicestack:voicestack src/ /app/src/

# Copy scripts and config templates (for reference)
COPY --chown=voicestack:voicestack scripts/.env.production.* /app/config/

# Copy entrypoint script
COPY --chown=voicestack:voicestack scripts/entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Create directories for runtime data
RUN mkdir -p /app/voices /app/models && \
    chown -R voicestack:voicestack /app

# Switch to non-root user
USER voicestack

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src:$PYTHONPATH" \
    # Default to production environment
    ENV=production \
    LOG_LEVEL=INFO \
    HOST=0.0.0.0

# Health check (will be service-specific via entrypoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${ASR_PORT:-5001}/health || curl -f http://localhost:${TTS_PORT:-5002}/health || exit 1

# Expose both ports (only the active service port will be used)
EXPOSE 5001 5002

# Use entrypoint script to determine which service to run
ENTRYPOINT ["/app/entrypoint.sh"]
