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

# Install hatch
RUN pip install --no-cache-dir hatch

# Copy scripts for PyTorch installation and license acceptance
COPY scripts/install_torch.sh scripts/accept_coqui_license.sh ./scripts/
RUN chmod +x scripts/*.sh

# Create production environment with ASR + TTS + server (NO dev deps)
RUN hatch env create prod-asr && \
    hatch env create prod-tts

# Merge both prod venvs into a single unified venv
# prod-asr has: server + asr deps, prod-tts has: server + tts deps
# We install everything into one venv for the unified image
RUN python -m venv /build/.venv && \
    /build/.venv/bin/pip install --no-cache-dir -e ".[server,asr,tts]"

# Install PyTorch (auto-detects CPU/CUDA)
RUN bash scripts/install_torch.sh

# Pre-accept Coqui TTS license to prevent interactive prompts
RUN bash scripts/accept_coqui_license.sh

# Fix shebangs to match production path
RUN sed -i 's|#!/build/.venv/bin/python|#!/app/.venv/bin/python|g' /build/.venv/bin/* && \
    # Clean up unnecessary files to reduce image size
    find /build/.venv -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true && \
    find /build/.venv -type d -name 'tests' -exec rm -rf {} + 2>/dev/null || true && \
    find /build/.venv -name '*.pyc' -delete 2>/dev/null || true && \
    find /build/.venv -name '*.pyo' -delete 2>/dev/null || true

# Remove the intermediate hatch envs (not needed)
RUN rm -rf /build/.venv-asr /build/.venv-tts

# -----------------------------------------------------------------------------
# Stage 2: Production - Minimal Runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS production

LABEL maintainer="vyscava@gmail.com"
LABEL description="Voice Stack unified ASR + TTS service"
LABEL version="0.1.0"

# Install only runtime system dependencies (no build tools)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libportaudio2 \
        curl \
        ca-certificates \
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
