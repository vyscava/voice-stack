#!/bin/bash
# ============================================================================
# Voice Stack Docker Entrypoint
# ============================================================================
# Smart entrypoint for unified Docker image
# Determines which service to run based on SERVICE_MODE environment variable
#
# Usage:
#   SERVICE_MODE=asr ./entrypoint.sh
#   SERVICE_MODE=tts ./entrypoint.sh
#
# Environment Variables:
#   SERVICE_MODE: "asr" or "tts" (required)
#   HOST: Listen address (default: 0.0.0.0)
#   ASR_PORT: ASR service port (default: 5001)
#   TTS_PORT: TTS service port (default: 5002)
#   LOG_LEVEL: Logging level (default: info)
#   WORKERS: Number of uvicorn workers (default: 1)
# ============================================================================

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check required SERVICE_MODE variable
if [ -z "$SERVICE_MODE" ]; then
    log_error "SERVICE_MODE environment variable is required"
    log_error "Valid options: 'asr' or 'tts'"
    log_error ""
    log_error "Example:"
    log_error "  docker run -e SERVICE_MODE=asr voice-stack:latest"
    log_error "  docker run -e SERVICE_MODE=tts voice-stack:latest"
    exit 1
fi

# Convert to lowercase for comparison
SERVICE_MODE=$(echo "$SERVICE_MODE" | tr '[:upper:]' '[:lower:]')

# Set defaults
HOST=${HOST:-0.0.0.0}
LOG_LEVEL=${LOG_LEVEL:-info}
WORKERS=${WORKERS:-1}

# Service-specific configuration
case "$SERVICE_MODE" in
    asr)
        SERVICE_NAME="Voice Stack ASR (Automatic Speech Recognition)"
        MODULE="asr.app:app"
        PORT=${ASR_PORT:-5001}
        DEFAULT_ENV_TEMPLATE="/app/config/.env.production.asr"
        ;;
    tts)
        SERVICE_NAME="Voice Stack TTS (Text-to-Speech)"
        MODULE="tts.app:app"
        PORT=${TTS_PORT:-5002}
        DEFAULT_ENV_TEMPLATE="/app/config/.env.production.tts"
        ;;
    *)
        log_error "Invalid SERVICE_MODE: '$SERVICE_MODE'"
        log_error "Valid options: 'asr' or 'tts'"
        exit 1
        ;;
esac

# Print startup banner
log_info "=========================================="
log_success "$SERVICE_NAME"
log_info "=========================================="
log_info "Mode:       $SERVICE_MODE"
log_info "Host:       $HOST"
log_info "Port:       $PORT"
log_info "Log Level:  $LOG_LEVEL"
log_info "Workers:    $WORKERS"
log_info "Module:     $MODULE"
log_info "=========================================="

# Check if .env file exists, if not suggest using template
if [ ! -f "/app/.env" ] && [ -f "$DEFAULT_ENV_TEMPLATE" ]; then
    log_info "No .env file found. Using defaults from environment variables."
    log_info "For production, mount a .env file or set environment variables."
    log_info "Template available at: $DEFAULT_ENV_TEMPLATE"
fi

# Ensure models directory exists (for downloaded models)
mkdir -p /app/models

# For TTS, ensure voices directory exists
if [ "$SERVICE_MODE" = "tts" ]; then
    mkdir -p /app/voices
    if [ -z "$(ls -A /app/voices)" ]; then
        log_info "Voices directory is empty. Add .wav files to /app/voices for voice cloning."
    else
        log_info "Found voice samples: $(ls -1 /app/voices | wc -l) files"
    fi
fi

# Print environment info
log_info "Python: $(python --version 2>&1)"
log_info "Working directory: $(pwd)"

# Handle signals for graceful shutdown
trap 'log_info "Received SIGTERM, shutting down..."; exit 0' SIGTERM
trap 'log_info "Received SIGINT, shutting down..."; exit 0' SIGINT

log_success "Starting $SERVICE_NAME..."
log_info ""

# Start the service using uvicorn
# Use exec to replace the shell process with uvicorn (proper signal handling)
exec uvicorn "$MODULE" \
    --host "$HOST" \
    --port "$PORT" \
    --log-level "$LOG_LEVEL" \
    --workers "$WORKERS" \
    --app-dir /app/src
