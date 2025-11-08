#!/bin/bash
# ============================================================================
# Coqui TTS License Pre-Acceptance Script
# ============================================================================
# Automatically accepts Coqui TTS license to prevent interactive prompts
# This is safe for Docker containers where interactive input is not possible
#
# Based on the mechanism in tests/conftest.py
#
# Usage:
#   ./accept_coqui_license.sh
#
# The Coqui TTS library checks for a "tos_agreed.txt" file before prompting
# for license acceptance. This script creates that file with "1" (agreement).
# ============================================================================

set -e

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_info "Pre-accepting Coqui TTS license agreement..."

# Determine TTS model cache directory
# Default location: ~/.local/share/tts
TTS_CACHE="${HOME}/.local/share/tts"

# Model directories that might trigger the license prompt
MODEL_DIRS=(
    "${TTS_CACHE}/tts_models--multilingual--multi-dataset--xtts_v2"
    "${TTS_CACHE}"  # Also create in root for safety
)

# Create license agreement files
for MODEL_DIR in "${MODEL_DIRS[@]}"; do
    # Create directory if it doesn't exist
    mkdir -p "$MODEL_DIR"

    TOS_FILE="${MODEL_DIR}/tos_agreed.txt"

    if [ -f "$TOS_FILE" ]; then
        log_info "License agreement already exists: $TOS_FILE"
    else
        # Create the agreement file with "1" (indicates agreement)
        echo "1" > "$TOS_FILE"
        log_success "Created license agreement: $TOS_FILE"
    fi
done

log_success "Coqui TTS license pre-acceptance complete"
log_info "Containers will not prompt for license agreement on first run"

exit 0
