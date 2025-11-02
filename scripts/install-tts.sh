#!/bin/bash
# Voice Stack TTS Service Installation Script
#
# This script installs the Voice Stack TTS (Text-to-Speech) service
# as a systemd service. It can be run as root (for system-wide installation) or
# as a regular user (for user-specific installation).
#
# Usage:
#   sudo ./install-tts.sh              # Install as system service
#   ./install-tts.sh                    # Install as user service
#

set -e

# Source common setup functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common-setup.sh"

# Configuration
SERVICE_NAME="voice-stack-tts"
SERVICE_PORT=5002
HATCH_ENV="prod-tts"
VENV_PATH=".venv-tts"

# Print banner
echo "=========================================="
echo "Voice Stack TTS Service Installation"
echo "=========================================="
echo ""

# Detect user and repository location
detect_user
get_repo_root
check_privileges

# Check system dependencies
check_system_dependencies

# Ensure hatch is installed
ensure_hatch_installed

# Stop service if it's already running
stop_service_if_running "$SERVICE_NAME"

# Install system dependencies (if not already done)
log_info "Checking system dependencies..."
if [ -f "$REPO_ROOT/scripts/install_system_deps.sh" ]; then
    log_info "Installing system dependencies..."
    if [ "$EUID" -eq 0 ]; then
        bash "$REPO_ROOT/scripts/install_system_deps.sh"
    else
        log_warn "System dependencies installation requires root privileges."
        log_warn "Please run: sudo $REPO_ROOT/scripts/install_system_deps.sh"
        log_warn "Continuing without system dependencies check..."
    fi
else
    log_warn "System dependencies script not found. Skipping."
fi

# Setup Hatch environment for TTS
setup_hatch_environment "$HATCH_ENV" "$VENV_PATH"

# Install PyTorch with appropriate backend
install_pytorch

# Create .env file if it doesn't exist
ENV_TEMPLATE="$REPO_ROOT/scripts/.env.production.tts"
if [ -f "$ENV_TEMPLATE" ]; then
    setup_env_file "$ENV_TEMPLATE"
else
    log_warn "Production .env template not found at $ENV_TEMPLATE"
    log_warn "Please create a .env file manually in $REPO_ROOT"
fi

# Ensure voices directory exists
VOICES_DIR="$REPO_ROOT/voices"
if [ ! -d "$VOICES_DIR" ]; then
    log_info "Creating voices directory at $VOICES_DIR"
    mkdir -p "$VOICES_DIR"
    if [ "$EUID" -eq 0 ] && [ -n "$SUDO_USER" ]; then
        chown "$ACTUAL_USER:$ACTUAL_USER" "$VOICES_DIR"
    fi
    log_success "Voices directory created"
fi

# Fix ownership of all files
fix_ownership

# Install systemd service
TEMPLATE_FILE="$SCRIPT_DIR/voice-stack-tts.service.template"
install_systemd_service "$SERVICE_NAME" "$TEMPLATE_FILE" "$ACTUAL_USER" "$ACTUAL_USER" "$REPO_ROOT" "$ACTUAL_HOME"

# Enable and start the service
enable_and_start_service "$SERVICE_NAME"

# Print service information
print_service_info "$SERVICE_NAME" "$SERVICE_PORT"

log_info "Next steps:"
echo "  1. Review and update .env file at: $REPO_ROOT/.env"
echo "  2. Customize TTS settings (model, device, voices, etc.)"
echo "  3. Add voice samples to: $VOICES_DIR"
echo "  4. Restart service after changes: systemctl restart $SERVICE_NAME"
echo "  5. Test the service: curl http://localhost:${SERVICE_PORT}/health"
echo ""
log_success "Installation complete!"
