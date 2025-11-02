#!/bin/bash
# Voice Stack ASR Service Update Script
#
# This script updates the Voice Stack ASR service after code changes.
# It stops the service, updates dependencies, and restarts the service.
#
# Usage:
#   sudo ./update-asr.sh               # Update system service
#   ./update-asr.sh                     # Update user service
#

set -e

# Source common setup functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common-setup.sh"

# Configuration
SERVICE_NAME="voice-stack-asr"
SERVICE_PORT=5001
HATCH_ENV="prod-asr"
VENV_PATH=".venv-asr"

# Print banner
echo "=========================================="
echo "Voice Stack ASR Service Update"
echo "=========================================="
echo ""

# Detect user and repository location
detect_user
get_repo_root
check_privileges

# Check if service exists
if ! service_exists "$SERVICE_NAME"; then
    log_error "$SERVICE_NAME is not installed!"
    log_info "Please install it first with: ./scripts/install-asr.sh"
    exit 1
fi

# Stop the service
stop_service_if_running "$SERVICE_NAME"

# Pull latest changes
log_info "Pulling latest changes from git..."
cd "$REPO_ROOT"

if [ "$EUID" -eq 0 ] && [ -n "$SUDO_USER" ]; then
    sudo -u "$SUDO_USER" git pull --rebase
else
    git pull --rebase
fi

log_success "Git repository updated"

# Ensure hatch is installed
ensure_hatch_installed

# Rebuild Hatch environment
log_info "Rebuilding Hatch environment (this may take a few minutes)..."
cd "$REPO_ROOT"

# Prune the old environment
log_info "Removing old environment..."
if [ -d "$VENV_PATH" ]; then
    if [ "$EUID" -eq 0 ] && [ -n "$SUDO_USER" ]; then
        sudo -u "$SUDO_USER" hatch env prune "$HATCH_ENV" || rm -rf "$VENV_PATH"
    else
        hatch env prune "$HATCH_ENV" || rm -rf "$VENV_PATH"
    fi
fi

# Create new environment
setup_hatch_environment "$HATCH_ENV" "$VENV_PATH"

# Reinstall PyTorch (in case CUDA requirements changed)
install_pytorch

# Fix ownership
fix_ownership

# Restart the service
log_info "Starting $SERVICE_NAME..."
if [ "$INSTALL_AS_USER_SERVICE" = "true" ]; then
    systemctl --user start "$SERVICE_NAME"
else
    systemctl start "$SERVICE_NAME"
fi

# Wait for service to start
sleep 2

# Check service status
if service_is_active "$SERVICE_NAME"; then
    log_success "$SERVICE_NAME is running!"
else
    log_error "$SERVICE_NAME failed to start. Check logs with:"
    if [ "$INSTALL_AS_USER_SERVICE" = "true" ]; then
        echo "  systemctl --user status $SERVICE_NAME"
        echo "  journalctl --user -u $SERVICE_NAME -n 50"
    else
        echo "  systemctl status $SERVICE_NAME"
        echo "  journalctl -u $SERVICE_NAME -n 50"
    fi
    exit 1
fi

# Test health endpoint
log_info "Testing health endpoint..."
if command -v curl &> /dev/null; then
    sleep 3  # Give service a moment to fully start
    if curl -s -f "http://localhost:${SERVICE_PORT}/health" > /dev/null; then
        log_success "Health check passed!"
    else
        log_warn "Health check failed - service may still be starting"
    fi
else
    log_warn "curl not found, skipping health check"
fi

echo ""
log_success "=========================================="
log_success "$SERVICE_NAME update complete!"
log_success "=========================================="
echo ""
log_info "Service status:"
if [ "$INSTALL_AS_USER_SERVICE" = "true" ]; then
    systemctl --user status "$SERVICE_NAME" --no-pager || true
else
    systemctl status "$SERVICE_NAME" --no-pager || true
fi
echo ""
log_success "Update complete!"
