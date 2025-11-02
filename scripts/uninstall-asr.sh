#!/bin/bash
# Voice Stack ASR Service Uninstallation Script
#
# This script removes the Voice Stack ASR service from systemd.
# It does NOT remove the repository files or virtual environment.
#
# Usage:
#   sudo ./uninstall-asr.sh            # Uninstall system service
#   ./uninstall-asr.sh                  # Uninstall user service
#

set -e

# Source common setup functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common-setup.sh"

# Configuration
SERVICE_NAME="voice-stack-asr"
VENV_PATH=".venv-asr"

# Print banner
echo "=========================================="
echo "Voice Stack ASR Service Uninstallation"
echo "=========================================="
echo ""

# Detect user and repository location
detect_user
get_repo_root
check_privileges

# Check if service exists
if ! service_exists "$SERVICE_NAME"; then
    log_warn "$SERVICE_NAME is not installed"
    log_info "Nothing to uninstall"
    exit 0
fi

# Confirm uninstallation
log_warn "This will remove the $SERVICE_NAME systemd service."
log_warn "The repository files and virtual environment will NOT be removed."
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_info "Uninstallation cancelled"
    exit 0
fi

# Remove the systemd service
remove_systemd_service "$SERVICE_NAME"

# Ask if user wants to remove the virtual environment
echo ""
log_info "Virtual environment at $REPO_ROOT/$VENV_PATH is still present."
read -p "Do you want to remove it? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -d "$REPO_ROOT/$VENV_PATH" ]; then
        log_info "Removing virtual environment..."
        rm -rf "$REPO_ROOT/$VENV_PATH"
        log_success "Virtual environment removed"
    else
        log_info "Virtual environment not found"
    fi
fi

echo ""
log_success "=========================================="
log_success "$SERVICE_NAME uninstallation complete!"
log_success "=========================================="
echo ""
log_info "Repository files are still present at: $REPO_ROOT"
log_info "You can reinstall the service later with: ./scripts/install-asr.sh"
echo ""
