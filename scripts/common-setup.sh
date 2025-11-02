#!/bin/bash
# Common functions for Voice Stack service installation scripts
# This file is sourced by install/update/uninstall scripts

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running with sufficient privileges for system service installation
check_privileges() {
    if [ "$EUID" -ne 0 ] && [ "$INSTALL_AS_USER_SERVICE" != "true" ]; then
        log_warn "Not running as root. Will install as user service instead of system service."
        log_warn "To install as system service, run with sudo."
        INSTALL_AS_USER_SERVICE="true"
    fi
}

# Detect the actual user (even when run with sudo)
detect_user() {
    if [ -n "$SUDO_USER" ]; then
        ACTUAL_USER="$SUDO_USER"
        ACTUAL_UID=$(id -u "$SUDO_USER")
        ACTUAL_GID=$(id -g "$SUDO_USER")
        ACTUAL_HOME=$(eval echo ~"$SUDO_USER")
    else
        ACTUAL_USER="$(whoami)"
        ACTUAL_UID=$(id -u)
        ACTUAL_GID=$(id -g)
        ACTUAL_HOME="$HOME"
    fi
    log_info "Detected user: $ACTUAL_USER (UID: $ACTUAL_UID)"
}

# Get the repository root directory
get_repo_root() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    log_info "Repository root: $REPO_ROOT"
}

# Check if a systemd service exists (system or user)
service_exists() {
    local service_name="$1"

    if [ "$INSTALL_AS_USER_SERVICE" = "true" ]; then
        systemctl --user list-unit-files | grep -q "^${service_name}"
    else
        systemctl list-unit-files | grep -q "^${service_name}"
    fi
}

# Check if a systemd service is active
service_is_active() {
    local service_name="$1"

    if [ "$INSTALL_AS_USER_SERVICE" = "true" ]; then
        systemctl --user is-active --quiet "$service_name"
    else
        systemctl is-active --quiet "$service_name"
    fi
}

# Stop a systemd service if it's running
stop_service_if_running() {
    local service_name="$1"

    if service_exists "$service_name"; then
        if service_is_active "$service_name"; then
            log_info "Stopping $service_name..."
            if [ "$INSTALL_AS_USER_SERVICE" = "true" ]; then
                systemctl --user stop "$service_name" || true
            else
                systemctl stop "$service_name" || true
            fi
            log_success "$service_name stopped"
        else
            log_info "$service_name is not running"
        fi
    else
        log_info "$service_name does not exist"
    fi
}

# Check system dependencies
check_system_dependencies() {
    log_info "Checking system dependencies..."

    local missing_deps=()

    # Check for Python 3.10+
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3 (3.10 or higher)")
    else
        python_version=$(python3 --version 2>&1 | awk '{print $2}')
        python_major=$(echo "$python_version" | cut -d. -f1)
        python_minor=$(echo "$python_version" | cut -d. -f2)
        if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 10 ]); then
            missing_deps+=("python3 (3.10 or higher, found $python_version)")
        fi
    fi

    # Check for pip
    if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
        missing_deps+=("pip3")
    fi

    # Check for hatch
    if ! command -v hatch &> /dev/null; then
        log_warn "Hatch not found. Will attempt to install it."
    fi

    # Check for FFmpeg
    if ! command -v ffmpeg &> /dev/null; then
        missing_deps+=("ffmpeg")
    fi

    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_error "Missing required dependencies:"
        for dep in "${missing_deps[@]}"; do
            echo "  - $dep"
        done
        log_info "Please install missing dependencies first."
        log_info "You can run: ./scripts/install_system_deps.sh"
        exit 1
    fi

    log_success "All system dependencies are installed"
}

# Install hatch if not present
ensure_hatch_installed() {
    if ! command -v hatch &> /dev/null; then
        log_info "Installing hatch..."
        if [ "$EUID" -eq 0 ] && [ -n "$SUDO_USER" ]; then
            # Install as the actual user, not root
            sudo -u "$SUDO_USER" pip3 install --user hatch
        else
            pip3 install --user hatch
        fi

        # Add to PATH if needed
        export PATH="$ACTUAL_HOME/.local/bin:$PATH"

        if ! command -v hatch &> /dev/null; then
            log_error "Failed to install hatch. Please install it manually: pip3 install hatch"
            exit 1
        fi
        log_success "Hatch installed successfully"
    else
        log_info "Hatch is already installed"
    fi
}

# Create and configure Hatch environment
setup_hatch_environment() {
    local env_name="$1"  # prod-asr or prod-tts
    local venv_path="$2" # .venv-asr or .venv-tts

    log_info "Setting up Hatch environment: $env_name"

    cd "$REPO_ROOT"

    # Remove existing environment if it exists
    if [ -d "$venv_path" ]; then
        log_info "Removing existing environment at $venv_path..."
        rm -rf "$venv_path"
    fi

    # Create the environment
    log_info "Creating Hatch environment (this may take a few minutes)..."
    if [ "$EUID" -eq 0 ] && [ -n "$SUDO_USER" ]; then
        # Run as the actual user
        sudo -u "$SUDO_USER" -E hatch env create "$env_name"
    else
        hatch env create "$env_name"
    fi

    log_success "Hatch environment created: $venv_path"
}

# Install PyTorch with appropriate backend
install_pytorch() {
    log_info "Installing PyTorch..."

    cd "$REPO_ROOT"

    if [ -f "./scripts/install_torch.sh" ]; then
        if [ "$EUID" -eq 0 ] && [ -n "$SUDO_USER" ]; then
            sudo -u "$SUDO_USER" bash ./scripts/install_torch.sh
        else
            bash ./scripts/install_torch.sh
        fi
        log_success "PyTorch installation completed"
    else
        log_warn "PyTorch installation script not found. Skipping."
        log_warn "You may need to install PyTorch manually for GPU support."
    fi
}

# Create systemd service file
install_systemd_service() {
    local service_name="$1"       # voice-stack-asr or voice-stack-tts
    local template_file="$2"       # path to template file
    local service_user="$3"        # user to run service as
    local service_group="$4"       # group to run service as
    local install_dir="$5"         # installation directory
    local home_dir="$6"            # home directory of service user

    log_info "Installing systemd service: $service_name"

    # Determine service file location
    if [ "$INSTALL_AS_USER_SERVICE" = "true" ]; then
        SERVICE_DIR="$home_dir/.config/systemd/user"
        mkdir -p "$SERVICE_DIR"
    else
        SERVICE_DIR="/etc/systemd/system"
    fi

    local service_file="$SERVICE_DIR/${service_name}.service"

    # Replace placeholders in template
    sed -e "s|__SERVICE_USER__|${service_user}|g" \
        -e "s|__SERVICE_GROUP__|${service_group}|g" \
        -e "s|__INSTALL_DIR__|${install_dir}|g" \
        -e "s|__HOME_DIR__|${home_dir}|g" \
        "$template_file" > "$service_file"

    # Set correct permissions
    chmod 644 "$service_file"

    log_success "Service file created: $service_file"

    # Reload systemd
    log_info "Reloading systemd daemon..."
    if [ "$INSTALL_AS_USER_SERVICE" = "true" ]; then
        systemctl --user daemon-reload
    else
        systemctl daemon-reload
    fi

    log_success "Systemd daemon reloaded"
}

# Enable and start systemd service
enable_and_start_service() {
    local service_name="$1"

    log_info "Enabling $service_name..."
    if [ "$INSTALL_AS_USER_SERVICE" = "true" ]; then
        systemctl --user enable "$service_name"
    else
        systemctl enable "$service_name"
    fi
    log_success "$service_name enabled"

    log_info "Starting $service_name..."
    if [ "$INSTALL_AS_USER_SERVICE" = "true" ]; then
        systemctl --user start "$service_name"
    else
        systemctl start "$service_name"
    fi
    log_success "$service_name started"

    # Wait a moment for service to start
    sleep 2

    # Check service status
    if service_is_active "$service_name"; then
        log_success "$service_name is running!"
    else
        log_error "$service_name failed to start. Check logs with:"
        if [ "$INSTALL_AS_USER_SERVICE" = "true" ]; then
            echo "  systemctl --user status $service_name"
            echo "  journalctl --user -u $service_name -n 50"
        else
            echo "  systemctl status $service_name"
            echo "  journalctl -u $service_name -n 50"
        fi
        exit 1
    fi
}

# Remove systemd service
remove_systemd_service() {
    local service_name="$1"

    if ! service_exists "$service_name"; then
        log_info "$service_name does not exist, skipping removal"
        return 0
    fi

    log_info "Removing $service_name..."

    # Stop the service
    stop_service_if_running "$service_name"

    # Disable the service
    log_info "Disabling $service_name..."
    if [ "$INSTALL_AS_USER_SERVICE" = "true" ]; then
        systemctl --user disable "$service_name" || true
    else
        systemctl disable "$service_name" || true
    fi

    # Remove service file
    if [ "$INSTALL_AS_USER_SERVICE" = "true" ]; then
        SERVICE_FILE="$ACTUAL_HOME/.config/systemd/user/${service_name}.service"
    else
        SERVICE_FILE="/etc/systemd/system/${service_name}.service"
    fi

    if [ -f "$SERVICE_FILE" ]; then
        rm -f "$SERVICE_FILE"
        log_success "Service file removed: $SERVICE_FILE"
    fi

    # Reload systemd
    log_info "Reloading systemd daemon..."
    if [ "$INSTALL_AS_USER_SERVICE" = "true" ]; then
        systemctl --user daemon-reload
    else
        systemctl daemon-reload
    fi

    log_success "$service_name removed successfully"
}

# Create .env file if it doesn't exist
setup_env_file() {
    local env_template="$1"
    local env_file="$REPO_ROOT/.env"

    if [ -f "$env_file" ]; then
        log_warn ".env file already exists at $env_file"
        log_warn "Please review and update it manually if needed"
        log_warn "Template available at: $env_template"
    else
        log_info "Creating .env file from template..."
        cp "$env_template" "$env_file"

        # Set proper ownership
        if [ "$EUID" -eq 0 ] && [ -n "$SUDO_USER" ]; then
            chown "$ACTUAL_USER:$ACTUAL_USER" "$env_file"
        fi

        log_success ".env file created at $env_file"
        log_warn "Please review and update .env with your configuration!"
    fi
}

# Set proper ownership of repository files
fix_ownership() {
    if [ "$EUID" -eq 0 ] && [ -n "$SUDO_USER" ]; then
        log_info "Setting proper ownership of $REPO_ROOT..."
        chown -R "$ACTUAL_USER:$ACTUAL_USER" "$REPO_ROOT"
        log_success "Ownership set to $ACTUAL_USER:$ACTUAL_USER"
    fi
}

# Print service status and useful commands
print_service_info() {
    local service_name="$1"
    local port="$2"
    local health_endpoint="http://localhost:${port}/health"

    echo ""
    log_success "=========================================="
    log_success "$service_name installation complete!"
    log_success "=========================================="
    echo ""
    log_info "Service management commands:"

    if [ "$INSTALL_AS_USER_SERVICE" = "true" ]; then
        echo "  Start:   systemctl --user start $service_name"
        echo "  Stop:    systemctl --user stop $service_name"
        echo "  Restart: systemctl --user restart $service_name"
        echo "  Status:  systemctl --user status $service_name"
        echo "  Logs:    journalctl --user -u $service_name -f"
    else
        echo "  Start:   sudo systemctl start $service_name"
        echo "  Stop:    sudo systemctl stop $service_name"
        echo "  Restart: sudo systemctl restart $service_name"
        echo "  Status:  sudo systemctl status $service_name"
        echo "  Logs:    sudo journalctl -u $service_name -f"
    fi

    echo ""
    log_info "Health check: $health_endpoint"
    echo ""
}

# Export all functions
export -f log_info log_success log_warn log_error
export -f check_privileges detect_user get_repo_root
export -f service_exists service_is_active stop_service_if_running
export -f check_system_dependencies ensure_hatch_installed
export -f setup_hatch_environment install_pytorch
export -f install_systemd_service enable_and_start_service remove_systemd_service
export -f setup_env_file fix_ownership print_service_info
