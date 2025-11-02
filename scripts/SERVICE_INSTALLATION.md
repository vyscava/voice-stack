# Voice Stack Service Installation Guide

This guide explains how to install, update, and manage Voice Stack ASR and TTS services as systemd services on Debian/Ubuntu Linux.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Installing ASR Service](#installing-asr-service)
  - [Installing TTS Service](#installing-tts-service)
  - [Installing Both Services](#installing-both-services)
- [Configuration](#configuration)
- [Service Management](#service-management)
- [Updating Services](#updating-services)
- [Uninstalling Services](#uninstalling-services)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

## Overview

Voice Stack services can be installed as:

- **System services**: Run as a system-wide service (requires root/sudo)
- **User services**: Run as a user-specific service (no root required)

Services are fully independent - you can install ASR without TTS, or vice versa.

### Architecture

```
/path/to/voice-stack/          # Repository root (wherever you cloned it)
├── .venv-asr/                  # ASR virtual environment
├── .venv-tts/                  # TTS virtual environment
├── .env                        # Configuration file
├── src/                        # Source code
├── scripts/                    # Installation scripts
├── voices/                     # TTS voice samples
└── models/                     # Downloaded models (optional)
```

Models are cached to:
- **Faster-Whisper/Whisper**: `~/.cache/huggingface/hub/`
- **Coqui TTS**: `~/.local/share/tts/`

## Prerequisites

### System Requirements

**Operating System:**
- Debian 11+ or Ubuntu 20.04+
- systemd-based Linux distribution

**Hardware Requirements:**

**For CPU-only:**
- 4+ CPU cores
- 8GB+ RAM (16GB+ recommended)
- 10GB+ free disk space

**For GPU (recommended for production):**
- NVIDIA GPU with CUDA 12.1+ support
- 8GB+ VRAM (for TTS, larger ASR models)
- NVIDIA drivers 525+
- CUDA 12.1+ and cuDNN installed

### Software Dependencies

**Required:**
- Python 3.10 or higher
- pip3
- git
- FFmpeg
- libsndfile1
- libportaudio2

**Installation:**

```bash
# Update package list
sudo apt update

# Install system dependencies using the provided script
sudo ./scripts/install_system_deps.sh

# Or install manually
sudo apt install -y python3 python3-pip python3-dev git \
    ffmpeg libsndfile1 libportaudio2 build-essential
```

**For GPU support:**
```bash
# Install NVIDIA drivers, CUDA toolkit, and cuDNN
# See: https://developer.nvidia.com/cuda-downloads

# Verify CUDA installation
nvidia-smi
nvcc --version
```

## Installation

### Quick Start

**1. Clone the repository:**
```bash
# Choose your installation location
cd /opt  # System-wide installation
# OR
cd ~     # User-specific installation

# Clone the repository
git clone https://github.com/vyscava/voice-stack.git
cd voice-stack
```

**2. Install system dependencies:**
```bash
sudo ./scripts/install_system_deps.sh
```

**3. Install the service(s):**
```bash
# Install ASR service
sudo ./scripts/install-asr.sh

# OR install TTS service
sudo ./scripts/install-tts.sh

# OR install both
sudo ./scripts/install-asr.sh
sudo ./scripts/install-tts.sh
```

### Installing ASR Service

```bash
cd /path/to/voice-stack

# Install as system service (recommended)
sudo ./scripts/install-asr.sh

# OR install as user service (no sudo)
./scripts/install-asr.sh
```

**What the script does:**
1. Checks system dependencies
2. Installs Hatch (Python environment manager)
3. Creates `.venv-asr` virtual environment
4. Installs ASR dependencies (Faster-Whisper, Whisper, etc.)
5. Installs PyTorch with CUDA support (if available)
6. Creates `.env` configuration file from template
7. Installs and starts systemd service

**After installation:**
```bash
# Check service status
sudo systemctl status voice-stack-asr

# Check logs
sudo journalctl -u voice-stack-asr -f

# Test health endpoint
curl http://localhost:5001/health
```

### Installing TTS Service

```bash
cd /path/to/voice-stack

# Install as system service (recommended)
sudo ./scripts/install-tts.sh

# OR install as user service (no sudo)
./scripts/install-tts.sh
```

**What the script does:**
1. Checks system dependencies
2. Installs Hatch (Python environment manager)
3. Creates `.venv-tts` virtual environment
4. Installs TTS dependencies (Coqui TTS, etc.)
5. Installs PyTorch with CUDA support (if available)
6. Creates `.env` configuration file from template
7. Creates `voices/` directory for voice samples
8. Installs and starts systemd service

**After installation:**
```bash
# Check service status
sudo systemctl status voice-stack-tts

# Check logs
sudo journalctl -u voice-stack-tts -f

# Test health endpoint
curl http://localhost:5002/health
```

### Installing Both Services

You can install both services on the same machine:

```bash
cd /path/to/voice-stack

# Install both services
sudo ./scripts/install-asr.sh
sudo ./scripts/install-tts.sh
```

**Note:** Both services share the same `.env` file but use separate virtual environments.

## Configuration

### Environment Configuration

Configuration is managed through the `.env` file in the repository root.

**For ASR:**
```bash
# Copy and edit the ASR production template
cp scripts/.env.production.asr .env
nano .env

# Key settings to review:
# - ASR_DEVICE=cuda (or cpu)
# - ASR_MODEL=medium (or large-v3 for best accuracy)
# - ASR_ENGINE=fasterwhisper (recommended)
# - ASR_VAD_ENABLED=true (recommended for performance)
```

**For TTS:**
```bash
# Copy and edit the TTS production template
cp scripts/.env.production.tts .env
nano .env

# Key settings to review:
# - TTS_DEVICE=cuda (or cpu)
# - TTS_MODEL=tts_models/multilingual/multi-dataset/xtts_v2
# - TTS_VOICES_DIR=voices
# - TTS_AUTO_LANG=true
```

**If installing both services:**
Both services can share a single `.env` file with combined settings:

```bash
# Manually merge the templates or create a combined .env
cat scripts/.env.production.asr scripts/.env.production.tts > .env
nano .env  # Remove duplicate lines and customize
```

**After modifying `.env`:**
```bash
# Restart the service(s) to apply changes
sudo systemctl restart voice-stack-asr
sudo systemctl restart voice-stack-tts
```

### GPU Configuration

**Enable CUDA support:**

1. Install CUDA and drivers:
```bash
# Check if CUDA is available
nvidia-smi

# If not available, install NVIDIA drivers and CUDA toolkit
# See: https://developer.nvidia.com/cuda-downloads
```

2. Install PyTorch with CUDA:
```bash
cd /path/to/voice-stack
./scripts/install_torch.sh  # Auto-detects CUDA
```

3. Update `.env`:
```bash
# For ASR
ASR_DEVICE=cuda
ASR_COMPUTE_TYPE=int8_float16  # Recommended for CUDA

# For TTS
TTS_DEVICE=cuda
```

4. Restart services:
```bash
sudo systemctl restart voice-stack-asr
sudo systemctl restart voice-stack-tts
```

### Adding Voice Samples (TTS Only)

Voice samples enable voice cloning for TTS:

1. Prepare voice samples:
   - Format: WAV (22050Hz or 24000Hz, mono recommended)
   - Duration: 6-12 seconds
   - Quality: Clear audio, single speaker, minimal background noise

2. Add samples to `voices/` directory:
```bash
cd /path/to/voice-stack
cp /path/to/your/voice.wav voices/my-voice.wav
```

3. The voice name is the filename without extension:
   - `voices/john.wav` → voice name: `john`
   - `voices/jane-formal.wav` → voice name: `jane-formal`

4. Restart TTS service:
```bash
sudo systemctl restart voice-stack-tts
```

5. Use the voice in API requests:
```bash
curl -X POST http://localhost:5002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world!",
    "voice": "my-voice"
  }' \
  --output speech.mp3
```

## Service Management

### Systemctl Commands

**System services (requires sudo):**
```bash
# Start service
sudo systemctl start voice-stack-asr
sudo systemctl start voice-stack-tts

# Stop service
sudo systemctl stop voice-stack-asr
sudo systemctl stop voice-stack-tts

# Restart service
sudo systemctl restart voice-stack-asr
sudo systemctl restart voice-stack-tts

# Check status
sudo systemctl status voice-stack-asr
sudo systemctl status voice-stack-tts

# Enable auto-start on boot (already enabled by install script)
sudo systemctl enable voice-stack-asr
sudo systemctl enable voice-stack-tts

# Disable auto-start
sudo systemctl disable voice-stack-asr
sudo systemctl disable voice-stack-tts
```

**User services (no sudo):**
```bash
# Add --user flag to all commands
systemctl --user start voice-stack-asr
systemctl --user stop voice-stack-asr
systemctl --user restart voice-stack-asr
systemctl --user status voice-stack-asr
```

### Viewing Logs

**Real-time logs (follow mode):**
```bash
# System service
sudo journalctl -u voice-stack-asr -f
sudo journalctl -u voice-stack-tts -f

# User service
journalctl --user -u voice-stack-asr -f
journalctl --user -u voice-stack-tts -f
```

**Recent logs (last 50 lines):**
```bash
sudo journalctl -u voice-stack-asr -n 50
sudo journalctl -u voice-stack-tts -n 50
```

**Logs since specific time:**
```bash
sudo journalctl -u voice-stack-asr --since "1 hour ago"
sudo journalctl -u voice-stack-asr --since "2024-01-15 10:00:00"
```

### Health Checks

Both services expose health check endpoints:

```bash
# ASR health check
curl http://localhost:5001/health
curl http://localhost:5001/healthz
curl http://localhost:5001/healthcheck

# TTS health check
curl http://localhost:5002/health
curl http://localhost:5002/healthz
curl http://localhost:5002/healthcheck
```

**Expected response:**
```json
{"status": "healthy"}
```

## Updating Services

When you update the code (e.g., `git pull`), you need to update the services:

### Updating ASR Service

```bash
cd /path/to/voice-stack

# Update service (pulls git changes, rebuilds environment, restarts)
sudo ./scripts/update-asr.sh

# OR for user service
./scripts/update-asr.sh
```

**What the update script does:**
1. Stops the service
2. Pulls latest changes (`git pull --rebase`)
3. Removes old virtual environment
4. Creates new virtual environment with updated dependencies
5. Reinstalls PyTorch (in case CUDA requirements changed)
6. Restarts the service
7. Verifies the service is running

### Updating TTS Service

```bash
cd /path/to/voice-stack

# Update service
sudo ./scripts/update-tts.sh

# OR for user service
./scripts/update-tts.sh
```

### Updating Both Services

```bash
cd /path/to/voice-stack

# Update both services
sudo ./scripts/update-asr.sh
sudo ./scripts/update-tts.sh
```

**Important:** The update scripts use `git pull --rebase` to avoid merge commits. Make sure you have no uncommitted local changes before updating.

## Uninstalling Services

### Uninstalling ASR Service

```bash
cd /path/to/voice-stack

# Uninstall system service
sudo ./scripts/uninstall-asr.sh

# OR uninstall user service
./scripts/uninstall-asr.sh
```

**What the script does:**
1. Stops the service
2. Disables the service
3. Removes systemd service file
4. Optionally removes virtual environment (asks for confirmation)

**Note:** Repository files and models are NOT removed.

### Uninstalling TTS Service

```bash
cd /path/to/voice-stack

# Uninstall system service
sudo ./scripts/uninstall-tts.sh

# OR uninstall user service
./scripts/uninstall-tts.sh
```

### Complete Removal

To completely remove Voice Stack:

```bash
cd /path/to/voice-stack

# Uninstall services
sudo ./scripts/uninstall-asr.sh
sudo ./scripts/uninstall-tts.sh

# Remove repository
cd ..
rm -rf voice-stack

# Optional: Remove cached models
rm -rf ~/.cache/huggingface/hub/
rm -rf ~/.local/share/tts/
```

## Troubleshooting

### Service Won't Start

**1. Check service status:**
```bash
sudo systemctl status voice-stack-asr
sudo journalctl -u voice-stack-asr -n 100
```

**2. Common issues:**

**Missing dependencies:**
```bash
# Reinstall system dependencies
sudo ./scripts/install_system_deps.sh
```

**Python version mismatch:**
```bash
# Check Python version (must be 3.10+)
python3 --version

# If too old, install Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-dev
```

**Virtual environment issues:**
```bash
# Rebuild virtual environment
cd /path/to/voice-stack
rm -rf .venv-asr .venv-tts
sudo ./scripts/install-asr.sh  # or install-tts.sh
```

**Permission issues:**
```bash
# Fix ownership
cd /path/to/voice-stack
sudo chown -R $USER:$USER .
```

**Port already in use:**
```bash
# Check what's using the port
sudo lsof -i :5001  # ASR
sudo lsof -i :5002  # TTS

# Change port in .env
nano .env
# Set ASR_PORT=5003 or TTS_PORT=5004
sudo systemctl restart voice-stack-asr
```

### Service Crashes or Restarts

**1. Check memory usage:**
```bash
# Monitor memory while service runs
htop
# or
free -h
watch -n 1 free -h
```

**2. Reduce memory usage:**

For ASR:
```bash
# In .env, use smaller model
ASR_MODEL=small  # instead of medium or large
ASR_NUM_OF_WORKERS=1  # reduce workers
```

For TTS:
```bash
# In .env
TTS_MAX_CHARS=120  # reduce from 180
```

**3. Check for CUDA out-of-memory errors:**
```bash
# View logs
sudo journalctl -u voice-stack-tts -n 100 | grep -i "cuda\|memory"

# If CUDA OOM, switch to CPU or use smaller model
nano .env
# Set TTS_DEVICE=cpu or ASR_DEVICE=cpu
```

### Model Download Issues

**Models fail to download:**

```bash
# Check internet connectivity
ping huggingface.co

# Manually download models
cd /path/to/voice-stack
source .venv-asr/bin/activate
python3 -c "import faster_whisper; faster_whisper.WhisperModel('medium')"
```

### API Not Responding

**1. Check if service is running:**
```bash
sudo systemctl status voice-stack-asr
curl http://localhost:5001/health
```

**2. Check firewall:**
```bash
# Allow ports through firewall
sudo ufw allow 5001/tcp  # ASR
sudo ufw allow 5002/tcp  # TTS
```

**3. Check CORS settings:**
```bash
# In .env, allow your origin
CORS_ORIGINS=http://your-frontend-domain.com,https://your-frontend-domain.com
```

### Performance Issues

**Slow transcription/synthesis:**

1. Enable GPU acceleration:
```bash
# Verify CUDA is available
nvidia-smi
./scripts/install_torch.sh
# Set ASR_DEVICE=cuda and TTS_DEVICE=cuda in .env
```

2. Enable VAD for ASR (reduces processing time by 30-50%):
```bash
# In .env
ASR_VAD_ENABLED=true
```

3. Use Faster-Whisper instead of Whisper:
```bash
# In .env
ASR_ENGINE=fasterwhisper
```

4. Optimize model selection:
```bash
# Smaller models are faster but less accurate
ASR_MODEL=small  # or base, medium
```

## Advanced Topics

### Running Behind Nginx Reverse Proxy

**Example Nginx configuration:**

```nginx
# /etc/nginx/sites-available/voice-stack

server {
    listen 80;
    server_name your-domain.com;

    # ASR Service
    location /asr/ {
        proxy_pass http://localhost:5001/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # For file uploads
        client_max_body_size 100M;
    }

    # TTS Service
    location /tts/ {
        proxy_pass http://localhost:5002/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable and restart Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/voice-stack /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Custom Model Paths

To use a specific model location:

```bash
# In .env
ASR_MODEL_LOCATION=/mnt/models/whisper
```

Then copy models to that directory:
```bash
mkdir -p /mnt/models/whisper
# Download and place models there
```

### User Services (systemd --user)

If you don't have sudo access, install as user service:

```bash
cd /path/to/voice-stack

# Install as user service
./scripts/install-asr.sh
./scripts/install-tts.sh

# Manage with --user flag
systemctl --user start voice-stack-asr
systemctl --user status voice-stack-asr
journalctl --user -u voice-stack-asr -f
```

**Enable user services to start on boot:**
```bash
# Enable lingering (allows user services to run without login)
sudo loginctl enable-linger $USER

# Enable services
systemctl --user enable voice-stack-asr
systemctl --user enable voice-stack-tts
```

### Multiple Instances

To run multiple instances on different ports:

1. Clone repository to different location
2. Modify `.env` with different ports
3. Install with custom service names (manually edit service templates)

### Monitoring with Prometheus

Add Prometheus metrics endpoint to the services (requires code modification) or use node_exporter for system metrics.

### Logging to File

To log to file in addition to journald:

1. Create log directory:
```bash
mkdir -p /path/to/voice-stack/logs
```

2. Modify uvicorn command in service file to add `--log-config`:
```bash
# Create logging config
# See: https://docs.python.org/3/library/logging.config.html
```

## Support

For issues, questions, or contributions:

- GitHub Issues: https://github.com/vyscava/voice-stack/issues
- Main README: [../README.md](../README.md)
- API Documentation: See main README

## License

MIT License - see [../LICENSE](../LICENSE)
