#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" == "Darwin" ]]; then
  # macOS
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew not found. Install from https://brew.sh/"
    exit 1
  fi
  brew update
  brew install ffmpeg portaudio libsndfile
else
  # Debian/Ubuntu
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    python3-dev \
    build-essential
fi
echo "System deps installed."
