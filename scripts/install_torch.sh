#!/usr/bin/env bash
set -euo pipefail

# Decide your backend:
#   GPU (NVIDIA CUDA) on Debian -> install CUDA build
#   macOS Apple Silicon -> CPU build (MPS works with upstream torch)
#   CPU-only Linux -> CPU build

OS="$(uname -s)"
ARCH="$(uname -m)"

if [[ "${OS}" == "Darwin" ]]; then
  # macOS (Apple Silicon or Intel) — upstream PyPI has universal wheels
  pip install --upgrade "torch>=2.3" "torchaudio>=2.3" --index-url https://download.pytorch.org/whl/cpu
else
  # Linux: choose CUDA or CPU
  CUDA=${CUDA:-"auto"}  # "auto" | "cu121" | "cpu"
  if [[ "${CUDA}" == "auto" ]]; then
    # naive detection: if nvidia-smi exists, prefer CUDA 12.1 wheels
    if command -v nvidia-smi >/dev/null 2>&1; then
      CUDA="cu121"
    else
      CUDA="cpu"
    fi
  fi

  if [[ "${CUDA}" == "cpu" ]]; then
    pip install --upgrade "torch>=2.3" "torchaudio>=2.3" --index-url https://download.pytorch.org/whl/cpu
  elif [[ "${CUDA}" == "cu121" ]]; then
    pip install --upgrade "torch>=2.3" "torchaudio>=2.3" --index-url https://download.pytorch.org/whl/cu121
  else
    echo "Unknown CUDA choice: ${CUDA}. Use CUDA=cpu or CUDA=cu121."
    exit 1
  fi
fi

echo "Torch installed."
