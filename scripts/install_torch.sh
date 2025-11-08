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
  CUDA=${CUDA:-"auto"}  # "auto" | "cu124" | "cu121" | "cu118" | "cpu"
  if [[ "${CUDA}" == "auto" ]]; then
    # Smart detection: parse actual CUDA version from nvidia-smi
    if command -v nvidia-smi >/dev/null 2>&1; then
      CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
      CUDA_MAJOR=$(echo "${CUDA_VERSION}" | cut -d. -f1)

      echo "Detected CUDA version: ${CUDA_VERSION}"

      # Choose PyTorch wheel based on detected CUDA version
      if [[ "${CUDA_MAJOR}" -ge 12 ]]; then
        CUDA="cu124"  # cu124 works for CUDA 12.x and 13.x
        echo "Using cu124 wheels (compatible with CUDA 12.x/13.x)"
      elif [[ "${CUDA_MAJOR}" == 11 ]]; then
        CUDA="cu118"  # for CUDA 11.x systems
        echo "Using cu118 wheels (compatible with CUDA 11.x)"
      else
        CUDA="cpu"
        echo "CUDA version too old, falling back to CPU"
      fi
    else
      CUDA="cpu"
    fi
  fi

  if [[ "${CUDA}" == "cpu" ]]; then
    pip install --upgrade "torch>=2.3" "torchaudio>=2.3" --index-url https://download.pytorch.org/whl/cpu
  elif [[ "${CUDA}" == "cu124" ]]; then
    pip install --upgrade "torch>=2.3" "torchaudio>=2.3" --index-url https://download.pytorch.org/whl/cu124
  elif [[ "${CUDA}" == "cu121" ]]; then
    pip install --upgrade "torch>=2.3" "torchaudio>=2.3" --index-url https://download.pytorch.org/whl/cu121
  elif [[ "${CUDA}" == "cu118" ]]; then
    pip install --upgrade "torch>=2.3" "torchaudio>=2.3" --index-url https://download.pytorch.org/whl/cu118
  else
    echo "Unknown CUDA choice: ${CUDA}. Use CUDA=cpu, CUDA=cu118, CUDA=cu121, or CUDA=cu124."
    exit 1
  fi
fi

echo "Torch installed."
