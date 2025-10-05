
# 🗣️ Voice Stack — ASR + TTS Microservices

This repository contains modular **speech services** built around **Automatic Speech Recognition (ASR)** and **Text-to-Speech (TTS)** using modern Coqui TTS models such as `XTTS-v2`.

It’s designed for **local development** on macOS and **production deployment** on Linux (Debian 12 + NVIDIA GPU).

---

## 🚀 Features

- **ASR**: Automatic Speech Recognition microservice with FastAPI.
- **TTS**: Text-to-Speech microservice using Coqui `XTTS-v2`.
- **Cross-platform**: Works on macOS (CPU) and Debian (GPU) with minimal changes.
- **Voice cloning support**: Load and manage `.wav` voice samples under `/voices`.
- **Hatch + Nox toolchain**: Unified dev, test, and build system.
- **CI/CD ready**: GitLab pipelines supported for testing and packaging.

---

## 🧰 Prerequisites

### macOS (development)
- Python **3.11+**
- [Hatch](https://hatch.pypa.io/latest/) (>=1.13)
- [Nox](https://nox.thea.codes/en/stable/)
- [FFmpeg](https://ffmpeg.org/download.html) — for audio processing
  Install via Homebrew:
  ```bash
  brew install ffmpeg
  ```

### Debian 12 (production)
- Python **3.11+**
- CUDA 12.x + cuDNN (for GPU acceleration)
- FFmpeg
- NVIDIA drivers correctly installed

---

## 🧑‍💻 Development Setup

1. **Clone the repository**
   ```bash
   git clone https://gitlab.vitorgarbim.me/data-platforms/voice-stack.git
   cd voice-stack
   ```

2. **Install Hatch (if not already)**
   ```bash
   pip install --upgrade hatch
   ```

3. **Create and activate the dev environment**
   ```bash
   hatch shell
   ```

4. **Run ASR and TTS in development mode**
   ```bash
   hatch run asr:dev
   hatch run tts:dev
   ```

   By default, these commands:
   - Start ASR on port **5001**
   - Start TTS on port **5002**
   - Enable hot reload under `/src/asr` and `/src/tts`

5. **Format, lint, and type-check code**
   ```bash
   hatch fmt
   hatch lint
   hatch typecheck
   ```

6. **Run tests**
   ```bash
   hatch test
   ```

7. **Deactivate environment**
   ```bash
   exit
   ```

---

## ⚙️ Environment Variables

| Variable | Description | Default |
|-----------|--------------|----------|
| `ASR_DEVICE` | `cuda` or `cpu` | `cpu` |
| `ASR_PORT` | Port for ASR service | `5001` |
| `TTS_DEVICE` | `cuda` or `cpu` | `cpu` |
| `TTS_PORT` | Port for TTS service | `5002` |
| `TTS_MODEL` | Coqui model name | `tts_models/multilingual/multi-dataset/xtts_v2` |
| `TTS_AUTO_LANG` | Auto language detection | `1` |
| `TTS_MAX_CHARS` | Max chars per chunk | `180` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

---

## 🧪 Build & Packaging

### Build the project
```bash
hatch build
```
This creates a wheel and source distribution under `dist/`.

### Run tests in isolation
```bash
nox -s tests
```

### Format and lint with Nox
```bash
nox -s format lint
```

---

## 📁 Project Layout

```
voice-stack/
├── pyproject.toml
├── noxfile.py
├── .vscode/
│   ├── settings.json
│   └── launch.json
├── src/
│   ├── asr/
│   │   ├── __init__.py
│   │   └── app.py
│   ├── tts/
│   │   ├── __init__.py
│   │   └── engine_xtts.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   ├── test_asr.py
│   ├── test_tts.py
│   └── test_utils.py
└── README.md
```

---

## 🧩 Common Commands

| Task | Command |
|------|----------|
| Create environment | `hatch shell` |
| Start ASR (dev) | `hatch run asr:dev` |
| Start TTS (dev) | `hatch run tts:dev` |
| Run tests | `hatch test` |
| Lint | `hatch lint` |
| Format | `hatch fmt` |
| Type check | `hatch typecheck` |
| Build package | `hatch build` |

---

## 🧠 Notes

- The repo is structured for **cross-platform reproducibility**: develop on macOS (CPU) → deploy on Linux (GPU).
- Voice samples go in `/voices` and can follow naming like `Bethania_pt.wav`, `Bethania_en.wav`.
- Both services are FastAPI-based and can be containerized later for production.

---

## 📜 License

MIT License © 2025 Vitor Bordini Garbim
