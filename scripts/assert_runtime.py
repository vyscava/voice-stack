#!/usr/bin/env python3
"""Assert the invariants an installed voice-stack environment must satisfy.

Run by BOTH Dockerfiles (production and CI) so the two cannot drift, and usable
standalone after a local install:

    .venv/bin/python scripts/assert_runtime.py

Why this file exists
--------------------
The unit suites mock ``torch``. A torch that cannot be imported at all therefore
passes every test, and on 2026-08-13 CI was fully green on an image that
crash-looped for three days (voice-stack#4). These are the checks that would
have caught it, bound to the artifact rather than to a pipeline.

Each check is here because it has actually failed, or because the obvious
weaker version of it silently passes:

* **variant, not just version** - ``torch==2.9.1`` on PyPI is the CUDA wheel.
  A version pin is satisfied by the wrong build, so the build variant is
  asserted explicitly.
* **torchaudio moves with torch** - its extension is ABI-bound; a mismatched
  pair dies at import with "Could not load this library: _torchaudio.abi3.so".
* **torchcodec must LOAD, not merely exist** - transformers reports
  availability via ``importlib.util.find_spec``, which is metadata only. An
  unloadable torchcodec passes that check and passes ``import torchcodec``;
  only touching the decoder loads the shared object.
* **the TTS engine import** - ``from TTS.api import TTS`` walks
  coqui-TTS -> transformers -> torch._dynamo, which is where #4 exploded.

What it deliberately does NOT do: download weights or run inference. It is a
build-time check. Inference breaks -- ``silero_vad.read_audio()`` on
torchaudio >= 2.9 is the live example -- import cleanly and fail only when handed
real data, and are the smoke stage's job, not this file's.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

EXPECTED_VARIANT = "+cpu"

# Image variants (Dockerfile targets). "full" is the default because
# Dockerfile.ci runs this script without arguments and installs everything.
VARIANTS = ("full", "kokoro")
DEFAULT_KOKORO_MODEL = "/app/models/kokoro/kokoro-v1.0.onnx"
DEFAULT_KOKORO_VOICES = "/app/models/kokoro/voices-v1.0.bin"


def check_torch(failures: list[str]) -> bool:
    """torch + torchaudio import, and are the CPU build. Returns False if unimportable."""
    # Wrapped like everything else: an unimportable torch is the rarest failure
    # but also the one where a raw traceback is least useful, and it would skip
    # the summary that tells you which OTHER checks were never reached.
    try:
        import torch
        import torchaudio
    except Exception as exc:  # noqa: BLE001
        failures.append(f"torch/torchaudio could not be imported at all: {exc!r}")
        return False

    print(f"torch {torch.__version__} | torchaudio {torchaudio.__version__}")

    # Variant, not just version: PyPI serves torch==2.9.1 as the CUDA build.
    for name, version in (("torch", torch.__version__), ("torchaudio", torchaudio.__version__)):
        if not version.endswith(EXPECTED_VARIANT):
            failures.append(
                f"{name} is {version}, expected a {EXPECTED_VARIANT} build "
                "(install from https://download.pytorch.org/whl/cpu)"
            )
    return True


def check_asr(failures: list[str]) -> None:
    try:
        import faster_whisper  # noqa: F401
        import silero_vad  # noqa: F401

        print("faster_whisper + silero_vad import OK")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"ASR imports failed: {exc!r}")


def check_xtts(failures: list[str]) -> None:
    try:
        import transformers

        print(f"transformers {transformers.__version__}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"transformers could not be imported: {exc!r}")

    # torchcodec has to LOAD. find_spec-style checks pass on a broken install.
    try:
        import torchcodec
        from torchcodec.decoders import AudioDecoder  # noqa: F401

        print(f"torchcodec {torchcodec.__version__} loaded (AudioDecoder importable)")
    except Exception as exc:  # noqa: BLE001 - any failure here is fatal
        failures.append(f"torchcodec could not load its extension: {exc!r}")

    # The exact call that broke in #4.
    try:
        from TTS.api import TTS  # noqa: F401

        print("TTS.api import OK")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"TTS engine import failed: {exc!r}")


def check_kokoro(failures: list[str], *, require_model: bool) -> None:
    """kokoro-onnx imports, and (when the model is baked in) the session builds.

    Building the ONNX session and reading the voices is not inference: it is the
    step that fails on a wrong-architecture onnxruntime wheel or a truncated
    model file, both of which import cleanly.
    """
    try:
        import onnxruntime
        from kokoro_onnx import Kokoro

        print(f"onnxruntime {onnxruntime.__version__} | kokoro_onnx import OK")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"Kokoro engine imports failed: {exc!r}")
        return

    model = Path(os.environ.get("TTS_KOKORO_MODEL_PATH", DEFAULT_KOKORO_MODEL))
    voices = Path(os.environ.get("TTS_KOKORO_VOICES_PATH", DEFAULT_KOKORO_VOICES))
    if not (model.is_file() and voices.is_file()):
        if require_model:
            failures.append(f"Kokoro model files missing: {model} / {voices}")
        else:
            print("Kokoro model not present here; session check skipped")
        return
    try:
        engine = Kokoro(str(model), str(voices))
        n = len(engine.get_voices())
        print(f"Kokoro session built from {model.name}, {n} voices")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"Kokoro session could not be built: {exc!r}")


def check_absent(failures: list[str], modules: tuple[str, ...]) -> None:
    """The slim image must not quietly grow the heavy engine back."""
    failures.extend(
        f"{name} is installed but must not be in this image"
        for name in modules
        if importlib.util.find_spec(name) is not None
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    args = parser.parse_args(argv)

    failures: list[str] = []
    print(f"variant: {args.variant}")

    torch_ok = check_torch(failures)
    if torch_ok:
        check_asr(failures)
    if args.variant == "full":
        check_xtts(failures)
        check_kokoro(failures, require_model=False)
    else:
        check_kokoro(failures, require_model=True)
        check_absent(failures, ("TTS", "transformers"))

    if failures:
        print("\nRUNTIME ASSERTIONS FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("all runtime assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
