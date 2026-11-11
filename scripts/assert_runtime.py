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

* **variant, not just version** - ``torch==2.15.0`` on PyPI is the CUDA wheel.
  A version pin is satisfied by the wrong build, so the build variant is
  asserted explicitly.
* **a final-release interpreter** - #4's root cause (found in #13): the full
  image ran on Ubuntu 22.04's python3.11, which is 3.11.0~rc1 and lacks
  ``sys.get_int_max_str_digits``. Newer torch registers a polyfill for it
  behind ``sys.version_info >= (3, 11)``, so ``import torch._dynamo`` died
  half way. Any non-final interpreter now fails the build.
* **torch._dynamo imported directly, first** - #4's "Duplicate dispatch rule"
  is what a SECOND ``import torch._dynamo`` raises after a first attempt died
  partway and was swallowed by a library. Importing it here, unwrapped and
  before transformers touches it, turns a repeat into the REAL exception
  (that is exactly how #13 found the rc1 interpreter).
* **torchaudio's ops run, not just import** - torchaudio 2.11 is its last
  release and is paired with a newer torch (see pyproject.toml). Its extension
  is ABI-bound, so the two calls coqui-tts makes (MelSpectrogram, resample)
  are executed against this torch.
* **torchcodec must LOAD, not merely exist** - transformers reports
  availability via ``importlib.util.find_spec``, which is metadata only. An
  unloadable torchcodec passes that check and passes ``import torchcodec``;
  only touching the decoder loads the shared object.
* **the TTS engine import** - ``from TTS.api import TTS`` walks
  coqui-TTS -> transformers -> torch._dynamo, which is where #4 exploded.
* **the VAD session runs** - VAD is the Silero ONNX model faster-whisper
  bundles. The service treats a VAD that fails to load as "transcribe without
  VAD" and keeps going, so a broken onnxruntime or missing asset would be
  silent at runtime. Here it is fatal.
* **the slim image has no torch** - since #13 the ``kokoro`` variant carries no
  torch, torchaudio, openai-whisper, coqui or transformers. Any of them coming
  back (a new transitive dependency) fails the build instead of silently
  re-importing the advisories they carry.

What it deliberately does NOT do: download weights or run model inference. It
is a build-time check; inference is the smoke stage's job.
"""

from __future__ import annotations

import argparse
import importlib
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

# What the slim image must never contain (import names).
KOKORO_FORBIDDEN = ("torch", "torchaudio", "torchcodec", "whisper", "TTS", "transformers")


def check_interpreter(failures: list[str]) -> None:
    """The interpreter must be a final release, not an alpha/beta/rc."""
    v = sys.version_info
    print(f"python {sys.version.split()[0]} ({v.releaselevel})")
    if v.releaselevel != "final":
        failures.append(
            f"python {sys.version.split()[0]} is a {v.releaselevel} build; torch._dynamo assumes a final "
            "release (Ubuntu 22.04's python3.11 is 3.11.0~rc1 and broke it in #4)"
        )


def check_torch(failures: list[str]) -> bool:
    """torch + torchaudio import, are the CPU build, and torchaudio's ops run.

    Returns False if torch is unimportable.
    """
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

    # Variant, not just version: PyPI serves torch==2.15.0 as the CUDA build.
    for name, version in (("torch", torch.__version__), ("torchaudio", torchaudio.__version__)):
        if not version.endswith(EXPECTED_VARIANT):
            failures.append(
                f"{name} is {version}, expected a {EXPECTED_VARIANT} build "
                "(install from https://download.pytorch.org/whl/cpu)"
            )

    # Unwrapped by any library, and before transformers can import it: see the
    # module docstring for why the order matters.
    try:
        importlib.import_module("torch._dynamo")
        print("torch._dynamo import OK")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"import torch._dynamo failed (this is the #4 import path): {exc!r}")

    # The two torchaudio calls coqui-tts' XTTS makes, run against this torch.
    try:
        wave = torch.zeros(1, 22050)
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_fft=1024, n_mels=80)(wave)
        res = torchaudio.functional.resample(wave, 22050, 16000)
        if res.shape[-1] != 16000:
            failures.append(f"torchaudio resample returned {tuple(res.shape)}, expected 16000 samples")
        print(f"torchaudio ops OK (mel {tuple(mel.shape)}, resample {tuple(res.shape)})")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"torchaudio ops failed against torch {torch.__version__}: {exc!r}")
    return True


def check_asr(failures: list[str]) -> None:
    """faster-whisper imports and its bundled Silero ONNX VAD session runs."""
    try:
        import logging
        import pathlib

        import numpy as np

        # Go through the service's own loader, not faster_whisper.vad directly:
        # load_silero_model swallows every failure into (None, None) so the
        # service keeps transcribing WITHOUT VAD. Only this check turns that
        # silent degradation (for example a renamed VadOptions field in a
        # future faster-whisper) into a failed build.
        src = pathlib.Path(__file__).resolve().parent.parent / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from utils.audio.audio_helper import load_silero_model

        model, speech_timestamps = load_silero_model(logging.getLogger("assert_runtime"))
        if model is None or speech_timestamps is None:
            failures.append("load_silero_model returned (None, None): the service would run WITHOUT VAD")
            return
        ts = speech_timestamps(np.zeros(16000, dtype=np.float32), model, sampling_rate=16000)
        if ts:
            failures.append(f"Silero VAD reported speech in one second of silence: {ts!r}")
        print("faster_whisper import OK | service VAD loader returns a working Silero ONNX session")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"ASR / VAD check failed: {exc!r}")


def check_whisper_torch(failures: list[str]) -> None:
    """The PyTorch Whisper engine (ASR_ENGINE=whisper) imports. Full image only."""
    try:
        import whisper  # noqa: F401

        print("openai-whisper import OK")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"openai-whisper import failed: {exc!r}")


def check_xtts(failures: list[str]) -> None:
    try:
        import transformers

        print(f"transformers {transformers.__version__}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"transformers could not be imported: {exc!r}")

    # torchcodec has to LOAD and DECODE. find_spec-style checks pass on a broken
    # install, and PyPI's x86_64 wheel is a CUDA build whose extension cannot
    # load on a CPU host (libnvrtc.so.13), so the variant is asserted too.
    try:
        import io

        import numpy as np
        import soundfile as sf
        import torchcodec
        from torchcodec.decoders import AudioDecoder

        if not str(torchcodec.__version__).endswith(EXPECTED_VARIANT):
            failures.append(
                f"torchcodec is {torchcodec.__version__}, expected a {EXPECTED_VARIANT} build "
                "(install from https://download.pytorch.org/whl/cpu)"
            )
        buf = io.BytesIO()
        sf.write(buf, (0.1 * np.sin(np.arange(2205) / 5.0)).astype("float32"), 22050, format="WAV")
        samples = AudioDecoder(buf.getvalue()).get_all_samples()
        if samples.sample_rate != 22050 or samples.data.shape[-1] != 2205:
            failures.append(
                f"torchcodec decoded {tuple(samples.data.shape)} @ {samples.sample_rate} Hz, expected 2205 @ 22050"
            )
        print(f"torchcodec {torchcodec.__version__} loaded and decoded a WAV ({tuple(samples.data.shape)})")
    except Exception as exc:  # noqa: BLE001 - any failure here is fatal
        failures.append(f"torchcodec could not load its extension or decode: {exc!r}")

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
    """The slim image must not quietly grow the heavy stack back."""
    failures.extend(
        f"{name} is installed but must not be in this image"
        for name in modules
        if importlib.util.find_spec(name) is not None
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    args = parser.parse_args(argv)

    failures: list[str] = []
    print(f"variant: {args.variant}")

    check_interpreter(failures)
    check_asr(failures)
    if args.variant == "full":
        if check_torch(failures):
            check_whisper_torch(failures)
            check_xtts(failures)
        check_kokoro(failures, require_model=False)
    else:
        check_kokoro(failures, require_model=True)
        check_absent(failures, KOKORO_FORBIDDEN)

    if failures:
        print("\nRUNTIME ASSERTIONS FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("all runtime assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
