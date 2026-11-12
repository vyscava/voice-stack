#!/usr/bin/env python3
"""Assert the invariants an installed voice-stack environment must satisfy.

Run by every Dockerfile (the CPU images, the CI image and the GPU images in
docker/) so they cannot drift, and usable standalone after a local install:

    .venv/bin/python scripts/assert_runtime.py
    .venv/bin/python scripts/assert_runtime.py --variant cuda   # or rocm, kokoro

Why this file exists
--------------------
The unit suites mock ``torch``. A torch that cannot be imported at all therefore
passes every test, and on 2026-08-13 CI was fully green on an image that
crash-looped for three days (the torch 2.13 outage). These are the checks that would
have caught it, bound to the artifact rather than to a pipeline.

Each check is here because it has actually failed, or because the obvious
weaker version of it silently passes:

* **variant, not just version** - ``torch==2.15.0`` on PyPI is the CUDA wheel.
  A version pin is satisfied by the wrong build, so the build variant is
  asserted explicitly.
* **a final-release interpreter** - the root cause of the torch 2.13 outage: the full
  image ran on Ubuntu 22.04's python3.11, which is 3.11.0~rc1 and lacks
  ``sys.get_int_max_str_digits``. Newer torch registers a polyfill for it
  behind ``sys.version_info >= (3, 11)``, so ``import torch._dynamo`` died
  half way. Any non-final interpreter now fails the build.
* **torch._dynamo imported directly, first** - the outage's "Duplicate dispatch rule"
  is what a SECOND ``import torch._dynamo`` raises after a first attempt died
  partway and was swallowed by a library. Importing it here, unwrapped and
  before transformers touches it, turns a repeat into the REAL exception
  (that is exactly how the rc1 interpreter was found).
* **torchaudio's ops run, not just import** - torchaudio 2.11 is its last
  release and is paired with a newer torch (see pyproject.toml). Its extension
  is ABI-bound, so the two calls coqui-tts makes (MelSpectrogram, resample)
  are executed against this torch.
* **torchcodec must LOAD, not merely exist** - transformers reports
  availability via ``importlib.util.find_spec``, which is metadata only. An
  unloadable torchcodec passes that check and passes ``import torchcodec``;
  only touching the decoder loads the shared object.
* **the TTS engine import** - ``from TTS.api import TTS`` walks
  coqui-TTS -> transformers -> torch._dynamo, which is where the torch 2.13 outage exploded.
* **the VAD session runs** - VAD is the Silero ONNX model faster-whisper
  bundles. The service treats a VAD that fails to load as "transcribe without
  VAD" and keeps going, so a broken onnxruntime or missing asset would be
  silent at runtime. Here it is fatal.
* **the slim image has no torch** - the ``kokoro`` variant carries no
  torch, torchaudio, openai-whisper, coqui or transformers. Any of them coming
  back (a new transitive dependency) fails the build instead of silently
  re-importing the advisories they carry.

* **GPU images: the GPU build, proven WITHOUT a GPU** - the ``cuda`` and
  ``rocm`` variants run on a CI runner that has no GPU, so they check
  everything that can be checked there and nothing that needs a device:
  torch is the CUDA/HIP build (``torch.version.cuda`` / ``torch.version.hip``),
  CTranslate2 is the build linked against cuBLAS / HIP and the libraries it
  loads resolve, onnxruntime is the GPU package with its CUDA provider whose
  shared object actually loads (CUDA image), and the rest of the stack still
  imports and runs on CPU. ``torch.cuda.is_available()`` is False here and is
  deliberately not asserted.

What it deliberately does NOT do: download weights, run model inference, or
touch a GPU. It is a build-time check; inference is the smoke stage's job, and
nothing in CI has a GPU.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from pathlib import Path

# Image variants. "full" is the default because Dockerfile.ci runs this script
# without arguments and installs everything. "kokoro" and "full" are targets of
# the root Dockerfile; "cuda" and "rocm" are docker/Dockerfile.cuda / .rocm.
VARIANTS = ("full", "kokoro", "cuda", "rocm")

# Which build of each torch package a variant must carry, as the prefix of the
# version's local part ("2.15.0+cu132" -> "cu132"), and the index that has it.
# ROCm torchcodec is "cpu": PyTorch publishes no ROCm torchcodec, and it only
# decodes audio on the CPU anyway. CUDA torchaudio is "cpu" too: torchaudio
# 2.11 (its last release) stops at cu130, and on import it REFUSES to run
# against a torch built for another CUDA version ("PyTorch has CUDA version
# 13.2 whereas TorchAudio has CUDA version 13.0", measured in the first
# CI pipeline of the CUDA build). Its CPU build skips that check, and the two calls coqui-tts makes
# (MelSpectrogram, resample) are plain torch ops that run on GPU tensors.
TORCH_BUILDS: dict[str, dict[str, str]] = {
    "full": {"torch": "cpu", "torchaudio": "cpu", "torchcodec": "cpu"},
    "cuda": {"torch": "cu", "torchaudio": "cpu", "torchcodec": "cu"},
    "rocm": {"torch": "rocm", "torchaudio": "rocm", "torchcodec": "cpu"},
}
INDEX_HINT = {"cpu": "https://download.pytorch.org/whl/cpu", "cu": "a CUDA index", "rocm": "a ROCm index"}

# The library each GPU build of CTranslate2 links (the PyPI x86_64 wheel
# dlopen()s cuBLAS 12; the ROCm wheel from the CTranslate2 GitHub release links
# HIP). Its presence in the bundled libctranslate2 is how "GPU support is
# compiled in" is told apart from a CPU-only wheel without a GPU to ask.
CT2_GPU_MARKER = {"cuda": b"libcublas.so.12", "rocm": b"libamdhip64.so"}
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
            "release (Ubuntu 22.04's python3.11 is 3.11.0~rc1 and broke it in the torch 2.13 outage)"
        )


def build_of(version: str) -> str:
    """The local part of a version: "2.15.0+cu132" -> "cu132", "2.15.0" -> ""."""
    return version.partition("+")[2]


def check_build(failures: list[str], name: str, version: str, want: str) -> None:
    """``name`` must be the ``want`` build (a local-version prefix), not merely the right version."""
    if not build_of(version).startswith(want):
        failures.append(f"{name} is {version}, expected a +{want} build (install from {INDEX_HINT[want]})")


def file_contains(path: Path, needle: bytes, chunk: int = 1 << 24) -> bool:
    """Whether ``needle`` occurs in the file, read in chunks (libctranslate2 is ~440 MB)."""
    overlap = len(needle) - 1
    tail = b""
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            if needle in tail + block:
                return True
            tail = block[-overlap:] if overlap else b""
    return False


def check_torch(failures: list[str], variant: str = "full") -> bool:
    """torch + torchaudio import, are this variant's build, and torchaudio's ops run.

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
    builds = TORCH_BUILDS[variant]
    check_build(failures, "torch", torch.__version__, builds["torch"])
    check_build(failures, "torchaudio", torchaudio.__version__, builds["torchaudio"])

    # The GPU toolkit torch was compiled for. Readable with no GPU present.
    cuda, hip = torch.version.cuda, getattr(torch.version, "hip", None)
    print(f"torch.version.cuda={cuda} torch.version.hip={hip}")
    if variant == "cuda":
        if not cuda or hip:
            failures.append(f"torch is not a CUDA build (version.cuda={cuda!r}, version.hip={hip!r})")
        else:
            # Loads libcudnn from the nvidia-cudnn wheel: a missing or
            # mismatched cuDNN fails here, not on the first GPU request.
            try:
                cudnn = torch.backends.cudnn.version()
                print(f"cuDNN {cudnn} loads")
                if not cudnn:
                    failures.append("torch.backends.cudnn.version() is empty: cuDNN did not load")
            except Exception as exc:  # noqa: BLE001
                failures.append(f"cuDNN could not be loaded by torch: {exc!r}")
    elif variant == "rocm":
        if not hip:
            failures.append(f"torch is not a ROCm/HIP build (version.hip={hip!r}, version.cuda={cuda!r})")
    elif cuda or hip:
        failures.append(f"torch is a GPU build in a CPU image (version.cuda={cuda!r}, version.hip={hip!r})")

    # Unwrapped by any library, and before transformers can import it: see the
    # module docstring for why the order matters.
    try:
        importlib.import_module("torch._dynamo")
        print("torch._dynamo import OK")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"import torch._dynamo failed (the import path that crashed in the torch 2.13 outage): {exc!r}")

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


def check_xtts(failures: list[str], variant: str = "full") -> None:
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

        check_build(failures, "torchcodec", str(torchcodec.__version__), TORCH_BUILDS[variant]["torchcodec"])
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

    # The exact call that broke in the torch 2.13 outage.
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


def check_ctranslate2_gpu(failures: list[str], variant: str) -> None:
    """CTranslate2 (faster-whisper's engine) is the GPU build and its GPU libraries resolve.

    No GPU here, so the device is never asked for. What is checked instead:
    the bundled libctranslate2 is the one linked against cuBLAS (CUDA) or HIP
    (ROCm), the device query runs and answers (0 devices) instead of crashing,
    and for CUDA the cuBLAS 12 library it dlopen()s at the first GPU call
    resolves on this image's library path. The ROCm build links HIP, hipBLAS
    and hipRAND directly, so `import ctranslate2` succeeding already proves
    those resolve.
    """
    try:
        import ctranslate2

        spec = importlib.util.find_spec("ctranslate2")
        pkg_dir = Path(spec.origin).parent if spec and spec.origin else None
        libs = sorted(pkg_dir.parent.glob("ctranslate2.libs/libctranslate2*.so*")) if pkg_dir else []
        if not libs:
            failures.append("CTranslate2: no bundled libctranslate2 found next to the package")
            return
        marker = CT2_GPU_MARKER[variant]
        if not any(file_contains(lib, marker) for lib in libs):
            failures.append(f"CTranslate2 {ctranslate2.__version__} is not the {variant} build ({marker!r} not linked)")
            return
        count = ctranslate2.get_cuda_device_count()
        print(f"ctranslate2 {ctranslate2.__version__}: {variant} build ({marker.decode()}), {count} GPU(s) visible")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"CTranslate2 GPU check failed: {exc!r}")
        return

    if variant == "cuda":
        import ctypes

        try:
            ctypes.CDLL("libcublas.so.12")
            print("libcublas.so.12 (for CTranslate2) resolves")
        except OSError as exc:
            failures.append(f"libcublas.so.12 does not resolve, ASR_DEVICE=cuda would fail: {exc}")


def check_onnxruntime_variant(failures: list[str], variant: str) -> None:
    """Which onnxruntime the image has, and (CUDA) that its CUDA provider loads.

    onnxruntime and onnxruntime-gpu install the same `onnxruntime` package, so
    both present means one overwrote the other's files: the image would run
    whichever was installed last. Exactly one distribution is required.
    """
    from importlib import metadata

    def installed(dist: str) -> bool:
        try:
            metadata.version(dist)
            return True
        except metadata.PackageNotFoundError:
            return False

    want = "onnxruntime-gpu" if variant == "cuda" else "onnxruntime"
    other = "onnxruntime" if variant == "cuda" else "onnxruntime-gpu"
    if not installed(want) or installed(other):
        failures.append(f"expected {want} and not {other} (have {want}={installed(want)}, {other}={installed(other)})")
    try:
        import onnxruntime as ort

        providers = list(ort.get_available_providers())
        print(f"onnxruntime providers: {providers}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"onnxruntime could not be imported: {exc!r}")
        return
    if variant != "cuda":
        return
    if "CUDAExecutionProvider" not in providers:
        failures.append(f"CUDAExecutionProvider missing from onnxruntime-gpu: {providers}")
        return
    # The provider list is static; the provider library is what needs cuBLAS
    # 13, cuDNN 9, cuFFT and cuRAND. It also links the DRIVER, libcuda.so.1,
    # which the NVIDIA container toolkit injects at run time and which no
    # image (and no GPU-less CI runner) has, so it cannot simply be loaded
    # here. Instead every library it links must resolve except that one.
    lib = Path(ort.__file__).parent / "capi" / "libonnxruntime_providers_cuda.so"
    missing = unresolved_libraries(failures, lib)
    if missing is None:
        return
    if missing - DRIVER_PROVIDED:
        failures.append(f"{lib.name} links libraries that do not resolve: {sorted(missing - DRIVER_PROVIDED)}")
    else:
        print(f"{lib.name}: every linked library resolves except the driver ({sorted(missing)})")


# Libraries the GPU driver supplies at run time (the NVIDIA container toolkit
# mounts them in); never present at build time.
DRIVER_PROVIDED = frozenset({"libcuda.so.1"})


def parse_ldd_missing(output: str) -> set[str]:
    """Sonames ``ldd`` reports as "not found"."""
    return {line.split("=>")[0].strip() for line in output.splitlines() if "not found" in line}


def unresolved_libraries(failures: list[str], lib: Path) -> set[str] | None:
    """The shared libraries ``lib`` links (directly or not) that do not resolve, per ldd.

    None (with a failure recorded) if ldd itself could not run on it.
    """
    import subprocess

    if not lib.is_file():
        failures.append(f"{lib} does not exist")
        return None
    try:
        out = subprocess.run(["ldd", str(lib)], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        failures.append(f"ldd {lib.name} failed: {exc!r}")
        return None
    return parse_ldd_missing(out)


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
    elif args.variant in ("cuda", "rocm"):
        if check_torch(failures, args.variant):
            check_whisper_torch(failures)
            check_xtts(failures, args.variant)
        check_ctranslate2_gpu(failures, args.variant)
        check_onnxruntime_variant(failures, args.variant)
        check_kokoro(failures, require_model=True)
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
