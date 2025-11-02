"""
audio_helper.py — Audio decoding, normalization, and VAD helpers.

What this module provides
-------------------------
1) Decoding
   - Fast path via libsndfile/soundfile for PCM-ish formats (WAV/FLAC/AIFF/OGG-Vorbis).
   - Robust fallback via FFmpeg for compressed/“tricky” formats (Opus/MP3/M4A…).

2) Normalization
   - Resample to 16 kHz, downmix to mono, ensure float32 in [-1, 1].
   - Matches common ASR model requirements (e.g., Whisper-family).

3) VAD (Silero)
   - Lazy load of Silero VAD (CPU by default).
   - Apply VAD to PCM16 audio and stitch speech-only regions together.
   - Keeps dependencies optional: torch/torchaudio are imported inside functions.
"""

from __future__ import annotations

import io
import logging
import sys
import types
from typing import Any

import numpy as np
import numpy.typing as npt
import soundfile as sf


def decode_with_soundfile(*, raw_bytes: bytes) -> tuple[npt.NDArray[np.float32], int]:
    """
    Decode audio bytes using libsndfile via the `soundfile` library.

    This is the *fast path* decoder for uncompressed or common formats
    (e.g., WAV, FLAC, AIFF, OGG-Vorbis). It uses libsndfile internally,
    which directly supports these codecs without spawning an external
    process like FFmpeg.

    Parameters
    ----------
    raw_bytes : bytes
        Raw audio file contents (e.g., result of `f.read()`).

    Returns
    -------
    tuple[np.ndarray, int]
        A tuple containing:
        - A 1D NumPy array of float32 samples (mono) in the range [-1.0, 1.0].
          Stereo files are automatically averaged across channels.
        - The detected sample rate (Hz).

    Notes
    -----
    - The audio data is returned as `float32` for model compatibility
      and consistency with FFmpeg decoding.
    - If the file has multiple channels (e.g., stereo), it is collapsed
      to mono by averaging channels. Most ASR models only support mono.
    - For unsupported formats (e.g., MP3, M4A, Opus), you should use
      `decode_with_ffmpeg` instead.
    """

    # Decode bytes directly from memory (no temp file needed).
    # dtype="float32" ensures normalized samples in [-1, 1].
    audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)

    # If audio has more than one channel (e.g., stereo -> shape (N, 2)),
    # average channels to produce a single mono waveform.
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    return audio, sr


def _torch_available() -> bool:
    """
    Cheap probe to check if PyTorch is importable.

    Notes
    -----
    - We import inside the function to keep module import time minimal.
    - Returning False means: use CPU-only fallbacks (e.g., resampy).
    """
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def pick_torch_device(prefer: tuple[str, ...] = ("mps", "cuda", "cpu")) -> str:
    """
    Decide which torch device string to use: "mps" (Apple GPU), "cuda" (NVIDIA), or "cpu".

    Strategy
    --------
    1) Try Apple Metal (MPS) on macOS if available.
    2) Try CUDA if available.
    3) Fall back to CPU.

    Parameters
    ----------
    prefer : tuple[str, ...]
        Ordered device preferences. You can override to ("cuda", "mps", "cpu"), etc.

    Returns
    -------
    str
        One of {"mps", "cuda", "cpu"} representing the *best* available device.

    Behavior if torch missing
    -------------------------
    - If PyTorch is not installed/importable, returns "cpu".
    """
    if not _torch_available():
        return "cpu"

    import torch  # local import, lazy

    for dev in prefer:
        if dev == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            is_available = getattr(mps_backend, "is_available", None)
            if callable(is_available) and is_available():
                return "mps"
        elif dev == "cuda":
            if torch.cuda.is_available():
                return "cuda"
        elif dev == "cpu":
            return "cpu"
    return "cpu"


def _to_torch_device(x: Any, device: str) -> Any:
    """
    Move a torch object (Tensor/Module/Transform) to the requested device.

    - For "cpu", this is a no-op except ensuring .to("cpu") existence.
    - For "mps"/"cuda", we call `.to(device)`.

    This is a tiny wrapper to keep call-sites clean and avoid repeating guards.
    """
    try:
        return x.to(device)
    except AttributeError:
        return x  # e.g., if x is a plain Python object


def _device_to_str(dev: Any) -> str:
    if isinstance(dev, str):
        return dev.split(":", 1)[0]
    dev_type = getattr(dev, "type", None)
    if isinstance(dev_type, str):
        return dev_type
    return "cpu"


def _resample_torch_audio(
    *,
    audio_f32: npt.NDArray[np.float32],
    sr: int,
    target_sr: int = 16000,
    prefer: tuple[str, ...] = ("mps", "cuda", "cpu"),
) -> tuple[npt.NDArray[np.float32], int]:
    """
    Resample using TorchAudio's highly optimized C++ resampler.

    Device behavior
    ---------------
    - Automatically selects "mps" (Apple GPU), "cuda" (NVIDIA GPU), or "cpu".
    - On GPU/MPS, tensors and the Resample transform are moved to that device.
    - If torch/torchaudio are unavailable, the caller should fall back.

    Parameters
    ----------
    audio_f32 : np.ndarray (float32)
        Mono waveform in [-1, 1].
    sr : int
        Current sample rate.
    target_sr : int
        Desired sample rate (default 16 kHz).
    prefer : tuple[str, ...]
        Ordered device preferences.

    Returns
    -------
    (np.ndarray, int)
        Resampled mono waveform (float32) and the new sample rate.
    """
    import torch
    import torchaudio

    device = pick_torch_device(prefer)
    x = torch.from_numpy(np.ascontiguousarray(audio_f32)).to(device=device, dtype=torch.float32)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        resampler = _to_torch_device(resampler, device)
        x = resampler(x)
        sr = target_sr

    return x.to("cpu").numpy().astype("float32", copy=False), sr


def resample_to_16k_mono(
    *,
    audio_f32: npt.NDArray[np.float32],
    sr: int,
    prefer: tuple[str, ...] = ("mps", "cuda", "cpu"),
) -> tuple[npt.NDArray[np.float32], int]:
    """
    Ensure an audio array is mono float32 @ 16 kHz.

    This function standardizes audio for speech processing or model input.
    It collapses stereo to mono (if needed) and resamples to 16 kHz using:

      1) TorchAudio (with device selection: MPS/CUDA/CPU) — primary path
      2) resampy (CPU, Python-level)                      — fallback (high quality but slower)

    Parameters
    ----------
    audio_f32 : np.ndarray
        The input audio samples (float32). Can be mono or stereo.
    sr : int
        The current sample rate (Hz).
    prefer : tuple[str, ...]
        Ordered device preferences for torch (default: ("mps", "cuda", "cpu")).

    Returns
    -------
    tuple[np.ndarray, int]
        - A 1D NumPy array (mono) in float32, resampled to 16 kHz.
        - The new sample rate (always 16000).

    Why 16 kHz?
    -----------
    - 16 kHz is the standard sample rate for speech recognition models
      such as Whisper, Wav2Vec2, and DeepSpeech.
    - It provides sufficient frequency resolution for human speech while
      keeping model input sizes and compute costs manageable.

    Notes on performance
    --------------------
    - TorchAudio path typically runs in ~10–30 ms for a few seconds of audio on CPU,
      and can be faster on MPS/CUDA.
    - resampy provides excellent quality but is slower (Python-heavy).
    """

    # Collapse stereo to mono if needed by averaging channels.
    if audio_f32.ndim == 2:
        audio_f32 = audio_f32.mean(axis=1)

    # Already 16 kHz? Just standardize dtype and return.
    if sr == 16000:
        return audio_f32.astype("float32", copy=False), 16000

    # Try TorchAudio first (GPU/MPS/CPU)
    if _torch_available():
        try:
            y, new_sr = _resample_torch_audio(audio_f32=audio_f32, sr=sr, target_sr=16000, prefer=prefer)
            return y, new_sr
        except Exception:
            pass  # fall through to resampy

    # Fallback: resampy (CPU)
    try:
        import resampy  # local import to keep it optional

        y = resampy.resample(audio_f32, sr, 16000).astype("float32", copy=False)
        return y, 16000
    except Exception as e:
        raise RuntimeError("No resampler available (torch/torchaudio or resampy)") from e


def load_silero_model(log: logging.Logger) -> tuple[Any, Any]:
    """
    Lazily import and instantiate Silero VAD, preferring an accelerated torch device.

    Why lazy?
    ---------
    - Avoids importing PyTorch/torchaudio at process start when VAD isn't used.
    - Tries MPS (Apple), then CUDA (NVIDIA), then CPU. Falls back gracefully to CPU.

    Returns
    -------
    (model, get_speech_timestamps) | (None, None)
        `model` is the Silero VAD module, already `.eval()`-ed and moved to a device
        that appears to support execution. If an accelerated device fails a quick
        capability check, the model is transparently moved to CPU.
        `get_speech_timestamps` is the callable to run VAD.
        If anything fails (missing deps, incompatible ops), returns (None, None).

    Logging
    -------
    Emits messages indicating which device is used. If acceleration is unavailable
    or unsupported, logs a warning and uses CPU.
    """

    try:
        # Prevent Silero from breaking into ipdb
        def _noop(*_args: Any, **_kwargs: Any) -> None:
            return None

        fake_ipdb = types.ModuleType("ipdb")
        fake_ipdb.set_trace = _noop
        sys.modules["ipdb"] = fake_ipdb
    except Exception as e:
        print(f"Warning: failed to neutralize ipdb: {e}")

    try:
        import torch
        import torchaudio  # noqa: F401 (we may need it for resampling later)
        from silero_vad import get_speech_timestamps, load_silero_vad

        # 1) Load model (official API loads CPU by default), eval mode
        model = load_silero_vad()
        eval_fn = getattr(model, "eval", None)
        if callable(eval_fn):
            try:
                eval_fn()
            except Exception:
                pass

        # 2) Try to move to an accelerated device (MPS -> CUDA -> CPU).
        #    If anything fails, we revert to CPU.
        device = pick_torch_device(("mps", "cuda", "cpu"))
        chosen = "cpu"
        if device != "cpu":
            try:
                model = _to_torch_device(model, device)
                # Quick smoke test: run a tiny forward to ensure ops exist on the device.
                # We use ~0.25s of zeros at 16 kHz.
                test = torch.zeros(4000, dtype=torch.float32, device=device)
                # Some silero wrappers accept waveform + sr directly; others need only waveform at 16k
                speech_fn: Any = get_speech_timestamps
                with torch.no_grad():
                    _ = speech_fn(test, model, sampling_rate=16000)
                chosen = device
            except Exception as e:
                log.warning("Silero VAD acceleration (%s) unavailable, falling back to CPU: %s", device, e)
                model = _to_torch_device(model, "cpu")
                chosen = "cpu"
        else:
            model = _to_torch_device(model, "cpu")

        log.info("Silero VAD loaded (%s).", chosen.upper())
        return model, get_speech_timestamps

    except Exception as e:
        log.warning("Silero VAD unavailable, proceeding without VAD: %s", e)
        return None, None


def _concat_chunks_numpy(wav_np: npt.NDArray[np.float32], timestamps: list[dict[Any, Any]]) -> npt.NDArray[np.float32]:
    """
    Concatenate detected speech segments into one contiguous waveform (float32).

    Why this exists
    ---------------
    - Silero returns a list of {start, end} indices over the input tensor.
    - We stitch those regions together using NumPy for speed and simplicity.

    Parameters
    ----------
    wav_np : np.ndarray (float32, (N,))
        The source mono waveform as a 1-D array in [-1, 1].
    timestamps : list[dict]
        Output from `get_speech_timestamps` containing 'start'/'end' (Python ints).

    Returns
    -------
    np.ndarray (float32, (M,))
        Concatenated speech-only waveform. If no timestamps or invalid inputs,
        returns the original `wav_np` (fail-safe).
    """
    if not isinstance(wav_np, np.ndarray) or wav_np.ndim != 1 or wav_np.size == 0:
        return wav_np
    if not timestamps:
        return wav_np

    n = wav_np.shape[0]
    pieces: list[npt.NDArray[np.float32]] = []
    for t in timestamps:
        # Clamp to valid bounds
        s = max(0, min(n, int(t.get("start", 0))))
        e = max(0, min(n, int(t.get("end", 0))))
        if e > s:
            pieces.append(wav_np[s:e])

    if not pieces:
        return wav_np
    return np.concatenate(pieces, axis=0)


def apply_vad_silero(
    *,
    pcm16: npt.NDArray[np.int16],
    sr: int,
    silero_model: Any,
    get_speech_timestamps_fn: Any,
    log: logging.Logger,
) -> npt.NDArray[np.int16]:
    """
    Apply Silero VAD to PCM16 mono audio and return speech-only PCM16.

    Contract
    --------
    - Input: PCM16 mono, sample rate `sr` (any). If `sr != 16000`, we resample to 16 kHz.
    - Output: PCM16 mono with non-speech removed (concatenated back-to-back).

    Device selection and fallbacks
    ------------------------------
    - If the Silero model appears to live on MPS/CUDA, we attempt to execute on that device.
    - If device execution fails (unsupported op/dtype), we transparently fall back to CPU.
    - Resampling uses TorchAudio on the chosen device when available; falls back to resampy.

    Why convert types?
    ------------------
    - Silero expects a float32 tensor in [-1, 1] and works best at 16 kHz.
    - We therefore:
        1) Convert int16 → float32 [-1, 1].
        2) Resample to 16 kHz if needed (TorchAudio preferred; resampy fallback).
        3) Run VAD to get speech timestamps.
        4) Stitch speech regions with NumPy.
        5) Convert float32 → int16 for downstream pipelines that expect PCM16.

    Fail-safe behavior
    ------------------
    - If the model/timestamps function is None, or any error occurs, we return the original `pcm16`.

    Parameters
    ----------
    pcm16 : np.ndarray (int16, (N,))
        Mono PCM16 samples.
    sr : int
        Current sample rate.
    silero_model : Any
        Loaded model from `load_silero_model` (or None).
    get_speech_timestamps_fn : Callable
        Function returned by `load_silero_model` (or None).
    log : logging.Logger
        Logger for diagnostics.

    Returns
    -------
    np.ndarray (int16, (M,))
        Speech-only PCM16. If VAD unavailable/failed, returns input unchanged.
    """
    if silero_model is None or get_speech_timestamps_fn is None:
        return pcm16

    try:
        import torch

        # torch/torchaudio might not be present; we handle fallback later.
        try:
            import torchaudio as torchaudio_module
        except Exception:
            torchaudio_module = None

        # Helper: infer device from the model if possible (jit/script or nn.Module).
        def _infer_model_device(m: Any) -> str:
            # Try parameters first (nn.Module)
            try:
                params_attr = getattr(m, "parameters", None)
                if callable(params_attr):
                    for param in params_attr():
                        dev = getattr(param, "device", None)
                        if dev is not None:
                            return _device_to_str(dev)
                        break
            except Exception:
                pass

            # Try buffers (some modules have no params)
            try:
                buffers_attr = getattr(m, "buffers", None)
                if callable(buffers_attr):
                    bufs_iter = buffers_attr()
                    for buf in bufs_iter:
                        dev = getattr(buf, "device", None)
                        if dev is not None:
                            return _device_to_str(dev)
                        break
            except Exception:
                pass

            # Fallback: attribute on custom/torchscript objects
            dev = getattr(m, "device", None)
            if dev is not None:
                return _device_to_str(dev)

            return "cpu"

        # 1) int16 → float32 in [-1, 1], ensure contiguous 1-D array
        audio_f32 = pcm16.astype(np.float32) / 32768.0

        # Choose execution device (prefer model's device; fall back to CPU)
        exec_device = _infer_model_device(silero_model)
        if exec_device not in {"cpu", "cuda", "mps"}:
            # If it's a torch.device object or similar, stringify may have returned 'device(type=\'cpu\')'
            exec_device = "cpu"

        def _run_on_device(device: str) -> npt.NDArray[np.int16]:
            """
            Attempt the full pipeline on the requested device.
            Falls back to CPU at the caller if any op fails here.
            """
            x = torch.from_numpy(np.ascontiguousarray(audio_f32.reshape(-1))).to(device=device, dtype=torch.float32)
            _sr = sr

            # 2) Resample to 16 kHz on the same device when possible
            if _sr != 16000:
                if torchaudio_module is not None:
                    try:
                        resampler = torchaudio_module.transforms.Resample(orig_freq=_sr, new_freq=16000)
                        resampler = _to_torch_device(resampler, device)
                        x = resampler(x)
                        _sr = 16000
                    except Exception:
                        # Fallback to resampy on CPU
                        import resampy

                        y = resampy.resample(audio_f32, _sr, 16000).astype("float32", copy=False)
                        x = torch.from_numpy(y).to(device=device)
                        _sr = 16000
                else:
                    # No torchaudio, fallback to resampy (CPU) then move to device
                    import resampy

                    y = resampy.resample(audio_f32, _sr, 16000).astype("float32", copy=False)
                    x = torch.from_numpy(y).to(device=device)
                    _sr = 16000

            # 3) Run Silero to get speech timestamps
            with torch.no_grad():
                ts = get_speech_timestamps_fn(x, silero_model, sampling_rate=_sr)

            # 4) Stitch speech segments and convert back to PCM16
            speech_np_f32 = _concat_chunks_numpy(x.detach().to("cpu").numpy(), ts)
            speech_i16 = (np.clip(speech_np_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
            return speech_i16

        # Try on the model's device; if it fails (unsupported op), retry on CPU.
        try:
            return _run_on_device(exec_device)
        except Exception as dev_err:
            if exec_device != "cpu":
                log.warning("[VAD:silero] device '%s' failed (%s). Retrying on CPU.", exec_device, dev_err)
                # Move model to CPU if needed
                try:
                    silero_model = _to_torch_device(silero_model, "cpu")
                except Exception:
                    pass
                return _run_on_device("cpu")
            raise  # will be caught by outer except

    except Exception as e:
        # Any failure should not break the ASR pipeline; just return original audio.
        log.warning("[VAD:silero] error, returning original audio: %s", e)
        return pcm16


def wav_bytes_to_pcm16le_bytes(wav_bytes: bytes) -> bytes:
    """
    Convert a WAV container (any subtype readable by soundfile) into raw PCM16 LE bytes (no header).
    Keeps channel layout (interleaved if >1ch).
    """
    data, _ = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)

    # Normalize to float32 mono/stereo → int16 LE bytes
    if isinstance(data, np.ndarray):
        data = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    else:
        arr = np.array(data, dtype=np.float32, copy=False)
        data = (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    return data
