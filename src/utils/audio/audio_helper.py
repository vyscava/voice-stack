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
   - Silero v6 as the ONNX model bundled with faster-whisper, run on onnxruntime.
   - Apply VAD to PCM16 audio and stitch speech-only regions together.
   - No torch on this path: the slim image ships without it.
   Resampling uses resampy only (no torchaudio, which is end of line).
"""

from __future__ import annotations

import io
import logging
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
    """Cheap probe: is PyTorch importable? (Imported lazily to keep module import fast.)"""
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


def resample_to_16k_mono(*, audio_f32: npt.NDArray[np.float32], sr: int) -> tuple[npt.NDArray[np.float32], int]:
    """
    Return the audio as mono float32 at 16 kHz, the rate speech models expect.

    Stereo is averaged to mono. Resampling uses resampy (band-limited sinc,
    CPU). It used to try torchaudio first, but torchaudio is end of line (2.11
    is its last release and it no longer moves with torch), the slim image has
    no torch at all and already took this path in production, and a few
    seconds of speech resample in milliseconds either way.
    """
    if audio_f32.ndim == 2:
        audio_f32 = audio_f32.mean(axis=1)
    if sr == 16000:
        return audio_f32.astype("float32", copy=False), 16000

    import resampy  # local import: only ASR images ship it

    return resampy.resample(audio_f32, sr, 16000).astype("float32", copy=False), 16000


# Silero's own get_speech_timestamps() defaults (silero-vad 6.x). faster-whisper's
# VadOptions defaults are tuned for long-form chunking (2 s silence, 400 ms pad)
# and would change what this service trims, so the silero values are passed
# explicitly: same v6 model, same parameters, only the runtime changed.
SILERO_VAD_DEFAULTS: dict[str, Any] = {
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 100,
    "speech_pad_ms": 30,
}


def _onnx_speech_timestamps(
    audio: npt.NDArray[np.float32], model: Any, sampling_rate: int = 16000
) -> list[dict[Any, Any]]:
    """Run faster-whisper's ONNX Silero VAD; returns [{start, end}] in samples.

    ``model`` is accepted for the (model, fn) calling convention and is the same
    cached session faster-whisper uses internally (``get_vad_model`` is an
    lru_cache), so loading it up front is what makes a broken install fail at
    startup instead of on the first request.
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    del model  # faster-whisper resolves the cached session itself
    ts: list[dict[Any, Any]] = get_speech_timestamps(
        np.ascontiguousarray(audio, dtype=np.float32),
        vad_options=VadOptions(**SILERO_VAD_DEFAULTS),
        sampling_rate=sampling_rate,
    )
    return ts


def load_silero_model(log: logging.Logger) -> tuple[Any, Any]:
    """
    Load the Silero VAD model that ships inside faster-whisper (ONNX, CPU).

    Why not the ``silero-vad`` package
    ----------------------------------
    It is the same Silero v6 network, but packaged as TorchScript, so it pulled
    torch and torchaudio into every image that runs ASR. faster-whisper already
    bundles the ONNX export (``faster_whisper/assets/silero_vad_v6.onnx``) and
    runs it on onnxruntime, which it depends on anyway. Using it keeps torch out
    of the slim image entirely.

    Returns
    -------
    (model, get_speech_timestamps) | (None, None)
        ``model`` is the onnxruntime-backed Silero session; ``get_speech_timestamps``
        is called as ``fn(audio_f32_16k, model, sampling_rate=16000)`` and
        returns sample-index segments. On any failure (missing package, missing
        asset, onnxruntime unable to build the session) returns (None, None) and
        logs at ERROR, because transcribing without VAD is a silent quality and
        latency regression, not a neutral fallback.
    """
    try:
        from faster_whisper.vad import get_vad_model

        model = get_vad_model()
        # Prove the session runs, not just that it was constructed: ~0.25 s of
        # silence must come back as "no speech", not raise.
        _onnx_speech_timestamps(np.zeros(4000, dtype=np.float32), model, sampling_rate=16000)
        log.info("Silero VAD loaded (ONNX, CPU).")
        return model, _onnx_speech_timestamps
    except Exception as e:
        log.error("Silero VAD unavailable, proceeding WITHOUT VAD: %s", e)
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
) -> tuple[npt.NDArray[np.int16], list[dict[str, float]]]:
    """
    Apply Silero VAD to PCM16 mono audio and return speech-only PCM16 with timestamps.

    Contract
    --------
    - Input: PCM16 mono, sample rate `sr` (any). If `sr != 16000`, we resample to 16 kHz (resampy).
    - Output: Tuple of (PCM16 mono with non-speech removed, list of speech segment timestamps)

    Pipeline (NumPy + onnxruntime, no torch):
        1) Convert int16 -> float32 [-1, 1].
        2) Resample to 16 kHz if needed.
        3) Run VAD to get speech timestamps (sample indices at 16 kHz).
        4) Stitch speech regions with NumPy.
        5) Convert float32 -> int16 for downstream pipelines that expect PCM16.

    Fail-safe behavior
    ------------------
    - If the model/timestamps function is None, or any error occurs, we return the original `pcm16`
      with empty timestamps list.

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
    tuple[np.ndarray, list[dict]]
        - Speech-only PCM16 (int16, (M,)). If VAD unavailable/failed, returns input unchanged.
        - List of speech segments with timestamps in seconds: [{"start": float, "end": float}, ...]
          These timestamps are relative to the original audio (before VAD processing).
          Empty list if VAD was not applied or failed.
    """
    if silero_model is None or get_speech_timestamps_fn is None:
        return pcm16, []

    try:
        audio_f32 = pcm16.astype(np.float32).reshape(-1) / 32768.0
        if sr != 16000:
            import resampy

            audio_f32 = resampy.resample(audio_f32, sr, 16000).astype(np.float32, copy=False)

        ts = get_speech_timestamps_fn(audio_f32, silero_model, sampling_rate=16000)

        timestamps_sec = [{"start": float(t["start"]) / 16000.0, "end": float(t["end"]) / 16000.0} for t in ts]
        speech_np_f32 = _concat_chunks_numpy(audio_f32, ts)
        speech_i16 = (np.clip(speech_np_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
        return speech_i16, timestamps_sec

    except Exception as e:
        # Any failure should not break the ASR pipeline; just return original audio with no timestamps.
        log.warning("[VAD:silero] error, returning original audio: %s", e)
        return pcm16, []


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
