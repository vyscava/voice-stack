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
from typing import Any

import ffmpeg
import numpy as np
import resampy
import soundfile as sf


def decode_with_soundfile(*, raw_bytes: bytes) -> tuple[np.ndarray[Any, Any], int]:
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


def decode_with_ffmpeg(*, raw_bytes: bytes, fmt_hint: str | None = None) -> tuple[np.ndarray[Any, Any], int]:
    """
    Decode arbitrary audio bytes to a normalized float32 NumPy array using FFmpeg.

    This function takes any supported audio format (e.g., MP3, M4A, OGG, OPUS, WAV),
    decodes it through FFmpeg, downmixes to mono, resamples to 16 kHz, and returns
    the samples as normalized float32 values in the range [-1.0, 1.0].

    Parameters
    ----------
    raw_bytes : bytes
        The raw binary content of the input audio file or stream.
    fmt_hint : str | None
        Optional hint for FFmpeg about the input format (e.g., "mp3", "m4a", "ogg").
        This can help FFmpeg correctly parse the input when there’s no filename
        or extension available.

    Returns
    -------
    tuple[np.ndarray, int]
        A tuple containing:
        - A 1-D NumPy array of float32 samples in the range [-1.0, 1.0].
        - The sample rate (always 16 000 Hz).

    Why float32?
    ------------
    Neural ASR models (e.g., Whisper, Wav2Vec2) expect normalized floating-point
    audio data. Raw PCM16 integers store samples as signed 16-bit integers in the
    range [-32768, 32767]. Converting to float32 by dividing by 32768.0 yields a
    normalized waveform that these models are trained on. It also avoids clipping
    or overflow when performing DSP or model inference in float precision.

    Example
    -------
    >>> with open("speech.mp3", "rb") as f:
    ...     audio, sr = decode_with_ffmpeg(raw_bytes=f.read(), fmt_hint="mp3")
    >>> audio.shape, sr
    ((16000,), 16000)
    """

    # Build FFmpeg input graph reading from stdin ("pipe:0").
    # - "f" specifies an optional input format hint if provided.
    in_kwargs = {"f": fmt_hint} if fmt_hint else {}
    stream = ffmpeg.input("pipe:0", **in_kwargs)  # type: ignore[attr-defined]

    # Configure FFmpeg output to:
    #   - Write to stdout ("pipe:1") instead of a file.
    #   - Output raw PCM16 little-endian samples (format="s16le").
    #   - Force mono (ac=1).
    #   - Resample to 16 kHz (ar=16000).
    #   - Use the PCM16 codec (acodec="pcm_s16le").
    stream = ffmpeg.output(  # type: ignore[attr-defined]
        stream,
        "pipe:1",
        format="s16le",  # raw samples, no container header
        acodec="pcm_s16le",  # 16-bit signed little-endian PCM
        ac=1,  # 1 audio channel (mono)
        ar=16000,  # 16 000 samples per second
    )

    # Run FFmpeg pipeline:
    #   - Feed raw_bytes to stdin
    #   - Capture stdout (decoded PCM bytes)
    #   - Capture stderr (logs, ignored here)
    pcm_bytes, _ = ffmpeg.run(  # type: ignore[attr-defined]
        stream,
        capture_stdout=True,
        capture_stderr=True,
        input=raw_bytes,
    )  # type: ignore[attr-defined]

    # Interpret the returned PCM bytes as 16-bit signed integers.
    audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)

    # Convert to float32 in [-1.0, 1.0]:
    # 32768.0 is the absolute value of the negative PCM16 limit (-32768).
    # Dividing by 32768.0 maps:
    #   -32768 → -1.0
    #   +32767 → +0.99997
    audio_f32 = audio_i16.astype(np.float32) / 32768.0

    return audio_f32, 16000


def resample_to_16k_mono(*, audio_f32: np.ndarray[Any, Any], sr: int) -> tuple[np.ndarray[Any, Any], int]:
    """
    Ensure an audio array is mono float32 @ 16 kHz.

    This function standardizes audio for speech processing or model input.
    It collapses stereo to mono (if needed) and resamples to 16 kHz using
    high-quality bandlimited sinc interpolation via `resampy`.

    Parameters
    ----------
    audio_f32 : np.ndarray
        The input audio samples (float32). Can be mono or stereo.
    sr : int
        The current sample rate (Hz).

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
    """

    # Collapse stereo to mono if needed by averaging channels.
    if audio_f32.ndim == 2:
        audio_f32 = audio_f32.mean(axis=1)

    # Resample only if the input sample rate differs from 16 kHz.
    if sr != 16000:
        audio_f32 = resampy.resample(audio_f32, sr, 16000)
        sr = 16000

    # Ensure final dtype is float32 (in case resampy returns float64).
    return audio_f32.astype("float32", copy=False), sr


def load_silero_model(log: logging.Logger) -> tuple[Any, Any]:
    """
    Lazily import and instantiate Silero VAD on CPU.

    Why lazy?
    ---------
    - Avoids importing PyTorch/torchaudio at process start when VAD isn't used.
    - Keeps GPU free (Silero on CPU is plenty fast for trimming).

    Returns
    -------
    (model, get_speech_timestamps) | (None, None)
        `model` is the Silero VAD module, already `.eval()`-ed.
        `get_speech_timestamps` is the callable to run VAD.
        If anything fails (missing deps, incompatible platform), returns (None, None).

    Logging
    -------
    Emits a friendly message if VAD is unavailable; callers should gracefully skip VAD in that case.
    """
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        from silero_vad import get_speech_timestamps, load_silero_vad

        model = load_silero_vad()  # Official API (no device kwarg); uses CPU by default
        model.eval()  # type: ignore[attr-defined]
        log.info("Silero VAD loaded (CPU).")
        return model, get_speech_timestamps
    except Exception as e:
        log.warning("Silero VAD unavailable, proceeding without VAD: %s", e)
        return None, None


def _concat_chunks_numpy(wav_np: np.ndarray, timestamps: list[dict]) -> np.ndarray[Any, Any]:
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
    pieces: list[np.ndarray] = []
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
    pcm16: np.ndarray,
    sr: int,
    silero_model,
    get_speech_timestamps_fn,
    log: logging.Logger,
) -> np.ndarray:
    """
    Apply Silero VAD to PCM16 mono audio and return speech-only PCM16.

    Contract
    --------
    - Input: PCM16 mono, sample rate `sr` (any). If `sr != 16000`, we resample to 16 kHz.
    - Output: PCM16 mono with non-speech removed (concatenated back-to-back).

    Why convert types?
    ------------------
    - Silero expects a float32 tensor on CPU in [-1, 1] and works best at 16 kHz.
    - We therefore:
        1) Convert int16 → float32 [-1, 1].
        2) Resample to 16 kHz if needed (via torchaudio).
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
        import torchaudio

        # 1) int16 → float32 in [-1, 1], ensure contiguous 1-D array
        audio_f32 = pcm16.astype(np.float32) / 32768.0
        wav = torch.from_numpy(np.ascontiguousarray(audio_f32.reshape(-1))).float().cpu()

        # 2) Resample to 16 kHz for Silero if needed (CPU)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
            sr = 16000

        # 3) Run Silero to get speech timestamps
        with torch.no_grad():
            ts = get_speech_timestamps_fn(wav, silero_model, sampling_rate=sr)

        # 4) Stitch speech segments and convert back to PCM16
        speech_np_f32 = _concat_chunks_numpy(wav.numpy(), ts)
        speech_i16 = (np.clip(speech_np_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
        return speech_i16

    except Exception as e:
        # Any failure should not break the ASR pipeline; just return original audio.
        log.warning("[VAD:silero] error, returning original audio: %s", e)
        return pcm16
