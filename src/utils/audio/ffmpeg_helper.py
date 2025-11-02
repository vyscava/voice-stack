import shutil
import subprocess
import tempfile
from typing import Any, cast

import ffmpeg
import numpy as np
import numpy.typing as npt

# Known audio file extensions (lowercase)
AUDIO_EXTENSIONS: tuple[str, ...] = (
    ".mp3",
    ".wav",
    ".aac",
    ".flac",
    ".ogg",
    ".wma",
    ".alac",
    ".m4a",
    ".opus",
    ".aiff",
    ".aif",
    ".pcm",
    ".ra",
    ".ram",
    ".mid",
    ".midi",
    ".ape",
    ".wv",
    ".amr",
    ".vox",
    ".tak",
    ".spx",
    ".m4b",
    ".mka",
)

# Map extension → FFmpeg input *container* format
# (NEVER map to codec like "opus"—only containers like "ogg")
EXT_TO_INPUT_FORMAT: dict[str, str | None] = {
    # PCM/Uncompressed
    "wav": "wav",
    "aiff": "aiff",
    "aif": "aiff",
    "pcm": "s16le",  # raw PCM little-endian
    # Lossy formats
    "mp3": "mp3",
    "aac": "aac",  # raw AAC may fail — let probe if that happens
    "wma": "asf",
    "amr": "amr",
    "spx": "ogg",  # Speex in OGG container
    "ra": None,  # RealAudio (rare; FFmpeg can probe)
    "ram": None,
    # Lossless
    "flac": "flac",
    "alac": "mov",  # ALAC is usually in M4A/MOV container
    "ape": "ape",
    "wv": "wv",
    "tak": "tak",
    # Containers / Modern streaming formats
    "ogg": "ogg",
    "oga": "ogg",
    "opus": "ogg",  # Opus streams live inside OGG container
    "m4a": None,  # MP4 family: probing is safer
    "m4b": None,
    "mka": "matroska",
    "mp4": None,
    "mov": None,
    "webm": "webm",
    # Misc
    "vox": None,
    "mid": None,
    "midi": None,  # not audio per se — skip or handle elsewhere
}


def _pcm_from_wav_cli(wav_bytes: bytes, sr: int, ch: int) -> bytes:
    # Minimal CLI path for PCM if no ffmpeg-python available
    return _encode_with_cli(wav_bytes, "pcm", sr, ch)


def _suffix_for(tf: str) -> str:
    return {
        "wav": ".wav",
        "pcm": ".s16le",
        "mp3": ".mp3",
        "flac": ".flac",
        "aac": ".aac",
        "opus": ".ogg",
    }.get(tf, ".wav")


def _resolve_input_format(fmt_hint: str | None) -> str | None:
    """
    Given an audio file extension (no dot), return a valid FFmpeg input format.
    If unknown or ambiguous, returns None (FFmpeg will probe automatically).
    """
    if not fmt_hint:
        return None
    return EXT_TO_INPUT_FORMAT.get(fmt_hint.lower().lstrip("."), None)


def _ffmpeg_python_decode(raw_bytes: bytes, input_format: str | None, use_alt_output: bool) -> npt.NDArray[np.int16]:
    """
    Try decoding with ffmpeg-python. If input_format is None, we don't pass 'f='
    (let FFmpeg probe). If use_alt_output, add -vn and larger probe/analyze.
    Returns int16 PCM bytes on success, raises ffmpeg.Error on failure.
    """
    in_kwargs: dict[str, Any] = {}
    if input_format:
        # Build FFmpeg input graph reading from stdin ("pipe:0").
        # - "f" specifies an optional input format hint if provided.
        in_kwargs["f"] = input_format

    if not use_alt_output:
        # Base (Normal) Graph
        stream = ffmpeg.input("pipe:0", **in_kwargs)
        # Configure FFmpeg output to:
        #   - Write to stdout ("pipe:1") instead of a file.
        #   - Output raw PCM16 little-endian samples (format="s16le").
        #   - Force mono (ac=1).
        #   - Resample to 16 kHz (ar=16000).
        #   - Use the PCM16 codec (acodec="pcm_s16le").
        stream = ffmpeg.output(
            stream,
            "pipe:1",
            format="s16le",  # raw samples, no container header
            acodec="pcm_s16le",  # 16-bit signed little-endian PCM
            ac=1,  # 1 audio channel (mono)
            ar=16000,  # 16 000 samples per second
        )
    else:
        # Alternative Graph
        stream = ffmpeg.input("pipe:0", **in_kwargs, re=None)
        stream = ffmpeg.output(
            stream,
            "pipe:1",
            format="s16le",  # raw samples, no container header
            acodec="pcm_s16le",  # 16-bit signed little-endian PCM
            ac=1,  # 1 audio channel (mono)
            ar=16000,  # 16 000 samples per second
            vn=None,  # ignore video channel if exists
        ).global_args(
            "-hide_banner",
            "-loglevel",
            "error",
            "-probesize",
            "10000000",
            "-analyzeduration",
            "10000000",
        )

    # Run FFmpeg pipeline:
    #   - Feed raw_bytes to stdin
    #   - Capture stdout (decoded PCM bytes)
    #   - Capture stderr (logs, ignored here)
    pcm_bytes, _ = ffmpeg.run(
        stream,
        capture_stdout=True,
        capture_stderr=True,
        input=raw_bytes,
        cmd="ffmpeg",
    )

    # Interpret the returned PCM bytes as 16-bit signed integers.
    audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    if audio_i16.size == 0:
        raise RuntimeError("ffmpeg produced no audio samples")
    return audio_i16


def _ffmpeg_cli_decode(raw_bytes: bytes) -> npt.NDArray[np.int16]:
    """
    Last-resort CLI fallback: write input to a temp file, run `ffmpeg -i` to raw PCM,
    read stdout. We don’t pass `-f` for input — we let FFmpeg fully probe.
    """
    with (
        tempfile.NamedTemporaryFile(suffix=".bin", delete=True) as fin,
        tempfile.NamedTemporaryFile(suffix=".s16le", delete=True) as fout,
    ):
        fin.write(raw_bytes)
        fin.flush()

        # Let ffmpeg probe the container/codec; transcode to mono 16k PCM16
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",  # overwrite output
            "-i",
            fin.name,  # input file
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            fout.name,
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", "ignore")
            raise RuntimeError(f"ffmpeg CLI decode failed:\n{err}")

        pcm_bytes = fout.read()
        audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        if audio_i16.size == 0:
            raise RuntimeError("ffmpeg CLI produced no samples")
        return audio_i16


def decode_with_ffmpeg(
    *,
    raw_bytes: bytes,
    fmt_hint: str | None = None,
    _attempt: int = 0,
    _max_attempts: int = 4,
) -> tuple[npt.NDArray[np.float32], int]:
    """
    Decode arbitrary audio bytes to mono 16 kHz float32 via a recursive fallback chain:

      Attempt 0: ffmpeg-python (probe, no input format hint)
      Attempt 1: ffmpeg-python (normalized input format hint, e.g. opus→ogg, alac→mov)
      Attempt 2: ffmpeg-python (alt graph: -vn + larger probe/analyze)
      Attempt 3: ffmpeg CLI fallback (temp files, full probing)

    Returns
    -------
    (audio_f32, 16000)

    Notes
    -----
    - Requires a working `ffmpeg` binary on PATH.
    - Normalization uses `resolve_input_format(fmt_hint)` if available; pass None to skip.
    """
    if not raw_bytes:
        raise ValueError("Empty audio buffer")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg CLI not found on PATH. Install it (e.g., `brew install ffmpeg`).")

    # Base case: exceeded attempts → last resort CLI fallback
    if _attempt >= _max_attempts - 1:
        audio_i16 = _ffmpeg_cli_decode(raw_bytes)
        return (audio_i16.astype(np.float32) / 32768.0, 16000)

    # Checking if we can normalize the file format hint
    norm = _resolve_input_format(fmt_hint)

    # Determine strategy for this attempt
    try:
        if _attempt == 0:
            # No input format hint, let ffmpeg probe
            audio_i16 = _ffmpeg_python_decode(raw_bytes, input_format=None, use_alt_output=False)
        elif _attempt == 1:
            audio_i16 = _ffmpeg_python_decode(raw_bytes, input_format=norm, use_alt_output=False)
        elif _attempt == 2:
            audio_i16 = _ffmpeg_python_decode(raw_bytes, input_format=norm, use_alt_output=True)
        else:
            # Shouldn't hit due to the base case, but keep safe.
            audio_i16 = _ffmpeg_cli_decode(raw_bytes)

        # Success → return float32
        # Convert to float32 in [-1.0, 1.0]:
        # 32768.0 is the absolute value of the negative PCM16 limit (-32768).
        # Dividing by 32768.0 maps:
        #   -32768 → -1.0
        #   +32767 → +0.99997
        return (audio_i16.astype(np.float32) / 32768.0, 16000)

    except ffmpeg.Error:
        # Recursive step: advance attempt
        return decode_with_ffmpeg(
            raw_bytes=raw_bytes, fmt_hint=fmt_hint, _attempt=_attempt + 1, _max_attempts=_max_attempts
        )
    except Exception:
        # If a non-ffmpeg error happens, still advance attempts (keeps behavior consistent)
        return decode_with_ffmpeg(
            raw_bytes=raw_bytes, fmt_hint=fmt_hint, _attempt=_attempt + 1, _max_attempts=_max_attempts
        )


def encode_audio_from_wav_bytes(
    *,
    wav_bytes: bytes,
    target_format: str,  # "mp3"|"opus"|"aac"|"flac"|"wav"|"pcm"
    sample_rate: int = 24000,
    channels: int = 1,
) -> bytes:
    """
    Transcode WAV-in-memory to the requested target format using ffmpeg-python,
    falling back to ffmpeg CLI on error. For "pcm", returns raw s16le (no header).

    Requires a working ffmpeg binary on PATH.
    """
    tf = (target_format or "wav").lower()
    if tf not in {"mp3", "opus", "aac", "flac", "wav", "pcm"}:
        # Unknown target just return the original WAV
        return wav_bytes

    if not shutil.which("ffmpeg"):
        # No ffmpeg: only WAV is guaranteed; else return WAV
        return wav_bytes if tf != "pcm" else _pcm_from_wav_cli(wav_bytes, sample_rate, channels)

    try:
        stream = ffmpeg.input("pipe:0", f="wav")
        if tf == "wav":
            out = ffmpeg.output(stream, "pipe:1", format="wav", ac=channels, ar=sample_rate)
        elif tf == "pcm":
            out = ffmpeg.output(
                stream,
                "pipe:1",
                format="s16le",
                acodec="pcm_s16le",
                ac=channels,
                ar=sample_rate,
            )
        elif tf == "mp3":
            out = ffmpeg.output(stream, "pipe:1", format="mp3", ac=channels, ar=sample_rate)
        elif tf == "flac":
            out = ffmpeg.output(stream, "pipe:1", format="flac", ac=channels, ar=sample_rate)
        elif tf == "aac":
            # Prefer ADTS for raw AAC
            try:
                out = ffmpeg.output(stream, "pipe:1", format="adts", ac=channels, ar=sample_rate)
            except Exception:
                out = ffmpeg.output(stream, "pipe:1", format="aac", ac=channels, ar=sample_rate)
        elif tf == "opus":
            # libopus normally in OGG container
            try:
                out = ffmpeg.output(stream, "pipe:1", format="ogg", acodec="libopus", ac=channels, ar=sample_rate)
            except Exception:
                out = ffmpeg.output(stream, "pipe:1", format="opus", ac=channels, ar=sample_rate)
        else:
            out = ffmpeg.output(stream, "pipe:1", format="wav", ac=channels, ar=sample_rate)

        out = out.global_args("-hide_banner", "-loglevel", "error")
        encoded_stdout, _ = ffmpeg.run(out, capture_stdout=True, capture_stderr=True, input=wav_bytes, cmd="ffmpeg")
        encoded_bytes = cast(bytes, encoded_stdout)
        return encoded_bytes
    except Exception:
        # CLI fallback
        return _encode_with_cli(wav_bytes, tf, sample_rate, channels)


def _encode_with_cli(wav_bytes: bytes, tf: str, sr: int, ch: int) -> bytes:
    import tempfile

    with (
        tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as fin,
        tempfile.NamedTemporaryFile(suffix=_suffix_for(tf), delete=True) as fout,
    ):
        fin.write(wav_bytes)
        fin.flush()
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            fin.name,
            "-ac",
            str(ch),
            "-ar",
            str(sr),
        ]
        if tf == "pcm":
            cmd += ["-f", "s16le", "-acodec", "pcm_s16le", fout.name]
        elif tf == "mp3":
            cmd += ["-f", "mp3", fout.name]
        elif tf == "flac":
            cmd += ["-f", "flac", fout.name]
        elif tf == "aac":
            cmd += ["-f", "adts", fout.name]
        elif tf == "opus":
            cmd += ["-f", "ogg", "-acodec", "libopus", fout.name]
        else:
            cmd += ["-f", "wav", fout.name]

        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            return wav_bytes  # last-resort: return input WAV
        return fout.read()
