"""The ASR engine must return per-segment timestamps. Bazarr's subtitles depend on it.

Why this file exists
--------------------
``tests/unit/asr/test_api_bazarr.py`` has 16 tests covering ``srt``, ``vtt`` and
``tsv`` output, and every one of them passes with an engine that returns no
timestamps at all. ``conftest.py::asr_client`` replaces the engine with a Mock
whose timings are hardcoded in the fixture, so those tests exercise routing and
the formatters -- never the engine contract.

Proven, not assumed: zeroing ``start``/``end`` in ``mock_asr_engine`` (simulating
a timestamp-less engine) left all 16 passing. See voice-stack#6.

Without timestamps Bazarr still gets a 200 and still writes a subtitle file, but
every cue reads ``00:00:00,000 --> 00:00:00,001``. The failure is silent all the
way to the user's media player. This file is the gate that stops an engine swap
from shipping that.

What it deliberately does NOT assert
------------------------------------
Transcription accuracy. The fixture is espeak-ng output transcribed by
``whisper-tiny``, and it comes back partly garbled ("the lazy dog" -> "a very
dark"). That is fine and intentional: this is a CONTRACT test, not an accuracy
test. Asserting on text would make it flaky and would couple the Bazarr gate to
model quality, which is measured separately in the model bake-off.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

# Minimum segments needed to prove timestamps ADVANCE. One segment could carry
# start=end=0.0 and still look plausible; two cannot.
MIN_SEGMENTS = 2

# Deliberate wide gaps: the engine must produce SEPARATE segments for the
# timestamp contract to mean anything. One long run-on sentence would yield a
# single segment and prove nothing.
FIXTURE_SCRIPT = (
    "The quick brown fox jumps over the lazy dog.     "
    "Subtitles need accurate timestamps.     "
    "This is the third and final sentence."
)


@pytest.fixture(scope="session")
def speech_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthesize 16 kHz mono speech with espeak-ng.

    Generated rather than committed: a 297 KB WAV in git is a blob nobody can
    review, and `gitlab_commit_files` cannot carry binaries anyway. espeak-ng is
    deterministic, tiny, and already a common base-image package -- it just has
    to be present in the CI image.
    """
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    if espeak is None:
        pytest.skip("espeak-ng not installed; cannot synthesize the speech fixture")

    import numpy as np
    import resampy
    import soundfile as sf

    out_dir = tmp_path_factory.mktemp("asr-fixtures")
    raw = out_dir / "raw.wav"
    subprocess.run([espeak, "-v", "en-us", "-s", "145", "-w", str(raw), FIXTURE_SCRIPT], check=True)

    audio, sample_rate = sf.read(str(raw), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sample_rate != 16000:
        mono = resampy.resample(mono, sample_rate, 16000)
    mono = (mono / max(1e-9, float(np.abs(mono).max()))) * 0.8

    wav = out_dir / "speech_en_16k.wav"
    sf.write(str(wav), mono, 16000, subtype="PCM_16")
    return wav


def assert_timestamp_contract(segments: list[dict[str, Any]], *, audio_duration_s: float) -> None:
    """Raise AssertionError unless `segments` carry usable subtitle timings.

    Extracted so the gate can be tested against a known-bad input (see
    ``test_contract_helper_rejects_zeroed_timestamps``). A gate that has never
    been observed failing is an assumption, not a gate.
    """
    assert len(segments) >= MIN_SEGMENTS, f"expected >= {MIN_SEGMENTS} segments, got {len(segments)}"

    starts = [float(s["start"]) for s in segments]
    ends = [float(s["end"]) for s in segments]

    for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
        assert end > start, f"segment {i} has end ({end}) <= start ({start}) -- zero-length cue"

    assert starts == sorted(starts), f"segment starts are not monotonic: {starts}"

    distinct = {round(s, 3) for s in starts}
    assert len(distinct) >= MIN_SEGMENTS, f"segments share timestamps: {starts}"

    # A timestamp-less engine that emitted 0.0 for everything would pass the
    # checks above only if it also faked ordering. Coverage catches the rest:
    # real speech spans a meaningful fraction of its own clip.
    span = max(ends) - min(starts)
    assert (
        span > audio_duration_s * 0.5
    ), f"segments span only {span:.2f}s of a {audio_duration_s:.2f}s clip -- timings look synthetic"


def test_contract_helper_rejects_zeroed_timestamps() -> None:
    """The gate must FAIL on the exact mutation that the mocked tests survive.

    This is the self-verification. If this test ever passes silently, the gate
    below has stopped gating.
    """
    zeroed = [
        {"start": 0.0, "end": 0.0, "text": "the quick brown fox"},
        {"start": 0.0, "end": 0.0, "text": "subtitles need accurate timestamps"},
    ]
    with pytest.raises(AssertionError, match="zero-length cue"):
        assert_timestamp_contract(zeroed, audio_duration_s=9.29)


def test_contract_helper_rejects_single_collapsed_segment() -> None:
    """One segment covering nothing is the other way an engine degrades."""
    with pytest.raises(AssertionError, match="expected >= 2 segments"):
        assert_timestamp_contract([{"start": 0.0, "end": 1.0, "text": "hi"}], audio_duration_s=9.29)


@pytest.mark.integration
@pytest.mark.requires_models
def test_real_engine_returns_usable_timestamps(speech_wav: Path) -> None:
    """Run the REAL engine on real speech. No mocks anywhere in this test."""
    import soundfile as sf
    from faster_whisper import WhisperModel

    info = sf.info(str(speech_wav))
    assert info.samplerate == 16000, f"fixture must be 16 kHz, got {info.samplerate}"
    assert info.channels == 1, f"fixture must be mono, got {info.channels} channels"

    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    raw_segments, _ = model.transcribe(str(speech_wav), language="en")
    segments = [{"start": s.start, "end": s.end, "text": s.text} for s in raw_segments]

    assert_timestamp_contract(segments, audio_duration_s=info.frames / info.samplerate)


@pytest.mark.integration
@pytest.mark.requires_models
def test_srt_output_has_distinct_cue_times(speech_wav: Path) -> None:
    """End to end: engine timings must survive into the SRT Bazarr consumes."""
    import re

    import soundfile as sf
    from faster_whisper import WhisperModel

    from asr.engine.base import TranscribeResult

    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    raw_segments, _ = model.transcribe(str(speech_wav), language="en")
    segments = [{"start": s.start, "end": s.end, "text": s.text} for s in raw_segments]

    info = sf.info(str(speech_wav))
    assert_timestamp_contract(segments, audio_duration_s=info.frames / info.samplerate)

    result = TranscribeResult(
        text=" ".join(s["text"] for s in segments),
        segments=segments,
        language_code="en",
        language_name="English",
        confidence=0.9,
        duration_input_s=info.frames / info.samplerate,
        duration_after_vad_s=info.frames / info.samplerate,
        processing_ms=0,
        asr_ms=0,
        vad_used=False,
        engine="faster-whisper",
        model="tiny",
        vad_segments=[],
    )

    srt = result.to_srt()
    cue_times = re.findall(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", srt)

    assert len(cue_times) >= MIN_SEGMENTS, f"SRT has {len(cue_times)} cues, expected >= {MIN_SEGMENTS}"
    assert len(set(cue_times)) == len(cue_times), f"SRT cues share identical times: {cue_times}"
    assert cue_times[0][0] != cue_times[-1][0], "first and last cue start at the same time"
