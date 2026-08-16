#!/usr/bin/env python3
"""Validate voice-cloning reference clips BEFORE they are used to build a voice.

Why this exists
---------------
A reference corpus fails silently. Every check below corresponds to a defect that
produces a working-looking pipeline and a bad voice at the end:

  dead channel     A USB condenser mic with a cardioid capsule can still report
                   two input channels. If one carries nothing, a naive mono
                   downmix halves the amplitude of every clip and nothing errors.
  gain drift       Analog gain moved between clips. The speaker embedding wobbles
                   because loudness became a feature of the speaker.
  clipping         Irreversible. It survives every later resample and normalise.
  wrong rate/depth Recorded at the wrong setting; discovered after the session,
                   when the speaker is gone.
  silence          A clip that is mostly room tone teaches the model room tone.

Each of these is cheap to detect here and expensive to detect later. Run this
against the session directory before anyone trains anything.

Usage
-----
    python scripts/validate_voice_reference.py CLIPS_DIR
    python scripts/validate_voice_reference.py CLIPS_DIR --min-rate 48000 --strict-depth PCM_24

Exits non-zero if any clip fails, so it can gate a pipeline.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

# A channel carrying only converter noise sits far below this. Real speech,
# even quiet speech, sits far above it. The gap is wide enough that the exact
# value is not load-bearing.
DEAD_CHANNEL_RMS = 1e-4

# Full scale is 1.0. Samples at or above this are treated as clipped: real
# speech peaks below it, and a converter that pins reports exactly 1.0.
CLIP_LEVEL = 0.999

# A single pinned sample can occur naturally. A RUN of them is a flat top,
# which is clipping.
CLIP_RUN = 3

# Clips quieter than this are room tone, a mis-fire, or a missed cue.
MIN_RMS = 1e-3

# Gain drift tolerance. Clip RMS is compared against the session median in dB;
# beyond this the corpus teaches loudness as a speaker trait.
RMS_DRIFT_DB = 6.0

# A DC offset this large indicates an interface or capsule fault. It wastes
# headroom and biases every downstream frame.
MAX_DC_OFFSET = 0.01


@dataclass
class ClipReport:
    path: Path
    failures: list[str] = field(default_factory=list)
    rms: float = 0.0
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.failures


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _max_run(mask: np.ndarray) -> int:
    """Longest run of True in a 1-D boolean array."""
    if not mask.any():
        return 0
    # Reset a running count at every False, keep the maximum seen.
    best = run = 0
    for flag in mask:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def check_clip(
    path: Path,
    *,
    min_rate: int,
    strict_depth: str | None,
    min_duration_s: float,
) -> ClipReport:
    """Check one clip in isolation. Cross-clip checks happen in check_session."""
    report = ClipReport(path=path)

    info = sf.info(str(path))
    audio, rate = sf.read(str(path), dtype="float64", always_2d=True)

    report.duration_s = info.frames / info.samplerate if info.samplerate else 0.0

    if rate < min_rate:
        report.failures.append(f"sample rate {rate} Hz is below the required {min_rate} Hz")

    if strict_depth and info.subtype != strict_depth:
        report.failures.append(f"bit depth is {info.subtype}, expected {strict_depth}")

    if report.duration_s < min_duration_s:
        report.failures.append(
            f"duration {report.duration_s:.2f}s is below the minimum {min_duration_s:.2f}s"
        )

    # THE DEAD-CHANNEL CHECK. A multi-channel file must carry signal on EVERY
    # channel. Checking the mono downmix would pass a file whose second channel
    # is silent, which is the exact bug this guards against.
    n_channels = audio.shape[1]
    if n_channels > 1:
        for ch in range(n_channels):
            ch_rms = _rms(audio[:, ch])
            if ch_rms < DEAD_CHANNEL_RMS:
                report.failures.append(
                    f"channel {ch} is silent (rms {ch_rms:.2e}); a mono downmix would "
                    f"halve the level of every clip without erroring"
                )

    mono = audio.mean(axis=1)
    report.rms = _rms(mono)

    if report.rms < MIN_RMS:
        report.failures.append(f"clip is effectively silent (rms {report.rms:.2e})")

    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak >= CLIP_LEVEL:
        run = _max_run(np.abs(mono) >= CLIP_LEVEL)
        if run >= CLIP_RUN:
            report.failures.append(
                f"clipped: {run} consecutive samples at full scale (peak {peak:.4f})"
            )

    dc = float(np.mean(mono)) if mono.size else 0.0
    if abs(dc) > MAX_DC_OFFSET:
        report.failures.append(f"DC offset {dc:+.4f} exceeds {MAX_DC_OFFSET}")

    return report


def check_session(reports: list[ClipReport], *, drift_db: float = RMS_DRIFT_DB) -> None:
    """Cross-clip checks. Mutates reports in place, adding gain-drift failures.

    Level is compared against the session MEDIAN rather than the mean so that one
    very loud or very quiet clip cannot drag the reference it is judged against.
    """
    usable = [r for r in reports if r.rms > 0]
    if len(usable) < 2:
        return

    median = float(np.median([r.rms for r in usable]))
    if median <= 0:
        return

    for r in usable:
        delta_db = 20.0 * np.log10(r.rms / median)
        if abs(delta_db) > drift_db:
            r.failures.append(
                f"level is {delta_db:+.1f} dB from the session median, beyond "
                f"+/-{drift_db:.0f} dB; analog gain likely moved mid-session"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips_dir", type=Path, help="directory of .wav reference clips")
    parser.add_argument("--min-rate", type=int, default=48000, help="minimum sample rate in Hz (default 48000)")
    parser.add_argument("--strict-depth", default=None, help="required soundfile subtype, e.g. PCM_24")
    parser.add_argument("--min-duration", type=float, default=1.0, help="minimum clip duration in seconds")
    parser.add_argument("--drift-db", type=float, default=RMS_DRIFT_DB, help="allowed level drift from median, dB")
    args = parser.parse_args(argv)

    clips = sorted(args.clips_dir.glob("*.wav"))
    if not clips:
        print(f"FAIL: no .wav files found in {args.clips_dir}", file=sys.stderr)
        return 1

    reports = [
        check_clip(
            c,
            min_rate=args.min_rate,
            strict_depth=args.strict_depth,
            min_duration_s=args.min_duration,
        )
        for c in clips
    ]
    check_session(reports, drift_db=args.drift_db)

    failed = [r for r in reports if not r.ok]
    for r in reports:
        status = "ok  " if r.ok else "FAIL"
        print(f"{status} {r.path.name}  {r.duration_s:5.2f}s  rms={r.rms:.4f}")
        for f in r.failures:
            print(f"       - {f}")

    print(f"\n{len(reports) - len(failed)}/{len(reports)} clips usable")
    if failed:
        print(f"FAIL: {len(failed)} clip(s) must be re-recorded before training", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
