"""Every check in validate_voice_reference.py must be PROVEN able to fail.

A validator nobody has watched reject a bad file is an assumption, not a
validator. Each test below constructs the specific defect the check exists to
catch and asserts the check fires, and the happy-path test asserts a clean clip
passes so the checks are not merely always-on.

Clips are synthesised here rather than committed: a binary fixture in git is a
blob nobody can review, and the defects are easier to read as code than to hear.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from validate_voice_reference import (  # noqa: E402
    check_clip,
    check_session,
    main,
)

RATE = 48000
DUR = 2.0


def _tone(seconds: float = DUR, amp: float = 0.3, rate: int = RATE) -> np.ndarray:
    """A steady tone. Stands in for speech: it has energy and a stable level."""
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    return amp * np.sin(2 * np.pi * 220.0 * t)


def _write(path: Path, data: np.ndarray, rate: int = RATE, subtype: str = "PCM_24") -> Path:
    sf.write(str(path), data, rate, subtype=subtype)
    return path


def _check(path: Path, **kw):
    defaults = {"min_rate": RATE, "strict_depth": None, "min_duration_s": 1.0}
    defaults.update(kw)
    return check_clip(path, **defaults)


def test_clean_clip_passes(tmp_path: Path) -> None:
    """The happy path. Without this, every other test could pass by rejecting all."""
    report = _check(_write(tmp_path / "clean.wav", _tone()))
    assert report.ok, f"a clean clip was rejected: {report.failures}"


def test_dead_second_channel_is_caught(tmp_path: Path) -> None:
    """THE ONE THAT MATTERS. Channel 0 has speech, channel 1 is silent.

    The mono downmix of this file is a perfectly valid-looking quiet clip. Only
    a per-channel check sees the fault.
    """
    stereo = np.stack([_tone(), np.zeros(int(RATE * DUR))], axis=1)
    report = _check(_write(tmp_path / "dead_channel.wav", stereo))
    assert not report.ok
    assert any("channel 1 is silent" in f for f in report.failures), report.failures


def test_both_channels_live_passes(tmp_path: Path) -> None:
    """A genuine two-channel recording must NOT be flagged."""
    stereo = np.stack([_tone(), _tone(amp=0.25)], axis=1)
    report = _check(_write(tmp_path / "stereo_ok.wav", stereo))
    assert report.ok, report.failures


def test_clipping_is_caught(tmp_path: Path) -> None:
    """A flat top at full scale, which no resample or normalise can undo."""
    data = _tone(amp=1.5)
    report = _check(_write(tmp_path / "clipped.wav", np.clip(data, -1.0, 1.0)))
    assert not report.ok
    assert any("clipped" in f for f in report.failures), report.failures


def test_wrong_sample_rate_is_caught(tmp_path: Path) -> None:
    report = _check(_write(tmp_path / "slow.wav", _tone(rate=22050), rate=22050))
    assert not report.ok
    assert any("sample rate" in f for f in report.failures), report.failures


def test_wrong_bit_depth_is_caught(tmp_path: Path) -> None:
    report = _check(
        _write(tmp_path / "shallow.wav", _tone(), subtype="PCM_16"),
        strict_depth="PCM_24",
    )
    assert not report.ok
    assert any("bit depth" in f for f in report.failures), report.failures


def test_silent_clip_is_caught(tmp_path: Path) -> None:
    """A missed cue. Room tone teaches the model room tone."""
    report = _check(_write(tmp_path / "silent.wav", np.zeros(int(RATE * DUR))))
    assert not report.ok
    assert any("silent" in f for f in report.failures), report.failures


def test_short_clip_is_caught(tmp_path: Path) -> None:
    report = _check(_write(tmp_path / "short.wav", _tone(seconds=0.4)), min_duration_s=1.0)
    assert not report.ok
    assert any("duration" in f for f in report.failures), report.failures


def test_dc_offset_is_caught(tmp_path: Path) -> None:
    report = _check(_write(tmp_path / "dc.wav", _tone() + 0.05))
    assert not report.ok
    assert any("DC offset" in f for f in report.failures), report.failures


def test_gain_drift_across_clips_is_caught(tmp_path: Path) -> None:
    """Each clip is individually fine. The CORPUS is not.

    Nine clips at one level and one 20 dB down: only a cross-clip comparison
    can see that the analog gain moved.
    """
    reports = [_check(_write(tmp_path / f"c{i}.wav", _tone(amp=0.3))) for i in range(9)]
    quiet = _check(_write(tmp_path / "c9.wav", _tone(amp=0.03)))
    assert quiet.ok, "the quiet clip must pass in isolation, or this proves nothing"

    reports.append(quiet)
    check_session(reports)
    assert not quiet.ok
    assert any("from the session median" in f for f in quiet.failures), quiet.failures


def test_consistent_levels_do_not_trip_drift(tmp_path: Path) -> None:
    """Natural variation between takes must not be reported as drift."""
    amps = [0.28, 0.30, 0.32, 0.29, 0.31]
    reports = [_check(_write(tmp_path / f"v{i}.wav", _tone(amp=a))) for i, a in enumerate(amps)]
    check_session(reports)
    assert all(r.ok for r in reports), [r.failures for r in reports if not r.ok]


def test_main_exits_nonzero_on_a_bad_corpus(tmp_path: Path) -> None:
    """The script must GATE, not just report."""
    _write(tmp_path / "good.wav", _tone())
    _write(tmp_path / "bad.wav", np.zeros(int(RATE * DUR)))
    assert main([str(tmp_path)]) == 1


def test_main_exits_zero_on_a_good_corpus(tmp_path: Path) -> None:
    for i in range(3):
        _write(tmp_path / f"g{i}.wav", _tone())
    assert main([str(tmp_path)]) == 0


def test_main_fails_on_empty_directory(tmp_path: Path) -> None:
    """An empty directory must not read as 'nothing wrong'."""
    assert main([str(tmp_path)]) == 1
