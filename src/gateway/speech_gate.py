"""Decide whether a clip plausibly contains speech, BEFORE Whisper sees it.

Why this exists
---------------
Whisper does not return "I heard nothing". Handed silence it INVENTS text, and
it invents the same things repeatedly because of what it was trained on:

    1.5 s of digital silence  ->  "You"
    near-silence              ->  "Thank you for watching!"

Both were observed on this deployment. The second reached a peer agent as a
genuine question on 2026-08-17, and was caught only because that agent happened
to know the artifact. That is luck, not a control.

A length threshold cannot fix this. "Thank you for watching!" is 23 characters,
so any limit high enough to block it also blocks "What time is it?". A length
check filters EMPTY transcripts; a hallucination is by definition confident, and
confidence has no length.

So the guard has to measure the AUDIO, not the text. If the clip is too short or
too quiet to contain speech, it never reaches the model, and a model that is
never called cannot invent anything.

Measured with ffmpeg's volumedetect, which is already in the image for
torchcodec and handles whatever the browser sends (webm/opus) without the
gateway needing an audio stack of its own.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

from core.logging import logger_gateway as logger

# Below this, a clip is too short to be a question. Generous on purpose: a
# clipped "yes" is ~250 ms, and rejecting a real word is worse than admitting
# one dud.
MIN_DURATION_S = 0.35

# Mean level below this is a room with nobody talking. Digital silence measures
# -91 dBFS; a quiet room around -50; speech at conversational level well above
# -35. -45 sits in the gap with room to spare.
MIN_MEAN_DBFS = -45.0

# Known Whisper silence artifacts, normalised. This is a BACKSTOP for clips that
# pass the audio gate and still produce an invented transcript, not the primary
# defence -- a denylist can only ever catch what someone has already seen.
_HALLUCINATIONS = frozenset(
    {
        "you",
        "thank you",
        "thanks",
        "thank you for watching",
        "thanks for watching",
        "thank you for watching!",
        "thanks for watching!",
        "please subscribe",
        "subscribe",
        "bye",
        "bye bye",
        "the end",
        "silence",
    }
)

_MEAN_RE = re.compile(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")
_MAX_RE = re.compile(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")
_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class AudioProbe:
    duration_s: float | None
    mean_dbfs: float | None
    max_dbfs: float | None

    @property
    def measured(self) -> bool:
        """False when ffmpeg told us nothing usable."""
        return self.duration_s is not None or self.mean_dbfs is not None


def probe_audio(audio: bytes, *, timeout_s: float = 15.0) -> AudioProbe:
    """Measure duration and level. Never raises: an unmeasurable clip is not a
    reason to drop a request, only a reason not to reject it on this basis."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", "pipe:0", "-af", "volumedetect", "-f", "null", "-"],
            input=audio,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(f"speech gate: ffmpeg probe failed ({type(exc).__name__}); allowing the clip through")
        return AudioProbe(None, None, None)

    err = proc.stderr.decode("utf-8", errors="replace")

    duration = None
    if m := _DUR_RE.search(err):
        h, mnt, sec = m.groups()
        duration = int(h) * 3600 + int(mnt) * 60 + float(sec)

    mean = float(m.group(1)) if (m := _MEAN_RE.search(err)) else None
    peak = float(m.group(1)) if (m := _MAX_RE.search(err)) else None
    return AudioProbe(duration, mean, peak)


def looks_like_speech(
    audio: bytes,
    *,
    min_duration_s: float = MIN_DURATION_S,
    min_mean_dbfs: float = MIN_MEAN_DBFS,
) -> bool:
    """True if the clip is worth sending to ASR.

    FAILS OPEN. If ffmpeg cannot measure the clip we return True, because
    silently dropping a real question is a worse failure than occasionally
    letting a dud through -- the dud costs one wasted turn, the drop costs the
    user's actual request with no explanation.
    """
    probe = probe_audio(audio)
    if not probe.measured:
        return True

    if probe.duration_s is not None and probe.duration_s < min_duration_s:
        logger.info(f"speech gate: rejected, {probe.duration_s:.2f}s is shorter than {min_duration_s}s")
        return False

    if probe.mean_dbfs is not None and probe.mean_dbfs < min_mean_dbfs:
        logger.info(f"speech gate: rejected, mean level {probe.mean_dbfs:.1f} dBFS is below {min_mean_dbfs} dBFS")
        return False

    return True


def is_known_hallucination(transcript: str) -> bool:
    """True for text Whisper is known to invent from silence.

    Backstop only. A denylist catches what has already been seen once, which is
    exactly the class of thing that keeps producing new members.
    """
    normalised = re.sub(r"[^\w\s]", "", transcript).strip().lower()
    normalised = re.sub(r"\s+", " ", normalised)
    return normalised in _HALLUCINATIONS
