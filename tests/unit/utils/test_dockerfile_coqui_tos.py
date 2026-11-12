"""Every image that ships XTTS must accept the Coqui licence in its ENVIRONMENT.

The images also write tos_agreed.txt under ~/.local/share/tts, but deployments
mount the TTS models volume over that directory. A bind mount hides the file,
and XTTS then prompts on stdin and the container dies at start-up with
"EOFError: EOF when reading a line" (voice-stack#14). COQUI_TOS_AGREED=1 in the
runtime stage is what survives the mount, so it is asserted per image here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

# (Dockerfile, the runtime stage that ships XTTS)
XTTS_IMAGES = [
    ("Dockerfile", "full"),
    ("docker/Dockerfile.cuda", "voice-stack-cuda"),
    ("docker/Dockerfile.rocm", "voice-stack-rocm"),
]

_FROM = re.compile(r"^FROM\s+\S+(?:\s+AS\s+(\S+))?\s*$", re.IGNORECASE | re.MULTILINE)


def stage_text(dockerfile: str, stage: str) -> str:
    """Return the instructions of one named build stage (up to the next FROM)."""
    text = dockerfile
    matches = list(_FROM.finditer(text))
    for i, m in enumerate(matches):
        if m.group(1) == stage:
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            return text[m.end() : end]
    raise AssertionError(f"stage {stage!r} not found")


def sets_coqui_tos(stage: str) -> bool:
    """True when an ENV instruction in the stage sets COQUI_TOS_AGREED=1."""
    return re.search(r"^ENV\s+(?:.*\\\n\s*)*?.*\bCOQUI_TOS_AGREED=1\b", stage, re.MULTILINE) is not None


@pytest.mark.unit
@pytest.mark.parametrize(("path", "stage"), XTTS_IMAGES)
def test_xtts_image_accepts_licence_in_env(path: str, stage: str) -> None:
    text = (ROOT / path).read_text()
    assert sets_coqui_tos(stage_text(text, stage)), f"{path} stage {stage} lacks ENV COQUI_TOS_AGREED=1"


@pytest.mark.unit
def test_detector_rejects_stage_without_env() -> None:
    # The check must be able to fail: a comment or a RUN line is not ENV, and a
    # value other than 1 is not acceptance.
    df = (
        "FROM python:3.11-slim AS full\n"
        "# ENV COQUI_TOS_AGREED=1\n"
        "RUN echo COQUI_TOS_AGREED=1\n"
        "ENV COQUI_TOS_AGREED=0\n"
        "FROM python:3.11-slim AS other\n"
        "ENV COQUI_TOS_AGREED=1\n"
    )
    assert not sets_coqui_tos(stage_text(df, "full"))
    assert sets_coqui_tos(stage_text(df, "other"))


@pytest.mark.unit
def test_detector_accepts_continuation_line() -> None:
    df = "FROM x AS full\nENV A=1 \\\n    COQUI_TOS_AGREED=1 \\\n    B=2\n"
    assert sets_coqui_tos(stage_text(df, "full"))
