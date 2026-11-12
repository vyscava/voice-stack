"""ci/gpu-cache-prune.sh removes only this project's orphaned GPU builders.

The script runs against a stub ``docker`` on PATH that lists a fixed set of
containers and records every rm call, so the test needs no daemon.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import stat
import subprocess

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[3] / "ci" / "gpu-cache-prune.sh"

CONTAINERS = [
    "buildx_buildkit_gpu-cuda-893340",  # orphan from a cancelled cuda job
    "buildx_buildkit_gpu-rocm-894610",  # orphan from a cancelled rocm job
    "buildx_buildkit_other-gpu-cuda-1",  # another project's builder
    "buildx_buildkit_gpu-cuda-abc0",  # not a job-id suffix
    "buildx_buildkit_gpu-vulkan-12",  # not one of our variants
    "kb-ingest",
]

STUB = """#!/bin/sh
echo "$*" >> "$DOCKER_LOG"
case "$1 $2" in
  "ps -a") printf '%s\\n' {names} ;;
  "buildx du") printf 'ID: x\\nSize: 1MB\\nDescription: unrelated\\n' ;;
esac
exit 0
"""


def _run(tmp_path: pathlib.Path, *args: str) -> tuple[str, list[str]]:
    if shutil.which("sh") is None:
        pytest.skip("needs a POSIX sh")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    docker = bindir / "docker"
    docker.write_text(STUB.format(names=" ".join(CONTAINERS)))
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "docker.log"
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}", "DOCKER_LOG": str(log)}
    out = subprocess.run(
        ["sh", str(SCRIPT), *args], cwd=tmp_path, env=env, capture_output=True, text=True, check=True
    ).stdout
    calls = log.read_text().splitlines() if log.exists() else []
    return out, calls


@pytest.mark.unit
def test_apply_removes_only_our_orphan_builders_and_state_volumes(tmp_path: pathlib.Path) -> None:
    out, calls = _run(tmp_path, "--apply")
    removed = [c for c in calls if c.startswith(("rm ", "volume rm "))]
    assert removed == [
        "rm -f buildx_buildkit_gpu-cuda-893340",
        "volume rm buildx_buildkit_gpu-cuda-893340_state",
        "rm -f buildx_buildkit_gpu-rocm-894610",
        "volume rm buildx_buildkit_gpu-rocm-894610_state",
    ]
    assert "orphan GPU builders: 2" in out


@pytest.mark.unit
def test_dry_run_lists_orphans_and_removes_nothing(tmp_path: pathlib.Path) -> None:
    out, calls = _run(tmp_path)
    assert "orphan GPU builders: 2" in out
    assert "buildx_buildkit_gpu-rocm-894610" in out
    assert not [c for c in calls if c.startswith(("rm ", "volume rm ", "builder prune"))]
