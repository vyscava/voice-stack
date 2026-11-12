"""Publishing the GPU images.

Two things are pinned here:

1. The CI wiring in .gitlab/ci/gpu-images.yml. The deploy jobs have no
   `needs`, so they wait for every job in an earlier stage: a GPU build placed
   anywhere before `deploy` on main would hold (or, failing, block) every
   production deploy. The publish jobs must sit in a stage after `deploy`, with
   no `needs` that would let them start early, and must be allowed to fail.
   Merge-request jobs must never push.

2. ci/gpu-image-publish.sh, run against a stub ``docker`` on PATH that records
   its calls: it pushes the image the check built under the moving tag, and
   refuses when that image is missing or the registry is not configured.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import stat
import subprocess
from typing import Any

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
PUBLISH = ROOT / "ci" / "gpu-image-publish.sh"
VARIANTS = ("cuda", "rocm")


# ---------------------------------------------------------------------------
# CI wiring
# ---------------------------------------------------------------------------


def _resolved_jobs() -> dict[str, dict[str, Any]]:
    """gpu-images.yml jobs with `extends` merged (GitLab's deep-merge, enough for these keys)."""
    raw = yaml.safe_load((ROOT / ".gitlab" / "ci" / "gpu-images.yml").read_text())

    def resolve(name: str) -> dict[str, Any]:
        job = dict(raw[name])
        parents = job.pop("extends", [])
        merged: dict[str, Any] = {}
        for parent in [parents] if isinstance(parents, str) else parents:
            merged.update(resolve(parent))
        merged.update(job)
        return merged

    return {name: resolve(name) for name in raw if not name.startswith(".") and isinstance(raw[name], dict)}


def _stages() -> list[str]:
    return yaml.safe_load((ROOT / ".gitlab-ci.yml").read_text())["stages"]


@pytest.mark.unit
@pytest.mark.parametrize("variant", VARIANTS)
def test_publish_job_runs_strictly_after_the_deploy(variant: str) -> None:
    job = _resolved_jobs()[f"publish:gpu-image:{variant}"]
    stages = _stages()
    assert stages.index(job["stage"]) > stages.index("deploy")
    # `needs` (even []) would let the job start before the deploy stage ends.
    assert "needs" not in job
    assert job["allow_failure"] is True


@pytest.mark.unit
@pytest.mark.parametrize("variant", VARIANTS)
def test_publish_job_pushes_only_after_the_check_and_only_on_main(variant: str) -> None:
    job = _resolved_jobs()[f"publish:gpu-image:{variant}"]
    assert job["script"] == [
        'sh ci/gpu-image-check.sh "$GPU_VARIANT"',
        'sh ci/gpu-image-publish.sh "$GPU_VARIANT"',
    ]
    assert job["rules"], "publish job has no rules"
    for rule in job["rules"]:
        assert rule["if"].startswith("$CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH"), rule


@pytest.mark.unit
@pytest.mark.parametrize("variant", VARIANTS)
def test_publish_is_manual_only(variant: str) -> None:
    """Publishing costs two cold ~20-minute builds and ~20 GB of pushes, so it
    never starts by itself: every rule that creates the job makes it manual."""
    job = _resolved_jobs()[f"publish:gpu-image:{variant}"]
    assert [rule.get("when") for rule in job["rules"]] == ["manual"]


@pytest.mark.unit
@pytest.mark.parametrize("variant", VARIANTS)
def test_merge_request_job_never_publishes(variant: str) -> None:
    job = _resolved_jobs()[f"build:gpu-image:{variant}"]
    assert not any("gpu-image-publish" in line for line in job["script"])
    assert not any("docker login" in line or "docker push" in line for line in job["script"])
    for rule in job["rules"]:
        assert "$CI_DEFAULT_BRANCH" not in rule["if"] and "main" not in rule["if"], rule


# ---------------------------------------------------------------------------
# ci/gpu-image-publish.sh
# ---------------------------------------------------------------------------

STUB = """#!/bin/sh
echo "$*" >> "$DOCKER_LOG"
case "$1 $2" in
  "image inspect") [ "$3" = "$PRESENT_IMAGE" ] && exit 0 || exit 1 ;;
  "login -u") cat > "$DOCKER_LOG.stdin" ;;
  "push "*)
    echo "The push refers to repository [$2]"
    [ -n "$PUSH_DIGEST" ] && echo "cuda: digest: $PUSH_DIGEST size: 4242" ;;
esac
exit 0
"""

ENV = {
    "GPU_IMAGE": "voice-stack-gpu-cuda:job-42",
    "PRESENT_IMAGE": "voice-stack-gpu-cuda:job-42",
    "DOCKER_REGISTRY": "reg.example:5000",
    "DOCKER_REGISTRY_USER": "ci",
    "DOCKER_REGISTRY_PASSWORD": "pw-from-ci",
    "PUSH_DIGEST": "sha256:" + "ab" * 32,
}


def _publish(tmp_path: pathlib.Path, variant: str, **overrides: str | None) -> tuple[int, str, list[str]]:
    if shutil.which("sh") is None:
        pytest.skip("needs a POSIX sh")
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    docker = bindir / "docker"
    docker.write_text(STUB)
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "docker.log"
    env = {k: v for k, v in os.environ.items() if not k.startswith("DOCKER_REGISTRY")}
    env.update(ENV)
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    env.update({"PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}", "DOCKER_LOG": str(log)})
    proc = subprocess.run(["sh", str(PUBLISH), variant], cwd=tmp_path, env=env, capture_output=True, text=True)
    calls = log.read_text().splitlines() if log.exists() else []
    return proc.returncode, proc.stdout + proc.stderr, calls


@pytest.mark.unit
def test_publish_pushes_the_checked_image_under_the_moving_tag(tmp_path: pathlib.Path) -> None:
    rc, out, calls = _publish(tmp_path, "cuda")
    assert rc == 0, out
    assert "login -u ci --password-stdin reg.example:5000" in calls
    assert (tmp_path / "docker.log.stdin").read_text().strip() == "pw-from-ci"
    assert "tag voice-stack-gpu-cuda:job-42 reg.example:5000/voice-stack:cuda" in calls
    assert [c for c in calls if c.startswith("push ")] == ["push reg.example:5000/voice-stack:cuda"]
    published = (tmp_path / "gpu-image-published.txt").read_text().strip()
    assert published == "published reg.example:5000/voice-stack:cuda@sha256:" + "ab" * 32
    assert calls[-1] == "logout reg.example:5000"
    # The password never reaches argv (and so never the job log).
    assert not any("pw-from-ci" in c for c in calls)


@pytest.mark.unit
def test_publish_refuses_when_the_checked_image_is_missing(tmp_path: pathlib.Path) -> None:
    rc, out, calls = _publish(tmp_path, "rocm", GPU_IMAGE="voice-stack-gpu-rocm:job-7")
    assert rc != 0
    assert "not found" in out
    assert not [c for c in calls if c.startswith(("login", "push", "tag"))]


@pytest.mark.unit
@pytest.mark.parametrize("missing", ["DOCKER_REGISTRY", "DOCKER_REGISTRY_USER", "DOCKER_REGISTRY_PASSWORD"])
def test_publish_refuses_without_registry_settings(tmp_path: pathlib.Path, missing: str) -> None:
    rc, out, calls = _publish(tmp_path, "cuda", **{missing: None})
    assert rc != 0
    assert missing in out
    assert not [c for c in calls if c.startswith("push")]


@pytest.mark.unit
def test_publish_rejects_an_unknown_variant(tmp_path: pathlib.Path) -> None:
    rc, out, calls = _publish(tmp_path, "vulkan")
    assert rc == 2
    assert not calls


@pytest.mark.unit
def test_publish_fails_when_the_push_reports_no_digest(tmp_path: pathlib.Path) -> None:
    rc, out, calls = _publish(tmp_path, "cuda", PUSH_DIGEST="")
    assert rc != 0
    assert "reported no digest" in out
    assert not (tmp_path / "gpu-image-published.txt").exists()
    assert calls[-1] == "logout reg.example:5000"
