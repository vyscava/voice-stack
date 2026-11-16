"""The GPU ASR deploy compose file and the settings it depends on.

A misspelled key is not an error anywhere: the container starts, the setting
keeps its default (here: an unpinned model, or the CPU), and the deploy goes
green on /health.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from core.settings import Settings

DEPLOY = Path(__file__).resolve().parents[3] / "compose" / "deploy"
NOT_SETTINGS = {"SERVICE_MODE", "HF_HOME", "HF_HUB_CACHE"}

pytestmark = [pytest.mark.unit, pytest.mark.asr]


def _gpu() -> dict:
    (service,) = yaml.safe_load((DEPLOY / "asr-gpu.yml").read_text(encoding="utf-8"))["services"].values()
    return service


def test_every_gpu_setting_is_a_real_setting() -> None:
    unknown = set(_gpu()["environment"]) - set(Settings.model_fields) - NOT_SETTINGS
    assert not unknown, f"not read by the service, so silently ignored: {sorted(unknown)}"


def test_runs_on_the_gpu_with_pinned_weights() -> None:
    env = _gpu()["environment"]
    assert env["ASR_DEVICE"] == "cuda"
    assert env["ASR_COMPUTE_TYPE"] == "float16"
    # A full 40-hex commit, not a branch name: "main" would follow upstream.
    assert re.fullmatch(r"[0-9a-f]{40}", env["ASR_MODEL_REVISION"])


def test_reserves_one_nvidia_gpu() -> None:
    (device,) = _gpu()["deploy"]["resources"]["reservations"]["devices"]
    assert device["driver"] == "nvidia" and device["count"] == 1 and "gpu" in device["capabilities"]


def test_the_model_cache_is_on_the_mounted_volume() -> None:
    service = _gpu()
    mount = next(v.split(":")[1] for v in service["volumes"] if v.startswith("asr-gpu-models:"))
    assert service["environment"]["ASR_MODEL_LOCATION"] == mount


def test_the_model_never_unloads() -> None:
    # A reload costs seconds on the first turn of a call; 0 disables the idle unload.
    assert _gpu()["environment"]["ASR_IDLE_TIMEOUT_MINUTES"] == "0"
