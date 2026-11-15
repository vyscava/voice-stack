"""The streaming ASR deploy compose file: every setting it sets must be one the service reads.

A misspelled key is not an error anywhere: the container starts, the setting
keeps its default, and the deploy goes green. For streaming that default is
"disabled", so every call would silently fall back to batch ASR.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.settings import Settings

DEPLOY = Path(__file__).resolve().parents[3] / "compose" / "deploy"
# Read by the entrypoint or libraries, not by Settings.
NOT_SETTINGS = {"SERVICE_MODE", "HF_HOME", "HF_HUB_CACHE"}


def _service(name: str) -> dict:
    return yaml.safe_load((DEPLOY / name).read_text(encoding="utf-8"))["services"]


@pytest.mark.unit
@pytest.mark.asr
def test_every_stream_setting_is_a_real_setting() -> None:
    (service,) = _service("asr-stream.yml").values()
    unknown = set(service["environment"]) - set(Settings.model_fields) - NOT_SETTINGS
    assert not unknown, f"not read by the service, so silently ignored: {sorted(unknown)}"
    assert service["environment"]["ASR_STREAM_ENABLED"] == "true"


@pytest.mark.unit
@pytest.mark.asr
def test_stream_joins_the_network_the_agent_already_uses() -> None:
    stream = yaml.safe_load((DEPLOY / "asr-stream.yml").read_text(encoding="utf-8"))
    agent = yaml.safe_load((DEPLOY / "agent.yml").read_text(encoding="utf-8"))
    assert (
        stream["networks"]["asr"] == agent["networks"]["asr"] == {"external": True, "name": "voice-stack-asr_default"}
    )


@pytest.mark.unit
@pytest.mark.asr
def test_the_model_cache_is_on_the_mounted_volume() -> None:
    """The full image sets HF_HUB_CACHE, which wins over HF_HOME: the compose file must override it.

    Otherwise the 2.4 GB model downloads into the container layer and again
    after every deploy, while the deploy goes green on /health.
    """
    (service,) = _service("asr-stream.yml").values()
    mount = next(v.split(":")[1] for v in service["volumes"] if v.startswith("asr-stream-models:"))
    cache = service["environment"]["HF_HUB_CACHE"]
    assert cache == mount or cache.startswith(mount + "/")
