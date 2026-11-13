"""The deploy compose file must pass every brain setting through to the container.

A setting the agent reads but the compose file never forwards is silently replaced
by its default: the deploy succeeds and the agent talks to the brain without the
configured value (for example an API key that never arrives, so every turn is 401).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from core.settings import Settings

COMPOSE = Path(__file__).resolve().parents[3] / "compose" / "deploy" / "agent.yml"
BRAIN_PREFIXES = ("AGENT_BRAIN", "AGENT_LLM_", "AGENT_NATS_")
BRAIN_EXTRAS = {"AGENT_SELF_NAME", "AGENT_PEER"}


def _agent_environment() -> dict:
    services = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]
    (service,) = [s for s in services.values() if "environment" in s]
    return service["environment"]


def test_every_brain_setting_is_forwarded():
    brain_settings = {name for name in Settings.model_fields if name.startswith(BRAIN_PREFIXES) or name in BRAIN_EXTRAS}
    assert "AGENT_LLM_API_KEY" in brain_settings
    missing = brain_settings - set(_agent_environment())
    assert not missing, f"settings never reach the container: {sorted(missing)}"


def test_api_key_falls_back_to_the_settings_default_not_empty():
    # An empty bearer is not the same as "no key needed": keep the settings default when unset.
    assert _agent_environment()["AGENT_LLM_API_KEY"] == "${AGENT_LLM_API_KEY:-unused}"
