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


def test_stt_and_preemptive_settings_are_forwarded_with_safe_defaults():
    env = _agent_environment()
    # Streaming by default against the streaming ASR stack (batch is the fallback, and the
    # stack env can still set AGENT_STT_MODE=batch); preemptive generation stays off.
    assert env["AGENT_STT_MODE"] == "${AGENT_STT_MODE:-stream}"
    assert env["AGENT_ASR_STREAM_URL"] == "${AGENT_ASR_STREAM_URL:-ws://voice-stack-asr-stream:5001/v1/stream}"
    assert env["AGENT_ASR_STREAM_FINAL_TIMEOUT_S"] == "${AGENT_ASR_STREAM_FINAL_TIMEOUT_S:-2.0}"
    assert env["AGENT_PREEMPTIVE_GENERATION"] == "${AGENT_PREEMPTIVE_GENERATION:-false}"


def test_api_key_falls_back_to_the_settings_default_not_empty():
    # An empty bearer is not the same as "no key needed": keep the settings default when unset.
    assert _agent_environment()["AGENT_LLM_API_KEY"] == "${AGENT_LLM_API_KEY:-unused}"


def test_stream_url_points_at_the_streaming_asr_stack():
    """The agent's default stream URL must name the service and port the streaming stack really runs."""
    import re

    stream = yaml.safe_load((COMPOSE.parent / "asr-stream.yml").read_text(encoding="utf-8"))
    (service,) = stream["services"].values()
    url = _agent_environment()["AGENT_ASR_STREAM_URL"]
    host, port, path = re.fullmatch(r"\$\{AGENT_ASR_STREAM_URL:-ws://([^:]+):(\d+)(/.*)\}", url).groups()
    assert host == service["container_name"]
    assert port == service["environment"]["ASR_PORT"]
    assert path == "/v1/stream"


def test_deployed_brain_gets_one_session_per_call():
    """The deployed Hermes api_server merges header-less calls that open with the same words."""
    env = _agent_environment()
    assert env["AGENT_LLM_SESSION_HEADER"] == "${AGENT_LLM_SESSION_HEADER:-X-Hermes-Session-Id}"


def test_deployed_voices_are_overridable_with_the_chosen_defaults():
    env = _agent_environment()
    assert env["AGENT_VOICE_EN"] == "${AGENT_VOICE_EN:-bf_isabella}"
    assert env["AGENT_VOICE_PT"] == "${AGENT_VOICE_PT:-pf_dora}"
