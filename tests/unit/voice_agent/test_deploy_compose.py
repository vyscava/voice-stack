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


def test_outbound_call_settings_pass_through():
    env = _agent_environment()
    expected = {
        "AGENT_CALL_TOKEN": "${AGENT_CALL_TOKEN:-}",
        "AGENT_CALL_TRUNK_NAME": "${AGENT_CALL_TRUNK_NAME:-asterisk-out}",
        "AGENT_CALL_TO": "${AGENT_CALL_TO:-}",
        "AGENT_CALL_RING_TIMEOUT_S": "${AGENT_CALL_RING_TIMEOUT_S:-40}",
        "AGENT_CALL_MAX_DURATION_S": "${AGENT_CALL_MAX_DURATION_S:-1800}",
        "AGENT_CALL_QUIET_HOURS": "${AGENT_CALL_QUIET_HOURS:-}",
        "AGENT_CALL_TIMEZONE": "${AGENT_CALL_TIMEZONE:-UTC}",
        "AGENT_CALL_MAX_PER_DAY": "${AGENT_CALL_MAX_PER_DAY:-3}",
        "AGENT_CALL_MIN_INTERVAL_MIN": "${AGENT_CALL_MIN_INTERVAL_MIN:-30}",
    }
    assert set(expected) <= set(Settings.model_fields)
    assert {name: env[name] for name in expected} == expected


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


def test_every_agent_asr_setting_reaches_the_container():
    """A setting with no compose line is silently stuck at its default in the deploy."""
    env = _agent_environment()
    missing = sorted(n for n in Settings.model_fields if n.startswith("AGENT_ASR_") and n not in env)
    assert not missing, f"not forwarded by compose/deploy/agent.yml: {missing}"
    assert env["AGENT_ASR_PRIMARY_URL"] == "${AGENT_ASR_PRIMARY_URL:-}"  # empty = today's single batch STT


def test_the_deployed_brain_timeout_allows_a_tool_using_brain():
    # A tool-using agent brain can take ~20 s before its first words: 20 s cut it to the apology.
    assert _agent_environment()["AGENT_BRAIN_TIMEOUT_S"] == "${AGENT_BRAIN_TIMEOUT_S:-45}"


def test_the_deploy_retires_the_ticks_and_delays_the_ack():
    """The deployed brain speaks its own preamble (~2.2 s): canned filler only covers a slower brain."""
    from core.settings import Settings

    env = _agent_environment()
    assert env["AGENT_ACK_AFTER_S"] == "${AGENT_ACK_AFTER_S:-2.5}"
    assert env["AGENT_TICKS"] == "${AGENT_TICKS:-false}"
    # Both are real settings, so the values reach the agent.
    assert {"AGENT_ACK_AFTER_S", "AGENT_TICKS"} <= set(Settings.model_fields)


def test_the_deploy_ticks_through_long_tool_runs():
    """Retired before the first words, the ticks stay on for a long tool run (an agent call)."""
    from core.settings import Settings

    env = _agent_environment()
    assert env["AGENT_TOOL_TICKS"] == "${AGENT_TOOL_TICKS:-true}"
    assert env["AGENT_TOOL_TICKS_DELAY_S"] == "${AGENT_TOOL_TICKS_DELAY_S:-2.0}"
    assert {"AGENT_TOOL_TICKS", "AGENT_TOOL_TICKS_DELAY_S"} <= set(Settings.model_fields)
