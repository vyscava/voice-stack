"""voice_agent_errors_total: LiveKit's session errors counted by component and recoverability.

A TTS that answers 503 on every request leaves the caller in silence while the
agent stays up; the error event is the only signal of it.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from livekit.agents.llm.llm import LLMError
from livekit.agents.stt.stt import STTError
from livekit.agents.tts.tts import TTSError
from prometheus_client import REGISTRY

from voice_agent import metrics


def _count(component: str, recoverable: str) -> float:
    return (
        REGISTRY.get_sample_value("voice_agent_errors_total", {"component": component, "recoverable": recoverable})
        or 0.0
    )


def _event(cls: type, recoverable: bool) -> SimpleNamespace:
    return SimpleNamespace(
        error=cls(timestamp=time.time(), label="x", error=RuntimeError("503"), recoverable=recoverable)
    )


def test_errors_are_counted_by_component_and_recoverability() -> None:
    before = {k: _count(*k) for k in [("tts", "false"), ("stt", "true"), ("brain", "false"), ("other", "unknown")]}
    metrics.record_error(_event(TTSError, recoverable=False))
    metrics.record_error(_event(STTError, recoverable=True))
    metrics.record_error(_event(LLMError, recoverable=False))
    metrics.record_error(SimpleNamespace(error=RuntimeError("something else")))
    assert _count("tts", "false") == before[("tts", "false")] + 1
    assert _count("stt", "true") == before[("stt", "true")] + 1
    assert _count("brain", "false") == before[("brain", "false")] + 1
    assert _count("other", "unknown") == before[("other", "unknown")] + 1


def test_the_worker_counts_session_errors() -> None:
    """The worker subscribes the counter to the session's error event."""
    import inspect

    from voice_agent import worker

    source = inspect.getsource(worker)
    assert '@session.on("error")' in source
    assert "metrics.record_error(" in source
