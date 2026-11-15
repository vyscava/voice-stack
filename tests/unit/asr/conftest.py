"""Fixtures for the ASR unit tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from asr.stream import factory as stream_factory
from tests.unit.asr.stream_fakes import FakeEngine, config


@pytest.fixture
def fake_engine(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeEngine]:
    """An enabled /v1/stream backed by a FakeEngine instead of the real model."""
    engine = FakeEngine()
    monkeypatch.setattr(stream_factory, "get_stream_config", lambda: config(enabled=True))
    stream_factory.set_stream_engine(engine)
    # The existing tests are written for one stream per process; tests of the
    # multi-session limit set their own.
    stream_factory.set_max_sessions(1)
    yield engine
    stream_factory.set_max_sessions(None)
    stream_factory.set_stream_engine(None)
