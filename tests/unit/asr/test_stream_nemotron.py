"""Unit tests for the Nemotron streaming session/engine against a fake processor and model.

The fakes reproduce the parts of the Transformers API the code relies on (the
processor's chunk-size properties, BatchFeature-like inputs, a generate() that
pulls mel chunks from a generator and keeps per-call state on the instance).
No model is downloaded. transformers itself is only needed for TextStreamer.
"""

from __future__ import annotations

import contextlib
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("transformers")

from asr.stream.nemotron import NemotronStreamEngine, NemotronStreamSession  # noqa: E402

SR = 16000
HOP, N_FFT, WIN = 160, 512, 400
LOOKAHEAD, SUBSAMPLING = 3, 8
FIRST_FRAMES = 1 + SUBSAMPLING * LOOKAHEAD  # 25
NEXT_FRAMES = SUBSAMPLING * (LOOKAHEAD + 1)  # 32


class FakeInputs(dict):  # type: ignore[type-arg]
    """Stands in for transformers' BatchFeature: a dict with attribute access and .to()."""

    def __getattr__(self, name: str) -> Any:
        return self[name]

    def to(self, *args: Any, **kwargs: Any) -> FakeInputs:
        return self


class FakeTokenizer:
    def decode(self, ids: list[int], **kwargs: Any) -> str:
        return " ".join(f"w{i}" for i in ids)


class FakeProcessor:
    """Same chunk arithmetic as Nemotron3_5AsrProcessor for lookahead 3."""

    feature_extractor = SimpleNamespace(hop_length=HOP, n_fft=N_FFT, win_length=WIN, sampling_rate=SR)
    tokenizer = FakeTokenizer()
    num_mel_frames_first_audio_chunk = FIRST_FRAMES
    num_mel_frames_per_audio_chunk = NEXT_FRAMES
    num_samples_first_audio_chunk = (FIRST_FRAMES - 1) * HOP + WIN // 2
    num_samples_per_audio_chunk = NEXT_FRAMES * HOP + WIN

    def __init__(self) -> None:
        self.streaming_calls: list[int] = []

    def __call__(self, audio: Any, *, sampling_rate: int, is_streaming: bool = False, **kwargs: Any) -> FakeInputs:
        assert sampling_rate == SR
        if not is_streaming:
            return FakeInputs(input_features=np.zeros((1, 10, 4)), prompt_ids=np.array([101]))
        self.streaming_calls.append(len(audio))
        if kwargs["is_first_audio_chunk"]:
            assert len(audio) == self.num_samples_first_audio_chunk
            frames = FIRST_FRAMES + 2  # the card slices the first chunk's features down itself
        else:
            assert len(audio) == self.num_samples_per_audio_chunk
            frames = NEXT_FRAMES
        return FakeInputs(input_features=np.zeros((1, frames, 4)), num_lookahead_tokens=LOOKAHEAD, prompt_ids=[101])

    def decode(self, sequences: Any, skip_special_tokens: bool = True) -> list[str]:
        return ["Olá. <pt-BR>" if not skip_special_tokens else "Olá."]


class FakeModel:
    """generate() keeps per-call state on self and deletes it after, like the real mixin does."""

    device = "cpu"
    dtype = None

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.first_chunk_seen = threading.Event()
        self.chunk_frames: list[int] = []

    def generate(self, **kwargs: Any) -> Any:
        features = kwargs["input_features"]
        self._streaming = not isinstance(features, np.ndarray)
        try:
            if not self._streaming:
                return SimpleNamespace(sequences=[[1, 2]])
            assert kwargs["num_lookahead_tokens"] == LOOKAHEAD
            streamer = kwargs["streamer"]
            for i, chunk in enumerate(features):
                if self.fail:
                    raise RuntimeError("boom")
                assert self._streaming is True, "another generate() call clobbered this one's state"
                self.chunk_frames.append(chunk.shape[1])
                streamer.put(np.array([i]))
                self.first_chunk_seen.set()
            streamer.end()
            return SimpleNamespace(sequences=[[0]])
        finally:
            del self._streaming  # AttributeError if a concurrent call already deleted it


def _audio(seconds: float) -> Any:
    return np.full(int(seconds * SR), 0.1, dtype=np.float32)


def _chunk_end(k: int) -> int:
    """End sample of chunk k (k >= 1) under the model card's indexing."""
    return (FIRST_FRAMES + (k - 1) * NEXT_FRAMES) * HOP - N_FFT // 2 + FakeProcessor.num_samples_per_audio_chunk


@pytest.mark.unit
@pytest.mark.asr
def test_session_streams_exact_chunks_and_flush_covers_all_audio() -> None:
    model, processor = FakeModel(), FakeProcessor()
    session = NemotronStreamSession(model, processor)
    for _ in range(10):
        session.feed(_audio(0.1))
    text = session.finish()

    assert model.chunk_frames[0] == FIRST_FRAMES
    assert set(model.chunk_frames[1:]) == {NEXT_FRAMES}
    # The silence pad pushed the last chunk past every real sample.
    assert _chunk_end(len(model.chunk_frames) - 1) >= SR
    assert text == " ".join(f"w{i}" for i in range(len(model.chunk_frames)))


@pytest.mark.unit
@pytest.mark.asr
def test_session_shorter_than_first_chunk_still_decodes() -> None:
    model = FakeModel()
    session = NemotronStreamSession(model, FakeProcessor())
    session.feed(_audio(0.05))  # 800 samples < the 4040-sample first chunk
    assert session.finish().startswith("w0")
    assert model.chunk_frames[0] == FIRST_FRAMES


@pytest.mark.unit
@pytest.mark.asr
def test_session_close_before_audio_never_generates() -> None:
    model = FakeModel()
    session = NemotronStreamSession(model, FakeProcessor())
    session.close()
    session.close()  # idempotent
    assert model.chunk_frames == []


@pytest.mark.unit
@pytest.mark.asr
def test_session_surfaces_a_decode_failure() -> None:
    session = NemotronStreamSession(FakeModel(fail=True), FakeProcessor())
    session.feed(_audio(0.5))
    with pytest.raises(RuntimeError, match="streaming decode failed"):
        session.finish()


def _engine(model: FakeModel) -> NemotronStreamEngine:
    engine = NemotronStreamEngine.__new__(NemotronStreamEngine)  # skip the model download
    engine._model = model
    engine._processor = FakeProcessor()
    engine._torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
    return engine


@pytest.mark.unit
@pytest.mark.asr
def test_language_detection_during_a_stream_does_not_corrupt_it() -> None:
    """The real generate() keeps per-call state on the model; overlapping calls must not share it."""
    model = FakeModel()
    engine = _engine(model)
    session = engine.new_session()
    session.feed(_audio(0.5))
    assert model.first_chunk_seen.wait(5), "the stream never started decoding"

    # A whole batch generate() runs and cleans up while the streaming one is mid-utterance.
    assert engine.detect_language(_audio(2.0)) == "pt"

    session.feed(_audio(1.0))
    assert session.finish().startswith("w0 w1")
