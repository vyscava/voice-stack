"""
Streaming ASR backed by NVIDIA Nemotron 3.5 ASR streaming through Hugging Face Transformers.

This follows the streaming recipe on the model card
(https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b, "Streaming
transcription"): ``model.generate`` is given the first chunk's processor
outputs plus a GENERATOR of per-chunk ``input_features``, and decoded text
comes back through a streamer. The card's generator walks a finished file;
here it walks a buffer that grows as WebSocket frames arrive, blocking on a
queue until the next chunk is complete. The chunk arithmetic
(``num_samples_first_audio_chunk``, ``num_mel_frames_*``, the
``hop_length``/``n_fft`` start offset) is the card's, unchanged.

One utterance = one ``generate`` call on one daemon thread. ``finish()`` pads
silence so the last real chunk is emitted, ends the generator and joins.

Language: the prompt is "auto", and in that mode the model emits a special
``<xx-YY>`` token after each sentence's final punctuation. The streamer drops
special tokens from the text; the collector reads them first and keeps the
last one, so the language costs no extra decode.

Concurrency: the model's ``generate`` keeps per-call state ON THE MODEL
INSTANCE (``_prompt_ids``, a wrapped ``get_audio_features``, ``_streaming``,
``_stream_exhausted``, ...; see transformers' ``generation_nemotron3_5_asr`` and
``generation_nemotron_asr_streaming``) and deletes it when the call returns.
Utterances can overlap briefly (a session still winding down, for example
after a cancelled close, while the next one starts), and two ``generate``
calls on one instance would corrupt each other. Every session therefore runs
on ``copy.copy(model)``: a new instance ``__dict__``
for that per-call state, the same parameter and submodule objects (no extra
weights in memory).

transformers and torch are imported lazily: importing this module costs
nothing, and the images that do not ship the ``stream`` extra still import
the ASR app.
"""

from __future__ import annotations

import copy
import queue
import threading
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from asr.stream.base import STREAM_SAMPLE_RATE, AudioArray, StreamConfig, parse_language_tag
from core.logging import logger_asr as logger

# Silence appended on finish so the final chunk (current frames + lookahead)
# fills up and is decoded. At least this much, more for long lookaheads.
FLUSH_PAD_SECONDS = 0.6
# How long finish()/close() wait for the generate thread before giving up.
JOIN_TIMEOUT_SECONDS = 30.0
# The streaming prompt. "auto" lets one connection carry any supported language.
STREAM_LANGUAGE = "auto"

_END = None  # queue sentinel: no more audio for this utterance


def _make_collector(tokenizer: Any, on_text: Callable[[str], None], on_language: Callable[[str], None]) -> Any:
    """A transformers TextStreamer that hands finalized text to ``on_text`` and language tags to ``on_language``.

    TextIteratorStreamer (the card's choice) is this same class with a queue
    in ``on_finalized_text``; a callback avoids a second consumer thread. The
    text is decoded with special tokens skipped, so the ``<xx-YY>`` tags are
    picked out of the raw token ids in ``put`` before that happens.
    """
    from transformers import TextStreamer

    class _Collector(TextStreamer):
        def put(self, value: Any) -> None:
            ids = value[0] if len(value.shape) > 1 else value
            for token in tokenizer.convert_ids_to_tokens([int(i) for i in ids.tolist()]):
                language = parse_language_tag(str(token))
                if language is not None:
                    on_language(language)
            super().put(value)

        def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
            if text:
                on_text(text)

    return _Collector(tokenizer, skip_prompt=False, skip_special_tokens=True)


class NemotronStreamSession:
    """One utterance. ``feed``/``finish``/``close`` are called from executor threads, one at a time."""

    def __init__(self, model: Any, processor: Any) -> None:
        self._model = model
        self._processor = processor
        self._sr = STREAM_SAMPLE_RATE
        self._queue: queue.Queue[AudioArray | None] = queue.Queue()
        self._audio: AudioArray = np.zeros(0, dtype=np.float32)
        self._received = 0  # real (unpadded) samples fed
        self._text: list[str] = []
        self._language: str | None = None
        self._text_lock = threading.Lock()
        self._error: BaseException | None = None
        self._ended = False
        self._thread = threading.Thread(target=self._run, name="nemotron-stream", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ API
    def feed(self, samples: AudioArray) -> str:
        self._raise_if_failed()
        if self._ended or samples.size == 0:
            return self.text
        self._received += int(samples.size)
        self._queue.put(samples.astype(np.float32, copy=False))
        return self.text

    def finish(self) -> str:
        self._raise_if_failed()
        if not self._ended:
            self._queue.put(np.zeros(self._pad_samples(), dtype=np.float32))
            self._end()
        self._thread.join(JOIN_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            raise TimeoutError("streaming decode did not finish in time")
        self._raise_if_failed()
        return self.text

    def close(self) -> None:
        self._end()
        self._thread.join(JOIN_TIMEOUT_SECONDS)

    @property
    def language(self) -> str | None:
        with self._text_lock:
            return self._language

    @property
    def text(self) -> str:
        """The hypothesis with whitespace collapsed.

        In language="auto" mode the model emits an ``<xx-YY>`` tag (a special
        token, preceded by a word-boundary token) after every sentence. The
        streamer strips the tag but keeps the boundary, which left a double
        space mid-transcript ("seven thirty.  Could you"), seen on the first
        real-model run.
        """
        with self._text_lock:
            return " ".join("".join(self._text).split())

    # ------------------------------------------------------------ internals
    def _end(self) -> None:
        if not self._ended:
            self._ended = True
            self._queue.put(_END)

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"streaming decode failed: {self._error}") from self._error

    def _pad_samples(self) -> int:
        """Silence needed so every real sample lands in a yielded chunk.

        The generator only yields a chunk once audio extends PAST its end (the
        card's ``end_idx < audio.shape[0]``), so pad by at least one full
        chunk plus the STFT window, never less than FLUSH_PAD_SECONDS, and
        enough to complete the first chunk of a very short utterance.
        """
        p = self._processor
        pad = max(
            int(FLUSH_PAD_SECONDS * self._sr), int(p.num_samples_per_audio_chunk) + int(p.feature_extractor.n_fft)
        )
        shortfall = int(p.num_samples_first_audio_chunk) - self._received
        return pad + max(0, shortfall)

    def _pull(self) -> bool:
        """Block for the next piece of audio. False once the utterance has ended."""
        item = self._queue.get()
        if item is None:
            return False
        self._audio = np.concatenate([self._audio, item])
        return True

    def _chunk_inputs(self, audio: AudioArray, *, first: bool) -> Any:
        model = self._model
        inputs = self._processor(
            audio,
            sampling_rate=self._sr,
            is_streaming=True,
            is_first_audio_chunk=first,
            language=STREAM_LANGUAGE,
            return_tensors="pt",
        )
        return inputs.to(model.device, dtype=model.dtype)

    def _features(self, first_inputs: Any) -> Iterator[Any]:
        p = self._processor
        yield first_inputs.input_features[:, : p.num_mel_frames_first_audio_chunk, :]

        mel_frame_idx = int(p.num_mel_frames_first_audio_chunk)
        hop_length = int(p.feature_extractor.hop_length)
        n_fft = int(p.feature_extractor.n_fft)
        start_idx = mel_frame_idx * hop_length - n_fft // 2
        while True:
            end_idx = start_idx + int(p.num_samples_per_audio_chunk)
            while end_idx >= self._audio.shape[0]:
                if not self._pull():
                    return
            yield self._chunk_inputs(self._audio[start_idx:end_idx], first=False).input_features
            mel_frame_idx += int(p.num_mel_frames_per_audio_chunk)
            start_idx = mel_frame_idx * hop_length - n_fft // 2

    def _append_text(self, piece: str) -> None:
        with self._text_lock:
            self._text.append(piece)

    def _set_language(self, language: str) -> None:
        with self._text_lock:
            self._language = language  # the last tag of the utterance wins

    def _run(self) -> None:
        try:
            first_len = int(self._processor.num_samples_first_audio_chunk)
            while self._audio.shape[0] < first_len:
                if not self._pull():
                    return  # closed before a single chunk arrived: nothing to decode
            first_inputs = self._chunk_inputs(self._audio[:first_len], first=True)
            streamer = _make_collector(self._processor.tokenizer, self._append_text, self._set_language)
            self._model.generate(
                **{**first_inputs, "input_features": self._features(first_inputs), "streamer": streamer}
            )
        except Exception as exc:  # surfaced to the caller by _raise_if_failed
            logger.exception("Streaming decode failed")
            self._error = exc


class NemotronStreamEngine:
    """Loads the model once; hands out one session per utterance."""

    def __init__(self, config: StreamConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForRNNT, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Streaming ASR needs transformers and torch (the 'stream' extra), which this install "
                "does not include. Install voice-stack[stream] or leave ASR_STREAM_ENABLED off."
            ) from exc

        if config.threads > 0:
            torch.set_num_threads(config.threads)
        dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
        load_kwargs = self.model_load_kwargs(config, dtype)

        logger.info(
            f"Loading streaming ASR model {config.model} (dtype={config.dtype}, "
            f"device_map={load_kwargs.get('device_map', 'none (cpu)')}, "
            f"lookahead={config.lookahead}, threads={config.threads})"
        )
        self._processor = AutoProcessor.from_pretrained(config.model)
        self._processor.set_num_lookahead_tokens(config.lookahead)
        self._model = AutoModelForRNNT.from_pretrained(config.model, **load_kwargs)
        self._model.eval()
        logger.info(f"Streaming ASR model loaded; latency {self._processor.streaming_latency_ms} ms per chunk")

    @staticmethod
    def model_load_kwargs(config: StreamConfig, dtype: Any) -> dict[str, Any]:
        """from_pretrained() arguments. On CPU, NO device_map: the model loads on the CPU anyway,
        and any device_map (even "cpu") makes transformers require the accelerate package."""
        kwargs: dict[str, Any] = {"dtype": dtype}
        if config.device != "cpu":
            kwargs["device_map"] = "auto"
        return kwargs

    def _isolated_model(self) -> Any:
        """A shallow copy that shares weights but keeps generate()'s per-call attributes to itself."""
        return copy.copy(self._model)

    def new_session(self) -> NemotronStreamSession:
        return NemotronStreamSession(self._isolated_model(), self._processor)
