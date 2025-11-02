# WhisperEngine – Parameter Cheat Sheet

## Overview

This module wraps the Faster-Whisper ASR engine, providing an in-memory
transcription interface for float32 waveforms or raw bytes.

The Whisper ASR (Automatic Speech Recognition) pipeline converts speech
(audio) into text using a Transformer model. Several decoding options control
accuracy, speed, and output granularity. The parameters below are the core
knobs exposed to the engine.

## Core Parameters

1) language : `str | None`
    Language code (ISO-639-1 or similar), e.g. "en", "pt", "fr", "zh".
    - If provided, decoding is restricted to that language’s vocabulary.
    - If None, the model automatically detects the spoken language.
    - Examples: "en" → English, "pt" → Portuguese, "zh" → Chinese.

2) task : `str`
    Either "transcribe" or "translate".
    - "transcribe" → Converts speech to same-language text (default).
    - "translate" → Converts non-English speech into English text.
    Example: Portuguese audio
        task="transcribe" → "Olá, tudo bem?"
        task="translate"  → "Hello, how are you?"

3) beam_size : `int`
    Width of beam search (number of hypotheses tracked during decoding).
    - Larger values improve accuracy but slow down inference.
    - Typical values: 1 (fast) → 5 (balanced) → 10 (max accuracy).
    - When `beam_size > 1`, the model performs deterministic search.
    - When `beam_size == 1`, the model may use sampling (temperature/best_of).

4) temperature : `float`
    Sampling temperature controlling output randomness.
    - `0.0` → deterministic (best for ASR; default).
    - `>0.0` → increases diversity (used for translation/creative tasks).
    - Only effective when `beam_size == 1` (sampling mode).

5) best_of : `int`
    Number of parallel sampling candidates when beam_size == 1.
    - The model runs several stochastic passes and keeps the most likely.
    - Ignored when beam_size > 1 (beam search mode).
    - Example: best_of=3 + temperature>0 → 3 samples, keep best.

6) word_timestamps : `bool`
    If True, compute and return per-word timestamps.
    - Increases processing time slightly.
    - Useful for subtitles or alignment tools.


## Optional / Advanced Parameters

> Supported internally by Faster-Whisper**

1) vad : `bool`
    Voice Activity Detection pre-processing toggle.
    - When True, trims long silences before sending to the model.
    - Typically implemented upstream (e.g., Silero VAD).

2) patience : `float`
    Beam search patience factor controlling early stopping.
    - `1.0` → default (stops when no better beam found).
    - `>1.0` → allows beams to continue growing, slightly improving accuracy.

3) length_penalty : `float`
    Penalty applied to longer sequences.
    - `>1.0` discourages long transcriptions; `<1.0` encourages them.
    - Rarely needed; default = `1.0`.

4) repetition_penalty : `float`
    Penalty factor to discourage repetitive outputs.
    - `>1.0` reduces “word loops” in long-form transcription.

5) no_repeat_ngram_size : `int`
    Prevents repeating any n-gram of this size.
    - Example: 3 → bans repeating the same three-word sequences.

6) suppress_blank : `bool`
    Whether to suppress blank tokens between words.
    - Default True for cleaner outputs.

7) suppress_tokens : `list[int] | None`
    List of token IDs to never generate (e.g., special tokens like <|music|>).

8) sampling_topk : `int`
    Limits sampling to the top-K most probable tokens.
    - Only relevant in sampling mode (beam_size == 1, temperature > 0).

9) sampling_temperature : `float`
    Internal alias of `temperature`; both control sampling randomness.

10) max_length : `int`
    Maximum number of tokens per segment (default ≈448).

11) return_scores : `bool`
    If True, returns per-segment log-probability scores.

12) return_no_speech_prob : `bool`
    If True, includes model’s “no speech detected” probability.

## Usage Guidelines

- Recommended defaults for transcription:

    ```py
    language=None           # auto-detect
    task="transcribe"       # same-language transcription
    beam_size=5             # balanced speed/accuracy
    temperature=0.0         # deterministic decoding
    best_of=1               # unused when beam_size>1
    word_timestamps=False   # faster if not needed
    ```

- Recommended defaults for translation:

    ```py
    language=None
    task="translate"
    beam_size=5
    temperature=0.0
    best_of=1
    ```

- Performance notes:

  - Resample all audio to 16 kHz mono float32 before calling the model.
  - Keep `beam_size` modest (≤5) for near real-time workloads.
  - Use caching if multiple calls reuse the same audio or parameters.
  - Disable `word_timestamps` unless alignment is required.


## Example:

```py
result = engine.transcribe_waveform(
    audio_f32=audio,              # np.ndarray (float32)
    sr=16000,                     # 16 kHz mono
    request_language="en",
    task="transcribe",
    beam_size=5,
    temperature=0.0,
    best_of=1,
    word_timestamps=False,
)

print(result["text"])
# "Hello world!"
```

## Decoding Mode Matrix (beam_size × temperature × best_of)

Whisper uses two broad decoding modes:

1) Beam search (deterministic, accuracy-focused)
   - Active when: beam_size > 1
   - Ignores: temperature, best_of
   - Tunables: beam_size, patience, length_penalty, repetition_penalty, no_repeat_ngram_size

2) Sampling (stochastic, diversity-focused)
   - Active when: beam_size == 1
   - Uses: temperature (>0 enables randomness), best_of (parallel samples)
   - Tunables: sampling_topk, sampling_temperature (= temperature), repetition_penalty, no_repeat_ngram_size

### Quick rules:

- If you care about stable accuracy → prefer beam_size=5, temperature=0.0, best_of=1.
- If you need multiple creative candidates → set beam_size=1, temperature>0, best_of>1.
- Setting temperature>0 while beam_size>1 has no effect (still beam search).

### Truth table

|beam_size | temperature | best_of | Mode        | Notes
|----------|-------------|---------|-------------|-------------------------------
|> 1       | any         | any     | Beam search | temperature/best_of ignored
|= 1       | 0.0         | 1       | Greedy      | fastest deterministic sampling
|= 1       | 0.0         | >1      | Greedy (xN) | multiple greedy passes; keep best
|= 1       | > 0.0       | 1       | Sampling    | single stochastic sample
|= 1       | > 0.0       | >1      | Sampling    | run N samples; keep bes (accuracy↑)

### Practical presets

- Balanced accuracy (default recommendation)
  > task="transcribe", beam_size=5, temperature=0.0, best_of=1

- Faster, okay accuracy
  > task="transcribe", beam_size=3, temperature=0.0, best_of=1

- Creative / multiple candidates (summaries, translations, drafts)
  > task="translate", beam_size=1, temperature=0.7, best_of=3

- Word-level timestamps (heavier)
  > word_timestamps=True  # adds compute; keep off unless required

### Performance tips

- Beam size drives latency roughly linearly; start at 3–5.
- For real-time-ish pipelines, disable word_timestamps and keep beam_size ≤ 3.
- Temperature>0 only matters when beam_size == 1 (sampling mode).
- best_of>1 multiplies sampling cost (runs N candidates); set modestly (2–3).
