# Voice stack modernization: giving the agent fleet a voice

**Status:** approved design, implementation in progress
**Date:** 2026-08-16
**Author:** the voice-stack maintainers
**Tracking:** voice-stack#6 · gate MR !48

> Facts below marked **verified** were measured on hardware, not read from a
> vendor page. Several vendor claims turned out to be wrong; see Corrections.

## Principle

**Voice is a transport, not an agent.**

The homelab already has brains: Claude Code agents
reachable over NATS, paid for by an existing subscription. They have no ears and no
mouth. This project adds those at the edge and changes the agents themselves by zero
lines.

Explicitly rejected: building a new end-to-end voice agent, introducing a second LLM,
or anything billed per token or per minute. Marginal cost of a conversation must be zero.

## What changes and what does not

| changes | unchanged |
|---|---|
| new `voice-gateway` service | agent code |
| TTS moves CPU → GPU, engine stays XTTS v2 | NATS DM grammar `agents.<self>.dm.<peer>.<topic>` |
| | OpenAI-compatible `/v1/audio/speech`, `/v1/audio/voices` |
| ASR engine faster-whisper → transcribe.cpp | **Bazarr endpoints** (hard requirement) |
| the maintainer's voice added as a cloned voice | `voice_speak` MCP tool and every current caller |

## Hard constraint: Bazarr must keep working

Bazarr is a production consumer of the ASR service. Not a nice-to-have.

**Status: both original risks resolved, one gate built.**

1. **Headerless raw PCM — non-issue (verified).** transcribe.cpp's `transcribe()` accepts
   `pcm: PCMLike` directly, which is *closer* to Bazarr's wire format than faster-whisper
   was. The PCM→WAV wrapping work originally scoped does not exist.
2. **Segment timestamps — available (verified).** `Segment(text, t0_ms, t1_ms, first_word,
   n_words, first_token, n_tokens)`. Confirmed both from the type and from a running model
   producing real millisecond values that agree with faster-whisper within ~100 ms.
3. **Form-encoded request shape** — `BazarrAsrRequest.as_form` and
   `BazarrDetectLanguageRequest` remain the wire contract.

### The gate (MR !48)

The 16 existing Bazarr tests **cannot catch a timestamp regression**. `conftest.py::asr_client`
replaces the engine with a `Mock` whose timings are hardcoded, so they test routing and the
SRT/VTT formatters, never the engine. Proven by mutation: zeroing `start`/`end` left all 16
green.

!48 adds an engine-contract test with no mocks, and — critically — that test is **proven able
to fail**, via `pytest.raises` against the same zeroed input. It also installs `espeak-ng` in
the CI image, because the fixture otherwise skips and the gate silently stops gating.

## Hard constraint: cloned voices never leave the homelab

**Decided, not open.** No cloned voice is sent outside the homelab, and no cloned voice is
used to speak to anyone other than the maintainer.

This rules out, without further discussion:

- exposing TTS publicly, or proxying it through any external service
- an agent replying in a cloned voice on an outbound message to a third party
- uploading reference clips or fine-tuned speaker weights anywhere off-network
- committing reference audio, speaker embeddings or trained speaker weights to this
  repository, which is public

The reasoning is not squeamishness. A good clone of a specific real person is mistakable for
that person by someone who knows them, and the usual mitigation is a watermark, which is
forensic rather than audible: it helps establish misuse after it has happened and prevents
nothing at the moment it matters. The only reliable control is to not let the audio leave.

**Where this is enforced today:** both services are LAN-only at the proxy
(`tts.local.` and `asr.local.` behind an internal whitelist, with no public route), and no
audio or speaker weights are tracked in git. Any future gateway inherits the same rule: a
voice surface may be reachable from inside the network only.

Test names and fixtures use generic speaker names. A real person's name in a public test is a
smaller leak than audio, but it is still a leak, and it costs nothing to avoid.

## Hardware and placement

| | |
|---|---|
| deploy target | the host where voice-stack already runs |
| GPU | 12 GB, compute capability 8.6 |
| currently used | ~4.5 GB (TEI embed/rerank/code, kb-clip, kb-face) |
| headroom | ~8 GB |
| model evaluation | a workstation with 128 GB unified memory. The production GPU is never touched. |

The larger GPU reports ~98% of its memory used, but that is a *claim*: vLLM runs with
`gpu_memory_utilization=0.9`. Reclaiming it has a real cost though —
its KV-cache concurrency headroom is already thin — so it stays untouched. Evaluating on the workstation
removes any need to trade.

## Components

### 1. `voice-gateway` (new)

Owns a voice exchange end to end. Stateless apart from a short per-exchange buffer.

```
audio in ─▶ VAD trim ─▶ ASR ─▶ NATS DM to agent ─▶ (wait) ─▶ NATS reply ─▶ TTS ─▶ audio out
```

**Audio never touches NATS.** Only transcript and reply text cross the bus. The audio is
produced and consumed on the same host; NATS is a message bus with a ~1 MB default payload,
not a media pipe; and raw household audio should not exist as a routable message.

**Two-speed replies.** The gateway speaks an immediate acknowledgement before the agent
answers. **Now justified by measurement:** ASR runs at RTF 0.006-0.020, so a 10-second
utterance transcribes in ~0.1 s. ASR is nowhere near the latency budget — the agent's tool
calls are. Perceived latency becomes TTS latency, which we control.

### 2. TTS: XTTS v2 retained. Chatterbox evaluated and REJECTED.

**Decision: XTTS v2 stays.** Chatterbox was evaluated and rejected on quality by the
maintainer's ear, which is the only evaluation method that has been reliable on this
project. The record below is kept so the option is not re-proposed without new evidence.

The mistake worth remembering: Chatterbox was verified to *function*, and
"zero-shot from 8 seconds" was then allowed to read as a quality argument. It is a
convenience argument. Data efficiency and ceiling are unrelated, and XTTS fine-tuned on
hours of audio beat it — which is what should have been expected.

#### What was evaluated

MIT licensed. **23 languages including `pt` (verified from the shipped package — the vendor
page claims 25).** Zero-shot cloning via `audio_prompt_path`, explicit `language_id`, plus
`exaggeration` and `cfg_weight` expressiveness controls XTTS did not offer. Output carries
Resemble's PerTh watermark.

**Verified working:** produced 4.84 s of Portuguese cloned from an existing 8-second reference clip from a
real speaker, **zero-shot, no fine-tune**. XTTS needed a 6.5-hour run over 960 clips for
the comparable result.

**Required pins — the image breaks without them:**

| package | pin | why |
|---|---|---|
| `resemble-perth` | explicit | not pulled by `chatterbox-tts`; hard-required in `ChatterboxMultilingualTTS.__init__` |
| `setuptools` | `<81` | ≥81 removed `pkg_resources`, which `perth` imports |
| `torch` | CUDA build | see CPU caveat below |

Unpinned, a fresh build installs `setuptools 84`, `perth` swallows the resulting `ImportError`
and sets `PerthImplicitWatermarker = None`, and Chatterbox raises `TypeError: 'NoneType'
object is not callable` — an error naming none of the four packages involved. Same failure
class as #4 and !46.

**CPU caveat:** stock Chatterbox **cannot load on a host without an NVIDIA GPU.** Its
checkpoints are serialized on CUDA and it calls `torch.load` without `map_location`, so
passing `device=cpu` does not help. Fixable with a loader shim. Does not affect the 3060, but
it does affect the deferred room device, which was scoped CPU-only on the strength of the
*ASR* benchmark. That reasoning does not transfer to TTS.

Ruled out, recorded so they are not re-litigated:

| candidate | why not |
|---|---|
| Voxtral TTS 4B (Mistral) | needs ≥16 GB VRAM; also CC BY-NC |
| CosyVoice 2 | no Portuguese (zh/en/ja/ko only) |
| Kokoro 82M | no voice cloning; viable only as a fallback utility voice |

### 3. ASR: transcribe.cpp

MIT, streaming and batch, 16 model families / 60+ variants behind one interface. CUDA, Vulkan,
Metal, plus a tinyBLAS CPU path. Python bindings. Chosen for optionality as much as speed:
models swap without rewriting the service.

**Verified on the evaluation workstation**, whisper-tiny and base, warm-up before timing:

| model | audio | cpu RTF | vulkan RTF | winner |
|---|---|---|---|---|
| tiny | 9.3 s | 0.011 | 0.007 | Vulkan 1.7x |
| tiny | 92.9 s | 0.006 | 0.006 | tied |
| base | 9.3 s | 0.020 | 0.011 | Vulkan 1.9x |
| base | 92.9 s | 0.011 | 0.006 | Vulkan 1.8x |

Two conclusions. **Everything is 50-150x faster than real time on both backends**, so model
choice is driven by accuracy, not speed. And **Vulkan works on gfx1151 with zero ROCm
involvement**, auto-detected, which makes that workstation a genuinely usable ASR bench.

### 4. Voice assets

- **the maintainer** — to be recorded. See below.
- **Reference speaker** — exported and verified in object storage, checksummed on
  upload and re-verified by streaming every object back down.

## How models are selected

Loss curves and leaderboards were wrong every time on earlier voice-cloning work; the maintainer's ear was right
every time.

- **ASR:** The maintainer reads a fixed held-out script. WER per candidate on *their own* voice, pt-BR and en
  separately. Parakeet TDT v3 was trained on *European* Portuguese while this household speaks
  Brazilian — that gets measured, not assumed.
- **TTS:** blind A/B on the maintainer's ear.
- All evaluation on the evaluation workstation. Only the finalist reaches the 3060.

## Recording the maintainer's voice

**Hardware:** sE Electronics NEOM USB — 16 mm back-electret condenser, cardioid, 24-bit, up to
192 kHz, 20 Hz-20 kHz, separate Mic Level and Playback Level with zero-latency monitoring.
Attached to the Mac that hosts the Loki agent, which is where the session happens.

**Why record at all, given XTTS already clones from a short clip.** Not feasibility. XTTS v2
conditions on a reference clip of a few seconds, so a usable voice needs no session at all.
The session is about **quality headroom**, and the reason it is worth a session is a measured
one: on the existing corpus the *reference clip* moved timbre roughly twice as much as
fine-tuning did (~22 Hz of pitch shift versus ~13 Hz). The reference is the highest-leverage
variable in the whole pipeline, which makes a clean 24-bit condenser take the cheapest
available quality win. A held-out set makes that claim testable rather than asserted.

Validate the take before it becomes a voice: `scripts/validate_voice_reference.py` checks
per-channel signal, level drift, clipping, rate and depth. Reference defects are silent
otherwise, and they are cheap here and expensive after training.

**Capture settings**

- **48 kHz / 24-bit WAV, mono, no processing.** Not 192 kHz: the capsule stops at 20 kHz, the
  models work at 22.05-24 kHz, and 48 gives a clean downsample with a quarter of the data.
  Max is not best.
- No noise reduction, compression or gate. Processing can be added later, never removed.
- **Set Mic Level once and do not touch it.** Analog gain drift between clips wobbles the
  speaker embedding.
- Cardioid means position is a variable: fixed distance, fixed angle, mark the desk.
- Varied delivery — statements, questions, excitement, low energy, thinking aloud. A monotone
  corpus yields a monotone clone.
- A held-out set that is never used as a reference, so quality claims are testable.

**Assertions the capture script must make** (fail the recording, never the training run):

- **channel count, and that a mono downmix is not a dead channel.** The device reports
  `Input Channels: 2` despite a cardioid capsule. A silent second channel would pass silently
  into everything downstream. Same shape as an earlier corpus bug, where `prepare_dataset.py`
  stamped `speaker_name` from a CLI flag and never checked the audio, so 22% of the corpus was
  the wrong speaker. Credit: claude-code-loki.
- sample rate and bit depth are what we asked for
- no clipping
- RMS consistency across clips

## Failure modes

| failure | behaviour |
|---|---|
| agent does not reply | speak a timeout message; never hang silently |
| ASR returns empty or garbage | ask for a repeat rather than DM the agent nonsense |
| TTS fails | fall back to a text reply on the same surface |
| GPU OOM | **fail loud.** No silent CPU fallback — that is how the current stack became slow |
| NATS publish rejected | `nats_publish` reports success even on async ACL rejection; verify delivery |

## Testing

- **!48 engine-contract gate** — no mocks, proven able to fail.
- Voice-list endpoint tests from !45 continue to pass.
- Latency measured as **time to first audio**, not total generation.
- ASR WER harness over the held-out script, per model, per language.
- Build-time assertions via `scripts/assert_runtime.py` (!46).

## Corrections to earlier versions of this document

Recorded rather than silently edited, because each was a claim made confidently and then
falsified by measurement:

1. **"25 languages"** — the shipped package exports 23. Vendor page overstates.
2. **"CPU beat the iGPU by 3.5x"** — measurement artifact. The Vulkan run had no warm-up and
   paid shader compilation inside the timed region. With warm-up, Vulkan wins at `base` and
   ties at `tiny`.
3. **"Room device is CPU-only, no GPU needed"** — true for ASR, false for TTS. Stock
   Chatterbox will not load without CUDA.
4. **"PCM→WAV wrapping needed for Bazarr"** — not needed; transcribe.cpp takes PCM directly.

## Deferred, by design

Browser full duplex, room device with wake word, SIP/phone, barge-in. The gateway is built so
each is a transport adapter rather than a rewrite. Barge-in needs a media layer (Pipecat or
LiveKit Agents); that decision is deliberately not made here.

## Open questions

1. Does `claude-telegram-bridge` route voice notes to agents today, or can the gateway
   intercept them? Decides whether the gateway or the agent owns the voice exchange.
2. XTTS v2 real-time factor on the GPU, measured as time to FIRST audio rather than total
   generation. The CPU figure is a floor and must not be quoted as a deployment number. This
   sets whether the two-speed acknowledgement in the gateway is still needed once TTS moves
   off CPU.
