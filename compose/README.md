# `compose/deploy/`: a real production deployment

**These files are not a template. They will not work for you unmodified.**

They are the stack definitions of a real voice-stack deployment, deployed by
CI (`.gitlab/ci/deploy.yml`) to a 4-core CPU host. They are kept in the
repository as a worked example of a production deployment.

## What is in them that will not work for you

| reference | where | what it is |
|---|---|---|
| `${VOICE_STACK_IMAGE}` | all three | the image, pinned by digest by the deploy job from a private registry |
| `ASR_CPU_THREADS: "4"`, `TTS_KOKORO_THREADS: "4"` | `asr.yml`, `tts.yml` | sized to the host's 4 CPUs; set yours to the cores the container gets |
| `${GATEWAY_*}` | `gateway.yml` | bus identity, NATS URL and timeouts, supplied by the deployer from its secret store |
| external networks | `gateway.yml` | joins the ASR and TTS stacks' own networks, which must already exist |

### A note on `gateway.yml`

It joins `voice-stack-asr_default` and `voice-stack-tts_default` as
**external** networks, so the ASR and TTS stacks must already exist under
exactly those project names. That is the price of addressing them by container
name instead of by address: names survive a host renumbering and addresses do
not.

It also needs a NATS server and an agent listening on the other end. If you
have neither, you do not want this file at all: the ASR and TTS services stand
alone without it.

## What you probably want instead

The generic, self-contained deployment lives at the repository root:

```bash
cp .env.example .env
docker compose up -d
```

That builds the image locally, needs no private registry, and serves ASR on
`localhost:5001` and TTS on `localhost:5002`. See the main
[README](../README.md) for the full quick start.

## Why keep these here at all

A sanitised example shows you the happy path. A real one shows you what
somebody actually had to do: the thread counts and memory limits they landed
on, the healthcheck timing for a model that loads at startup, the comments
explaining why a setting is what it is.
