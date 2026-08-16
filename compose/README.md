# `compose/` — the author's homelab deployment

**These files are not a template. They will not work for you unmodified.**

They are the real Portainer stack definitions running voice-stack in the
author's homelab, kept in the repository as a worked example of a production
deployment rather than as something to copy.

## What is in them that will not work for you

| reference | where | what it is |
|---|---|---|
| `registry.local.vitorgarbim.me/voice-stack:latest` | `asr.yml:15`, `tts.yml:15` | a private container registry |
| `192.168.50.4` | `asr.yml:53`, `tts.yml:50` | an internal DNS server, hardcoded |
| `192.168.50.15`, hostname `openwebui` | `asr.yml:2,11`, `tts.yml:2,11` | the host these run on |
| `https://gitlab.vitorgarbim.me/...` | `asr.yml:9`, `tts.yml:9` | the private GitLab this mirrors from |
| named volumes | `asr.yml:73-75`, `tts.yml:70-74` | Portainer-managed volumes |
| `${DNS_RESOLVER}` | `gateway.yml` | an internal resolver, supplied as a stack variable rather than hardcoded |
| external networks | `gateway.yml` | joins the ASR and TTS stacks' own networks, which must already exist |

Those are private-network addresses (RFC 1918) and an internal hostname. They
are not secrets and they are not reachable from the internet, but they are also
not useful to you — they will simply fail to resolve.

### A note on `gateway.yml`

It joins `voice-stack-asr_default` and `voice-stack-tts_default` as **external** networks, so it
only comes up if those stacks already exist under exactly those names. That is the price of
addressing them by container name instead of by address: names survive a host renumbering and
addresses do not.

It also needs a NATS server and an agent listening on the other end. If you have neither, you do
not want this file at all — the ASR and TTS services stand alone without it.

## What you probably want instead

The generic, self-contained deployment lives at the repository root:

```bash
cp .env.example .env
docker compose up -d
```

That builds the image locally, needs no private registry, and serves ASR on
`localhost:5001` and TTS on `localhost:5002`.

`docker compose` reads `.env` from the project directory automatically, and every
setting written as `${VAR:-default}` in `docker-compose.yml` takes its value from
there. Settings not listed in that file do not reach the containers at all — if you
need one that is missing, add it to `docker-compose.yml` as `${VAR:-default}`
rather than only to `.env`. See the main [README](../README.md) for the full quick
start.

## Why keep these here at all

A sanitised example config shows you the happy path. A real one shows you what
somebody actually had to do: the resource limits they landed on, the healthcheck
that catches the failure mode they hit, the comments explaining why a setting is
what it is. If you are deploying this seriously, the diff between these files and
`docker-compose.yml` is the interesting part.
