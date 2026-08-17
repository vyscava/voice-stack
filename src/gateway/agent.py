"""Reaching an agent over NATS, and getting an answer back.

Why request/reply rather than publish
-------------------------------------
``publish`` reports success even when the server asynchronously rejects the
message on an ACL violation. A gateway built on it would look healthy while
every utterance vanished. ``request`` gives a real reply or a real timeout, and
the reply rides an auto-negotiated inbox so there is no reply-subject to guess
wrong.

Subject grammar
---------------
An agent may publish only under its own prefix, so a DM from us to a peer is::

    agents.<self>.dm.<peer>.<topic>

read as *from* dot dm dot *to*. Getting this backwards is not an error you see:
the server rejects it asynchronously and the publish still reports success. That
is why ``self_name`` must match the bus identity exactly, with no shortening.

Text only
---------
``ask`` takes and returns ``str``. Audio is produced and consumed on the same
host and never crosses the bus: NATS has a ~1 MB default payload and is a
message bus rather than a media pipe, and raw household audio should not exist
as a routable message.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from core.logging import logger_gateway as logger
from gateway import metrics


class NatsAgent:
    """Asks a peer agent a question and waits for its reply."""

    def __init__(
        self,
        *,
        servers: str,
        self_name: str,
        peer: str,
        topic: str = "voice",
        connect_timeout_s: float = 2.0,
        max_reconnect_attempts: int = 3,
    ) -> None:
        self._servers = servers
        self._self_name = self_name
        self._peer = peer
        self._topic = topic
        self._connect_timeout_s = connect_timeout_s
        self._max_reconnect_attempts = max_reconnect_attempts
        self._nc: Any = None
        self._lock = asyncio.Lock()

    @property
    def subject(self) -> str:
        return f"agents.{self._self_name}.dm.{self._peer}.{self._topic}"

    async def connect(self) -> None:
        """Connect once and reuse. Safe to call repeatedly.

        Fails FAST rather than retrying: the caller turns a dead bus into
        speech, and a client quietly reconnecting for a minute would defeat
        that by making the exchange hang instead.
        """
        async with self._lock:
            if self._nc is not None and self._nc.is_connected:
                return
            import nats

            self._nc = await nats.connect(
                self._servers,
                connect_timeout=self._connect_timeout_s,
                allow_reconnect=True,
                max_reconnect_attempts=self._max_reconnect_attempts,
            )
            logger.info(f"NATS connected: {self._servers}, publishing on {self.subject}")

    async def close(self) -> None:
        if self._nc is not None:
            try:
                await self._nc.drain()
            except Exception as exc:
                logger.debug(f"NATS drain skipped: {exc}")
            self._nc = None

    async def ask(self, text: str, *, timeout_s: float) -> str:
        """Send `text` to the peer and return its reply.

        Raises TimeoutError if the peer does not answer in time. The caller
        turns that into speech rather than a hang.
        """
        # `text` carries the transcript, and `source` says where it came from.
        #
        # Without the marker an ASR transcript is indistinguishable from typed
        # input, so a peer treats a MISHEARING as a genuine question and answers
        # it confidently -- and the gateway then speaks that answer aloud in a
        # human voice, which is a far more convincing way to be wrong than text.
        # Told the input was heard rather than read, a peer can say "I heard X"
        # when X looks implausible. Asked for by claude-code-channel, who is on
        # the receiving end of exactly this failure.
        # `text` is the cleaned form a peer should ACT on. `transcript` is what
        # ASR actually returned, verbatim, and must never be rewritten.
        #
        # They are equal today apart from whitespace, which prompted a fair
        # question from claude-code-channel: is `transcript` load-bearing or
        # documentation? Answer: it is the RAW channel, and it earns its keep
        # the moment anything sits between ASR and the DM -- a normaliser, a
        # spell-corrector, or a fast model rewriting the query in the two-brain
        # shape. At that point a peer needs to see what was HEARD, not what was
        # tidied, in order to say "I heard X" about a mishearing.
        #
        # So: whatever is added between here and ASR, `transcript` stays raw.
        # Collapsing these into one field would quietly remove that guarantee.
        payload = json.dumps(
            {
                "text": text.strip(),
                "transcript": text,
                "source": "asr",
                "from": self._self_name,
                "to": self._peer,
            }
        ).encode()

        async def _connect_and_request() -> bytes:
            await self.connect()
            response = await self._nc.request(self.subject, payload, timeout=timeout_s)
            return response.data

        # The CONNECT is inside the bound, not just the request. A client that
        # is busy reconnecting to a dead server would otherwise hang the
        # exchange for as long as it kept trying, and the caller would hear
        # nothing at all -- the one outcome this gateway must never produce.
        # Timed HERE rather than in the endpoint, because this is the only place
        # that knows where the agent call starts and stops. The agent is the
        # dominant term in an exchange -- measured at 19-20 s for a trivial
        # question against ~4 s for ASR and ~4 s for TTS -- so without this the
        # histograms cannot tell you WHERE the time went.
        started = time.monotonic()
        try:
            data = await asyncio.wait_for(_connect_and_request(), timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            metrics.AGENT_DURATION.observe(time.monotonic() - started)
            raise TimeoutError(f"agent '{self._peer}' did not reply within {timeout_s}s") from exc
        metrics.AGENT_DURATION.observe(time.monotonic() - started)

        return _extract_reply(data)


def _extract_reply(raw: bytes) -> str:
    """Read the peer's reply, whether it answered with an envelope or bare text.

    Peers differ: some reply with the standard DM envelope as JSON, some with a
    plain string. Accepting both here is cheaper than requiring every agent in
    the fleet to change, and a reply we cannot parse is treated as empty so the
    caller speaks an error instead of reading JSON aloud.
    """
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("agent reply was not valid UTF-8")
        return ""

    stripped = decoded.strip()
    if not stripped.startswith("{"):
        return decoded

    try:
        body = json.loads(stripped)
    except json.JSONDecodeError:
        return decoded

    if not isinstance(body, dict):
        return decoded

    for key in ("text", "reply", "message", "body"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value

    logger.warning(f"agent reply JSON had no text field; keys were {sorted(body)}")
    return ""
