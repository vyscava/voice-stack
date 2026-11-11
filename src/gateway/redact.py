"""Keep credentials embedded in a URL out of logs and HTTP responses.

``GATEWAY_NATS_URL`` carries the NATS password (``nats://user:pw@host:4222``).
Anything that prints that URL, or an exception that may quote it, goes through
here first.

Design rule: over-masking is fine, under-masking is not. The userinfo is
located at the LAST ``@`` of the whole remainder (a password may itself contain
``/``, ``?``, ``#``, ``,`` or ``@``), and anything that does not parse cleanly
as ``scheme://[userinfo@]host[:port][/path]`` is replaced by ``***``.
"""

from __future__ import annotations

import re

_SCHEME = r"(?:nats|tls|wss?)://"
_SCHEME_RE = re.compile(_SCHEME, re.IGNORECASE)
# Split a server list only where the NEXT item starts with a scheme, so a comma
# inside a password never splits it.
_LIST_SPLIT_RE = re.compile(r",(?=\s*" + _SCHEME + ")", re.IGNORECASE)
_HOSTPORT_RE = re.compile(r"(?:\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9._-]+)(?::[0-9]{1,5})?(?:/.*)?", re.DOTALL)
_URL_IN_TEXT_RE = re.compile(_SCHEME + r"\S*", re.IGNORECASE)


def _redact_one(item: str) -> str:
    item = item.strip()
    match = _SCHEME_RE.match(item)
    prefix = match.group(0) if match else ""
    rest = item[len(prefix) :]
    at = rest.rfind("@")
    userinfo, remainder = (rest[:at], rest[at + 1 :]) if at >= 0 else (None, rest)
    if not _HOSTPORT_RE.fullmatch(remainder):
        return "***"
    if userinfo is None:
        return f"{prefix}{remainder}"
    user, colon, _ = userinfo.partition(":")
    return f"{prefix}{user}:***@{remainder}" if colon else f"{prefix}***@{remainder}"


def redact_url(servers: str) -> str:
    """Return ``servers`` (one URL or several, comma-separated) with passwords masked.

    ``user:password@`` becomes ``user:***@``. A user-only URL becomes ``***@``
    because nats-py reads a lone userinfo as an auth TOKEN. A URL without
    credentials is unchanged. Anything unparseable becomes ``***``.
    """
    return ",".join(_redact_one(part) for part in _LIST_SPLIT_RE.split(str(servers)))


def scrub_urls(text: object) -> str:
    """Mask credentials in every ``scheme://...@...`` found inside free text.

    For exception messages: nats-py is unpinned, so a future version may quote
    the server URL in an error.
    """
    return _URL_IN_TEXT_RE.sub(lambda m: redact_url(m.group(0)) if "@" in m.group(0) else m.group(0), str(text))
