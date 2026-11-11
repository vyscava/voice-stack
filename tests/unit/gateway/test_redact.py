"""Credentials embedded in a URL must never reach a log, an error or a response.

Over-masking is acceptable; under-masking is not. The password may contain any
printable character, including the URL delimiters / ? # , @ and :.
"""

from __future__ import annotations

import random

import pytest

from gateway.redact import redact_url, scrub_urls

SECRET = "S3cRt-sentinel"

# The reviewer's cases: the secret sits after a / ? # or , that a naive netloc split cuts at.
REVIEWER_CASES = [
    ("nats://S3c#Rt@h:4222", "S3c#Rt"),
    ("tls://u:#S3cRt#@h:4222", "#S3cRt#"),
    ("nats://u:xQ7/zK8wV@h:4222", "xQ7/zK8wV"),
    ("nats://u:xQ7?zK8wV@h:4222", "xQ7?zK8wV"),
    ("nats://U#sEr9@h:4222", "U#sEr9"),
    ("nats://u:,S3cRt,@h:4222", "S3cRt"),
]


@pytest.mark.parametrize(("url", "secret"), REVIEWER_CASES)
def test_reviewer_cases_do_not_leak(url, secret):
    out = redact_url(url)
    assert secret not in out
    assert "@h:4222" in out


def test_exact_shapes():
    assert redact_url("tls://u:#S3cRt#@h:4222") == "tls://u:***@h:4222"
    assert redact_url("nats://u:xQ7/zK8wV@h:4222") == "nats://u:***@h:4222"
    assert redact_url("nats://U#sEr9@h:4222") == "nats://***@h:4222"
    assert redact_url("nats://u:,S3cRt,@h:4222") == "nats://u:***@h:4222"
    assert redact_url("nats://u:pw@h:4222/some/path") == "nats://u:***@h:4222/some/path"
    assert redact_url("nats://h:4222") == "nats://h:4222"
    assert redact_url("nats://u:pw@[::1]:4222") == "nats://u:***@[::1]:4222"


def test_list_splits_only_before_a_scheme():
    assert redact_url("nats://u:a@h1:4222,tls://u:b@h2:4222") == "nats://u:***@h1:4222,tls://u:***@h2:4222"
    assert redact_url("nats://u:a@h1:4222, wss://u:b@h2:4222") == "nats://u:***@h1:4222,wss://u:***@h2:4222"
    # a comma that is not followed by a scheme belongs to the password
    assert SECRET not in redact_url(f"nats://u:x,{SECRET},y@h:4222,nats://h2:4222")


@pytest.mark.parametrize(
    "bad",
    [f"nats://u:{SECRET}@", f"nats://u:{SECRET}@h:port", f"nats://u:{SECRET}@[::1", f"nats://u:{SECRET}@h:4222 x", ""],
)
def test_malformed_fails_closed(bad):
    assert redact_url(bad) == "***"


def test_fuzz_full_printable_ascii_never_leaks():
    rng = random.Random(20261111)
    alphabet = [chr(c) for c in range(32, 127)]
    for _ in range(20000):
        user = "".join(rng.choice([c for c in alphabet if c != ":"]) for _ in range(rng.randint(1, 8)))
        secret = "".join(rng.choice(alphabet) for _ in range(rng.randint(8, 16)))
        scheme = rng.choice(["nats", "tls", "ws", "wss", "NATS", "Tls"])
        tail = rng.choice(["", "/p", "/p/q"])
        url = f"{scheme}://{user}:{secret}@nats.example.org:4222{tail}"
        out = redact_url(url)
        assert secret not in out, (url, out)
        # user-only form: the whole userinfo is a token
        token = secret.replace(":", "x")  # a ':' would make it user:password, not a token
        out2 = redact_url(f"{scheme}://{token}@nats.example.org:4222{tail}")
        assert token not in out2, (token, out2)
        # inside a server list
        out3 = redact_url(f"{url},nats://other:{secret}@b:4222")
        assert secret not in out3, (url, out3)


def test_scrub_urls_masks_urls_inside_free_text():
    text = f"connect failed for nats://voice-gateway:{SECRET}@nats.local:4222 (timeout)"
    out = scrub_urls(text)
    assert SECRET not in out
    assert "connect failed for" in out
    assert scrub_urls("plain error, nothing to hide") == "plain error, nothing to hide"
    assert scrub_urls("could not reach nats://h:4222") == "could not reach nats://h:4222"
    assert SECRET not in scrub_urls(OSError(f"tls://u:{SECRET}@h:4222"))
