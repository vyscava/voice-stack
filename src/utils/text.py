"""
Text utilities shared by TTS/ASR:
- Unicode normalization
- conservative sentence splitting
- word-wrapped chunking with terminal punctuation
"""

from __future__ import annotations

import re
import unicodedata

# sentence split on [.?!…] + whitespace
_SENT_SPLIT = re.compile(r"(?<=[\.\?\!…])\s+")
_WS = re.compile(r"\s+")
# control chars that tend to break tokenizers/kernels
_BAD_CTRL = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F]")

# common odd glyphs → simpler equivalents
SUBS = {
    "—": "-",
    "–": "-",
    "―": "-",
    "…": "...",
    "\u00a0": " ",  # nbsp → space
    "\u2060": "",  # word-joiner → drop
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
}


def normalize_text(s: str) -> str:
    """
    NFKC normalize + replace odd punctuation + drop control chars + collapse whitespace.
    Helps avoid tokenizer blowups and bizarre grapheme sequences.
    """
    if not s:
        return s
    s = unicodedata.normalize("NFKC", s)
    for k, v in SUBS.items():
        s = s.replace(k, v)
    s = _BAD_CTRL.sub("", s)
    s = _WS.sub(" ", s).strip()
    return s


def ensure_terminal_punct(s: str) -> str:
    """Ensure a chunk ends in . ? ! to keep prosody sane in TTS."""
    return s if s and s[-1] in ".?!" else (s + ".")


def split_sentences(text: str) -> list[str]:
    """Conservative sentence split on [.?!…] + whitespace, post normalization."""
    text = normalize_text(text)
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts or [text]


def wrap_words(s: str, max_chars: int) -> list[str]:
    """Word-wrap a long sentence into <= max_chars pieces without breaking words."""
    if len(s) <= max_chars:
        return [s]
    out, cur, n = [], [], 0
    for w in s.split(" "):
        need = len(w) + (1 if n else 0)
        if n + need > max_chars and cur:
            out.append(" ".join(cur))
            cur, n = [w], len(w)
        else:
            cur.append(w)
            n += need
    if cur:
        out.append(" ".join(cur))
    return out


def build_safe_chunks(text: str, max_chars: int) -> list[str]:
    """
    Build TTS-safe chunks:
      - split to sentences
      - word-wrap sentences larger than max_chars
      - ensure terminal punctuation
    """
    chunks: list[str] = []
    for sent in split_sentences(text):
        if len(sent) <= max_chars:
            chunks.append(ensure_terminal_punct(sent))
        else:
            for sub in wrap_words(sent, max_chars):
                chunks.append(ensure_terminal_punct(sub))
    return chunks
