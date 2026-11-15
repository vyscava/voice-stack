"""What the agent says on its own while the brain prepares a reply.

Two stages, both only while the brain has said nothing yet:

* an **acknowledgement** ("Okay.", "Certo.") shortly after the caller stops,
  so the caller hears a reply in about half a second instead of waiting for
  the brain's first words;
* a **thinking line** ("Hmm, let me think.") when the wait gets long, for a
  tool call or a hand-off to a deeper model.

Phrases follow the language ASR detected for the turn. Acknowledgements rotate
so consecutive turns do not start with the same word.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator


class Filler(str):
    """A line the agent spoke itself, not the brain. A str, so TTS takes it as is."""

    kind: str = "filler"


class Ack(Filler):
    kind = "ack"


class Think(Filler):
    kind = "think"


def parse_phrases(raw: str) -> tuple[str, ...]:
    """``"Okay.|Sure."`` -> ``("Okay.", "Sure.")``; blanks dropped."""
    return tuple(p.strip() for p in raw.split("|") if p.strip())


def _lang_key(lang: str | None) -> str:
    return "pt" if (lang or "").strip().lower().startswith("pt") else "en"


class FillerPicker:
    """Per-call phrase choice: rotating acknowledgements, one thinking line per language."""

    def __init__(self, *, acks: dict[str, tuple[str, ...]], think: dict[str, str]) -> None:
        self._acks = {k: v for k, v in acks.items() if v}
        self._think = {k: v for k, v in think.items() if v}
        self._cycles: dict[str, Iterator[str]] = {k: itertools.cycle(v) for k, v in self._acks.items()}

    def ack(self, lang: str | None) -> Ack | None:
        cycle = self._cycles.get(_lang_key(lang)) or self._cycles.get("en")
        return Ack(next(cycle)) if cycle else None

    def think(self, lang: str | None) -> Think | None:
        text = self._think.get(_lang_key(lang)) or self._think.get("en")
        return Think(text) if text else None

    def phrases(self) -> tuple[str, ...]:
        """Every line this picker can say, longest first (for stripping history)."""
        every = {p for v in self._acks.values() for p in v} | set(self._think.values())
        return tuple(sorted(every, key=len, reverse=True))
