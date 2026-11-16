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


class Progress(Filler):
    kind = "progress"


Phrases = str | tuple[str, ...]


def _phrases(value: Phrases) -> tuple[str, ...]:
    return parse_phrases(value) if isinstance(value, str) else tuple(p for p in value if p)


class FillerPicker:
    """Per-call phrase choice; each kind rotates through its phrases per language.

    A setting with one phrase behaves as before; ``|``-separated phrases take
    turns, so a long call does not repeat the same line.
    """

    def __init__(
        self,
        *,
        acks: dict[str, Phrases],
        think: dict[str, Phrases],
        progress: dict[str, Phrases] | None = None,
    ) -> None:
        self._kinds: dict[type[Filler], dict[str, tuple[str, ...]]] = {
            Ack: {k: _phrases(v) for k, v in acks.items() if _phrases(v)},
            Think: {k: _phrases(v) for k, v in think.items() if _phrases(v)},
            Progress: {k: _phrases(v) for k, v in (progress or {}).items() if _phrases(v)},
        }
        self._cycles: dict[tuple[type[Filler], str], Iterator[str]] = {
            (kind, lang): itertools.cycle(phrases)
            for kind, by_lang in self._kinds.items()
            for lang, phrases in by_lang.items()
        }

    def _pick(self, kind: type[Filler], lang: str | None) -> Filler | None:
        cycle = self._cycles.get((kind, _lang_key(lang))) or self._cycles.get((kind, "en"))
        return kind(next(cycle)) if cycle else None

    def ack(self, lang: str | None) -> Filler | None:
        return self._pick(Ack, lang)

    def think(self, lang: str | None) -> Filler | None:
        return self._pick(Think, lang)

    def progress(self, lang: str | None) -> Filler | None:
        return self._pick(Progress, lang)

    def phrases(self) -> tuple[str, ...]:
        """Every line this picker can say, longest first (for stripping history)."""
        every = {p for by_lang in self._kinds.values() for v in by_lang.values() for p in v}
        return tuple(sorted(every, key=len, reverse=True))
