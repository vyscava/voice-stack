"""Fillers: per-language phrases, rotating acks, and the lines a turn can open with."""

from __future__ import annotations

from voice_agent.fillers import Ack, FillerPicker, Think, parse_phrases


def _picker(**kw) -> FillerPicker:
    defaults = {
        "acks": {"en": ("Okay.", "Sure."), "pt": ("Tá.", "Certo.")},
        "think": {"en": "Hmm, let me think.", "pt": "Hum, deixa eu pensar."},
    }
    defaults.update(kw)
    return FillerPicker(**defaults)


def test_parse_phrases_drops_blanks():
    assert parse_phrases(" Okay. | |Sure.|") == ("Okay.", "Sure.")
    assert parse_phrases("") == ()


def test_acks_rotate_and_follow_the_language():
    p = _picker()
    assert [p.ack("en"), p.ack("en"), p.ack("en")] == ["Okay.", "Sure.", "Okay."]
    assert p.ack("pt-BR") == "Tá."
    assert isinstance(p.ack("en"), Ack)


def test_unknown_language_uses_english():
    p = _picker()
    assert p.ack("fr") == "Okay."
    assert p.ack(None) == "Sure."
    assert p.think("de") == "Hmm, let me think."


def test_think_follows_the_language():
    p = _picker()
    assert p.think("pt") == "Hum, deixa eu pensar."
    assert isinstance(p.think("en"), Think)


def test_missing_portuguese_falls_back_to_english():
    p = _picker(acks={"en": ("Okay.",), "pt": ()}, think={"en": "Hmm.", "pt": ""})
    assert p.ack("pt") == "Okay."
    assert p.think("pt") == "Hmm."


def test_nothing_configured_says_nothing():
    p = _picker(acks={"en": (), "pt": ()}, think={"en": "", "pt": ""})
    assert p.ack("en") is None
    assert p.think("en") is None
    assert p.phrases() == ()


def test_phrases_lists_every_line_longest_first():
    phrases = _picker().phrases()
    assert set(phrases) == {"Okay.", "Sure.", "Tá.", "Certo.", "Hmm, let me think.", "Hum, deixa eu pensar."}
    assert [len(x) for x in phrases] == sorted((len(x) for x in phrases), reverse=True)


def test_progress_and_think_lines_rotate_per_language_and_are_stripped_from_history():
    picker = FillerPicker(
        acks={"en": ("Okay.",)},
        think={"en": "Hmm.|Let me see."},
        progress={"en": "Still checking.|Almost there.", "pt": "Ainda verificando."},
    )
    assert [picker.progress("en"), picker.progress("en"), picker.progress("en")] == [
        "Still checking.",
        "Almost there.",
        "Still checking.",
    ]
    assert picker.progress("pt-BR") == "Ainda verificando."
    assert picker.progress("pt").kind == "progress"
    assert [picker.think("en"), picker.think("en")] == ["Hmm.", "Let me see."]
    assert {"Still checking.", "Almost there.", "Ainda verificando.", "Let me see."} <= set(picker.phrases())


def test_no_progress_phrases_means_no_line():
    assert FillerPicker(acks={}, think={}).progress("en") is None
