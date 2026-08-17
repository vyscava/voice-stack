"""Silence must never become a question.

Whisper does not report "I heard nothing" -- it invents text, and reliably
invents the SAME text. Both cases below were observed on the live deployment:

    1.5 s of digital silence -> "You"
    near-silence             -> "Thank you for watching!"

The second reached a peer agent as a genuine question on 2026-08-17 and was
caught only because that agent happened to recognise the artifact.
"""

from __future__ import annotations

import pytest

from gateway.speech_gate import (
    AudioProbe,
    is_known_hallucination,
    looks_like_speech,
    probe_audio,
)


class TestHallucinationBackstop:
    @pytest.mark.parametrize(
        "text",
        [
            "You",
            "you",
            "Thank you for watching!",
            "thanks for watching",
            "Thank you for watching.",
            "  Thanks!  ",
            "Bye bye",
            "Please subscribe",
        ],
    )
    def test_known_artifacts_are_caught(self, text):
        assert is_known_hallucination(text), f"{text!r} should be recognised as a silence artifact"

    @pytest.mark.parametrize(
        "text",
        [
            "What is the capital of France?",
            "thank you for fixing the deploy",  # contains the phrase, is not the phrase
            "you should restart the service",
            "subscribe me to the newsletter please",
            "Está tudo bem?",
        ],
    )
    def test_real_speech_is_not_caught(self, text):
        """The denylist must be exact-match after normalising.

        A substring check here would eat 'thank you for fixing the deploy',
        which is a real thing a person says.
        """
        assert not is_known_hallucination(text)

    def test_punctuation_and_case_do_not_evade_it(self):
        assert is_known_hallucination("THANK YOU FOR WATCHING!!!")
        assert is_known_hallucination("thank   you  for   watching")


class TestLengthGuardCannotDoThisJob:
    def test_the_hallucination_is_longer_than_any_sane_length_limit(self):
        """Why the audio gate exists at all.

        'Thank you for watching!' is 23 characters. A length threshold high
        enough to block it also blocks 'What time is it?' (16). The two are not
        separable by length, which is why the guard has to measure AUDIO.
        """
        artifact = "Thank you for watching!"
        real_question = "What time is it?"
        assert len(artifact) > len(real_question)
        assert not is_known_hallucination(real_question)
        assert is_known_hallucination(artifact)


class TestAudioGate:
    def test_too_short_is_rejected(self, monkeypatch):
        monkeypatch.setattr(
            "gateway.speech_gate.probe_audio",
            lambda *a, **k: AudioProbe(duration_s=0.1, mean_dbfs=-20.0, max_dbfs=-5.0),
        )
        assert not looks_like_speech(b"x")

    def test_too_quiet_is_rejected(self, monkeypatch):
        """Digital silence measures about -91 dBFS."""
        monkeypatch.setattr(
            "gateway.speech_gate.probe_audio",
            lambda *a, **k: AudioProbe(duration_s=2.0, mean_dbfs=-91.0, max_dbfs=-90.0),
        )
        assert not looks_like_speech(b"x")

    def test_normal_speech_passes(self, monkeypatch):
        monkeypatch.setattr(
            "gateway.speech_gate.probe_audio",
            lambda *a, **k: AudioProbe(duration_s=3.8, mean_dbfs=-24.0, max_dbfs=-3.0),
        )
        assert looks_like_speech(b"x")

    def test_quiet_but_real_speech_passes(self, monkeypatch):
        """A softly spoken question must not be discarded.

        Dropping a real request with no explanation is a worse failure than
        letting one dud through, so the floor sits well below conversational
        level.
        """
        monkeypatch.setattr(
            "gateway.speech_gate.probe_audio",
            lambda *a, **k: AudioProbe(duration_s=1.2, mean_dbfs=-40.0, max_dbfs=-18.0),
        )
        assert looks_like_speech(b"x")

    def test_fails_open_when_unmeasurable(self, monkeypatch):
        """ffmpeg failing is not evidence of silence.

        If the probe tells us nothing, allow the clip: the cost of a wrong
        reject is the user's actual question vanishing without explanation.
        """
        monkeypatch.setattr(
            "gateway.speech_gate.probe_audio",
            lambda *a, **k: AudioProbe(None, None, None),
        )
        assert looks_like_speech(b"x")

    def test_probe_never_raises_when_ffmpeg_is_missing(self, monkeypatch):
        def boom(*a, **k):
            raise FileNotFoundError("ffmpeg")

        monkeypatch.setattr("subprocess.run", boom)
        probe = probe_audio(b"x")
        assert not probe.measured
