"""Custom voice cloning, local checkpoints, and unknown-voice handling.

Every test here fails against the previous behaviour:

* voices in TTS_VOICES_DIR were never loaded -- the setting was read into
  `self.voices_dir` and used nowhere, so the documented "drop a .wav in
  voices/" feature did not exist.
* an unrecognised voice was silently replaced with the first builtin speaker,
  so a typo returned HTTP 200 and audio in the wrong voice. The API layer's
  KeyError -> 404 VoiceNotFoundError branch was unreachable.
* TTS_MODEL could only name a Coqui model id, so a fine-tuned checkpoint could
  not be served at all.

The engine is exercised without loading a real model: TTSCoqui.__init__ calls
_load_model, so tests build the object with __new__ and drive the individual
methods. That keeps them runnable in CI, where torch and the XTTS weights are
absent -- the same reason this class of bug survived the existing suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _make_engine(tmp_path: Path, *, builtin: list[str] | None = None, voices_dir: Path | None = None):
    """Build a TTSCoqui without loading a model."""
    from tts.engine.coqui import TTSCoqui

    eng = TTSCoqui.__new__(TTSCoqui)
    eng.voices_dir = str(voices_dir if voices_dir is not None else tmp_path / "voices")
    eng.voice_to_preset = {name: name for name in (builtin or [])}
    eng.voice_to_wavs = {}
    return eng


def _wav(path: Path) -> Path:
    """Minimal real .wav so discovery is not fooled by an empty file."""
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x00" * 2205)
    return path


class TestCustomVoiceDiscovery:
    def test_flat_wav_becomes_a_voice(self, tmp_path):
        vd = tmp_path / "voices"
        _wav(vd / "alice.wav")
        eng = _make_engine(tmp_path, voices_dir=vd)

        eng._load_custom_voices()

        assert "alice" in eng.voice_to_wavs
        assert eng.voice_to_wavs["alice"] == [str(vd / "alice.wav")]

    def test_directory_of_wavs_becomes_one_voice(self, tmp_path):
        vd = tmp_path / "voices"
        _wav(vd / "bob" / "ref1.wav")
        _wav(vd / "bob" / "ref2.wav")
        eng = _make_engine(tmp_path, voices_dir=vd)

        eng._load_custom_voices()

        assert sorted(eng.voice_to_wavs) == ["bob"]
        assert len(eng.voice_to_wavs["bob"]) == 2

    def test_missing_voices_dir_is_not_an_error(self, tmp_path):
        eng = _make_engine(tmp_path, voices_dir=tmp_path / "nope")

        eng._load_custom_voices()

        assert eng.voice_to_wavs == {}

    def test_non_wav_files_ignored(self, tmp_path):
        vd = tmp_path / "voices"
        vd.mkdir()
        (vd / "notes.txt").write_text("not audio")
        (vd / "sample.mp3").write_bytes(b"\x00")
        eng = _make_engine(tmp_path, voices_dir=vd)

        eng._load_custom_voices()

        assert eng.voice_to_wavs == {}

    def test_custom_voices_appear_in_list_voices(self, tmp_path):
        vd = tmp_path / "voices"
        _wav(vd / "alice.wav")
        eng = _make_engine(tmp_path, builtin=["Claribel Dervla"], voices_dir=vd)
        eng._load_custom_voices()

        ids = [v.id for v in eng.list_voices().data]

        assert "Claribel Dervla" in ids
        assert "alice" in ids

    def test_custom_voice_not_duplicated_when_shadowing_builtin(self, tmp_path):
        vd = tmp_path / "voices"
        _wav(vd / "Claribel Dervla.wav")
        eng = _make_engine(tmp_path, builtin=["Claribel Dervla"], voices_dir=vd)
        eng._load_custom_voices()

        ids = [v.id for v in eng.list_voices().data]

        assert ids.count("Claribel Dervla") == 1


class TestLocalCheckpointResolution:
    def test_coqui_model_id_is_not_treated_as_a_path(self):
        from tts.engine.coqui import TTSCoqui

        assert TTSCoqui._resolve_local_model("tts_models/multilingual/multi-dataset/xtts_v2") is None

    def test_nonexistent_path_falls_through_to_model_manager(self):
        from tts.engine.coqui import TTSCoqui

        assert TTSCoqui._resolve_local_model("/no/such/model/dir") is None

    def test_directory_resolves_to_model_and_config(self, tmp_path):
        from tts.engine.coqui import TTSCoqui

        (tmp_path / "model.pth").write_bytes(b"\x00")
        (tmp_path / "config.json").write_text("{}")

        resolved = TTSCoqui._resolve_local_model(str(tmp_path))

        assert resolved == (tmp_path / "model.pth", tmp_path / "config.json")

    def test_checkpoint_file_resolves_to_sibling_config(self, tmp_path):
        from tts.engine.coqui import TTSCoqui

        ckpt = tmp_path / "best_model.pth"
        ckpt.write_bytes(b"\x00")
        (tmp_path / "config.json").write_text("{}")

        resolved = TTSCoqui._resolve_local_model(str(ckpt))

        assert resolved == (ckpt, tmp_path / "config.json")

    def test_missing_config_raises_rather_than_failing_deep_in_the_loader(self, tmp_path):
        from tts.engine.coqui import TTSCoqui
        from tts.exceptions import ModelLoadError

        (tmp_path / "model.pth").write_bytes(b"\x00")

        with pytest.raises(ModelLoadError):
            TTSCoqui._resolve_local_model(str(tmp_path))

    def test_directory_without_checkpoint_raises(self, tmp_path):
        from tts.engine.coqui import TTSCoqui
        from tts.exceptions import ModelLoadError

        (tmp_path / "config.json").write_text("{}")

        with pytest.raises(ModelLoadError):
            TTSCoqui._resolve_local_model(str(tmp_path))


class TestUnknownVoiceIsRejected:
    """The regression that matters: an unknown voice must NOT be substituted.

    These call the engine's own `resolve_voice` rather than a copy of its
    logic. A re-implemented check would keep passing while the production path
    regressed, which is the failure mode that let this bug ship in the first
    place.

    Asserting on the raised KeyError rather than on log output, too: code that
    logs a warning and then substitutes anyway would satisfy a log assertion
    while shipping exactly the behaviour this replaces.
    """

    def test_unknown_voice_raises_keyerror(self, tmp_path):
        eng = _make_engine(tmp_path, builtin=["Claribel Dervla", "Ana Florence"])

        with pytest.raises(KeyError):
            eng.resolve_voice("NOT_A_REAL_VOICE")

    def test_unknown_voice_does_not_return_the_first_builtin(self, tmp_path):
        """The precise old behaviour: it returned Claribel Dervla and HTTP 200."""
        eng = _make_engine(tmp_path, builtin=["Claribel Dervla", "Ana Florence"])

        try:
            result = eng.resolve_voice("NOT_A_REAL_VOICE")
        except KeyError:
            return  # correct
        pytest.fail(f"unknown voice was silently substituted, resolved to {result!r}")

    def test_builtin_voice_selected_by_name(self, tmp_path):
        eng = _make_engine(tmp_path, builtin=["Claribel Dervla"])

        assert eng.resolve_voice("Claribel Dervla") is None

    def test_custom_voice_returns_its_reference_wavs(self, tmp_path):
        vd = tmp_path / "voices"
        _wav(vd / "alice.wav")
        eng = _make_engine(tmp_path, builtin=["Claribel Dervla"], voices_dir=vd)
        eng._load_custom_voices()

        assert eng.resolve_voice("alice") == [str(vd / "alice.wav")]

    def test_no_voices_at_all_raises_runtimeerror(self, tmp_path):
        eng = _make_engine(tmp_path)

        with pytest.raises(RuntimeError):
            eng.resolve_voice("anything")
