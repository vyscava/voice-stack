"""Unit tests for utils.language.language_helper module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from utils.language.language_codes import LanguageCode
from utils.language.language_helper import detect_lang, latin_heuristic, script_heuristic


@pytest.mark.unit
class TestDetectLang:
    """Tests for detect_lang function."""

    def test_empty_string_returns_none(self) -> None:
        """Test empty string returns None."""
        assert detect_lang("") is None

    def test_too_short_text_returns_none(self) -> None:
        """Test text shorter than 8 characters returns None."""
        assert detect_lang("hello") is None
        assert detect_lang("1234567") is None

    def test_exactly_8_chars_calls_detector(self) -> None:
        """Test exactly 8 characters calls langdetect."""
        with patch("utils.language.language_helper._ld_detect") as mock_detect:
            mock_detect.return_value = "en"
            result = detect_lang("12345678")
            assert result == LanguageCode.EN
            mock_detect.assert_called_once()

    def test_langdetect_success_en(self) -> None:
        """Test successful English detection."""
        with patch("utils.language.language_helper._ld_detect") as mock_detect:
            mock_detect.return_value = "en"
            result = detect_lang("This is some English text that should be detected.")
            assert result == LanguageCode.EN

    def test_langdetect_success_es(self) -> None:
        """Test successful Spanish detection."""
        with patch("utils.language.language_helper._ld_detect") as mock_detect:
            mock_detect.return_value = "es"
            result = detect_lang("Este es un texto en español.")
            assert result == LanguageCode.ES

    def test_langdetect_with_region_code_strips_region(self) -> None:
        """Test region codes are stripped (e.g., en-us → en)."""
        with patch("utils.language.language_helper._ld_detect") as mock_detect:
            mock_detect.return_value = "en-us"
            result = detect_lang("Some English text here.")
            assert result == LanguageCode.EN

    def test_langdetect_chinese_normalizes_to_zh_cn(self) -> None:
        """Test Chinese variants normalize to ZH_CN."""
        with patch("utils.language.language_helper._ld_detect") as mock_detect:
            # Test various Chinese codes
            for zh_variant in ["zh", "zh-cn", "zh-tw", "zh-hk"]:
                mock_detect.return_value = zh_variant
                result = detect_lang("Some Chinese text.")
                assert result == LanguageCode.ZH_CN

    def test_langdetect_exception_returns_none(self) -> None:
        """Test that exceptions from langdetect are caught."""
        with patch("utils.language.language_helper._ld_detect") as mock_detect:
            mock_detect.side_effect = Exception("Detector failed")
            result = detect_lang("Some text that causes error.")
            assert result is None

    def test_langdetect_none_result_returns_unknown(self) -> None:
        """Test that None from langdetect returns UNKNOWN."""
        with patch("utils.language.language_helper._ld_detect") as mock_detect:
            mock_detect.return_value = None
            result = detect_lang("Indecisive text.")
            # None gets converted to empty string, which returns UNKNOWN
            assert result == LanguageCode.UNKNOWN

    def test_langdetect_unrecognized_returns_unknown(self) -> None:
        """Test unrecognized language codes return UNKNOWN."""
        with patch("utils.language.language_helper._ld_detect") as mock_detect:
            mock_detect.return_value = "xyz"  # Invalid code
            result = detect_lang("Some text in unknown language.")
            assert result == LanguageCode.UNKNOWN


@pytest.mark.unit
class TestScriptHeuristic:
    """Tests for script_heuristic function."""

    def test_empty_string_returns_none(self) -> None:
        """Test empty string returns None."""
        assert script_heuristic("") is None

    def test_latin_only_returns_none(self) -> None:
        """Test Latin script text returns None."""
        assert script_heuristic("Hello world") is None

    def test_chinese_cjk_detected(self) -> None:
        """Test Chinese CJK characters are detected."""
        # CJK Unified Ideographs
        assert script_heuristic("你好") == LanguageCode.ZH_CN
        assert script_heuristic("世界") == LanguageCode.ZH_CN
        assert script_heuristic("Hello 你好") == LanguageCode.ZH_CN

    def test_japanese_hiragana_detected(self) -> None:
        """Test Japanese hiragana is detected."""
        # Hiragana characters
        assert script_heuristic("こんにちは") == LanguageCode.JA
        assert script_heuristic("Hello こんにちは") == LanguageCode.JA

    def test_japanese_katakana_detected(self) -> None:
        """Test Japanese katakana is detected."""
        # Katakana characters
        assert script_heuristic("カタカナ") == LanguageCode.JA
        assert script_heuristic("Test カタカナ") == LanguageCode.JA

    def test_korean_hangul_detected(self) -> None:
        """Test Korean Hangul is detected."""
        # Hangul characters
        assert script_heuristic("안녕하세요") == LanguageCode.KO
        assert script_heuristic("Hello 안녕") == LanguageCode.KO

    def test_arabic_detected(self) -> None:
        """Test Arabic script is detected."""
        # Arabic characters
        assert script_heuristic("مرحبا") == LanguageCode.AR
        assert script_heuristic("Hello مرحبا") == LanguageCode.AR

    def test_cyrillic_detected_as_russian(self) -> None:
        """Test Cyrillic script is detected as Russian."""
        # Cyrillic characters
        assert script_heuristic("Привет") == LanguageCode.RU
        assert script_heuristic("Hello Привет") == LanguageCode.RU

    def test_devanagari_detected_as_hindi(self) -> None:
        """Test Devanagari script is detected as Hindi."""
        # Devanagari characters
        assert script_heuristic("नमस्ते") == LanguageCode.HI
        assert script_heuristic("Hello नमस्ते") == LanguageCode.HI

    def test_first_non_latin_character_determines_script(self) -> None:
        """Test that first non-Latin character determines language."""
        # Korean should be detected first
        assert script_heuristic("안녕 你好") == LanguageCode.KO
        # Chinese should be detected first
        assert script_heuristic("你好 안녕") == LanguageCode.ZH_CN


@pytest.mark.unit
class TestLatinHeuristic:
    """Tests for latin_heuristic function."""

    def test_empty_string_returns_none(self) -> None:
        """Test empty string returns None."""
        assert latin_heuristic("") is None

    def test_plain_english_returns_none(self) -> None:
        """Test plain English without specific markers returns None."""
        assert latin_heuristic("Hello world this is a test") is None

    def test_portuguese_voce_detected(self) -> None:
        """Test Portuguese 'você' is detected."""
        assert latin_heuristic("Como você está?") == LanguageCode.PT
        assert latin_heuristic("você está bem") == LanguageCode.PT

    def test_portuguese_cao_detected(self) -> None:
        """Test Portuguese 'ção'/'ções' is detected."""
        assert latin_heuristic("Esta é uma informação importante") == LanguageCode.PT
        assert latin_heuristic("Temos várias opções aqui") == LanguageCode.PT

    def test_portuguese_que_detected(self) -> None:
        """Test Portuguese 'quê' is detected."""
        assert latin_heuristic("Por quê você fez isso?") == LanguageCode.PT

    def test_spanish_inverted_question_detected(self) -> None:
        """Test Spanish inverted question mark is detected."""
        assert latin_heuristic("¿Cómo estás?") == LanguageCode.ES

    def test_spanish_inverted_exclamation_detected(self) -> None:
        """Test Spanish inverted exclamation is detected."""
        assert latin_heuristic("¡Hola!") == LanguageCode.ES

    def test_spanish_enie_detected(self) -> None:
        """Test Spanish 'ñ' is detected."""
        assert latin_heuristic("mañana") == LanguageCode.ES
        assert latin_heuristic("español") == LanguageCode.ES

    def test_spanish_por_que_detected(self) -> None:
        """Test Spanish 'por qué' is detected."""
        assert latin_heuristic("No sé por qué pasó") == LanguageCode.ES
        assert latin_heuristic("qué bueno") == LanguageCode.ES

    def test_french_cedilla_detected(self) -> None:
        """Test French 'ç' is detected."""
        assert latin_heuristic("français") == LanguageCode.FR

    def test_french_ligature_detected(self) -> None:
        """Test French 'œ' ligature is detected."""
        assert latin_heuristic("cœur") == LanguageCode.FR

    def test_french_articles_detected(self) -> None:
        """Test French articles 'aux' and 'des' are detected."""
        # ' aux ' with spaces on both sides
        assert latin_heuristic("Je parle aux enfants de Paris") == LanguageCode.FR
        # ' des ' AND ' de ' both required
        assert latin_heuristic("Je parle des choses importantes de la vie") == LanguageCode.FR

    def test_german_eszett_detected(self) -> None:
        """Test German 'ß' is detected."""
        assert latin_heuristic("Straße") == LanguageCode.DE

    def test_german_umlauts_detected(self) -> None:
        """Test German umlauts are detected."""
        # Umlauts must be preceded by space
        assert latin_heuristic("schön über äpfel") == LanguageCode.DE
        assert latin_heuristic("das ist ü toll") == LanguageCode.DE
        assert latin_heuristic("für öffnen") == LanguageCode.DE

    def test_italian_gli_che_zione_detected(self) -> None:
        """Test Italian 'gli', 'che', 'zione' combination is detected."""
        assert latin_heuristic("gli studenti che hanno una informazione") == LanguageCode.IT

    def test_case_insensitive_matching(self) -> None:
        """Test that heuristics are case-insensitive."""
        assert latin_heuristic("VOCÊ está bem") == LanguageCode.PT
        assert latin_heuristic("MAÑANA") == LanguageCode.ES
        assert latin_heuristic("FRANÇAIS") == LanguageCode.FR

    def test_priority_order_portuguese_over_spanish(self) -> None:
        """Test that Portuguese is detected before Spanish when both markers present."""
        # If 'você' appears, should return PT even if Spanish markers also present
        text = "você está ¿bien?"
        assert latin_heuristic(text) == LanguageCode.PT

    def test_priority_order_spanish_over_french(self) -> None:
        """Test detection priority when multiple markers present."""
        # Spanish markers come before French in the code
        text = "¿Qué aux enfants?"
        assert latin_heuristic(text) == LanguageCode.ES


@pytest.mark.unit
class TestLanguageHelperIntegration:
    """Integration tests for language helper functions."""

    def test_script_heuristic_before_detect_lang(self) -> None:
        """Test that script_heuristic can be used before detect_lang."""
        # For CJK/non-Latin scripts, script_heuristic is faster
        text = "你好世界"
        script_result = script_heuristic(text)
        assert script_result == LanguageCode.ZH_CN

    def test_latin_heuristic_for_quick_detection(self) -> None:
        """Test that latin_heuristic provides quick detection for Latin scripts."""
        text = "você está bem"
        latin_result = latin_heuristic(text)
        assert latin_result == LanguageCode.PT

    def test_fallback_chain(self) -> None:
        """Test typical usage chain: script → latin → detect_lang."""
        # Example: if script_heuristic returns None, try latin_heuristic
        text = "Hello world"
        assert script_heuristic(text) is None
        assert latin_heuristic(text) is None
        # Would then fall back to detect_lang (which requires mocking for unit test)
