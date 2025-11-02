"""Unit tests for utils.language.language_codes module."""

from __future__ import annotations

import pytest

from utils.language.language_codes import DISPLAY_BY_CODE, LANG_ALIASES, LanguageCode


@pytest.mark.unit
class TestLanguageCodeEnum:
    """Tests for LanguageCode enum."""

    def test_enum_values_are_strings(self) -> None:
        """Test that all enum values are strings."""
        for lang in LanguageCode:
            assert isinstance(lang.value, str)

    def test_canonical_property(self) -> None:
        """Test canonical property returns string value."""
        assert LanguageCode.EN.canonical == "en"
        assert LanguageCode.ES.canonical == "es"
        assert LanguageCode.ZH_CN.canonical == "zh-cn"
        assert LanguageCode.UNKNOWN.canonical == "unknown"

    def test_display_name_property(self) -> None:
        """Test display_name property returns human-friendly name."""
        assert LanguageCode.EN.display_name == "English"
        assert LanguageCode.ES.display_name == "Spanish"
        assert LanguageCode.ZH_CN.display_name == "Chinese (Simplified)"
        assert LanguageCode.UNKNOWN.display_name == "Unknown"

    def test_all_enum_members_have_display_names(self) -> None:
        """Test all enum members have display names in DISPLAY_BY_CODE."""
        for lang in LanguageCode:
            assert lang.value in DISPLAY_BY_CODE
            assert isinstance(lang.display_name, str)
            assert len(lang.display_name) > 0


@pytest.mark.unit
class TestFromString:
    """Tests for LanguageCode.from_string method."""

    def test_from_string_none_returns_unknown(self) -> None:
        """Test that None input returns UNKNOWN."""
        assert LanguageCode.from_string(None) == LanguageCode.UNKNOWN

    def test_from_string_empty_returns_unknown(self) -> None:
        """Test that empty string returns UNKNOWN."""
        assert LanguageCode.from_string("") == LanguageCode.UNKNOWN

    def test_from_string_whitespace_returns_unknown(self) -> None:
        """Test that whitespace-only string returns UNKNOWN."""
        assert LanguageCode.from_string("   ") == LanguageCode.UNKNOWN

    def test_from_string_canonical_codes(self) -> None:
        """Test canonical codes are recognized."""
        assert LanguageCode.from_string("en") == LanguageCode.EN
        assert LanguageCode.from_string("es") == LanguageCode.ES
        assert LanguageCode.from_string("fr") == LanguageCode.FR
        assert LanguageCode.from_string("de") == LanguageCode.DE
        assert LanguageCode.from_string("zh-cn") == LanguageCode.ZH_CN

    def test_from_string_case_insensitive(self) -> None:
        """Test that matching is case-insensitive."""
        assert LanguageCode.from_string("EN") == LanguageCode.EN
        assert LanguageCode.from_string("En") == LanguageCode.EN
        assert LanguageCode.from_string("eN") == LanguageCode.EN
        assert LanguageCode.from_string("ZH-CN") == LanguageCode.ZH_CN

    def test_from_string_strips_whitespace(self) -> None:
        """Test that whitespace is stripped."""
        assert LanguageCode.from_string("  en  ") == LanguageCode.EN
        assert LanguageCode.from_string("\ten\n") == LanguageCode.EN

    def test_from_string_english_aliases(self) -> None:
        """Test English language aliases."""
        for alias in ["en", "en-us", "en_uk", "en-uk", "eng", "us", "uk"]:
            result = LanguageCode.from_string(alias)
            assert result == LanguageCode.EN, f"Alias '{alias}' should resolve to EN"

    def test_from_string_portuguese_aliases(self) -> None:
        """Test Portuguese language aliases."""
        for alias in ["pt", "pt-br", "ptbr", "pt_br", "br"]:
            result = LanguageCode.from_string(alias)
            assert result == LanguageCode.PT, f"Alias '{alias}' should resolve to PT"

    def test_from_string_spanish_aliases(self) -> None:
        """Test Spanish language aliases."""
        for alias in ["es", "spa"]:
            result = LanguageCode.from_string(alias)
            assert result == LanguageCode.ES, f"Alias '{alias}' should resolve to ES"

    def test_from_string_french_aliases(self) -> None:
        """Test French language aliases."""
        for alias in ["fr", "fra"]:
            result = LanguageCode.from_string(alias)
            assert result == LanguageCode.FR, f"Alias '{alias}' should resolve to FR"

    def test_from_string_german_aliases(self) -> None:
        """Test German language aliases."""
        for alias in ["de", "ger"]:
            result = LanguageCode.from_string(alias)
            assert result == LanguageCode.DE, f"Alias '{alias}' should resolve to DE"

    def test_from_string_italian_aliases(self) -> None:
        """Test Italian language aliases."""
        for alias in ["it", "ita"]:
            result = LanguageCode.from_string(alias)
            assert result == LanguageCode.IT, f"Alias '{alias}' should resolve to IT"

    def test_from_string_chinese_aliases(self) -> None:
        """Test Chinese language aliases."""
        for alias in ["zh", "zh-cn", "zhcn", "zho", "cn"]:
            result = LanguageCode.from_string(alias)
            assert result == LanguageCode.ZH_CN, f"Alias '{alias}' should resolve to ZH_CN"

    def test_from_string_other_languages(self) -> None:
        """Test other supported languages."""
        assert LanguageCode.from_string("pl") == LanguageCode.PL
        assert LanguageCode.from_string("tr") == LanguageCode.TR
        assert LanguageCode.from_string("ru") == LanguageCode.RU
        assert LanguageCode.from_string("nl") == LanguageCode.NL
        assert LanguageCode.from_string("cs") == LanguageCode.CS
        assert LanguageCode.from_string("ar") == LanguageCode.AR
        assert LanguageCode.from_string("hu") == LanguageCode.HU
        assert LanguageCode.from_string("ko") == LanguageCode.KO
        assert LanguageCode.from_string("ja") == LanguageCode.JA
        assert LanguageCode.from_string("hi") == LanguageCode.HI

    def test_from_string_unrecognized_returns_unknown(self) -> None:
        """Test that unrecognized codes return UNKNOWN."""
        assert LanguageCode.from_string("xyz") == LanguageCode.UNKNOWN
        assert LanguageCode.from_string("invalid") == LanguageCode.UNKNOWN
        assert LanguageCode.from_string("123") == LanguageCode.UNKNOWN


@pytest.mark.unit
class TestDisplayByCode:
    """Tests for DISPLAY_BY_CODE dictionary."""

    def test_all_enum_values_in_display_dict(self) -> None:
        """Test all enum values have entries in DISPLAY_BY_CODE."""
        for lang in LanguageCode:
            assert lang.value in DISPLAY_BY_CODE

    def test_display_names_are_nonempty_strings(self) -> None:
        """Test all display names are non-empty strings."""
        for name in DISPLAY_BY_CODE.values():
            assert isinstance(name, str)
            assert len(name) > 0

    def test_specific_display_names(self) -> None:
        """Test specific display names are correct."""
        assert DISPLAY_BY_CODE["en"] == "English"
        assert DISPLAY_BY_CODE["es"] == "Spanish"
        assert DISPLAY_BY_CODE["zh-cn"] == "Chinese (Simplified)"
        assert DISPLAY_BY_CODE["unknown"] == "Unknown"


@pytest.mark.unit
class TestLangAliases:
    """Tests for LANG_ALIASES dictionary."""

    def test_all_aliases_are_lowercase(self) -> None:
        """Test all alias keys are lowercase."""
        for alias in LANG_ALIASES.keys():
            assert alias == alias.lower()

    def test_all_alias_targets_are_valid(self) -> None:
        """Test all alias targets are valid canonical codes."""
        valid_codes = {lang.value for lang in LanguageCode}
        for alias, target in LANG_ALIASES.items():
            assert target in valid_codes, f"Alias '{alias}' targets invalid code '{target}'"

    def test_canonical_codes_map_to_themselves(self) -> None:
        """Test canonical codes map to themselves in aliases."""
        for lang in LanguageCode:
            if lang != LanguageCode.UNKNOWN:
                # Canonical codes should map to themselves
                assert lang.value in LANG_ALIASES
                assert LANG_ALIASES[lang.value] == lang.value

    def test_aliases_are_strings(self) -> None:
        """Test all aliases are strings."""
        for alias, target in LANG_ALIASES.items():
            assert isinstance(alias, str)
            assert isinstance(target, str)


@pytest.mark.unit
class TestLanguageCodeIntegration:
    """Integration tests for LanguageCode."""

    def test_round_trip_canonical(self) -> None:
        """Test round-trip from canonical code."""
        for lang in LanguageCode:
            if lang != LanguageCode.UNKNOWN:
                result = LanguageCode.from_string(lang.canonical)
                assert result == lang

    def test_round_trip_display_to_canonical(self) -> None:
        """Test getting canonical from display name lookup."""
        # This tests the coherence between display names and canonical codes
        for lang in LanguageCode:
            canonical = lang.canonical
            display = lang.display_name
            # Verify display name is in the dictionary
            assert DISPLAY_BY_CODE[canonical] == display

    def test_enum_is_iterable(self) -> None:
        """Test that LanguageCode can be iterated."""
        langs = list(LanguageCode)
        assert len(langs) > 0
        assert LanguageCode.EN in langs
        assert LanguageCode.UNKNOWN in langs

    def test_enum_members_are_unique(self) -> None:
        """Test that all enum values are unique."""
        values = [lang.value for lang in LanguageCode]
        assert len(values) == len(set(values))

    def test_string_comparison(self) -> None:
        """Test that LanguageCode can be compared with strings."""
        # LanguageCode(str, Enum) should allow string comparison
        assert LanguageCode.EN == "en"
        assert LanguageCode.ZH_CN == "zh-cn"
        assert LanguageCode.UNKNOWN == "unknown"
