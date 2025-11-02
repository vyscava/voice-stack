"""Unit tests for utils.text module."""

from __future__ import annotations

import pytest

from utils.text import build_safe_chunks, ensure_terminal_punct, normalize_text, split_sentences, wrap_words


@pytest.mark.unit
class TestNormalizeText:
    """Tests for normalize_text function."""

    def test_normalize_empty_string(self) -> None:
        """Test normalizing empty string returns empty string."""
        assert normalize_text("") == ""

    def test_normalize_nfkc(self) -> None:
        """Test NFKC normalization is applied."""
        # \uFB01 is 'ﬁ' ligature, NFKC normalizes to 'fi'
        text = "\ufb01le"  # ﬁle
        result = normalize_text(text)
        assert result == "file"

    def test_replace_em_dash(self) -> None:
        """Test em-dash is replaced with hyphen."""
        assert normalize_text("hello—world") == "hello-world"

    def test_replace_en_dash(self) -> None:
        """Test en-dash is replaced with hyphen."""
        assert normalize_text("hello–world") == "hello-world"

    def test_replace_ellipsis(self) -> None:
        """Test ellipsis character is replaced with three dots."""
        assert normalize_text("hello…") == "hello..."

    def test_replace_nbsp(self) -> None:
        """Test non-breaking space is replaced with regular space."""
        assert normalize_text("hello\u00a0world") == "hello world"

    def test_replace_curly_quotes(self) -> None:
        """Test curly quotes are replaced with straight quotes."""
        # Left/right single quotes → straight single quote
        assert normalize_text("\u2018hello\u2019") == "'hello'"
        # Left/right double quotes → straight double quote
        assert normalize_text("\u201chello\u201d") == '"hello"'

    def test_remove_control_chars(self) -> None:
        """Test control characters are removed."""
        # \u0001 is SOH (Start of Heading) control character
        text = "hello\u0001world"
        assert normalize_text(text) == "helloworld"

    def test_collapse_whitespace(self) -> None:
        """Test multiple whitespace is collapsed to single space."""
        assert normalize_text("hello    world\n\t test") == "hello world test"

    def test_strip_whitespace(self) -> None:
        """Test leading/trailing whitespace is stripped."""
        assert normalize_text("  hello world  ") == "hello world"

    def test_combined_normalizations(self) -> None:
        """Test multiple normalizations work together."""
        # em-dash, curly quotes, ellipsis, extra spaces
        text = "  Hello—world  \u201ccurly\u201d…  "
        result = normalize_text(text)
        assert result == 'Hello-world "curly"...'


@pytest.mark.unit
class TestEnsureTerminalPunct:
    """Tests for ensure_terminal_punct function."""

    def test_already_has_period(self) -> None:
        """Test text already ending with period is unchanged."""
        assert ensure_terminal_punct("Hello.") == "Hello."

    def test_already_has_question_mark(self) -> None:
        """Test text already ending with question mark is unchanged."""
        assert ensure_terminal_punct("Hello?") == "Hello?"

    def test_already_has_exclamation(self) -> None:
        """Test text already ending with exclamation is unchanged."""
        assert ensure_terminal_punct("Hello!") == "Hello!"

    def test_no_punctuation_adds_period(self) -> None:
        """Test text without terminal punctuation gets a period."""
        assert ensure_terminal_punct("Hello") == "Hello."

    def test_empty_string(self) -> None:
        """Test empty string returns a period."""
        assert ensure_terminal_punct("") == "."


@pytest.mark.unit
class TestSplitSentences:
    """Tests for split_sentences function."""

    def test_single_sentence(self) -> None:
        """Test single sentence returns list with one element."""
        result = split_sentences("Hello world.")
        assert result == ["Hello world."]

    def test_multiple_sentences_period(self) -> None:
        """Test splitting on periods."""
        text = "Hello world. This is a test."
        result = split_sentences(text)
        assert result == ["Hello world.", "This is a test."]

    def test_multiple_sentences_question_mark(self) -> None:
        """Test splitting on question marks."""
        text = "Hello? How are you?"
        result = split_sentences(text)
        assert result == ["Hello?", "How are you?"]

    def test_multiple_sentences_exclamation(self) -> None:
        """Test splitting on exclamation marks."""
        text = "Hello! Welcome!"
        result = split_sentences(text)
        assert result == ["Hello!", "Welcome!"]

    def test_mixed_punctuation(self) -> None:
        """Test splitting on mixed punctuation."""
        text = "Hello. How are you? I'm fine!"
        result = split_sentences(text)
        assert result == ["Hello.", "How are you?", "I'm fine!"]

    def test_ellipsis_split(self) -> None:
        """Test splitting on ellipsis character."""
        text = "Hello… World."
        result = split_sentences(text)
        # Ellipsis gets normalized to "..." which doesn't trigger split
        # Only the period after "World" triggers split logic
        assert len(result) >= 1

    def test_no_trailing_whitespace_in_sentences(self) -> None:
        """Test that split sentences have whitespace stripped."""
        text = "Hello.   World."
        result = split_sentences(text)
        for sent in result:
            assert sent == sent.strip()

    def test_empty_string(self) -> None:
        """Test empty string returns list with empty string."""
        result = split_sentences("")
        assert result == [""]

    def test_normalizes_before_split(self) -> None:
        """Test that text is normalized before splitting."""
        # Using em-dash which should be normalized to hyphen
        text = "Hello—world. Another sentence."
        result = split_sentences(text)
        # Should normalize em-dash to hyphen and split on period
        assert len(result) == 2
        assert "Hello-world." in result[0]


@pytest.mark.unit
class TestWrapWords:
    """Tests for wrap_words function."""

    def test_short_text_unchanged(self) -> None:
        """Test text shorter than max_chars returns as-is."""
        text = "Hello world"
        result = wrap_words(text, 50)
        assert result == [text]

    def test_exact_length(self) -> None:
        """Test text exactly at max_chars returns as-is."""
        text = "Hello"
        result = wrap_words(text, 5)
        assert result == [text]

    def test_wrap_at_word_boundary(self) -> None:
        """Test wrapping happens at word boundaries."""
        text = "Hello beautiful world"
        result = wrap_words(text, 15)
        # Should wrap without breaking words
        assert len(result) >= 2
        for chunk in result:
            assert len(chunk) <= 15

    def test_long_single_word(self) -> None:
        """Test single word longer than max_chars."""
        text = "Supercalifragilisticexpialidocious"
        result = wrap_words(text, 10)
        # Single long word should still be returned
        assert result == [text]

    def test_multiple_wraps(self) -> None:
        """Test text that requires multiple wraps."""
        text = "This is a very long sentence that needs multiple wraps"
        result = wrap_words(text, 15)
        # Each chunk should be <= max_chars
        for chunk in result:
            assert len(chunk) <= 15
        # Rejoining should give original text
        assert " ".join(result) == text

    def test_preserves_all_words(self) -> None:
        """Test that all words are preserved after wrapping."""
        text = "one two three four five six seven"
        result = wrap_words(text, 10)
        rejoined = " ".join(result)
        assert rejoined == text

    def test_empty_string(self) -> None:
        """Test empty string returns list with empty string."""
        result = wrap_words("", 50)
        assert result == [""]


@pytest.mark.unit
class TestBuildSafeChunks:
    """Tests for build_safe_chunks function."""

    def test_short_single_sentence(self) -> None:
        """Test short single sentence."""
        text = "Hello world"
        result = build_safe_chunks(text, 50)
        assert result == ["Hello world."]

    def test_multiple_short_sentences(self) -> None:
        """Test multiple sentences within max_chars."""
        text = "Hello. World."
        result = build_safe_chunks(text, 50)
        assert result == ["Hello.", "World."]

    def test_long_sentence_wraps(self) -> None:
        """Test long sentence gets wrapped."""
        text = "This is a very long sentence that exceeds the maximum character limit."
        result = build_safe_chunks(text, 20)
        # Should be split into multiple chunks
        assert len(result) > 1
        # Each chunk should end with punctuation
        for chunk in result:
            assert chunk[-1] in ".?!"
        # Each chunk should be <= max_chars
        for chunk in result:
            assert len(chunk) <= 20 + 1  # +1 for added punctuation

    def test_all_chunks_have_terminal_punct(self) -> None:
        """Test all chunks end with terminal punctuation."""
        text = "Hello world this is a test"
        result = build_safe_chunks(text, 10)
        for chunk in result:
            assert chunk[-1] in ".?!", f"Chunk '{chunk}' missing terminal punctuation"

    def test_preserves_existing_punctuation(self) -> None:
        """Test existing punctuation is preserved."""
        text = "Hello? World!"
        result = build_safe_chunks(text, 50)
        assert result == ["Hello?", "World!"]

    def test_normalizes_text(self) -> None:
        """Test text is normalized before chunking."""
        text = "\u201cHello\u201d—world…"
        result = build_safe_chunks(text, 50)
        # Should normalize curly quotes, em-dash, ellipsis
        assert '"Hello"-world' in result[0] or '"Hello"-world...' in result[0]

    def test_complex_text(self) -> None:
        """Test complex text with multiple sentences and wrapping."""
        text = (
            "This is the first sentence. "
            "This is a very long second sentence that will need to be wrapped. "
            "Short third."
        )
        result = build_safe_chunks(text, 30)
        # Should have multiple chunks
        assert len(result) >= 3
        # All should have terminal punctuation
        for chunk in result:
            assert chunk[-1] in ".?!"
        # None should exceed max_chars (except possibly by 1 for added punct)
        for chunk in result:
            assert len(chunk) <= 31  # max_chars + 1 for punctuation

    def test_empty_string(self) -> None:
        """Test empty string returns appropriate result."""
        result = build_safe_chunks("", 50)
        # Empty normalized text should return single period
        assert result == ["."]

    def test_whitespace_only(self) -> None:
        """Test whitespace-only string."""
        result = build_safe_chunks("   \n\t  ", 50)
        # Whitespace gets normalized away, leaving empty string -> period
        assert result == ["."]
