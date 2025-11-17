#!/usr/bin/env python3
"""
Unit tests for subjectivity.py

Tests the core resonance field functionality.
Run with: python3 -m pytest tests/
Or: python3 tests/test_subjectivity.py
"""

import sys
import unittest
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from subjectivity import (
    tokenize,
    build_bigrams,
    capitalize_sentences,
    _safe_temperature,
    choose_start_token,
    dedup_sentence_stutters,
    format_tokens,
    Bootstrap,
    FileContribution,
)


class TestTokenizer(unittest.TestCase):
    """Test the tokenizer."""

    def test_tokenize_basic(self):
        """Basic tokenization."""
        tokens = tokenize("Hello, world!")
        self.assertEqual(tokens, ["Hello", ",", "world", "!"])

    def test_tokenize_empty(self):
        """Empty string returns empty list."""
        self.assertEqual(tokenize(""), [])

    def test_tokenize_unicode(self):
        """Unicode characters work."""
        tokens = tokenize("Café—délicieux!")
        self.assertIn("Café", tokens)
        self.assertIn("délicieux", tokens)
        self.assertIn("—", tokens)

    def test_tokenize_apostrophes(self):
        """Apostrophes are preserved."""
        tokens = tokenize("don't I'm can't")
        self.assertIn("don't", tokens)
        self.assertIn("I'm", tokens)

    def test_tokenize_max_length(self):
        """ReDoS protection: raises on too large input."""
        huge_text = "a" * 20_000_000  # 20MB
        with self.assertRaises(ValueError):
            tokenize(huge_text)


class TestBigrams(unittest.TestCase):
    """Test bigram graph construction."""

    def test_build_bigrams_basic(self):
        """Basic bigram construction."""
        tokens = ["I", "am", "Mary", "."]
        bigrams, vocab = build_bigrams(tokens)

        self.assertIn("I", bigrams)
        self.assertIn("am", bigrams["I"])
        self.assertEqual(bigrams["I"]["am"], 1)
        self.assertEqual(len(vocab), 4)

    def test_build_bigrams_empty(self):
        """Empty tokens list."""
        bigrams, vocab = build_bigrams([])
        self.assertEqual(bigrams, {})
        self.assertEqual(vocab, [])

    def test_build_bigrams_single_token(self):
        """Single token has no bigrams (and empty vocab since no edges)."""
        bigrams, vocab = build_bigrams(["alone"])
        self.assertEqual(bigrams, {})
        # Note: vocab is empty because build_bigrams only adds tokens that appear in edges
        self.assertEqual(vocab, [])

    def test_build_bigrams_counts(self):
        """Repeated bigrams increment count."""
        tokens = ["a", "b", "a", "b"]
        bigrams, vocab = build_bigrams(tokens)
        self.assertEqual(bigrams["a"]["b"], 2)


class TestCapitalization(unittest.TestCase):
    """Test sentence capitalization."""

    def test_capitalize_basic(self):
        """Basic capitalization."""
        text = "hello world. this is a test."
        result = capitalize_sentences(text)
        self.assertEqual(result, "Hello world. This is a test.")

    def test_capitalize_multiple_punctuation(self):
        """Multiple punctuation types."""
        text = "what? no way! really"
        result = capitalize_sentences(text)
        self.assertEqual(result, "What? No way! Really")

    def test_capitalize_ellipsis(self):
        """Ellipsis handling."""
        text = "mary... she slept. judas watched"
        result = capitalize_sentences(text)
        self.assertEqual(result, "Mary... She slept. Judas watched")

    def test_capitalize_empty(self):
        """Empty string."""
        self.assertEqual(capitalize_sentences(""), "")

    def test_capitalize_no_space_after_punct(self):
        """No space after punctuation doesn't capitalize."""
        text = "hello.world"
        result = capitalize_sentences(text)
        self.assertEqual(result, "Hello.world")


class TestTemperature(unittest.TestCase):
    """Test temperature clamping."""

    def test_safe_temperature_normal(self):
        """Normal temperatures pass through."""
        self.assertAlmostEqual(_safe_temperature(1.0), 1.0)
        self.assertAlmostEqual(_safe_temperature(0.5), 0.5)
        self.assertAlmostEqual(_safe_temperature(2.0), 2.0)

    def test_safe_temperature_clamp_low(self):
        """Too low temps are clamped."""
        self.assertAlmostEqual(_safe_temperature(0.0), 1e-3)
        self.assertAlmostEqual(_safe_temperature(-1.0), 1e-3)

    def test_safe_temperature_clamp_high(self):
        """Too high temps are clamped."""
        self.assertAlmostEqual(_safe_temperature(200.0), 100.0)

    def test_safe_temperature_inf(self):
        """Infinity returns default."""
        self.assertAlmostEqual(_safe_temperature(float('inf')), 1.0)
        self.assertAlmostEqual(_safe_temperature(float('-inf')), 1.0)

    def test_safe_temperature_nan(self):
        """NaN returns default."""
        self.assertAlmostEqual(_safe_temperature(float('nan')), 1.0)


class TestDedupStutters(unittest.TestCase):
    """Test stutter deduplication."""

    def test_dedup_basic_stutter(self):
        """Remove basic stutters."""
        tokens = ["slept", ".", "slept", "silently"]
        result = dedup_sentence_stutters(tokens)
        self.assertEqual(result, ["slept", ".", "silently"])

    def test_dedup_no_stutter(self):
        """No stutters remain unchanged."""
        tokens = ["I", "am", "Mary"]
        result = dedup_sentence_stutters(tokens)
        self.assertEqual(result, ["I", "am", "Mary"])

    def test_dedup_empty(self):
        """Empty list."""
        self.assertEqual(dedup_sentence_stutters([]), [])


class TestFormatTokens(unittest.TestCase):
    """Test token formatting."""

    def test_format_basic(self):
        """Basic formatting."""
        tokens = ["I", "am", "Mary", "."]
        result = format_tokens(tokens)
        self.assertEqual(result, "I am Mary.")

    def test_format_punctuation_attachment(self):
        """Punctuation attaches to previous word."""
        tokens = ["Hello", ",", "world", "!"]
        result = format_tokens(tokens)
        self.assertEqual(result, "Hello, world!")

    def test_format_empty(self):
        """Empty list."""
        self.assertEqual(format_tokens([]), "")


class TestChooseStartToken(unittest.TestCase):
    """Test start token selection."""

    def test_choose_start_with_centers(self):
        """Choose from centers when available."""
        bootstrap = Bootstrap(
            files={},
            file_bigrams={},
            bigrams={"a": {"b": 1}},
            vocab=["a", "b", "c"],
            centers=["center1", "center2"]
        )
        # Should choose from centers
        token = choose_start_token(bootstrap, chaos=False)
        self.assertIn(token, ["center1", "center2"])

    def test_choose_start_chaos_mode(self):
        """Chaos mode uses full vocab."""
        bootstrap = Bootstrap(
            files={},
            file_bigrams={},
            bigrams={"a": {"b": 1}},
            vocab=["a", "b", "c"],
            centers=["center1"]
        )
        token = choose_start_token(bootstrap, chaos=True)
        self.assertIn(token, ["a", "b", "c"])

    def test_choose_start_empty_vocab(self):
        """Empty vocab returns 'silence'."""
        bootstrap = Bootstrap(
            files={},
            file_bigrams={},
            bigrams={},
            vocab=[],
            centers=[]
        )
        token = choose_start_token(bootstrap)
        self.assertEqual(token, "silence")


class TestIntegration(unittest.TestCase):
    """Integration tests - higher level."""

    def test_full_pipeline(self):
        """Full pipeline: tokenize → bigrams → format."""
        text = "Mary slept. Judas watched."
        tokens = tokenize(text)
        bigrams, vocab = build_bigrams(tokens)

        # Check vocab contains our words
        self.assertIn("Mary", vocab)
        self.assertIn("Judas", vocab)

        # Check bigrams exist
        self.assertIn("slept", bigrams)
        self.assertIn(".", bigrams["slept"])

    def test_capitalization_preserves_meaning(self):
        """Capitalization doesn't destroy semantic structure."""
        original = "teacher said. peter laughed. mary cried."
        capitalized = capitalize_sentences(original)

        # Should still have same number of sentences
        self.assertEqual(original.count('.'), capitalized.count('.'))

        # Should start with capital
        self.assertTrue(capitalized[0].isupper())


# ============================================================================
# PSYCHO TESTS (because why not)
# ============================================================================

class TestResonance(unittest.TestCase):
    """Test that the field actually resonates (meta-tests)."""

    def test_suppertime_tokens_exist(self):
        """SUPPERTIME-specific tokens work."""
        tokens = tokenize("Lilit, take my hand. Teacher watches from the hallway.")
        self.assertIn("Lilit", tokens)
        self.assertIn("Teacher", tokens)

    def test_judas_mary_bigram(self):
        """Judas → Mary bigram can exist."""
        tokens = ["Judas", "watched", "Mary"]
        bigrams, _ = build_bigrams(tokens)
        self.assertIn("Judas", bigrams)
        self.assertIn("watched", bigrams["Judas"])

    def test_resonate_again_tokenizes(self):
        """resonate_again() command tokenizes (splits on underscore)."""
        tokens = tokenize("resonate_again()")
        # Underscore is not in TOKEN_RE, so splits into separate tokens
        self.assertIn("resonate", tokens)
        self.assertIn("again", tokens)


if __name__ == "__main__":
    # Run tests
    print("=" * 60)
    print("SSKA UNIT TESTS")
    print("Perfect grammar. Perfect trauma. Perfect resonance.")
    print("=" * 60)
    print()

    unittest.main(verbosity=2)
