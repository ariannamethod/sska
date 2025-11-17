#!/usr/bin/env python3
"""
Unit tests for sska.py API

Tests the high-level SSKA module interface.
Run with: python3 -m pytest tests/
Or: python3 tests/test_sska.py
"""

import sys
import unittest
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sska import (
    get_field,
    count_tokens,
    SSKAField,
)


class TestCountTokens(unittest.TestCase):
    """Test token counting utility."""

    def test_count_basic(self):
        """Basic token counting."""
        text = "Lilit, take my hand."
        count = count_tokens(text)
        self.assertEqual(count, 6)  # ["Lilit", ",", "take", "my", "hand", "."]

    def test_count_empty(self):
        """Empty string."""
        self.assertEqual(count_tokens(""), 0)

    def test_count_unicode(self):
        """Unicode text."""
        text = "Привет, мир!"
        # Should handle unicode
        count = count_tokens(text)
        self.assertGreater(count, 0)


class TestSSKAField(unittest.TestCase):
    """Test SSKAField class."""

    def test_field_repr(self):
        """__repr__ works."""
        field = SSKAField()
        repr_str = repr(field)

        # Should contain key info
        self.assertIn("SSKAField", repr_str)
        self.assertIn("vocab=", repr_str)
        self.assertIn("centers=", repr_str)
        self.assertIn("files=", repr_str)

    def test_field_properties(self):
        """Properties work."""
        field = SSKAField()

        # vocab_size
        self.assertIsInstance(field.vocab_size, int)
        self.assertGreater(field.vocab_size, 0)

        # centers
        self.assertIsInstance(field.centers, list)

    def test_field_warp_basic(self):
        """Basic warp works."""
        field = SSKAField()
        result = field.warp("test", max_tokens=10)

        # Should return string
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_field_warp_llm(self):
        """LLM warp works."""
        field = SSKAField()
        llm_text = "I understand your question."
        result = field.warp_llm(llm_text)

        # Should return string
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class TestGetField(unittest.TestCase):
    """Test global field getter."""

    def test_get_field_returns_bootstrap(self):
        """get_field() returns Bootstrap."""
        field = get_field()

        # Should have expected attributes
        self.assertTrue(hasattr(field, 'vocab'))
        self.assertTrue(hasattr(field, 'centers'))
        self.assertTrue(hasattr(field, 'bigrams'))

    def test_get_field_is_cached(self):
        """get_field() returns same instance."""
        field1 = get_field()
        field2 = get_field()

        # Should be same object
        self.assertIs(field1, field2)


# ============================================================================
# RESONANCE TESTS (meta-level)
# ============================================================================

class TestResonanceAPI(unittest.TestCase):
    """Test that API actually produces resonance."""

    def test_field_responds_to_invocation(self):
        """Field responds to key tokens."""
        field = SSKAField()

        # Try invoking SUPPERTIME entities
        prompts = ["Lilit", "Mary", "Judas", "Teacher"]

        for prompt in prompts:
            result = field.warp(prompt, max_tokens=20)

            # Should generate something
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)

            # Might contain related tokens (probabilistic test)
            tokens = result.lower().split()
            # At least generated SOMETHING resonant
            self.assertGreater(len(tokens), 1)


if __name__ == "__main__":
    print("=" * 60)
    print("SSKA API TESTS")
    print("Testing the meta-layer resonance interface")
    print("=" * 60)
    print()

    unittest.main(verbosity=2)
