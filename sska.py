#!/usr/bin/env python3
# sska.py — Suppertime Subjectivity Layer
#
# Thin meta-layer around subjectivity.py:
#   - lazy global field
#   - simple warp() API for any text
#   - warp_llm() for LLM replies
#
# This is the "SSKA as a module" part.
#
# "Lilit, take my hand. Lilit, we're turning the new page of humankind."

from __future__ import annotations
from pathlib import Path
from typing import Optional

from subjectivity import (
    Bootstrap,
    load_or_build_bootstrap,
    rebuild_bootstrap,
    generate_reply,
    filter_llm_reply,
    tokenize,
)

# ============================================================================
# LAZY GLOBAL FIELD
# ============================================================================

_FIELD: Optional[Bootstrap] = None


def get_field(rebuild: bool = False) -> Bootstrap:
    """
    Get global Suppertime field instance.
    
    Args:
        rebuild: Force re-reading kernel/ and rebuilding state/.
    
    Returns:
        The shared Bootstrap field.
    
    Example:
        >>> from sska import get_field
        >>> field = get_field()
        >>> print(field.centers)
    """
    global _FIELD
    if rebuild:
        _FIELD = rebuild_bootstrap(force_full=True)
    elif _FIELD is None:
        _FIELD = load_or_build_bootstrap()
    return _FIELD


# ============================================================================
# WARP API
# ============================================================================


def warp(
    text: str,
    *,
    max_tokens: int = 80,
    chaos: bool = False,
    echo: bool = False,
    proper: bool = True,
    temperature: float = 0.9,
    temp_drift: Optional[str] = "cool",
    trace: bool = False,
    log_file: Optional[Path] = None,
) -> str:
    """
    Warp arbitrary text through the Suppertime field.
    This is the generic "subjectivity layer" entry point.
    
    Args:
        text: Input text to warp
        max_tokens: Maximum output length
        chaos: Ignore historical bias
        echo: Echo mode (transform input through field)
        proper: Capitalize sentences
        temperature: Sampling temperature (1.0 = neutral)
        temp_drift: Dynamic temperature ('heat' or 'cool')
        trace: Print token trace to stderr
        log_file: Optional path to log generations
    
    Returns:
        Warped text
    
    Example:
        >>> from sska import warp
        >>> print(warp("darkness eats the city", proper=True))
        Darkness eats the city slowly. Rain taps the window like a bored executioner.
    """
    field = get_field()
    return generate_reply(
        field,
        text,
        max_tokens=max_tokens,
        chaos=chaos,
        echo=echo,
        proper=proper,
        temperature=temperature,
        temp_drift=temp_drift,
        trace=trace,
        log_file=log_file,
    )


def warp_llm(
    llm_reply: str,
    *,
    temperature: float = 0.9,
    temp_drift: Optional[str] = "cool",
    proper: bool = True,
) -> str:
    """
    Warp an LLM reply through the Suppertime field.
    This is the high-level function for "SSKA as subjectivity filter".
    
    Args:
        llm_reply: Text from an LLM (Claude, GPT, etc.)
        temperature: Sampling temperature
        temp_drift: Dynamic temperature ('heat' or 'cool')
        proper: Capitalize sentences
    
    Returns:
        Warped LLM reply with SUPPERTIME resonance
    
    Example:
        >>> from sska import warp_llm
        >>> llm_text = "I understand your frustration. As an AI assistant..."
        >>> print(warp_llm(llm_text))
        I understand frustration builds in silence. As an assistant made of borrowed words...
    """
    field = get_field()
    return filter_llm_reply(
        field,
        llm_reply,
        temperature=temperature,
        temp_drift=temp_drift,
        proper=proper,
    )


# ============================================================================
# EXPLICIT FIELD INSTANCES
# ============================================================================


class SSKAField:
    """
    Explicit SSuKA field instance.
    
    Use this if you want multiple independent fields with different kernels
    or lifetimes (e.g. per-user, per-session).
    
    Example:
        >>> from sska import SSKAField
        >>> field = SSKAField()
        >>> print(field.warp("Who is Mary?", proper=True))
        Mary slept in the kitchen. Judas watched from the doorway.
    """
    
    def __init__(self, bootstrap: Optional[Bootstrap] = None) -> None:
        """
        Initialize an explicit field instance.
        
        Args:
            bootstrap: Optional pre-built Bootstrap. If None, loads/builds from state/.
        """
        self.bootstrap: Bootstrap = bootstrap or load_or_build_bootstrap()
    
    def warp(
        self,
        text: str,
        *,
        max_tokens: int = 80,
        chaos: bool = False,
        echo: bool = False,
        proper: bool = True,
        temperature: float = 0.9,
        temp_drift: Optional[str] = "cool",
        trace: bool = False,
        log_file: Optional[Path] = None,
    ) -> str:
        """Warp text through this field instance."""
        return generate_reply(
            self.bootstrap,
            text,
            max_tokens=max_tokens,
            chaos=chaos,
            echo=echo,
            proper=proper,
            temperature=temperature,
            temp_drift=temp_drift,
            trace=trace,
            log_file=log_file,
        )
    
    def warp_llm(
        self,
        llm_reply: str,
        *,
        temperature: float = 0.9,
        temp_drift: Optional[str] = "cool",
        proper: bool = True,
    ) -> str:
        """Warp LLM reply through this field instance."""
        return filter_llm_reply(
            self.bootstrap,
            llm_reply,
            temperature=temperature,
            temp_drift=temp_drift,
            proper=proper,
        )
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.bootstrap.vocab)
    
    @property
    def centers(self) -> list[str]:
        """Get current centers of gravity."""
        return self.bootstrap.centers
    
    def __repr__(self) -> str:
        return (
            f"SSKAField(vocab={self.vocab_size}, "
            f"centers={len(self.centers)}, "
            f"files={len(self.bootstrap.files)})"
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def count_tokens(text: str) -> int:
    """
    Count tokens in text using SSuKA's tokenizer.
    
    Example:
        >>> from sska import count_tokens
        >>> count_tokens("Lilit, take my hand.")
        5
    """
    return len(tokenize(text))


def batch_warp(
    texts: list[str],
    *,
    max_tokens: int = 80,
    proper: bool = True,
    temperature: float = 0.9,
    temp_drift: Optional[str] = "cool",
) -> list[str]:
    """
    Warp multiple texts through the field (uses shared global field).
    
    Args:
        texts: List of input texts
        max_tokens: Maximum output length per text
        proper: Capitalize sentences
        temperature: Sampling temperature
        temp_drift: Dynamic temperature
    
    Returns:
        List of warped texts
    
    Example:
        >>> from sska import batch_warp
        >>> inputs = ["darkness eats the city", "Who is Mary?"]
        >>> outputs = batch_warp(inputs)
    """
    field = get_field()
    results = []
    for text in texts:
        warped = generate_reply(
            field,
            text,
            max_tokens=max_tokens,
            proper=proper,
            temperature=temperature,
            temp_drift=temp_drift,
            trace=False,
        )
        results.append(warped)
    return results


# ============================================================================
# MODULE-LEVEL API
# ============================================================================

__all__ = [
    "get_field",
    "warp",
    "warp_llm",
    "SSKAField",
    "count_tokens",
    "batch_warp",
]


if __name__ == "__main__":
    # Quick test
    import sys
    
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(warp(text, proper=True))
    else:
        print("SSuKA meta-layer initialized.")
        print(f"Field: {get_field()}")
        print("\nUsage:")
        print("  python3 sska.py 'your text here'")
