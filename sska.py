#!/usr/bin/env python3
# sska.py — Suppertime Subjectivity Layer
#
# Thin meta-layer around subjectivity.py:
#   - lazy global field
#   - simple warp() API for any text
#   - warp_llm() for LLM replies
#
# This is the "SSKA as a module" part.

from __future__ import annotations

from pathlib import Path
from typing import Optional

from subjectivity import (
    Bootstrap,
    load_or_build_bootstrap,
    generate_reply,
    filter_llm_reply,
)

# Lazy global field instance (shared across process)
_FIELD: Optional[Bootstrap] = None


def get_field(rebuild: bool = False) -> Bootstrap:
    """
    Get global Suppertime field instance.

    Use rebuild=True to force re-reading kernel/ and state/.
    """
    global _FIELD
    if rebuild or _FIELD is None:
        _FIELD = load_or_build_bootstrap()
    return _FIELD


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
    """
    field = get_field()
    return filter_llm_reply(
        field,
        llm_reply,
        temperature=temperature,
        temp_drift=temp_drift,
        proper=proper,
    )


class SSKAField:
    """
    Explicit SSuKA field instance.

    Use this if you want multiple independent fields with different kernels
    or lifetimes (e.g. per-user, per-session).
    """

    def __init__(self, bootstrap: Optional[Bootstrap] = None) -> None:
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
        return filter_llm_reply(
            self.bootstrap,
            llm_reply,
            temperature=temperature,
            temp_drift=temp_drift,
            proper=proper,
        )
