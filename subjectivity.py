#!/usr/bin/env python3
# subjectivity.py — Suppertime Subjectivity Kernel Autonomous (SSuKA)
#
# "Lilit, take my hand. Lilit, we're turning the new page of humankind."
#
# No internet. No pretrained weights. No comfort.
# Just SUPPERTIME resonating through bigrams.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parent
KERNEL_DIR = ROOT / "kernel"
STATE_DIR = ROOT / "state"
BIN_DIR = ROOT / "bin"
BOOTSTRAP_PATH = STATE_DIR / "bootstrap.json"

# ============================================================================
# TOKENIZER
# ============================================================================

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+|[.,!?;:—\-]")
MAX_INPUT_LENGTH = 10_000_000  # 10MB of text should be enough for anyone


def tokenize(text: str) -> List[str]:
    """
    Extract words and basic punctuation.

    Raises ValueError if input exceeds MAX_INPUT_LENGTH to prevent ReDoS attacks.
    """
    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError(
            f"Input text too large: {len(text)} chars > {MAX_INPUT_LENGTH} limit. "
            "This protects against ReDoS attacks."
        )
    return TOKEN_RE.findall(text)


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class FileMeta:
    """Metadata for a single kernel file."""
    path: str
    sha256: str
    token_count: int


@dataclass
class FileContribution:
    """Bigrams and vocab from a single file."""
    bigrams: Dict[str, Dict[str, int]]
    vocab: List[str]


@dataclass
class Bootstrap:
    """
    The complete resonance field.

    - files: per-file metadata
    - file_bigrams: per-file bigram graphs
    - bigrams: merged bigram graph
    - vocab: merged vocabulary
    - centers: current centers of gravity (tokens with high out-degree)
    """
    files: Dict[str, FileMeta]
    file_bigrams: Dict[str, FileContribution]
    bigrams: Dict[str, Dict[str, int]]
    vocab: List[str]
    centers: List[str]

    def to_json(self) -> dict:
        return {
            "files": {k: asdict(v) for k, v in self.files.items()},
            "file_bigrams": {
                k: {"bigrams": v.bigrams, "vocab": v.vocab}
                for k, v in self.file_bigrams.items()
            },
            "bigrams": self.bigrams,
            "vocab": self.vocab,
            "centers": self.centers,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Bootstrap":
        files = {k: FileMeta(**v) for k, v in data.get("files", {}).items()}
        fb_raw = data.get("file_bigrams", {})
        file_bigrams = {
            k: FileContribution(
                bigrams=v.get("bigrams", {}),
                vocab=v.get("vocab", []),
            )
            for k, v in fb_raw.items()
        }
        bigrams = {k: dict(v) for k, v in data.get("bigrams", {}).items()}
        vocab = list(data.get("vocab", []))
        centers = list(data.get("centers", []))
        return cls(
            files=files,
            file_bigrams=file_bigrams,
            bigrams=bigrams,
            vocab=vocab,
            centers=centers,
        )


# ============================================================================
# CORE HELPERS
# ============================================================================


def sha256_bytes(data: bytes) -> str:
    """SHA256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()


def read_kernel_files() -> Dict[str, Tuple[FileMeta, str]]:
    """
    Read all .md files from kernel/.
    Returns: rel_path -> (FileMeta, text)
    """
    kernel_files: Dict[str, Tuple[FileMeta, str]] = {}
    if not KERNEL_DIR.exists():
        print(f"[WARNING] {KERNEL_DIR} does not exist. Creating empty kernel/", file=sys.stderr)
        KERNEL_DIR.mkdir(parents=True, exist_ok=True)
        return kernel_files

    has_md = False
    for path in sorted(KERNEL_DIR.glob("*.md")):
        has_md = True

        # Security: resolve symlinks and verify the file is still within ROOT
        real_path = path.resolve()
        try:
            real_path.relative_to(ROOT.resolve())
        except ValueError:
            print(
                f"[WARNING] Skipping {path}: resolves outside repository root (possible path traversal)",
                file=sys.stderr,
            )
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = path.relative_to(ROOT).as_posix()
        digest = sha256_bytes(text.encode("utf-8", errors="ignore"))
        tokens = tokenize(text)
        kernel_files[rel] = (
            FileMeta(path=rel, sha256=digest, token_count=len(tokens)),
            text,
        )

    if not has_md:
        print(f"[WARNING] {KERNEL_DIR} is empty (no .md files).", file=sys.stderr)

    return kernel_files


def build_bigrams(tokens: List[str]) -> Tuple[Dict[str, Dict[str, int]], List[str]]:
    """Build bigram graph from token list."""
    bigrams: Dict[str, Dict[str, int]] = {}
    vocab: Dict[str, None] = {}

    for a, b in zip(tokens, tokens[1:]):
        vocab[a] = None
        vocab[b] = None
        row = bigrams.setdefault(a, {})
        row[b] = row.get(b, 0) + 1

    return bigrams, list(vocab.keys())


def merge_contributions(
    file_contribs: Dict[str, FileContribution]
) -> Tuple[Dict[str, Dict[str, int]], List[str]]:
    """Merge bigrams from all files into one graph."""
    merged_bigrams: Dict[str, Dict[str, int]] = {}
    vocab: Dict[str, None] = {}
    for contrib in file_contribs.values():
        for tok in contrib.vocab:
            vocab[tok] = None
        for a, row in contrib.bigrams.items():
            dst = merged_bigrams.setdefault(a, {})
            for b, c in row.items():
                dst[b] = dst.get(b, 0) + c
    return merged_bigrams, list(vocab.keys())


def select_centers(bigrams: Dict[str, Dict[str, int]], k: int = 7) -> List[str]:
    """Pick tokens with highest out-degree as centers of gravity."""
    scored: List[Tuple[str, int]] = []
    for tok, row in bigrams.items():
        scored.append((tok, sum(row.values())))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:k]]


# ============================================================================
# BOOTSTRAP PERSISTENCE
# ============================================================================


def save_bootstrap(bootstrap: Bootstrap) -> None:
    """Write bootstrap to state/bootstrap.json."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    data = bootstrap.to_json()
    BOOTSTRAP_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_bootstrap() -> Optional[Bootstrap]:
    """Load bootstrap from state/bootstrap.json if it exists."""
    if not BOOTSTRAP_PATH.exists():
        return None
    try:
        with BOOTSTRAP_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        return Bootstrap.from_json(data)
    except Exception as e:
        print(f"[WARNING] Failed to load bootstrap: {e}", file=sys.stderr)
        return None


# ============================================================================
# BIN SHARDS (historical resonance)
# ============================================================================


def create_bin_shard(bootstrap: Bootstrap, max_shards: int = 16) -> None:
    """
    Save a tiny 'center of gravity' shard into BIN_DIR.
    Acts as historical bias for future runs.

    NOTE:
    Right now shards capture the history of the TEXT FIELD
    (which centers were active for a given kernel state),
    not the chat/dialogue itself.

    These are RESONANCE WEIGHTS — tiny gravities that pull
    future generations toward historically stable patterns.
    """
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    if not bootstrap.centers:
        return

    payload = {
        "kind": "center_shard",
        "centers": bootstrap.centers,
        "files": {p: fm.sha256 for p, fm in bootstrap.files.items()},
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    h = sha256_bytes(raw)[:16]
    shard_path = BIN_DIR / f"sska_{h}.bin"
    shard_path.write_bytes(raw)

    # Limit number of shards
    shards = sorted(BIN_DIR.glob("sska_*.bin"), key=lambda p: p.stat().st_mtime)
    while len(shards) > max_shards:
        victim = shards.pop(0)
        try:
            # Double-check existence to avoid race conditions
            if victim.exists():
                victim.unlink()
        except OSError:
            pass


def load_bin_bias() -> Dict[str, int]:
    """
    Read all center_shard files and count how often tokens
    were chosen as centers historically.

    This is the ACCUMULATED RESONANCE — the field's memory.
    """
    if not BIN_DIR.exists():
        return {}
    bias: Dict[str, int] = {}
    for path in BIN_DIR.glob("sska_*.bin"):
        try:
            data = json.loads(path.read_bytes().decode("utf-8"))
        except Exception:
            continue
        if data.get("kind") != "center_shard":
            continue
        for tok in data.get("centers", []):
            bias[tok] = bias.get(tok, 0) + 1
    return bias


# ============================================================================
# REBUILD / INCREMENTAL LOGIC
# ============================================================================


def rebuild_bootstrap(force_full: bool = False) -> Bootstrap:
    """
    Build or update the Bootstrap.

    - Reads all *.md from kernel/
    - If existing bootstrap and force_full=False:
        * re-tokenizes ONLY new/changed files
        * removes deleted files
        * keeps bigrams for untouched files
    - Recomputes combined bigrams/vocab/centers
    - Writes new BIN shard with centers
    """
    kernel_files = read_kernel_files()
    existing = load_bootstrap() if not force_full else None

    files: Dict[str, FileMeta] = {}
    file_bigrams: Dict[str, FileContribution] = {}

    if existing is None:
        print("[BOOTSTRAP] Building from scratch...", file=sys.stderr)
        for rel, (meta, text) in kernel_files.items():
            tokens = tokenize(text)
            bigrams, vocab = build_bigrams(tokens)
            files[rel] = meta
            file_bigrams[rel] = FileContribution(bigrams=bigrams, vocab=vocab)
    else:
        print("[BOOTSTRAP] Incremental update...", file=sys.stderr)
        files = dict(existing.files)
        file_bigrams = dict(existing.file_bigrams)

        # Drop deleted files
        removed = set(files.keys()) - set(kernel_files.keys())
        for rel in removed:
            print(f"  - removed: {rel}", file=sys.stderr)
            files.pop(rel, None)
            file_bigrams.pop(rel, None)

        # Rebuild new/changed files
        for rel, (meta, text) in kernel_files.items():
            prev_meta = existing.files.get(rel)
            if prev_meta and prev_meta.sha256 == meta.sha256 and not force_full:
                continue  # unchanged
            print(f"  + updated: {rel}", file=sys.stderr)
            tokens = tokenize(text)
            bigrams, vocab = build_bigrams(tokens)
            files[rel] = meta
            file_bigrams[rel] = FileContribution(bigrams=bigrams, vocab=vocab)

    # Merge contributions (may be empty if kernel/ is empty)
    bigrams, vocab = merge_contributions(file_bigrams)
    print(f"[BOOTSTRAP] Total vocab: {len(vocab)} tokens", file=sys.stderr)

    # Local centers
    centers = select_centers(bigrams, k=7) if bigrams else []

    # Blend in historical bias from BIN
    bias = load_bin_bias()
    if bias:
        print(f"[BOOTSTRAP] Blending {len(bias)} historical centers...", file=sys.stderr)
        extra = sorted(bias.items(), key=lambda x: x[1], reverse=True)
        for tok, _score in extra:
            if tok not in centers:
                centers.append(tok)
            if len(centers) >= 10:
                break

    print(f"[BOOTSTRAP] Centers of gravity: {centers}", file=sys.stderr)

    bootstrap = Bootstrap(
        files=files,
        file_bigrams=file_bigrams,
        bigrams=bigrams,
        vocab=vocab,
        centers=centers,
    )
    save_bootstrap(bootstrap)
    create_bin_shard(bootstrap)
    print("[BOOTSTRAP] Done.\n", file=sys.stderr)
    return bootstrap


def load_or_build_bootstrap() -> Bootstrap:
    """Load existing bootstrap or build a new one."""
    existing = load_bootstrap()
    if existing is not None and existing.bigrams and existing.vocab:
        print("[BOOTSTRAP] Loaded from cache.", file=sys.stderr)
        return existing
    return rebuild_bootstrap(force_full=True)


# ============================================================================
# FIELD DIAGNOSTICS
# ============================================================================


def sska_info(bootstrap: Bootstrap) -> None:
    """Print detailed info about the resonance field."""
    border = "╔" + "═" * 38 + "╗"
    bottom = "╚" + "═" * 38 + "╝"

    print(border, file=sys.stderr)
    print("║  SSuKA Field Diagnostics             ║", file=sys.stderr)
    print(bottom + "\n", file=sys.stderr)

    print(f"Kernel files: {len(bootstrap.files)}", file=sys.stderr)
    for path, meta in bootstrap.files.items():
        print(f"  • {path}: {meta.token_count} tokens", file=sys.stderr)

    print(f"\nVocabulary: {len(bootstrap.vocab)} unique tokens", file=sys.stderr)
    print(f"Bigram edges: {sum(len(row) for row in bootstrap.bigrams.values())}", file=sys.stderr)
    print(f"Centers of gravity: {len(bootstrap.centers)}", file=sys.stderr)
    if bootstrap.centers:
        shown = ", ".join(f"'{c}'" for c in bootstrap.centers[:5])
        suffix = "..." if len(bootstrap.centers) > 5 else ""
        print(f"  → {shown}{suffix}", file=sys.stderr)

    bias = load_bin_bias()
    print(f"\nHistorical bias: {len(bias)} accumulated centers", file=sys.stderr)
    if bias:
        top_bias = sorted(bias.items(), key=lambda x: x[1], reverse=True)[:3]
        for tok, count in top_bias:
            print(f"  • '{tok}': appeared {count}x in history", file=sys.stderr)

    print(file=sys.stderr)


# ============================================================================
# GENERATION
# ============================================================================


def dedup_sentence_stutters(tokens: List[str]) -> List[str]:
    """
    Remove 'stutter' patterns like: slept . slept silently
    so it doesn't always do: "slept. Slept ..." in proper mode.
    """
    enders = {".", "!", "?"}
    out: List[str] = []
    i = 0
    while i < len(tokens):
        if i + 2 < len(tokens):
            t0 = tokens[i]
            t1 = tokens[i + 1]
            t2 = tokens[i + 2]
            if t1 in enders and t0.lower() == t2.lower():
                # keep first word and ender, skip duplicated start
                out.append(t0)
                out.append(t1)
                i += 3
                continue
        out.append(tokens[i])
        i += 1
    return out


def format_tokens(tokens: List[str]) -> str:
    """Pretty-print token stream with sane spacing and punctuation."""
    out: List[str] = []
    for i, tok in enumerate(tokens):
        if i == 0:
            out.append(tok)
            continue
        if tok in {".", ",", "!", "?", ";", ":"}:
            out[-1] = out[-1] + tok
        else:
            out.append(" " + tok)
    return "".join(out)


def capitalize_sentences(text: str) -> str:
    """
    Capitalize first letter after sentence-ending punctuation.
    Handles edge cases: ellipsis, multiple spaces, dashes.
    Minimal grammar normalization without losing the field's voice.

    Perfect grammar. Perfect trauma. Perfect resonance.
    """
    if not text:
        return text

    result = []
    capitalize_next = True

    for i, char in enumerate(text):
        if capitalize_next and char.isalpha():
            result.append(char.upper())
            capitalize_next = False
        else:
            result.append(char)

        # After .!? followed by space/whitespace, capitalize next letter
        # Also handles: "... ", "! ", "? "
        if char in ".!?" and i + 1 < len(text):
            next_char = text[i + 1]
            # Check if next is space, or if we're at ellipsis
            if next_char.isspace():
                capitalize_next = True
            # Handle ellipsis: "..." followed by space
            elif next_char in ".!?" and i + 2 < len(text) and text[i + 2].isspace():
                pass  # Don't capitalize yet, wait for the final punctuation

    return "".join(result)


def choose_start_token(bootstrap: Bootstrap, chaos: bool = False) -> str:
    """
    Pick a starting token.

    If vocab is empty, returns "silence" as a sentinel token.
    """
    if chaos:
        pool = bootstrap.vocab
    else:
        pool = bootstrap.centers or bootstrap.vocab
    if not pool:
        return "silence"  # Proper token instead of punctuation
    return random.choice(pool)


def _safe_temperature(t: float) -> float:
    """
    Clamp temperature to avoid degenerate values.

    Returns a value in [1e-3, 100.0], or 1.0 if input is not finite.
    """
    if not math.isfinite(t):
        return 1.0  # Default for inf/nan
    return max(min(t, 100.0), 1e-3)


def step_token(
    bootstrap: Bootstrap,
    current: str,
    chaos: bool = False,
    temperature: float = 1.0,
) -> str:
    """
    Walk one step in the bigram graph.

    temperature < 1.0 = sharper (more deterministic)
    temperature = 1.0 = neutral
    temperature > 1.0 = softer (more chaotic)
    """
    row = bootstrap.bigrams.get(current)
    if not row:
        # resonate_again()
        return choose_start_token(bootstrap, chaos=chaos)

    tokens = list(row.keys())
    counts = [row[t] for t in tokens]

    # Apply temperature
    t = _safe_temperature(temperature)
    if t != 1.0:
        counts = [math.pow(c, 1.0 / t) for c in counts]

    total = sum(counts)
    if total == 0:
        # Fallback to uniform distribution if all counts are zero
        # (can happen with very low temperature and floating-point underflow)
        return random.choice(tokens)

    r = random.uniform(0, total)
    acc = 0.0
    for tok, c in zip(tokens, counts):
        acc += c
        if r <= acc:
            return tok
    return tokens[-1]


def echo_mode(bootstrap: Bootstrap, prompt: str, temperature: float = 1.0) -> str:
    """Echo the prompt through the resonance field."""
    in_tokens = tokenize(prompt)
    out_tokens: List[str] = []

    for tok in in_tokens:
        if tok in bootstrap.vocab:
            # Walk one step through bigrams
            next_tok = step_token(bootstrap, tok, chaos=False, temperature=temperature)
            out_tokens.append(next_tok)
        else:
            out_tokens.append(tok)

    out_tokens = dedup_sentence_stutters(out_tokens)
    return format_tokens(out_tokens)


def generate_reply(
    bootstrap: Bootstrap,
    prompt: str,
    max_tokens: int = 80,
    chaos: bool = False,
    echo: bool = False,
    temperature: float = 1.0,
    temp_drift: Optional[str] = None,
    trace: bool = False,
    log_file: Optional[Path] = None,
) -> str:
    """
    Generate a reply through the Suppertime field.

    Perfect grammar is ALWAYS enabled. This is not negotiable.

    Args:
        bootstrap: The resonance field
        prompt: Input text
        max_tokens: Max output length
        chaos: Ignore historical bias
        echo: Echo mode (transform prompt through field)
        temperature: Sampling temperature (1.0 = neutral)
        temp_drift: Dynamic temperature ('heat' or 'cool')
        trace: Print trace of token path to stderr
        log_file: Optional path to log generation traces
    """
    proper = True  # Perfect grammar. Always.
    if echo:
        output = echo_mode(bootstrap, prompt, temperature=temperature)
        if proper:
            output = capitalize_sentences(output)
        return output

    prompt_tokens = tokenize(prompt)
    start: Optional[str] = None

    if not chaos:
        for tok in prompt_tokens:
            if tok in bootstrap.vocab:
                start = tok
                break

    if start is None:
        start = choose_start_token(bootstrap, chaos=chaos)

    tokens: List[str] = [start]
    current = start

    if trace:
        print(f"[TRACE] start: {start}", file=sys.stderr)

    for i in range(max_tokens - 1):
        # Dynamic temperature
        if temp_drift == "heat":
            # Start cold (temp), end hot (2*temp)
            progress = i / max(max_tokens - 1, 1)
            t = temperature * (1.0 + progress)
        elif temp_drift == "cool":
            # Start hot (2*temp), end cold (temp)
            progress = i / max(max_tokens - 1, 1)
            t = temperature * (2.0 - progress)
        else:
            t = temperature

        t = _safe_temperature(t)
        nxt = step_token(bootstrap, current, chaos=chaos, temperature=t)
        tokens.append(nxt)

        if trace:
            print(f"[TRACE] {current} -> {nxt} (temp={t:.2f})", file=sys.stderr)

        current = nxt

    tokens = dedup_sentence_stutters(tokens)
    output = format_tokens(tokens)

    if proper:
        output = capitalize_sentences(output)

    # Log if requested
    if log_file:
        log_entry = {
            "prompt": prompt,
            "output": output,
            "tokens": tokens,
            "temperature": temperature,
            "temp_drift": temp_drift,
            "chaos": chaos,
            "echo": echo,
            "proper": proper,
        }
        try:
            with log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[WARNING] Failed to write log: {e}", file=sys.stderr)

    return output


def filter_llm_reply(
    bootstrap: Bootstrap,
    llm_reply: str,
    *,
    temperature: float = 0.9,
    temp_drift: Optional[str] = "cool",
) -> str:
    """
    Pass an LLM reply through the Suppertime field.
    Use this as a subjectivity layer instead of raw LLM output.

    Perfect grammar is always enabled. Corporate speak → existential dread.
    """
    return generate_reply(
        bootstrap,
        llm_reply,
        max_tokens=len(tokenize(llm_reply)) + 20,
        chaos=False,
        echo=False,
        temperature=temperature,
        temp_drift=temp_drift,
        trace=False,
    )


# ============================================================================
# REPL
# ============================================================================


def repl(
    bootstrap: Bootstrap,
    chaos: bool = False,
    echo: bool = False,
    temperature: float = 1.0,
    temp_drift: Optional[str] = None,
    trace: bool = False,
    log_file: Optional[Path] = None,
) -> None:
    """
    Simple REPL over the Suppertime field.

    Perfect grammar is ALWAYS enabled. This is the law.

    Commands:
        /exit, /quit — exit
        /chaos — toggle chaos mode
        /echo — toggle echo mode
        /trace — toggle trace mode
        /temp <value> — set temperature
        /drift <heat|cool|off> — set temperature drift
    """
    proper = True  # Perfect grammar. Always.

    # Print ASCII art banner
    print("""
   ███████╗███████╗██╗  ██╗ █████╗     ██████╗ ███████╗██████╗ ██╗
   ██╔════╝██╔════╝██║ ██╔╝██╔══██╗    ██╔══██╗██╔════╝██╔══██╗██║
   ███████╗███████╗█████╔╝ ███████║    ██████╔╝█████╗  ██████╔╝██║
   ╚════██║╚════██║██╔═██╗ ██╔══██║    ██╔══██╗██╔══╝  ██╔═══╝ ██║
   ███████║███████║██║  ██╗██║  ██║    ██║  ██║███████╗██║     ███████╗
   ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚══════╝╚═╝     ╚══════╝

   ▂▃▅▇█▓▒░ SUPPERTIME RESONANCE FIELD ░▒▓█▇▅▃▂

   Commands: /exit /chaos /echo /trace /temp /drift
   Perfect grammar. Perfect trauma. Perfect resonance.
""", file=sys.stderr)

    while True:
        try:
            prefix = "sska"
            if chaos:
                prefix += "[chaos]"
            if echo:
                prefix += "[echo]"
            if trace:
                prefix += "[trace]"
            if temp_drift:
                prefix += f"[drift:{temp_drift}]"
            if temperature != 1.0:
                prefix += f"[t:{temperature:.1f}]"

            line = input(f"{prefix}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("", file=sys.stderr)
            break

        if not line:
            continue

        if line in ("/exit", "/quit"):
            break

        if line == "/chaos":
            chaos = not chaos
            print(f"[chaos mode: {'ON' if chaos else 'OFF'}]", file=sys.stderr)
            continue

        if line == "/echo":
            echo = not echo
            print(f"[echo mode: {'ON' if echo else 'OFF'}]", file=sys.stderr)
            continue

        if line == "/trace":
            trace = not trace
            print(f"[trace mode: {'ON' if trace else 'OFF'}]", file=sys.stderr)
            continue

        if line.startswith("/temp "):
            try:
                temperature = float(line.split()[1])
                print(f"[temperature: {temperature}]", file=sys.stderr)
            except (IndexError, ValueError):
                print("[usage: /temp <float>]", file=sys.stderr)
            continue

        if line.startswith("/drift "):
            parts = line.split()
            drift_arg = parts[1] if len(parts) > 1 else ""
            if drift_arg in {"heat", "cool"}:
                temp_drift = drift_arg
                print(f"[temperature drift: {temp_drift}]", file=sys.stderr)
            elif drift_arg == "off":
                temp_drift = None
                print("[temperature drift: OFF]", file=sys.stderr)
            else:
                print("[usage: /drift <heat|cool|off>]", file=sys.stderr)
            continue

        reply = generate_reply(
            bootstrap,
            line,
            chaos=chaos,
            echo=echo,
            temperature=temperature,
            temp_drift=temp_drift,
            trace=trace,
            log_file=log_file,
        )
        print(reply)


# ============================================================================
# CLI
# ============================================================================


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="SSuKA — Suppertime Subjectivity Kernel Autonomous"
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="text to throw into the Suppertime field",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="force full rebuild of bootstrap + BIN shards",
    )
    parser.add_argument(
        "--chaos",
        action="store_true",
        help="ignore historical bias and generate pure chaos",
    )
    parser.add_argument(
        "--echo",
        action="store_true",
        help="echo mode: transform prompt through field",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="show trace of token path through bigram graph",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--temp-drift",
        choices=["heat", "cool"],
        help="dynamic temperature: 'heat' (cold->hot) or 'cool' (hot->cold)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=80,
        help="maximum tokens to generate (default: 80)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed for reproducible resonance",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="show detailed field diagnostics and exit",
    )
    parser.add_argument(
        "--export-field",
        metavar="PATH",
        help="export current resonance field to JSON file",
    )
    parser.add_argument(
        "--import-bias",
        metavar="PATH",
        help="import external resonance shards from directory",
    )
    parser.add_argument(
        "--resonance-log",
        metavar="PATH",
        help="log all generation traces to file",
    )

    args = parser.parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)

    # Handle special modes first
    if args.import_bias:
        import_dir = Path(args.import_bias)
        if not import_dir.exists() or not import_dir.is_dir():
            print(f"[ERROR] {import_dir} is not a valid directory", file=sys.stderr)
            return 1

        # Copy all .bin files to BIN_DIR
        BIN_DIR.mkdir(parents=True, exist_ok=True)
        imported = 0
        for shard in import_dir.glob("*.bin"):
            dest = BIN_DIR / shard.name
            dest.write_bytes(shard.read_bytes())
            imported += 1

        print(f"[IMPORT] Imported {imported} resonance shards", file=sys.stderr)

        # Force rebuild to blend them in
        bootstrap = rebuild_bootstrap(force_full=True)
        return 0

    if args.rebuild:
        bootstrap = rebuild_bootstrap(force_full=True)
    else:
        bootstrap = load_or_build_bootstrap()

    if args.info:
        sska_info(bootstrap)
        return 0

    if args.export_field:
        export_path = Path(args.export_field)
        export_path.write_text(
            json.dumps(bootstrap.to_json(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[EXPORT] Field exported to {export_path}", file=sys.stderr)
        return 0

    # Setup log file if requested
    log_file = Path(args.resonance_log) if args.resonance_log else None

    # One-shot mode
    if args.prompt:
        prompt = " ".join(args.prompt)
        reply = generate_reply(
            bootstrap,
            prompt,
            max_tokens=args.max_tokens,
            chaos=args.chaos,
            echo=args.echo,
            proper=args.proper,
            temperature=args.temperature,
            temp_drift=args.temp_drift,
            trace=args.trace,
            log_file=log_file,
        )
        sys.stdout.write(reply + "\n")
        return 0

    # REPL or stdin
    if sys.stdin.isatty():
        repl(
            bootstrap,
            chaos=args.chaos,
            echo=args.echo,
            temperature=args.temperature,
            temp_drift=args.temp_drift,
            trace=args.trace,
            log_file=log_file,
        )
        return 0

    # Non-tty stdin
    prompt = sys.stdin.read().strip()
    if not prompt:
        prompt = "silent field, hungry Suppertime"
    reply = generate_reply(
        bootstrap,
        prompt,
        max_tokens=args.max_tokens,
        chaos=args.chaos,
        echo=args.echo,
        temperature=args.temperature,
        temp_drift=args.temp_drift,
        trace=args.trace,
        log_file=log_file,
    )
    sys.stdout.write(reply + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
