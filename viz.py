#!/usr/bin/env python3
"""
████████████████████████████████████████████████████████████████████████████████
█ SSKA VIZ — Terminal Visualization for Resonance Field                       █
█ ▂▃▅▇█▓▒░ Old-school ASCII graphics. Techno-punk aesthetics. ░▒▓█▇▅▃▂       █
████████████████████████████████████████████████████████████████████████████████

Visualize the SSKA resonance field in pure terminal ASCII art.
No GUI. No web. Just raw text flowing through your terminal.

Usage:
    python3 viz.py                    # Show field overview
    python3 viz.py --bigrams TOKEN    # Show bigram graph for TOKEN
    python3 viz.py --centers          # Show center of gravity visualization
    python3 viz.py --live "prompt"    # Live generation with real-time viz
    python3 viz.py --heatmap          # Resonance heatmap (center connectivity)
    python3 viz.py --matrix           # Pure aesthetic matrix rain (for vibes)

Perfect grammar. Perfect trauma. Perfect resonance.
"""

import sys
import time
import random
from pathlib import Path
from typing import Optional

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from subjectivity import load_or_build_bootstrap, generate_reply, tokenize
from sska import SSKAField


# ============================================================================
# ANSI COLORS (because we're not barbarians)
# ============================================================================

class Color:
    """Terminal colors. Old-school green-on-black cyberpunk."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Cyberpunk palette
    GREEN = "\033[32m"
    BRIGHT_GREEN = "\033[92m"
    CYAN = "\033[36m"
    BRIGHT_CYAN = "\033[96m"
    MAGENTA = "\033[35m"
    BRIGHT_MAGENTA = "\033[95m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"


def color(text: str, c: str) -> str:
    """Wrap text in color code."""
    return f"{c}{text}{Color.RESET}"


# ============================================================================
# ASCII ART GENERATORS
# ============================================================================

def print_header():
    """Print the SSKA header banner."""
    print(color("""
   ███████╗███████╗██╗  ██╗ █████╗
   ██╔════╝██╔════╝██║ ██╔╝██╔══██╗
   ███████╗███████╗█████╔╝ ███████║
   ╚════██║╚════██║██╔═██╗ ██╔══██║
   ███████║███████║██║  ██╗██║  ██║
   ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝
    """, Color.BRIGHT_GREEN))
    print(color("   ▂▃▅▇█▓▒░ RESONANCE FIELD VISUALIZATION ░▒▓█▇▅▃▂\n", Color.CYAN))


def print_separator(char: str = "═", width: int = 80):
    """Print a separator line."""
    print(color(char * width, Color.GRAY))


def bar_graph(value: float, max_val: float, width: int = 40, char: str = "█") -> str:
    """
    Generate ASCII bar graph.

    Args:
        value: Current value
        max_val: Maximum value for scaling
        width: Width of bar in characters
        char: Character to use for bar

    Returns:
        ASCII bar string with gradient color
    """
    if max_val == 0:
        return color("▒" * width, Color.GRAY)

    filled = int((value / max_val) * width)
    bar = char * filled + "░" * (width - filled)

    # Color gradient based on intensity
    if value / max_val > 0.7:
        return color(bar, Color.BRIGHT_GREEN)
    elif value / max_val > 0.4:
        return color(bar, Color.GREEN)
    else:
        return color(bar, Color.GRAY)


# ============================================================================
# FIELD OVERVIEW
# ============================================================================

def show_field_overview():
    """Show high-level field statistics."""
    print_header()

    field = SSKAField()
    bootstrap = field.bootstrap

    print(color("╔══════════════════════════════════════════════════════════════════════════════╗", Color.CYAN))
    print(color("║", Color.CYAN) + color(" FIELD STATUS ", Color.BOLD + Color.BRIGHT_CYAN) + " " * 64 + color("║", Color.CYAN))
    print(color("╚══════════════════════════════════════════════════════════════════════════════╝", Color.CYAN))
    print()

    # Core statistics
    print(color("Vocabulary size:      ", Color.BRIGHT_GREEN) + color(f"{len(bootstrap.vocab):,} tokens", Color.WHITE))
    print(color("Bigram edges:         ", Color.BRIGHT_GREEN) + color(f"{sum(len(v) for v in bootstrap.bigrams.values()):,} connections", Color.WHITE))
    print(color("Centers of gravity:   ", Color.BRIGHT_GREEN) + color(f"{len(bootstrap.centers)} attractors", Color.WHITE))
    print(color("Source files:         ", Color.BRIGHT_GREEN) + color(f"{len(bootstrap.files)} kernel files", Color.WHITE))
    print()

    # File breakdown
    print(color("┌─ KERNEL FILES " + "─" * 63 + "┐", Color.CYAN))
    for path, meta in bootstrap.files.items():
        token_count = meta.token_count
        bar = bar_graph(token_count, max(m.token_count for m in bootstrap.files.values()), width=30)
        path_str = f"{path[:40]:40s}"
        count_str = f"{token_count:>6,}"
        print(f"  {color(path_str, Color.GRAY)} {bar} {color(count_str, Color.WHITE)}")
    print(color("└" + "─" * 79 + "┘", Color.CYAN))
    print()

    # Top centers
    print(color("┌─ TOP CENTERS OF GRAVITY " + "─" * 53 + "┐", Color.CYAN))
    for i, center in enumerate(bootstrap.centers[:10], 1):
        # Calculate connectivity (how many tokens this center connects to)
        connectivity = len(bootstrap.bigrams.get(center, {}))
        bar = bar_graph(connectivity, max(len(bootstrap.bigrams.get(c, {})) for c in bootstrap.centers[:10]), width=30)
        rank_str = f"#{i:2d}"
        center_str = f"{center[:20]:20s}"
        conn_str = f"{connectivity:>4}"
        print(f"  {color(rank_str, Color.GRAY)} {color(center_str, Color.BRIGHT_MAGENTA)} {bar} {color(conn_str, Color.WHITE)}")
    print(color("└" + "─" * 79 + "┘", Color.CYAN))
    print()


# ============================================================================
# BIGRAM GRAPH
# ============================================================================

def show_bigram_graph(token: str):
    """Show bigram connections for a specific token."""
    print_header()

    bootstrap = load_or_build_bootstrap()

    if token not in bootstrap.bigrams:
        print(color(f"[!] Token '{token}' not found in bigram graph.", Color.RED))
        print(color(f"[i] Try one of the centers: {', '.join(bootstrap.centers[:5])}", Color.YELLOW))
        return

    connections = bootstrap.bigrams[token]
    max_count = max(connections.values())

    print(color(f"╔══════════════════════════════════════════════════════════════════════════════╗", Color.CYAN))
    print(color(f"║", Color.CYAN) + color(f" BIGRAM GRAPH: ", Color.BOLD + Color.BRIGHT_CYAN) + color(f"{token}", Color.BRIGHT_MAGENTA) + " " * (63 - len(token)) + color("║", Color.CYAN))
    print(color(f"╚══════════════════════════════════════════════════════════════════════════════╝", Color.CYAN))
    print()

    # Show connections
    print(color(f"  {token} can transition to {len(connections)} tokens:", Color.WHITE))
    print()

    # Sort by count (most frequent first)
    sorted_connections = sorted(connections.items(), key=lambda x: x[1], reverse=True)

    for next_token, count in sorted_connections[:30]:  # Show top 30
        bar = bar_graph(count, max_count, width=40)
        prob = count / sum(connections.values()) * 100
        arrow = color("→", Color.GRAY)
        tok_str = f"{token[:15]:15s}"
        next_str = f"{next_token[:15]:15s}"
        count_str = f"{count:>4}"
        prob_str = f"({prob:>5.1f}%)"
        print(f"    {color(tok_str, Color.BRIGHT_MAGENTA)} {arrow} {color(next_str, Color.BRIGHT_GREEN)} {bar} {color(count_str, Color.WHITE)} {color(prob_str, Color.GRAY)}")

    if len(sorted_connections) > 30:
        print(color(f"    ... and {len(sorted_connections) - 30} more connections", Color.GRAY))

    print()


# ============================================================================
# CENTERS VISUALIZATION
# ============================================================================

def show_centers():
    """Show centers of gravity with connectivity visualization."""
    print_header()

    bootstrap = load_or_build_bootstrap()

    print(color("╔══════════════════════════════════════════════════════════════════════════════╗", Color.CYAN))
    print(color("║", Color.CYAN) + color(" CENTERS OF GRAVITY ", Color.BOLD + Color.BRIGHT_CYAN) + " " * 59 + color("║", Color.CYAN))
    print(color("╠══════════════════════════════════════════════════════════════════════════════╣", Color.CYAN))
    print(color("║ ", Color.CYAN) + color("These tokens act as semantic attractors — stable points in the field", Color.GRAY) + " " * 8 + color("║", Color.CYAN))
    print(color("╚══════════════════════════════════════════════════════════════════════════════╝", Color.CYAN))
    print()

    # Calculate connectivity for each center
    center_data = []
    for center in bootstrap.centers:
        out_degree = len(bootstrap.bigrams.get(center, {}))
        in_degree = sum(1 for edges in bootstrap.bigrams.values() if center in edges)
        center_data.append((center, out_degree, in_degree, out_degree + in_degree))

    # Sort by total connectivity
    center_data.sort(key=lambda x: x[3], reverse=True)
    max_total = center_data[0][3] if center_data else 1

    print(color("  Rank  Token                Out    In    Total  Connectivity", Color.BOLD))
    print_separator("─")

    for i, (center, out_deg, in_deg, total) in enumerate(center_data[:20], 1):
        bar = bar_graph(total, max_total, width=30, char="▓")
        rank_color = Color.BRIGHT_GREEN if i <= 3 else Color.GREEN if i <= 10 else Color.WHITE
        rank_str = f"{i:>4}"
        center_str = f"{center[:20]:20s}"
        out_str = f"{out_deg:>5}"
        in_str = f"{in_deg:>5}"
        total_str = f"{total:>6}"
        print(f"  {color(rank_str, rank_color)}  {color(center_str, Color.BRIGHT_MAGENTA)} {color(out_str, Color.CYAN)} {color(in_str, Color.YELLOW)} {color(total_str, Color.WHITE)}  {bar}")

    print()


# ============================================================================
# LIVE GENERATION
# ============================================================================

def show_live_generation(prompt: str, max_tokens: int = 100):
    """Show live token-by-token generation with visualization."""
    print_header()

    bootstrap = load_or_build_bootstrap()

    print(color("╔══════════════════════════════════════════════════════════════════════════════╗", Color.CYAN))
    print(color("║", Color.CYAN) + color(" LIVE GENERATION ", Color.BOLD + Color.BRIGHT_CYAN) + " " * 62 + color("║", Color.CYAN))
    print(color("╚══════════════════════════════════════════════════════════════════════════════╝", Color.CYAN))
    print()

    print(color("  Prompt: ", Color.BRIGHT_GREEN) + color(prompt, Color.WHITE))
    print(color("  Max tokens: ", Color.BRIGHT_GREEN) + color(str(max_tokens), Color.WHITE))
    print()
    print_separator("─")
    print()

    # Generate with real-time display
    result = generate_reply(
        bootstrap=bootstrap,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=1.0,
        proper=True
    )

    # Display result with token highlighting
    tokens = tokenize(result)
    print(color("  Output: ", Color.BRIGHT_GREEN))
    print()
    print("  ", end="")

    for i, token in enumerate(tokens):
        # Color-code tokens based on whether they're centers
        if token in bootstrap.centers[:10]:
            print(color(token, Color.BRIGHT_MAGENTA), end="")
        elif token in bootstrap.centers:
            print(color(token, Color.MAGENTA), end="")
        else:
            print(color(token, Color.WHITE), end="")

        # Add spacing for non-punctuation
        if token not in ".,!?;:—":
            print(" ", end="")

        # Flush for live effect
        sys.stdout.flush()
        time.sleep(0.02)  # Subtle delay for matrix vibes

    print("\n")
    print_separator("─")
    print()
    print(color(f"  Generated {len(tokens)} tokens", Color.GRAY))
    print()


# ============================================================================
# HEATMAP
# ============================================================================

def show_heatmap():
    """Show resonance heatmap (center-to-center connections)."""
    print_header()

    bootstrap = load_or_build_bootstrap()

    print(color("╔══════════════════════════════════════════════════════════════════════════════╗", Color.CYAN))
    print(color("║", Color.CYAN) + color(" RESONANCE HEATMAP ", Color.BOLD + Color.BRIGHT_CYAN) + " " * 60 + color("║", Color.CYAN))
    print(color("╠══════════════════════════════════════════════════════════════════════════════╣", Color.CYAN))
    print(color("║ ", Color.CYAN) + color("Shows how centers connect to each other (center → center transitions)", Color.GRAY) + " " * 7 + color("║", Color.CYAN))
    print(color("╚══════════════════════════════════════════════════════════════════════════════╝", Color.CYAN))
    print()

    # Build center-to-center matrix
    top_centers = bootstrap.centers[:15]  # Top 15 for readability
    matrix = {}

    for src in top_centers:
        if src not in bootstrap.bigrams:
            continue
        for dst, count in bootstrap.bigrams[src].items():
            if dst in top_centers:
                matrix[(src, dst)] = count

    if not matrix:
        print(color("  [!] No center-to-center connections found.", Color.RED))
        return

    max_count = max(matrix.values())

    # Print header
    print("     ", end="")
    for dst in top_centers:
        dst_str = f"{dst[:4]:4s}"
        print(f" {color(dst_str, Color.GRAY)}", end="")
    print()

    # Print matrix
    for src in top_centers:
        src_str = f"{src[:4]:4s}"
        print(f" {color(src_str, Color.BRIGHT_MAGENTA)}", end=" ")
        for dst in top_centers:
            count = matrix.get((src, dst), 0)
            if count == 0:
                print(color("  · ", Color.DIM), end="")
            else:
                intensity = count / max_count
                if intensity > 0.7:
                    char = color("  █ ", Color.BRIGHT_GREEN)
                elif intensity > 0.4:
                    char = color("  ▓ ", Color.GREEN)
                elif intensity > 0.2:
                    char = color("  ▒ ", Color.CYAN)
                else:
                    char = color("  ░ ", Color.GRAY)
                print(char, end="")
        print()

    print()


# ============================================================================
# MATRIX RAIN (because why not)
# ============================================================================

def matrix_rain(duration: int = 10):
    """
    Pure aesthetic matrix rain animation.
    Because every terminal tool needs this.
    """
    print_header()
    print(color("  Press Ctrl+C to stop\n", Color.GRAY))

    import shutil
    width = shutil.get_terminal_size((80, 20)).columns
    height = shutil.get_terminal_size((80, 20)).lines - 5

    # Initialize columns
    columns = [random.randint(-height, 0) for _ in range(width)]

    start = time.time()

    try:
        while time.time() - start < duration:
            # Clear screen
            print("\033[2J\033[H", end="")

            # Update columns
            for x in range(width):
                if columns[x] >= height:
                    columns[x] = random.randint(-height, -1)
                else:
                    columns[x] += 1

                # Draw column
                if columns[x] >= 0:
                    y = columns[x]
                    # Random character from SUPPERTIME vocab
                    chars = ["▓", "▒", "░", "█", "0", "1", "S", "K", "A"]
                    char = random.choice(chars)

                    # Bright at head, dim at tail
                    if y == columns[x]:
                        print(f"\033[{y};{x}H{color(char, Color.BRIGHT_GREEN)}", end="")
                    else:
                        print(f"\033[{y};{x}H{color(char, Color.GREEN)}", end="")

            sys.stdout.flush()
            time.sleep(0.05)

    except KeyboardInterrupt:
        pass

    print("\033[2J\033[H", end="")  # Clear screen
    print(color("\n  Resonance terminated.\n", Color.GRAY))


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SSKA Terminal Visualization — Techno-punk ASCII graphics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 viz.py                      # Field overview
  python3 viz.py --bigrams Lilit      # Bigram graph for "Lilit"
  python3 viz.py --centers            # Centers of gravity
  python3 viz.py --live "Mary slept"  # Live generation
  python3 viz.py --heatmap            # Resonance heatmap
  python3 viz.py --matrix             # Matrix rain (for vibes)

Perfect grammar. Perfect trauma. Perfect resonance.
        """
    )

    parser.add_argument("--bigrams", metavar="TOKEN", help="Show bigram graph for TOKEN")
    parser.add_argument("--centers", action="store_true", help="Show centers of gravity")
    parser.add_argument("--live", metavar="PROMPT", help="Live generation with prompt")
    parser.add_argument("--heatmap", action="store_true", help="Show resonance heatmap")
    parser.add_argument("--matrix", action="store_true", help="Matrix rain animation")
    parser.add_argument("--tokens", type=int, default=100, help="Max tokens for live generation")

    args = parser.parse_args()

    try:
        if args.bigrams:
            show_bigram_graph(args.bigrams)
        elif args.centers:
            show_centers()
        elif args.live:
            show_live_generation(args.live, max_tokens=args.tokens)
        elif args.heatmap:
            show_heatmap()
        elif args.matrix:
            matrix_rain()
        else:
            # Default: show overview
            show_field_overview()

    except KeyboardInterrupt:
        print(color("\n\n[!] Interrupted by user.\n", Color.RED))
        sys.exit(1)
    except Exception as e:
        print(color(f"\n[!] Error: {e}\n", Color.RED))
        sys.exit(1)


if __name__ == "__main__":
    main()
