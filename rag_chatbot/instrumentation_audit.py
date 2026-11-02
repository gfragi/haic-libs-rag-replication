#!/usr/bin/env python3
"""
instrumentation_audit.py
------------------------
Measure haic instrumentation overhead in a Python source file.

Usage:
    python instrumentation_audit.py chatbot.py
    python instrumentation_audit.py compute_session_metrics.py

Output goes into the paper as evidence of low integration overhead.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Keywords that identify haic instrumentation lines
_HAIC_PATTERNS = re.compile(
    r"HaicLogger|log_decision|log_event|export_decisions_artifact"
    r"|haic_metrics|compute_metrics"
)


def audit(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()

    app_lines: list[tuple[int, str]] = []    # (lineno, text) — non-empty, non-comment
    haic_lines: list[tuple[int, str]] = []   # subset that match haic patterns

    for lineno, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        app_lines.append((lineno, raw))
        if _HAIC_PATTERNS.search(raw):
            haic_lines.append((lineno, raw))

    total = len(app_lines)
    n_haic = len(haic_lines)
    pct = (n_haic / total * 100) if total else 0.0

    sep = "─" * 70
    print()
    print(sep)
    print(f"  Instrumentation Audit: {path.name}")
    print(sep)
    print(f"  Total application lines   : {total}")
    print(f"  Instrumentation lines     : {n_haic}")
    print(f"  Overhead                  : {pct:.1f} %")
    print()
    print("  Instrumentation lines (with line numbers):")
    print()
    for lineno, text in haic_lines:
        print(f"  {lineno:4d}  {text.rstrip()}")
    print()
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit haic instrumentation overhead in a Python source file."
    )
    parser.add_argument("file", help="Python source file to audit")
    args = parser.parse_args()

    path = Path(args.file).expanduser().resolve()
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    audit(path)


if __name__ == "__main__":
    main()
