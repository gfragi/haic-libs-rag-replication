#!/usr/bin/env python3
"""
make_table6.py
--------------
Recompute Table 6 of the paper (HAIC metrics across the three RAG chatbot
sessions) from the session artifacts.

Usage:
    python make_table6.py                       # sessions/session_{1,2,3}.json
    python make_table6.py a.json b.json c.json  # any artifacts
    python make_table6.py --latex               # also print LaTeX table rows

Metrics are computed with haic_metrics.compute_metrics(rt_max_s=60) and no
task baseline (EL, EfficiencyScore and S are therefore not computed).
Std is the sample standard deviation (n - 1).
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import haic_metrics
from haic_metrics import compute_metrics, load_decisions_artifact

HERE = Path(__file__).parent
DEFAULT = [HERE / "sessions" / f"session_{i}.json" for i in (1, 2, 3)]
RT_MAX_S = 60.0

# (key, header, format) in table order; N is the number of logged decisions
COLUMNS = [
    ("N", "N", "{:.0f}"),
    ("F", "F", "{:.2f}"),
    ("D", "D (s)", "{:.2f}"),
    ("HCL", "HCL", "{:.3f}"),
    ("Tr", "Tr", "{:.3f}"),
    ("L", "L (ms)", "{:,.0f}"),
    ("A", "A", "{:.3f}"),
]


def session_row(path: Path) -> dict:
    artifact = load_decisions_artifact(path)
    result = compute_metrics(artifact, rt_max_s=RT_MAX_S)
    m = result["metrics"]
    return {
        "name": path.stem,
        "N": float(result["window_summary"]["counts"]["decisions_used"]),
        "F": m.get("F"),
        "D": m.get("D"),
        "HCL": m.get("HCL"),
        "Tr": m.get("Tr"),
        "L": m.get("ai_latency_mean_ms"),
        "A": m.get("A"),
        "notes": [m[k] for k in ("A_note", "Tr_note") if m.get(k)],
    }


def _fmt(fmt: str, v) -> str:
    return "n/a" if v is None else fmt.format(v)


def _stats(values: list) -> dict:
    xs = [v for v in values if v is not None]
    if not xs:
        return dict.fromkeys(("Mean", "Std", "Min", "Max"))
    return {
        "Mean": statistics.mean(xs),
        "Std": statistics.stdev(xs) if len(xs) > 1 else None,
        "Min": min(xs),
        "Max": max(xs),
    }


def _print_table(title: str, first_header: str, rows: list[tuple[str, list[str]]]) -> None:
    headers = [first_header] + [h for _, h, _ in COLUMNS]
    widths = [max(len(headers[i]), *(len(r[i]) for r in [[n] + c for n, c in rows])) for i in range(len(headers))]
    line = lambda cells: "  ".join(c.ljust(widths[0]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(cells))
    print(f"\n{title}")
    print(line(headers))
    print(line(["-" * w for w in widths]))
    for name, cells in rows:
        print(line([name] + cells))


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute Table 6 from RAG session artifacts.")
    parser.add_argument("artifacts", nargs="*", type=Path, default=DEFAULT)
    parser.add_argument("--latex", action="store_true", help="Also print LaTeX rows for the paper.")
    args = parser.parse_args()

    missing = [p for p in args.artifacts if not p.exists()]
    if missing:
        sys.exit("Missing artifact(s): " + ", ".join(map(str, missing)))

    sessions = [session_row(p) for p in args.artifacts]

    _print_table(
        "Per-session metrics",
        "Session",
        [(s["name"], [_fmt(f, s[k]) for k, _, f in COLUMNS]) for s in sessions],
    )

    stats = {k: _stats([s[k] for s in sessions]) for k, _, _ in COLUMNS}
    agg_rows = [(stat, [_fmt(f, stats[k][stat]) for k, _, f in COLUMNS]) for stat in ("Mean", "Std", "Min", "Max")]
    _print_table(f"Table 6: HAIC metrics across {len(sessions)} RAG chatbot sessions", "Stat.", agg_rows)

    print(
        f"\nhaic-metrics {haic_metrics.__version__}; HCL with RT_max = {RT_MAX_S:g} s; "
        "EL, ES and S not computed (no task baseline, no surrogate policy); Std uses n - 1."
    )
    for s in sessions:
        for note in s["notes"]:
            print(f"  {s['name']}: {note}")

    if args.latex:
        print("\n% LaTeX rows for Table 6 (paste between \\midrule and \\bottomrule)")
        for stat, cells in agg_rows:
            tex = [c.replace(",", "{,}") for c in cells]
            print(f"{stat:<4} & " + " & ".join(tex) + r" \\")


if __name__ == "__main__":
    main()
