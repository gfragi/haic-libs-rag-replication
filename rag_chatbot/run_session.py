#!/usr/bin/env python3
"""
run_session.py
--------------
Run one of the three scripted RAG demonstration sessions from the paper.

The questions come from questions.json and are sent one at a time: each is
shown on screen and sent when you press Enter, so the logged human duration
(duration_s) is the time you take before sending it. After each answer you
give the feedback yourself (y/n/s), exactly as in interactive use.

Usage:
    python run_session.py --session 1 --provider ollama --model llama3.2
    python run_session.py --session 2 --provider anthropic
    python run_session.py --session 3 --source abs     # arXiv abstract page instead of the PDF

Output (in sessions/):
    session_<N>.json          decisions artifact (input to make_table6.py)
    session_<N>.events.jsonl  raw event stream
    session_<N>.run.json      how the session was run (source, provider, model, versions)
"""

from __future__ import annotations

import argparse
import builtins
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import chatbot

HERE = Path(__file__).parent
SESSIONS_DIR = HERE / "sessions"
SOURCES = {
    "pdf": str(HERE / "data" / "2407.19098v2.pdf"),
    "abs": "https://arxiv.org/abs/2407.19098v2",
}


def _scripted_input(questions: list[str]):
    """Replace chatbot's input(): feed questions at the 'You:' prompt, pass others through."""
    remaining = list(questions)
    total = len(questions)

    def _input(prompt: str = "") -> str:
        if prompt != "You: ":
            return builtins.input(prompt)
        if not remaining:
            return "exit"
        q = remaining[0]
        print(f"\n[{total - len(remaining) + 1}/{total}] {q}")
        if builtins.input("  Press Enter to send (q = stop the session): ").strip().lower() == "q":
            return "exit"
        remaining.pop(0)
        print(f"You: {q}")
        return q

    return _input


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a scripted RAG demonstration session.")
    parser.add_argument("--session", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--source", choices=sorted(SOURCES), default="pdf",
                        help="pdf: data/2407.19098v2.pdf (default); abs: arXiv abstract page.")
    parser.add_argument("--provider", choices=["anthropic", "openai", "ollama"], default="anthropic")
    parser.add_argument("--model", default="", help="Model override (see chatbot.py defaults).")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing session file.")
    args = parser.parse_args()

    name = f"session_{args.session}"
    out_artifact = SESSIONS_DIR / f"{name}.json"
    if out_artifact.exists() and not args.force:
        sys.exit(f"{out_artifact} already exists; pass --force to overwrite.")

    source = SOURCES[args.source]
    if args.source == "pdf" and not Path(source).exists():
        sys.exit(f"{source} not found. Run: python download_source.py")

    questions = json.loads((HERE / "questions.json").read_text())["sessions"][name]

    with tempfile.TemporaryDirectory(prefix=f"haic_{name}_") as tmp:
        chatbot.LOG_DIR = Path(tmp)
        chatbot.input = _scripted_input(questions)
        sys.argv = ["chatbot.py", "--sources", source, "--provider", args.provider,
                    "--timeout", str(args.timeout)] + (["--model", args.model] if args.model else [])
        chatbot.main()

        artifacts = list(Path(tmp).glob("haic_decisions_*.json"))
        events = list(Path(tmp).glob("run_*.jsonl"))
        if len(artifacts) != 1:
            sys.exit(f"Expected one decisions artifact, found {len(artifacts)}.")

        artifact = json.loads(artifacts[0].read_text())
        n_turns = sum(1 for d in artifact["decisions"] if d.get("actor_type") == "human")
        complete = n_turns == len(questions)
        stem = name if complete else f"{name}.partial"

        SESSIONS_DIR.mkdir(exist_ok=True)
        shutil.copyfile(artifacts[0], SESSIONS_DIR / f"{stem}.json")
        if events:
            shutil.copyfile(events[0], SESSIONS_DIR / f"{stem}.events.jsonl")

    run_info = {
        "session": name,
        "questions": len(questions),
        "turns_completed": n_turns,
        "source": source if args.source == "abs" else "data/2407.19098v2.pdf (arXiv:2407.19098v2)",
        "provider": args.provider,
        "model": args.model or chatbot.DEFAULT_MODELS[args.provider],
        "haic_logging": version("haic-logging"),
        "haic_metrics": version("haic-metrics"),
        "python": sys.version.split()[0],
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (SESSIONS_DIR / f"{stem}.run.json").write_text(json.dumps(run_info, indent=2) + "\n")

    print(f"\nSaved sessions/{stem}.json ({n_turns}/{len(questions)} turns, "
          f"{len(artifact['decisions'])} decisions).")
    if not complete:
        print("Session incomplete: saved as .partial and ignored by make_table6.py. "
              "Rerun with --force to redo it.")


if __name__ == "__main__":
    main()
