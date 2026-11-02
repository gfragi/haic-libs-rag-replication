#!/usr/bin/env python3
"""
RAG Chatbot CLI
---------------
Usage:
    python chatbot.py
    python chatbot.py --sources url1,/path/to/file.pdf,notes.md
    python chatbot.py --provider openai --sources notes.md
    python chatbot.py --provider ollama --model mistral --sources report.pdf

At startup you are prompted for knowledge sources if none are given via
--sources.  Each source can be a URL or a local file (.txt, .md, .pdf).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from haic_logging import HaicLogger

# ── Constants ────────────────────────────────────────────────────────────────

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 4
LOG_DIR = Path(__file__).parent / "logs"

DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-haiku-5-5",
    "openai":    "gpt-4o-mini",
    "ollama":    "llama3",
}

_SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "context provided below. If the context does not contain enough information "
    "to answer fully, say so clearly."
)

# ── Text extraction ──────────────────────────────────────────────────────────


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; RAGBot/1.0; +https://github.com/rag-demo)"
    )
}


def _fetch_url(url: str) -> str:
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def extract_text(source: str) -> str:
    """Return raw text from a URL or local file path."""
    if source.startswith("http://") or source.startswith("https://"):
        return _fetch_url(source)
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {source}")
    if p.suffix.lower() == ".pdf":
        return _read_pdf(source)
    return _read_text(source)


# ── Chunking ─────────────────────────────────────────────────────────────────


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


# ── Embedding & retrieval ────────────────────────────────────────────────────


class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print("Loading embedding model…", flush=True)
        self.model = SentenceTransformer(model_name)
        self.chunks: list[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def add(self, chunks: list[str]) -> None:
        if not chunks:
            return
        vecs = self.model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
        self.chunks.extend(chunks)
        self.embeddings = (
            vecs if self.embeddings is None else np.vstack([self.embeddings, vecs])
        )

    def query(self, text: str, k: int = TOP_K) -> list[str]:
        if self.embeddings is None or not self.chunks:
            return []
        q_vec = self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)
        norms_store = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        norms_q = np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-10
        sims = (self.embeddings / norms_store) @ (q_vec / norms_q).T
        top_idx = np.argsort(sims.flatten())[::-1][:k]
        return [self.chunks[i] for i in top_idx]


# ── LLM backends ─────────────────────────────────────────────────────────────

GenerateFn = Callable[[str, str], tuple[str, float]]
"""Signature: generate_response(prompt, context) -> (answer, latency_ms)"""


def _make_anthropic(model: str, timeout: int) -> GenerateFn:
    try:
        import httpx
        from anthropic import Anthropic
    except ImportError:
        raise SystemExit("anthropic package not installed. Run: pip install anthropic")
    client = Anthropic(timeout=httpx.Timeout(timeout))

    def generate_response(prompt: str, context: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        msg = client.messages.create(
            model=model,
            max_tokens=1024,
            system=f"{_SYSTEM_INSTRUCTION}\n\nContext:\n{context}",
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text, (time.perf_counter() - t0) * 1000

    return generate_response


def _make_openai(model: str, timeout: int) -> GenerateFn:
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("openai package not installed. Run: pip install openai")
    client = OpenAI(timeout=timeout)

    def generate_response(prompt: str, context: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": f"{_SYSTEM_INSTRUCTION}\n\nContext:\n{context}"},
                {"role": "user",   "content": prompt},
            ],
        )
        return resp.choices[0].message.content, (time.perf_counter() - t0) * 1000

    return generate_response


def _make_ollama(model: str, timeout: int, base_url: str = "http://localhost:11434") -> GenerateFn:
    api_url = f"{base_url}/api/chat"

    def generate_response(prompt: str, context: str) -> tuple[str, float]:
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": f"{_SYSTEM_INSTRUCTION}\n\nContext:\n{context}"},
                {"role": "user",   "content": prompt},
            ],
        }
        t0 = time.perf_counter()
        resp = requests.post(api_url, json=payload, timeout=600)
        latency_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code == 404:
            raise RuntimeError(
                f"Model '{model}' not found in Ollama. Run: ollama pull {model}"
            )
        resp.raise_for_status()
        return resp.json()["message"]["content"], latency_ms

    return generate_response


def _friendly_error(exc: Exception) -> str:
    """Return a concise, single-line reason from provider API exceptions."""
    # OpenAI / Anthropic SDKs expose .message or .body
    if hasattr(exc, "message") and exc.message:
        return str(exc.message)
    if hasattr(exc, "body") and isinstance(exc.body, dict):
        inner = exc.body.get("error", {})
        if isinstance(inner, dict) and inner.get("message"):
            return inner["message"]
    return str(exc).splitlines()[0]


def make_generate_fn(provider: str, model: str, timeout: int) -> GenerateFn:
    """Return a generate_response(prompt, context) -> (answer, latency_ms) callable."""
    if provider == "anthropic":
        return _make_anthropic(model, timeout)
    if provider == "openai":
        return _make_openai(model, timeout)
    if provider == "ollama":
        return _make_ollama(model, timeout)
    raise ValueError(f"Unknown provider {provider!r}. Choose: anthropic, openai, ollama")


# ── Ingestion ────────────────────────────────────────────────────────────────


def ingest_sources(sources: list[str], store: VectorStore) -> int:
    for src in sources:
        src = src.strip()
        if not src:
            continue
        try:
            print(f"  Loading: {src}", flush=True)
            store.add(chunk_text(extract_text(src)))
        except Exception as exc:
            print(f"  [warning] Could not load '{src}': {exc}", file=sys.stderr)
    return len(store.chunks)


# ── Per-turn feedback ────────────────────────────────────────────────────────


def _ask_feedback() -> bool | None:
    """
    Ask after each AI response whether the answer is correct and complete
    with respect to the source document(s).
    Re-prompts once on invalid input, then defaults to None.

    Returns:
        True  → y (correct and complete)
        False → n (incorrect or incomplete)
        None  → s / invalid (skip — correct field left as null)
    """
    _VALID = {"y": True, "n": False, "s": None}
    for attempt in range(2):
        raw = input(
            "  Correct and complete w.r.t. the source? [y/n/s] (s=skip): "
        ).strip().lower()
        if raw in _VALID:
            return _VALID[raw]
        if attempt == 0:
            print("  Please enter y, n, or s.")
    return None


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_sources(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Chatbot CLI")
    parser.add_argument(
        "--sources",
        default="",
        help="Comma-separated list of URLs or file paths.",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "ollama"],
        default="anthropic",
        help="LLM backend to use (default: anthropic).",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model name override. Defaults: anthropic=claude-haiku-5-5, "
             "openai=gpt-4o-mini, ollama=llama3.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Read timeout in seconds for LLM API calls (default: 300).",
    )
    args = parser.parse_args()

    provider: str = args.provider
    model: str = args.model or DEFAULT_MODELS[provider]
    timeout: int = args.timeout

    # ── Collect sources ──────────────────────────────────────────────────────
    sources: list[str] = []
    if args.sources:
        sources = parse_sources(args.sources)
    else:
        print("Enter knowledge sources (URLs or file paths), one per line.")
        print("Press Enter on an empty line when done.\n")
        while True:
            line = input("  Source: ").strip()
            if not line:
                break
            sources.extend(parse_sources(line))

    if not sources:
        print("No sources provided. Exiting.")
        sys.exit(1)

    # ── Embed knowledge base ─────────────────────────────────────────────────
    store = VectorStore()
    print(f"\nIngesting {len(sources)} source(s)…")
    total_chunks = ingest_sources(sources, store)

    print(f"\nSources loaded: {total_chunks} chunks from {len(sources)} source(s)")
    print(f"Provider: {provider}  |  Model: {model}")
    print("Session started. Type your questions. Type 'exit' to end and export.\n")

    # ── Build the provider-agnostic generate function ─────────────────────────
    generate_response: GenerateFn = make_generate_fn(provider, model, timeout)

    # ── Set up logging ────────────────────────────────────────────────────────
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    haiclog = HaicLogger(
        log_dir=LOG_DIR,
        pilot_tag="rag-demo",
        app_name="haic_rag",
        app_version="0.1.0",
        model_name=model,
        model_type="llm",
    )

    # ── Chat loop ─────────────────────────────────────────────────────────────
    turn = 0
    last_input_time = time.perf_counter()
    turns: list[dict] = []         # {"question": ..., "response": ...}
    ai_decisions: list[dict] = []  # references into haiclog._decisions for backfill

    try:
        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            # ── Exit gate — always checked before any LLM call ───────────
            if query.lower() in {"exit", "quit"}:
                break
            if not query:
                continue

            turn += 1
            turn_id = f"turn_{turn}"

            now = time.perf_counter()
            duration_s = now - last_input_time
            last_input_time = now
            query_t = time.time()

            # Retrieve context + generate (provider-agnostic)
            chunks = store.query(query, k=TOP_K)
            context = "\n\n---\n\n".join(chunks) if chunks else "(no context retrieved)"

            try:
                answer, latency_ms = generate_response(query, context)
            except Exception as exc:
                reason = _friendly_error(exc)
                hint = (
                    "Is Ollama running? Try: ollama serve"
                    if provider == "ollama"
                    else "Try --provider ollama for local inference."
                )
                print(f"\nProvider error: {reason}. {hint}\n", file=sys.stderr)
                turn -= 1   # don't count this as a completed turn
                last_input_time = time.perf_counter()
                continue

            print(f"\nAssistant: {answer}\n")

            # Log human decision only for completed turns, stamped at query time
            # (correct backfilled after feedback below)
            human_decision = haiclog.log_decision(
                actor_type="human",
                action="query",
                object_id=turn_id,
                duration_s=duration_s,
                payload={"query": query},
                t=query_t,
            )

            # Log AI decision
            ai_decision = haiclog.log_decision(
                actor_type="ai",
                action="respond",
                object_id=turn_id,
                latency_ms=latency_ms,
                payload={"answer": answer, "num_chunks_used": len(chunks)},
            )

            # Explicit per-turn feedback — applied to both decisions for this turn.
            # Timer reset happens AFTER feedback so inter-turn wait isn't inflated.
            feedback = _ask_feedback()
            human_decision["correct"] = feedback
            ai_decision["correct"] = feedback

            turns.append({"question": query, "response": answer})
            ai_decisions.append(ai_decision)

            last_input_time = time.perf_counter()

    finally:
        haiclog.close()
        artifact_path = haiclog.export_decisions_artifact()
        print(f"\nSession ended. Decisions exported to: {artifact_path}")


if __name__ == "__main__":
    main()
