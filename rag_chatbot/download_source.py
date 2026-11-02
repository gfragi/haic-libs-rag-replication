#!/usr/bin/env python3
"""
download_source.py
------------------
Download the knowledge source used in the RAG demonstration: the HAIC
evaluation framework paper, arXiv:2407.19098v2 (PDF), and verify its SHA-256.

Usage:
    python download_source.py            # -> data/2407.19098v2.pdf

Standard library only, so it can run before the requirements are installed.
The PDF is not redistributed in this repository.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://arxiv.org/pdf/2407.19098v2"
SHA256 = "bd63d637d530062d22078f8da177f75999f7990dc267ccd9b1a078ad8638aef7"
DEST = Path(__file__).parent / "data" / "2407.19098v2.pdf"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if DEST.exists() and _sha256(DEST) == SHA256:
        print(f"Already present and verified: {DEST}")
        return

    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} …", flush=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "haic-libs-rag-example/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()

    tmp = DEST.with_suffix(".part")
    tmp.write_bytes(data)
    digest = _sha256(tmp)
    if digest != SHA256:
        tmp.unlink()
        sys.exit(f"Checksum mismatch: expected {SHA256}, got {digest}. Not saved.")
    tmp.replace(DEST)
    print(f"Saved and verified: {DEST} ({len(data):,} bytes)")


if __name__ == "__main__":
    main()
