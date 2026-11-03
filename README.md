# haic-libs RAG demonstration and replication materials

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23911785.svg)](https://doi.org/10.5281/zenodo.23911785)

Material that reproduces the demonstration results of the
[haic-libs](https://github.com/gfragi/haic-libs) paper: an instrumented
Retrieval-Augmented Generation (RAG) chatbot, the question sets and session
artifacts of its three evaluation sessions, and the scripts that recompute
the paper's tables from those artifacts.

Everything here uses the published library releases from PyPI,
`haic-metrics==0.3.0` and `haic-logging==0.1.0`
([DOI 10.5281/zenodo.23896317](https://doi.org/10.5281/zenodo.23896317)),
exactly as an outside application would.

## What reproduces what

| Result in the paper | Command | Folder |
|---|---|---|
| Table 4: instrumentation touchpoints and overhead | `python instrumentation_audit.py chatbot.py` | [`rag_chatbot/`](rag_chatbot/) |
| Table 6: HAIC metrics across the three RAG sessions | `python make_table6.py` | [`rag_chatbot/`](rag_chatbot/) |
| Re-running the three sessions | `python run_session.py --session N` | [`rag_chatbot/`](rag_chatbot/) |
| Cross-domain study (recomputation, checksums) | planned | [`cross_domain/`](cross_domain/) |

Setup, API keys, the feedback rule and the exact meaning of each logged
timing are in [rag_chatbot/README.md](rag_chatbot/README.md).

## Quick start

```bash
cd rag_chatbot
python3 -m venv .venv && source .venv/bin/activate     # Python 3.11 or later
pip install -r requirements.txt
python download_source.py                               # arXiv:2407.19098v2, SHA-256 checked
python instrumentation_audit.py chatbot.py              # Table 4
python make_table6.py                                   # Table 6 from sessions/
```

## Citation

Please cite the library and, for these materials, this repository (see
[`CITATION.cff`](CITATION.cff)):

- These materials: [10.5281/zenodo.23911785](https://doi.org/10.5281/zenodo.23911785)
  (all versions; v1.0.0 is [10.5281/zenodo.23911786](https://doi.org/10.5281/zenodo.23911786))
- haic-libs: [10.5281/zenodo.21197960](https://doi.org/10.5281/zenodo.21197960)
  (all versions; v0.3.0 is [10.5281/zenodo.23896317](https://doi.org/10.5281/zenodo.23896317))

## License

MIT. The knowledge source (arXiv:2407.19098) is not redistributed;
`download_source.py` fetches it from arXiv.
