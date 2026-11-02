# RAG chatbot demonstration

The instrumented Retrieval-Augmented Generation (RAG) chatbot used in the
paper's demonstration (Section 6.2, Tables 4 and 6). It is a command-line
chatbot that answers questions about user-supplied documents, logs every turn
with `haic-logging`, and is evaluated offline with `haic-metrics`.

- Retrieval: 512-character chunks (64 overlap), local `all-MiniLM-L6-v2`
  embeddings, cosine similarity, top 4 chunks.
- Generation: Anthropic, OpenAI or a local Ollama model (`--provider`).
- Instrumentation: five lines of `haic-logging` code in `chatbot.py`
  (see [Instrumentation overhead](#instrumentation-overhead)).

## Files

| File | Purpose |
|---|---|
| `chatbot.py` | The instrumented chatbot (interactive CLI) |
| `run_session.py` | Runs one of the paper's three scripted sessions and saves its artifact |
| `questions.json` | The question sets of sessions 1–3 |
| `download_source.py` | Downloads and verifies the knowledge source (arXiv:2407.19098v2 PDF) |
| `make_table6.py` | Recomputes Table 6 from the three session artifacts |
| `instrumentation_audit.py` | Counts application lines and `haic-libs` touchpoints (Table 4) |
| `requirements.in` / `requirements.txt` | Direct dependencies / fully pinned environment |
| `sessions/` | Session artifacts (`session_1.json` … `session_3.json`) |

## Setup

Python 3.11 or later.

```bash
cd rag_chatbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # pins haic-metrics==0.3.0, haic-logging==0.1.0
python download_source.py              # -> data/2407.19098v2.pdf (SHA-256 checked)
```

The PDF is not redistributed in this repository; `download_source.py` fetches
it from arXiv and refuses to save a file whose checksum differs. The first run
also downloads the embedding model (`all-MiniLM-L6-v2`, about 90 MB) from
Hugging Face.

### LLM provider and API key

The provider is chosen with `--provider`; keys are read from environment
variables. Do not put keys in files inside the repository.

| `--provider` | Environment variable | Default model |
|---|---|---|
| `anthropic` (default) | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `ollama` | none (local server on `localhost:11434`) | `llama3` |

```bash
export ANTHROPIC_API_KEY="..."         # or OPENAI_API_KEY="..."
```

The sessions reported in the paper used Ollama with `llama3.2`:

```bash
ollama pull llama3.2
ollama serve                           # if it is not already running
```

## Running a session

```bash
python run_session.py --session 1 --provider ollama --model llama3.2
python run_session.py --session 2 --provider ollama --model llama3.2
python run_session.py --session 3 --provider ollama --model llama3.2
```

Each session loads the knowledge source, then goes through its question set
(session 1: 5 questions; sessions 2 and 3: the same 10 questions):

1. The next question is shown. Press **Enter** to send it (`q` stops the session).
2. The chatbot retrieves context and prints its answer.
3. You rate the answer: **y**, **n** or **s** (see the rule below).

The session is saved to `sessions/session_<N>.json` (decisions artifact),
`sessions/session_<N>.events.jsonl` (raw events) and
`sessions/session_<N>.run.json` (source, provider, model, library versions).
An existing session file is not overwritten unless you pass `--force`. A
session stopped early is saved as `session_<N>.partial.json` and is not used
for Table 6.

Use `--source abs` to load the arXiv abstract page
(`https://arxiv.org/abs/2407.19098v2`, 10 chunks) instead of the full PDF
(218 chunks).

### Feedback rule

The answer to "Correct and complete w.r.t. the source?" sets the `correct`
field of **both** decisions of the turn (the human query and the AI answer):

| Key | Meaning | `correct` |
|---|---|---|
| `y` | The answer is **correct and complete with respect to the source document** | `True` |
| `n` | The answer is wrong, incomplete, or not supported by the source document | `False` |
| `s` | Skip: you cannot judge the answer | `None` (excluded from $T_r$) |

Judge each answer against the source document only, not against general
knowledge. An invalid key is asked again once and then recorded as `s`.

### What the timings measure

- `duration_s` (human): time from the end of the previous turn (your rating,
  or the session start) until you press Enter to send the question.
- `latency_ms` (AI): wall-clock time of the LLM call.

F, D and HCL therefore depend on how long you take before sending each
question, and $\bar{L}$ on the provider, model and hardware.

## Recomputing Table 6

```bash
python make_table6.py                  # reads sessions/session_{1,2,3}.json
python make_table6.py --latex          # also prints the LaTeX rows
```

It computes the metrics with `haic_metrics.compute_metrics(rt_max_s=60)` and
prints a per-session table and Table 6 (mean, sample standard deviation, min,
max of N, F, D, HCL, $T_r$, mean AI latency and A). EL, ES and S are not
computed: there is no no-AI baseline for open-ended question answering and no
surrogate policy.

## Interactive use

The chatbot also runs on any documents, without the scripted questions:

```bash
python chatbot.py --provider ollama --model llama3.2 --sources data/2407.19098v2.pdf
python chatbot.py --sources "https://example.com/page,notes.md"
```

Local files can be `.pdf`, `.txt` or `.md`; URLs are read as HTML pages (for
a PDF, download it first and pass the local path). Type `exit` to end the
session; the artifact is written to `logs/`.

## Instrumentation overhead

```bash
python instrumentation_audit.py chatbot.py
```

`chatbot.py` has 327 non-empty, non-comment lines, of which 5 call
`haic-libs` (1.5%):

| Line | Call | Purpose |
|---|---|---|
| 30 | `from haic_logging import HaicLogger` | Library import |
| 341 | `HaicLogger(...)` | Session initialization |
| 400 | `log_decision(actor_type="human", ...)` | User query |
| 410 | `log_decision(actor_type="ai", ...)` | AI response |
| 431 | `export_decisions_artifact()` | Export the decisions artifact |

`haiclog.close()` (line 430) ends the session; the audit script does not
count it.
