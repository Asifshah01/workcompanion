# WorkCompanion AI

A multi-agent study and research companion. Upload your own material, ask
questions, get answers that cite it, and have the app remember what you are
weak at.

The design premise is that **an answer you cannot trace is not worth much**. So
the retrieval pipeline refuses to invent evidence, labels the difference between
"this is in your notes" and "this is the model's own knowledge", and tells you
when it does not know.

---

## Table of contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [The agents](#the-agents)
- [How retrieval works](#how-retrieval-works)
- [Architecture](#architecture)
- [Memory and progress](#memory-and-progress)
- [Running the tests](#running-the-tests)
- [Docker](#docker)
- [Deploying to Streamlit Cloud](#deploying-to-streamlit-cloud)
- [Known limits](#known-limits)
- [Troubleshooting](#troubleshooting)

---

## What it does

Ten modes in one app:

| Mode | What it is for |
| --- | --- |
| **Dashboard** | What to work on next, weak areas, recent activity |
| **AI Tutor** | Explanations at four levels, plus a Socratic mode |
| **My Knowledge** | Upload, index and search your documents |
| **Research** | Deep multi-source synthesis and paper analysis |
| **Quiz** | Grounded questions, auto-graded, weak-topic report |
| **Flashcards** | Atomic citation-backed decks for spaced repetition |
| **Exam Mode** | Timed past-paper attempt under exam conditions |
| **Study Planner** | A spaced-repetition plan from your real constraints |
| **Progress** | Mastery per topic, history, recommendations |
| **Settings** | Runtime config, health checks, database tools |

Supported uploads: **PDF** (with OCR fallback for scans), **DOCX**, **PPTX**,
**TXT**, **MD**, **CSV**, **HTML**. Tables are preserved rather than flattened
into prose.

---

## Quick start

Requires **Python 3.11 or 3.12** (3.12 is what CI and Cloud use).

```bash
git clone https://github.com/Asifshah01/workcompanion.git
cd workcompanion

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

# Optional but recommended: enables OCR for scanned PDFs
pip install -r requirements-ocr.txt

cp .env.example .env      # Linux/macOS:  cp .env.example .env
# then paste your Groq key into .env as GROQ_API_KEY=...

streamlit run app.py
```

Open <http://localhost:8501>. Then go to **My Knowledge → Load demo material**
to index a small thermodynamics corpus and try a grounded question straight
away.

### Running without an API key

The app is fully usable with no key at all. It falls back to a deterministic
**offline provider** that extracts real sentences from your indexed documents
rather than inventing them. Output produced this way is labelled as offline in
the UI so it can never be mistaken for model output.

That also means the test suite needs no key, no network and no rate-limit
budget.

---

## Configuration

Everything lives in `.env` (copy from `.env.example`). Only one variable is
required:

```dotenv
GROQ_API_KEY=gsk_...
```

Everything else has a working default. The settings that matter most:

| Variable | Default | Notes |
| --- | --- | --- |
| `LLM_PROVIDER` | `groq` | Or `offline` to force the extractive provider |
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | Free-tier, ~0.4 s median latency |
| `GROQ_MAX_TOKENS` | `1000` | Free tier allows ~1000 **output** tokens/min |
| `GROQ_MAX_RETRIES` | `4` | With server-directed backoff |
| `DENSE_WEIGHT` / `SPARSE_WEIGHT` | `0.7` / `0.3` | Hybrid retrieval blend |
| `RETRIEVAL_TOP_K` | `6` | Chunks passed to the model |
| `RERANKER` | `local` | Or `cross_encoder` for higher quality |
| `MIN_RETRIEVAL_CONFIDENCE` | `0.35` | Below this the agent refuses |
| `ENABLE_CREWAI` | `true` | CrewAI is optional, never imported eagerly |
| `ENABLE_WEB_RESEARCH` | `false` | Needs a search provider configured |
| `DATABASE_URL` | `sqlite:///data/workcompanion.db` | Postgres-compatible |

The **Settings** page edits these at runtime. Changes are written to
`.env.local` and applied without a restart, so you can retune retrieval while
the app is running and see the confidence scores move.

### Storage

Vector store, SQLite and uploaded files all live under `DATA_DIR` (default
`./data`). Change it and everything follows.

---

## The agents

| Agent | Responsibility |
| --- | --- |
| **nexus** | Routes every request and extracts parameters from it |
| **tutor** | Adaptive teaching across beginner / undergraduate / graduate / expert |
| **socratic** | Scaffolded dialogue and misconception probing |
| **knowledge** | Cited, retrieval-grounded answers from your material |
| **research** | Paper analysis and document/web synthesis |
| **quiz** | Grounded generation, grading and weak-topic reports |
| **flashcards** | Atomic, citation-backed cards for spaced repetition |
| **planner** | Spaced-repetition plans from your real constraints |

Question-answering agents return a common `AgentResult` envelope (`answer`,
`confidence`, `confidence_level`, `grounding`, `sources`, `warnings`,
`latency_ms`, `tokens`) so the UI can render any of them uniformly.

Generation agents deliberately return the **artifact** instead — a `StudyPlan`
is far more useful to the planner page as a plan than as a string, and a
`QuizSet` has to keep its questions intact to be gradeable.

**CrewAI is optional.** It is used for three multi-agent flows —
`deep_research`, `exam_prep`, `study_session` — and is imported lazily. The app
runs with `ENABLE_CREWAI=false` and no CrewAI installed at all.

---

## How retrieval works

Ingestion:

1. **Load** the file, preserving tables and page numbers.
2. **Chunk semantically** — split on headings and sentence boundaries, not a
   fixed character count, and overlap so a claim split across a boundary stays
   retrievable.
3. **Attach metadata** — document, page, heading path, section, position.
4. **Deduplicate** by content hash, so re-uploading the same file is detected
   rather than silently doubling every chunk.
5. **Index** into dense embeddings plus a self-contained BM25 index.

Query:

1. **Rewrite and expand** the question (conversation-aware).
2. **Multi-query fan-out** — several paraphrases, merged by reciprocal rank.
3. **Hybrid search** — dense + sparse, blended `0.7 / 0.3`.
4. **Rerank** the candidates.
5. **Compress** to the sentences that actually answer the question.
6. **Score confidence** and check citations.

### Hallucination control

This is the part worth reading.

- **Confidence is gated on lexical alignment.** A question whose terms barely
  appear in the retrieved chunks gets its score multiplied down. Without this
  gate, off-topic questions scored *higher* than on-topic ones — retrieval
  found something plausible-looking and the model happily elaborated on it.
- **Below the confidence floor, the agent refuses** instead of answering. It
  returns a specific "your material does not cover this" message, not a vague
  hedge.
- **Citations are verifiable.** Each citation carries document, page and chunk
  id, and the claim checker verifies the answer's assertions against the cited
  text.
- **Grounding is labelled, not implied.** Every result declares which of
  *retrieved fact*, *model reasoning* or *general knowledge* it contains.
- **General knowledge is allowed but never disguised.** With
  `allow_general_knowledge=True`, an out-of-corpus question is answered — but
  with no citations, a deliberately low confidence, a `GENERAL_KNOWLEDGE`
  label and an explicit warning. Earlier this flag was threaded into the prompt
  but never consulted at the refusal gate, so it refused anyway.

---

## Architecture

```
workcompanion/
  config/       Settings, env loading, validation
  schemas/      Pydantic models shared across every layer
  llm/          Groq client, offline provider, retry and backoff
  loaders/      PDF/DOCX/PPTX/TXT/MD/CSV/HTML, OCR fallback
  rag/          chunking, embeddings, hybrid search, sparse index,
                query transform, reranker, compressor, citations, pipeline
  database/     SQLAlchemy models, repositories, session handling
  memory/       learner profile, conversation memory, progress tracker
  agents/       the eight agents + registry
  crew/         optional CrewAI flows and the Groq bridge
  tools/        web search and other external tools
  ui/           layout, theme, components, and one module per page
app.py          Streamlit entry point
```

Roughly 17k lines across 93 modules.

### Notable implementation choices

**Hybrid search without `rank_bm25`.** The sparse index is implemented in-repo
and persisted as JSON, scored document-scoped. Fewer moving parts, and
per-document removal is exact, which matters when a user re-indexes a file.

**One required secret.** `GROQ_API_KEY`. `BaseSettings` reads `os.environ`, so
Streamlit Cloud secrets work with no code change.

**Runtime reconfiguration.** Settings edits write `.env.local`, then reload
settings, reset the agent bundle and re-bootstrap — so retuning retrieval takes
effect on the next question.

**Navigation is queued, not assigned.** The sidebar is a widget, so writing the
page key directly gets overwritten on the next rerun. Pages call
`navigate(key)`, which queues the destination; the sidebar applies it.

---

## Memory and progress

- **Learner profile** — level, style, subject, weekly hours, and a rolling list
  of recorded misconceptions, injected into the tutor's prompt.
- **Conversation memory** — a bounded window of recent turns plus a rolling
  summary. Only the tail belongs in a prompt; the summary carries the rest.
- **Progress tracker** — accuracy per topic and subject, mastery, quiz history,
  flashcard counts, study minutes and activity by day.
- **Weak-area detection** — topics ranked by lowest mastery, which drives the
  dashboard recommendation and the quiz page's focus areas.

The Progress page does not just chart history; it tells you what to do next.

---

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

**134 tests, no API key and no network required.** The suite runs against the
offline provider on a throwaway data directory, so a green run means the
orchestration is correct rather than the model happened to be helpful.

| File | Covers |
| --- | --- |
| `tests/test_rag.py` | Chunking, hybrid search, BM25, citations, confidence |
| `tests/test_agents.py` | Every agent, the envelope, refusal behaviour |
| `tests/test_persistence.py` | Repositories, learner profile, progress |
| `tests/test_ui.py` | Page registry, navigation, transcript, settings |
| `tests/conftest.py` | Offline fixtures on temporary storage |

Several tests encode bugs that were found and fixed, and are worth not
re-breaking:

- Off-topic questions must not score higher than on-topic ones.
- Re-ingesting a file must not double the index.
- Forced re-index must replace, not accumulate.
- A session factory must rebind when the database changes — it silently kept
  returning sessions bound to the old database.

---

## Docker

```bash
docker build -t workcompanion .
docker run --rm -p 8501:8501 --env-file .env -v wc-data:/app/data workcompanion
```

Open <http://localhost:8501>.

The image includes Tesseract and libmagic. Mount `/app/data` to keep your
library across restarts.

---

## Deploying to Streamlit Cloud

1. Push the repo to GitHub (it is already a valid target).
2. In Streamlit Cloud: **New app → Deploy from GitHub**.
3. Select the repository and branch `main`.
4. Set **Python version** to `3.12` and **requirements file** to
   `requirements-cloud.txt` — it installs a CPU-only PyTorch, which is roughly
   5× smaller than the default CUDA wheels and avoids a build timeout.
5. Deploy, then open **Settings → Secrets** and add:

   ```toml
   GROQ_API_KEY = "gsk_..."
   ```

6. Reload. The app picks it up automatically.

`packages.toml` installs `tesseract-ocr` and `libmagic1`, and `runtime.txt`
pins `python-3.12.10`.

### On Streamlit Cloud

**Storage is ephemeral.** The vector store and SQLite reset on every restart and
on every code change. That is a platform limitation, not a bug. Use
**My Knowledge → Load demo material** to re-seed, or upload your own files.

---

## Known limits

Stated plainly rather than left to be discovered:

- **Groq free tier is ~1000 output tokens/minute.** Sustained multi-user use on
  the free tier will hit 429s. The client retries with server-directed backoff
  and a `max_tokens` clamp, but the ceiling is real. A paid key removes it.
- **Long answers truncate** at `GROQ_MAX_TOKENS`. When that happens it is
  *disclosed in the output* rather than silently cut.
- **Offline mode is extractive, not generative.** It quotes your documents. It
  will not explain a concept your notes never mention.
- **Cloud storage resets on restart.**
- **BM25 persistence is JSON**, which is fine at this scale but not at millions
  of chunks.

---

## Troubleshooting

**"Add a document in My Knowledge first" on every question.**
The index is empty. That message is the app refusing to answer without
evidence, which is the intended behaviour. Use **Load demo material** or upload
a file.

**Everything is labelled offline.**
`GROQ_API_KEY` is not set or not visible. Check **Settings → Environment**
(it reports `configured` / `missing`, never the value). On Cloud, secrets must
be added under **Settings → Secrets**, not in a local `.env`.

**429 rate limit errors.**
The free-tier ceiling. Lower `GROQ_MAX_TOKENS`, switch to
`GROQ_MODEL=qwen/qwen3.8-27b`, or use a paid key.

**An answer is refused even though my notes cover it.**
Check the retrieval confidence on the Progress page and loosen
`MIN_RETRIEVAL_CONFIDENCE` in Settings. Also confirm the right subject was
selected during ingestion — topic progress is subject-scoped.

**The first question is slow.**
The embedding model and, if enabled, the cross-encoder download on first use.
Subsequent requests are much faster.

**PDFs come back empty.**
Likely a scan. Install the OCR extras (`pip install -r requirements-ocr.txt`)
and make sure Tesseract is on `PATH`. Ingestion records whether OCR was used.

**Embeddings download on first run.**
`sentence-transformers` fetches its model once. Needs network access on first
start only.

---

## License

MIT