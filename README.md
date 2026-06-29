# Agentic RAG Pipeline for Agricultural Documents

An agentic Retrieval-Augmented Generation system that answers natural-
language questions over a mixed corpus of agricultural PDFs -- including
fully scanned, non-digital documents -- by retrieving evidence, verifying
that evidence is actually sufficient before answering, and logging every
decision into a normalized, auditable database.

## Architecture

```
                         User Query
                             |
                             v
              +---------------------------+
              |   Load Conversation        |
              |   History (app logic)      |
              +---------------------------+
                             |
                             v
              +---------------------------+
              |   Query Rewriter (LLM)     |
              |   rewrite using context    |
              +---------------------------+
                             |
                             v
              +---------------------------+
              |   Orchestrator (LLM)       |
              |   does this need RAG?      |
              +-------------+-------------+
                 No |               | Yes
                    v               v
        +-----------------+   +------------------------+
        |  Direct LLM      |   |  Retrieve: ChromaDB     |
        |  answer          |   |  vector search + BM25   |
        +-----------------+   |  (parallel, not an LLM   |
                    |          |  call)                  |
                    |          +-----------+--------------+
                    |                      v
                    |          +------------------------+
                    |          |  Re-rank: RRF fusion    |
                    |          |  (not an LLM call)      |
                    |          +-----------+--------------+
                    |                      v
                    |          +------------------------+
                    |          |  Relevance Evaluator    |
                    |          |  (LLM): sufficient /    |
                    |          |  partial / none         |
                    |          +-----------+--------------+
                    |             suff./partial |  none
                    |                      v     v
                    |          +-----------+  +---------------+
                    |          | Generate   |  | Retry limit?  |
                    |          | grounded,  |  | Yes -> rewrite|
                    |          | cited      |  | with feedback,|
                    |          | answer     |  | retry (max 2) |
                    |          | (LLM)      |  | No  -> Safe   |
                    |          +-----------+  | Response       |
                    |                      |  +---------------+
                    +----------------------+----+
                                 v
                       Return Response to User
                                 v
                  Save Query + Response + every
                  intermediate step (normalized DB)
```

This implements every node in the original architecture diagram
(`docs/RAG_architecture.pdf`), with two nodes deliberately extended
beyond their literal description after real failures during
development -- see `docs/architecture.md` for the full rationale.

## What makes this "agentic," not just RAG

- **Orchestrator** decides *whether* retrieval is even needed, before
  spending time on it.
- **Relevance Evaluator** checks retrieved evidence is actually
  sufficient *before* generation -- a 3-state verdict (`sufficient` /
  `partial` / `none`), not a blind retrieve-then-answer.
- **Retry loop** automatically rewrites the query using the evaluator's
  specific feedback and tries again (up to 2x) before falling back to
  an honest "not found" response, rather than guessing.

## The parsing challenge

| Document | Type | Extraction method |
|---|---|---|
| FAO Guidelines for Monitoring Crop Diseases | Digitally typeset | Direct text-layer extraction (PyMuPDF) |
| PARC Annual Report 2023-24 | Fully scanned, zero embedded fonts | OCR (Tesseract) |
| Punjab Agriculture Dept. Service Rules | Fully scanned, zero embedded fonts | OCR (Tesseract) |

Extraction strategy is decided **per page**, not per document: direct
text extraction is tried first; if a page yields under 50 characters,
it's rasterized and OCR'd instead. This correctly handles documents
with mixed content rather than assuming one method per file.

## Components

| Component | Technology |
|---|---|
| PDF text extraction | PyMuPDF + Tesseract OCR (adaptive, per-page) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) -- semantic, CPU-friendly, local |
| Vector store | ChromaDB (persistent, local) |
| Sparse retrieval | BM25 |
| Re-ranking | Reciprocal Rank Fusion (RRF) |
| LLM | Configurable: Groq / Qwen3.5 (local or remote) / Ollama |
| Logging | SQLite, normalized to 3NF, 6 linked tables |

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Tesseract OCR must also be installed as a system package:
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- macOS: `brew install tesseract`
- Linux: `sudo apt install tesseract-ocr`

Place source PDFs in `pdfs/` (or set `PDF_DIR`).

## Usage

```bash
export GROQ_API_KEY=gsk_...

python main.py --index              # build the index (run once)
python main.py --query "..."        # ask a single question
python main.py --chat               # multi-turn conversation
python main.py --inspect            # full trace of the last query
python main.py --stats              # database statistics
```

See `docs/test_questions.md` for a curated set of questions exercising
every distinct path through the pipeline.

## LLM backends

Set via `LLM_BACKEND`: `groq` (default) | `qwen_remote` | `qwen_local` | `ollama`.
Each backend implements the same `call(system_prompt, user_prompt) -> text`
interface, so the retrieval/evaluation logic never needs to know which
provider is active.


