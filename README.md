<<<<<<< HEAD
# Agentic RAG Pipeline — Agricultural Documents

## Project Structure

```
agentic_rag/
├── db_schema.py        # Normalized SQLite schema + helpers
├── pdf_ingestor.py     # PDF text extraction (direct + OCR fallback)
├── vector_store.py     # ChromaDB + TF-IDF embeddings (fully offline)
├── rag_pipeline.py     # All 5 agents + RRF + logging
├── main.py             # CLI entry point
└── requirements.txt    # pip dependencies
```

## Quick Start (Google Colab)

```python
# Cell 1: Install dependencies
!pip install chromadb groq pymupdf rank-bm25 scikit-learn pytesseract Pillow
!apt-get install -y tesseract-ocr -q

# Cell 2: Set Groq API key (free at console.groq.com)
import os
os.environ["GROQ_API_KEY"] = "gsk_your_key_here"

# Cell 3: Copy PDF files to /content/pdfs/
# Upload your PDFs through Colab's file panel

# Cell 4: Index PDFs (first time only — PARC report OCR takes ~8 min for all pages)
!python main.py --index --test   # --test = first 30 pages, fast
# !python main.py --index        # full index (slower)

# Cell 5: Ask questions
!python main.py --query "What diseases affect wheat in Central Asia?"
!python main.py --query "What research did PARC conduct on wheat in 2023-24?"
!python main.py --query "What are the qualifications for Director General Agriculture Punjab?"
```

## Pipeline Architecture

```
User Query
  │
  ▼ [LLM 1] Query Rewriter — rewrites using conversation history
  │
  ▼ [LLM 2] Orchestrator — needs RAG? or answer directly?
  │
  ├─ DIRECT ──────────────────────────────► [LLM] Direct Answer
  │
  └─ RAG
       │
       ▼ ChromaDB Vector Search (TF-IDF)
       ▼ BM25 Search (rank_bm25)
       ▼ RRF Reranking (Reciprocal Rank Fusion)
       │
       ▼ [LLM 3] Relevance Evaluator
       │
       ├─ RELEVANT ─────────────────────► [LLM] Grounded Answer
       │
       └─ NOT RELEVANT
              │
              ├─ retries left? → Query Rewriter (with feedback) → retrieval again
              └─ max retries → Safe Response
```

## Database Schema (3NF Normalized)

```
sessions ──────────┐
                   │
queries ───────────┤ (FK: session_id)
    │              │
    ├── pipeline_steps (FK: query_id)
    │       │
    │       └── llm_calls (FK: step_id)
    │
    ├── retrieved_docs (FK: query_id, step_id)
    │
    └── responses (FK: query_id, 1-to-1)
```

## Why These Tech Choices?

| Component | Choice | Why |
|-----------|--------|-----|
| LLM | `llama3-8b-8192` via Groq | Free tier, fast, 8K context |
| Vector DB | ChromaDB (local) | No server, persistent, Colab-friendly |
| Embeddings | TF-IDF (sklearn) | 100% offline, no HuggingFace download needed |
| OCR | pytesseract + PyMuPDF | PARC & Punjab PDFs are scanned (0 text layer) |
| BM25 | rank_bm25 | Exact keyword matching complements semantic search |
| Reranking | RRF algorithm | Merges vector + BM25 ranked lists without LLM |
| DB | SQLite | Zero setup, normalized, inspectable with DB Browser |

## PDF Handling Strategy

- **i5550e.pdf** (FAO Guidelines): 16 fonts → direct text extraction
- **PARC_Annual_Report_2023-24.pdf**: 0 fonts → scanned → OCR via pytesseract
- **PbAgriDeptExtenAdapReseWing_SR_2007.pdf**: 0 fonts → scanned → OCR

## Upgrade to Sentence-Transformers (when HuggingFace accessible)

In `vector_store.py`, replace `TFIDFEmbeddingFunction` with:
```python
from chromadb.utils import embedding_functions
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
```
=======
# Agriculture-Chatbot-using-Agentic-RAG
>>>>>>> bffd069d3dbd5ba57852fc0b8077fce24737af82
