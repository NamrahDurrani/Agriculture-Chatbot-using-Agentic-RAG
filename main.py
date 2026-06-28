"""
Main entry point for the Agentic RAG pipeline.

Usage:
    # Step 1: Index PDFs (run once)
    python main.py --index

    # Step 2: Ask questions interactively
    python main.py --chat

    # Step 3: Run a single query
    python main.py --query "What diseases affect wheat in Central Asia?"

    # Inspect DB after a run
    python main.py --inspect

    # Re-index from scratch (clear existing vectors)
    python main.py --index --reset

Environment:
    LLM_BACKEND=groq|ollama|qwen_local|qwen_remote   (optional, default: groq)
    GROQ_API_KEY=your_key_here  (required if LLM_BACKEND=groq)
    QWEN_REMOTE_BASE_URL=...    (required if LLM_BACKEND=qwen_remote, e.g.
                                 the current ngrok URL)
    QWEN_REMOTE_MODEL=...       (required if LLM_BACKEND=qwen_remote, must
                                 match exactly what GET {base_url}/v1/models
                                 reports)
    PDF_DIR=/path/to/pdfs       (optional, default: ./pdfs next to this file
                                 — works on Colab, desktop, anywhere)
"""
import sys
print("PYTHON:", sys.executable)
import os
import sys
import uuid
import argparse

# Add pipeline directory to path
sys.path.insert(0, os.path.dirname(__file__))

import db_schema
import pdf_ingestor
import vector_store
from rag_pipeline import AgenticRAGPipeline, inspect_last_query

# ── Default paths ─────────────────────────────────────────────────────────────
# Project-relative default: a "pdfs" folder next to this script. Works
# identically whether this runs in Colab (/content/pdfs), on a desktop
# (wherever you cloned the project), or anywhere else — no environment-
# specific absolute path baked in.
PDF_DIR = os.environ.get(
    "PDF_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdfs")
)

# Which PDFs to ingest (set to None to ingest all in PDF_DIR)
PDF_FILES = [
    "C:/Users/User/OneDrive/Desktop/RapidsAI/agenti_rag/pdfs/i5550e.pdf",                                # FAO Guidelines (text-based)
    "C:/Users/User/OneDrive/Desktop/RapidsAI/agenti_rag/pdfs/PARC Annual Report 2023-24_compressed.pdf", # PARC report (scanned → OCR)
    "C:/Users/User/OneDrive/Desktop/RapidsAI/agenti_rag/pdfs/PbAgriDeptExtenAdapReseWing_SR_2007_20070612.pdf",  # Punjab rules (scanned)
]

# For quick testing, limit pages per PDF (set None for full ingestion)
# PARC has 154 pages — full OCR takes ~8-10 minutes on CPU
MAX_PAGES_TESTING = 30   # Set to None for full ingestion


def do_index(reset: bool = False, test_mode: bool = False):
    """Ingest PDFs and build the vector index."""
    print("\n" + "="*60)
    print("INDEXING PHASE")
    print("="*60)

    db_schema.init_db()

    if reset:
        print("[INDEX] Resetting existing vector collection...")
        vector_store.reset_collection()

    current_size = vector_store.collection_size()
    if current_size > 0 and not reset:
        print(f"[INDEX] Collection already has {current_size} chunks.")
        ans = input("Re-index anyway? (y/N): ").strip().lower()
        if ans != "y":
            print("[INDEX] Skipping. Use --reset to force re-index.")
            return

    max_pages = MAX_PAGES_TESTING if test_mode else None
    if test_mode:
        print(f"[INDEX] TEST MODE: limiting to {max_pages} pages per PDF")

    # Ingest all PDFs
    all_chunks = pdf_ingestor.ingest_all_pdfs(
        pdf_dir=PDF_DIR,
        pdf_files=PDF_FILES,
        max_pages_per_pdf=max_pages,
        verbose=True,
    )

    if not all_chunks:
        print("[INDEX] ERROR: No chunks extracted! Check PDF paths.")
        return

    # Index into ChromaDB
    print(f"\n[INDEX] Adding {len(all_chunks)} chunks to ChromaDB...")
    added = vector_store.index_chunks(all_chunks, verbose=True)
    print(f"\n[INDEX] Done! {added} chunks indexed.")
    print(f"[INDEX] Collection size: {vector_store.collection_size()}")


def do_chat():
    """Interactive chat loop."""
    print("\n" + "="*60)
    print("AGENTIC RAG CHAT")
    print("Knowledge base: PARC Annual Report | FAO Guidelines | Punjab Agri Rules")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'inspect' to see last pipeline execution details")
    print("="*60 + "\n")

    if vector_store.collection_size() == 0:
        print("ERROR: Vector store is empty. Run --index first.")
        return

    pipeline  = AgenticRAGPipeline()
    session_id = str(uuid.uuid4())

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[CHAT] Exiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break
        if user_input.lower() == "inspect":
            inspect_last_query()
            continue

        response = pipeline.run(session_id=session_id, user_query=user_input)
        print(f"\nAssistant: {response}")


def do_single_query(query: str):
    """Run a single query and print the result."""
    if vector_store.collection_size() == 0:
        print("ERROR: Vector store is empty. Run --index first.")
        return

    pipeline   = AgenticRAGPipeline()
    session_id = str(uuid.uuid4())
    response   = pipeline.run(session_id=session_id, user_query=query)
    print(f"\nAnswer: {response}")
    print("\n" + "─"*60)
    inspect_last_query()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Agentic RAG Pipeline for Agricultural Documents"
    )
    parser.add_argument("--index",   action="store_true",
                        help="Ingest PDFs and build vector index")
    parser.add_argument("--reset",   action="store_true",
                        help="Clear existing vector index before indexing")
    parser.add_argument("--test",    action="store_true",
                        help="Limit to first 30 pages per PDF (for quick testing)")
    parser.add_argument("--chat",    action="store_true",
                        help="Start interactive chat")
    parser.add_argument("--query",   type=str,
                        help="Run a single query")
    parser.add_argument("--inspect", action="store_true",
                        help="Show last pipeline execution from DB")
    parser.add_argument("--stats",   action="store_true",
                        help="Show DB statistics")
    args = parser.parse_args()

    if args.index:
        do_index(reset=args.reset, test_mode=args.test)

    elif args.chat:
        do_chat()

    elif args.query:
        do_single_query(args.query)

    elif args.inspect:
        db_schema.init_db()
        inspect_last_query()

    elif args.stats:
        db_schema.init_db()
        conn = db_schema.get_connection()
        print("\n── Database Statistics ──────────────────────────")
        for table in ["sessions", "queries", "pipeline_steps",
                      "llm_calls", "retrieved_docs", "responses"]:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:20s}: {count} rows")
        conn.close()
        print(f"\n  vector_store chunks : {vector_store.collection_size()}")

    else:
        parser.print_help()
        print("\nQuick start:")
        print("  1. Set your Groq API key:  export GROQ_API_KEY=gsk_...")
        print("  2. Index PDFs (test mode): python main.py --index --test")
        print("  3. Ask a question:         python main.py --query 'What is yellow rust?'")
        print("  4. Chat interactively:     python main.py --chat")


if __name__ == "__main__":
    main()
