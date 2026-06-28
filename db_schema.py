"""
Normalized SQLite database schema for Agentic RAG pipeline logging.
3NF normalized - no transitive dependencies, all non-key attributes
depend only on the primary key.

Schema:
  sessions       → one per user session
  queries        → one per user question (FK → sessions)
  pipeline_steps → one row per named step (FK → queries)
  llm_calls      → one per LLM invocation (FK → pipeline_steps)
  retrieved_docs → one per retrieved chunk (FK → queries + pipeline_steps)
  responses      → one per final answer (FK → queries)
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "rag_pipeline.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    cur = conn.cursor()

    # ── 1. sessions ──────────────────────────────────────────────────────────
    # Represents one conversation session with a user.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id   TEXT PRIMARY KEY,
            created_at   TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # ── 2. queries ───────────────────────────────────────────────────────────
    # One row per question asked within a session.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            query_id         TEXT PRIMARY KEY,
            session_id       TEXT NOT NULL,
            original_query   TEXT NOT NULL,
            timestamp        TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)

    # ── 3. pipeline_steps ────────────────────────────────────────────────────
    # One row per named step executed for a query (e.g. "query_rewriter",
    # "orchestrator", "retrieval", "reranking", "evaluator", "main_llm").
    # step_order preserves execution sequence.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_steps (
            step_id      TEXT PRIMARY KEY,
            query_id     TEXT NOT NULL,
            step_name    TEXT NOT NULL,
            step_order   INTEGER NOT NULL,
            input_text   TEXT,
            output_text  TEXT,
            duration_ms  REAL,
            status       TEXT NOT NULL DEFAULT 'ok',   -- ok | error | skipped
            timestamp    TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (query_id) REFERENCES queries(query_id)
        )
    """)

    # ── 4. llm_calls ─────────────────────────────────────────────────────────
    # One row per call to an LLM (Groq). Separated from pipeline_steps so
    # token usage and model info are stored only where they apply.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS llm_calls (
            call_id           TEXT PRIMARY KEY,
            step_id           TEXT NOT NULL,
            model_name        TEXT NOT NULL,
            system_prompt     TEXT,
            user_prompt       TEXT,
            response_text     TEXT,
            prompt_tokens     INTEGER,
            completion_tokens INTEGER,
            total_tokens      INTEGER,
            timestamp         TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (step_id) REFERENCES pipeline_steps(step_id)
        )
    """)

    # ── 5. retrieved_docs ────────────────────────────────────────────────────
    # One row per document chunk surfaced during retrieval or reranking.
    # Separated from pipeline_steps to avoid repeating step metadata
    # across multiple chunks (1NF / 2NF compliance).
    cur.execute("""
        CREATE TABLE IF NOT EXISTS retrieved_docs (
            doc_id       TEXT PRIMARY KEY,
            query_id     TEXT NOT NULL,
            step_id      TEXT NOT NULL,
            chunk_text   TEXT NOT NULL,
            source_file  TEXT NOT NULL,
            page_num     INTEGER,
            vector_score REAL,
            bm25_score   REAL,
            rrf_score    REAL,
            final_rank   INTEGER,
            timestamp    TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (query_id)  REFERENCES queries(query_id),
            FOREIGN KEY (step_id)   REFERENCES pipeline_steps(step_id)
        )
    """)

    # ── 6. responses ─────────────────────────────────────────────────────────
    # The final answer returned to the user for each query.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            response_id    TEXT PRIMARY KEY,
            query_id       TEXT NOT NULL UNIQUE,   -- 1-to-1 with query
            final_response TEXT NOT NULL,
            used_rag       INTEGER NOT NULL,        -- 0 / 1 (boolean)
            retry_count    INTEGER NOT NULL DEFAULT 0,
            timestamp      TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (query_id) REFERENCES queries(query_id)
        )
    """)

    conn.commit()
    conn.close()
    print(f"[DB] Initialized at {DB_PATH}")


if __name__ == "__main__":
    init_db()
