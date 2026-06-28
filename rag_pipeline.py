"""
Agentic RAG Pipeline
=====================
Implements the full pipeline from your diagram:

  User Query
    → [1] Query Rewriter (LLM Call 1)
    → [2] Orchestrator (LLM Call 2): needs RAG?
       ├─ No  → Main LLM Call (direct)
       └─ Yes → [3] Retrieve (ChromaDB vector search)
              → [4] Re-rank (RRF algorithm)
              → [5] Document Relevance Evaluator (LLM Call 3)
                 ├─ Relevant → Main LLM Call (grounded)
                 └─ Not relevant → Retry?
                      ├─ Yes → Query Rewriter with evaluator feedback → [3]
                      └─ No  → Safe Response

Every step is logged to the normalized SQLite database.

Model: groq/llama3-8b-8192  (set GROQ_API_KEY env variable)

Hardening notes (small-model safety nets):
  - Query rewriter uses temperature=0.0, max_tokens=60, few-shot anchoring,
    and a post-hoc _sanitize_rewrite() that detects repetition-loop
    degeneration and falls back to the original query.
  - Relevance evaluator returns a 3-state verdict (sufficient/partial/none)
    instead of binary true/false, so topically-related-but-incomplete
    evidence still reaches generation with an explicit confidence hedge
    instead of being discarded or over-trusted.
"""

import os
import time
import uuid
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from groq import Groq
from rank_bm25 import BM25Okapi

import db_schema
import vector_store

# ── LLM backend selection ──────────────────────────────────────────────────────
# Set LLM_BACKEND=ollama to use a locally-deployed model instead of Groq.
# Default stays "groq" so existing behavior is unchanged unless you opt in.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "groq").lower()

# ── LLM backend selection ──────────────────────────────────────────────────────
# Set LLM_BACKEND=ollama      to use a remote/local Ollama server (/api/chat).
# Set LLM_BACKEND=qwen_local  to load Qwen3.5 directly via transformers on
#                              this machine's GPU.
# Set LLM_BACKEND=qwen_remote to call a remote OpenAI-compatible Qwen3.5
#                              server (transformers serve / vLLM / SGLang),
#                              e.g. a teammate's ngrok-tunneled instance.
# Default stays "groq" so existing behavior is unchanged unless you opt in.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "groq").lower()

if LLM_BACKEND == "ollama":
    from llm_client_ollama import OllamaClient
elif LLM_BACKEND == "qwen_local":
    from llm_client_qwen_local import QwenLocalClient, QWEN_MODEL_ID
elif LLM_BACKEND == "qwen_remote":
    from llm_client_qwen_remote import QwenRemoteClient, QWEN_REMOTE_MODEL
else:
    from groq import Groq

# ── Configuration ─────────────────────────────────────────────────────────────
GROQ_MODEL    = "llama3-8b-8192"
OLLAMA_MODEL  = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
TOP_K_VECTOR  = 10       # retrieve this many from ChromaDB
TOP_K_BM25    = 10       # retrieve this many from BM25
TOP_K_FINAL   = 5        # keep this many after RRF reranking
MAX_RETRIES   = 2        # max retrieval retry loops

# Whichever backend is active, this is the name that gets written to
# llm_calls.model_name — keeps DB logs honest about what actually ran,
# instead of always recording GROQ_MODEL regardless of backend.
if LLM_BACKEND == "ollama":
    ACTIVE_MODEL_NAME = OLLAMA_MODEL
elif LLM_BACKEND == "qwen_local":
    ACTIVE_MODEL_NAME = QWEN_MODEL_ID
elif LLM_BACKEND == "qwen_remote":
    ACTIVE_MODEL_NAME = QWEN_REMOTE_MODEL
else:
    ACTIVE_MODEL_NAME = GROQ_MODEL


# ── DB Logging helpers ────────────────────────────────────────────────────────

def _log_step(
    query_id: str,
    step_name: str,
    step_order: int,
    input_text: str = "",
    output_text: str = "",
    duration_ms: float = 0.0,
    status: str = "ok",
) -> str:
    """Insert a row into pipeline_steps, return step_id."""
    step_id = str(uuid.uuid4())
    conn = db_schema.get_connection()
    conn.execute(
        """INSERT INTO pipeline_steps
           (step_id, query_id, step_name, step_order,
            input_text, output_text, duration_ms, status)
           VALUES (?,?,?,?,?,?,?,?)""",
        (step_id, query_id, step_name, step_order,
         input_text[:4000], output_text[:4000], duration_ms, status),
    )
    conn.commit()
    conn.close()
    print(f"  [STEP {step_order}] {step_name} | {duration_ms:.0f}ms | {status}")
    return step_id


def _log_llm_call(
    step_id: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    response_text: str,
    usage: Dict,
) -> str:
    """Insert a row into llm_calls, return call_id."""
    call_id = str(uuid.uuid4())
    conn = db_schema.get_connection()
    conn.execute(
        """INSERT INTO llm_calls
           (call_id, step_id, model_name, system_prompt, user_prompt,
            response_text, prompt_tokens, completion_tokens, total_tokens)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (call_id, step_id, model_name,
         system_prompt[:2000], user_prompt[:4000], response_text[:4000],
         usage.get("prompt_tokens", 0),
         usage.get("completion_tokens", 0),
         usage.get("total_tokens", 0)),
    )
    conn.commit()
    conn.close()
    return call_id


def _log_retrieved_docs(
    query_id: str,
    step_id: str,
    docs: List[Dict],
):
    """Insert rows into retrieved_docs for each chunk."""
    conn = db_schema.get_connection()
    for doc in docs:
        doc_id = str(uuid.uuid4())
        conn.execute(
            """INSERT INTO retrieved_docs
               (doc_id, query_id, step_id, chunk_text, source_file,
                page_num, vector_score, bm25_score, rrf_score, final_rank)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (doc_id, query_id, step_id,
             doc.get("chunk_text", "")[:2000],
             doc.get("source_file", ""),
             doc.get("page_num", 0),
             doc.get("vector_score"),
             doc.get("bm25_score"),
             doc.get("rrf_score"),
             doc.get("final_rank")),
        )
    conn.commit()
    conn.close()


def _log_response(
    query_id: str,
    final_response: str,
    used_rag: bool,
    retry_count: int,
):
    """Insert the final response into responses table."""
    response_id = str(uuid.uuid4())
    conn = db_schema.get_connection()
    conn.execute(
        """INSERT INTO responses
           (response_id, query_id, final_response, used_rag, retry_count)
           VALUES (?,?,?,?,?)""",
        (response_id, query_id, final_response, int(used_rag), retry_count),
    )
    conn.commit()
    conn.close()


# ── Groq LLM wrapper ──────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable not set.\n"
                "Get a free key at https://console.groq.com"
            )
        self.client = Groq(api_key=api_key)

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> Tuple[str, Dict]:
        """
        Call Groq LLM. Returns (response_text, usage_dict).
        """
        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text  = response.choices[0].message.content.strip()
        usage = {
            "prompt_tokens":     response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens":      response.usage.total_tokens,
        }
        return text, usage


def get_llm_client():
    """
    Factory: returns the active LLM client based on LLM_BACKEND.

      LLM_BACKEND=groq        (default) -> LLMClient (Groq hosted API)
      LLM_BACKEND=ollama                -> OllamaClient (Ollama server,
                                            /api/chat schema)
      LLM_BACKEND=qwen_local             -> QwenLocalClient (Qwen3.5 loaded
                                            directly via transformers on
                                            this machine's GPU)
      LLM_BACKEND=qwen_remote            -> QwenRemoteClient (remote
                                            OpenAI-compatible server, e.g.
                                            ngrok-tunneled transformers
                                            serve / vLLM / SGLang)

    All expose the same .call(system_prompt, user_prompt, max_tokens,
    temperature) -> (text, usage_dict) interface, so nothing else in the
    pipeline needs to know which backend is active.
    """
    if LLM_BACKEND == "ollama":
        print(f"[PIPELINE] Using backend: Ollama ({OLLAMA_MODEL})")
        return OllamaClient(model=OLLAMA_MODEL)
    elif LLM_BACKEND == "qwen_local":
        print(f"[PIPELINE] Using backend: Qwen3.5 local ({QWEN_MODEL_ID})")
        return QwenLocalClient(model_id=QWEN_MODEL_ID)
    elif LLM_BACKEND == "qwen_remote":
        print(f"[PIPELINE] Using backend: Qwen3.5 remote ({QWEN_REMOTE_MODEL})")
        return QwenRemoteClient(model=QWEN_REMOTE_MODEL)
    else:
        print(f"[PIPELINE] Using backend: Groq ({GROQ_MODEL})")
        return LLMClient()


# ── BM25 Index (built once from all indexed chunks) ───────────────────────────

class BM25Index:
    """
    Lightweight BM25 index built from the ChromaDB collection documents.
    Re-built in memory each session (fast for ~thousands of chunks).
    """
    def __init__(self):
        self._corpus: List[str] = []
        self._metadata: List[Dict] = []
        self._bm25 = None

    def build_from_collection(self):
        """Pull all documents from ChromaDB and build BM25 index."""
        collection = vector_store._get_collection()
        if collection.count() == 0:
            print("[BM25] WARNING: Empty collection, BM25 index will be empty.")
            return

        # ChromaDB get() returns all docs (no query needed)
        results = collection.get(include=["documents", "metadatas"])
        self._corpus   = results["documents"]
        self._metadata = results["metadatas"]

        tokenized = [doc.lower().split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized)
        print(f"[BM25] Index built with {len(self._corpus)} documents")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Return top-k results with BM25 scores."""
        if self._bm25 is None or not self._corpus:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked[:top_k]:
            meta = self._metadata[idx] if idx < len(self._metadata) else {}
            results.append({
                "chunk_text":  self._corpus[idx],
                "source_file": meta.get("source_file", "unknown"),
                "page_num":    meta.get("page_num", 0),
                "bm25_score":  round(float(score), 4),
            })
        return results


# ── RRF Reranking ─────────────────────────────────────────────────────────────

def rrf_rerank(
    vector_results: List[Dict],
    bm25_results: List[Dict],
    top_k: int = 5,
    k: int = 60,
) -> List[Dict]:
    """
    Reciprocal Rank Fusion — combines two ranked lists.
    RRF score = Σ 1/(k + rank_i)

    Args:
        vector_results: Ranked list from vector search.
        bm25_results:   Ranked list from BM25 search.
        top_k:          How many results to return.
        k:              RRF constant (60 is standard).

    Returns:
        Merged, re-ranked list with rrf_score and final_rank.
    """
    rrf_scores: Dict[str, float] = {}
    doc_map:    Dict[str, Dict]  = {}

    def _key(doc: Dict) -> str:
        """Unique key for deduplication: file + page + first 100 chars."""
        return f"{doc['source_file']}|{doc['page_num']}|{doc['chunk_text'][:100]}"

    # Score from vector results
    for rank, doc in enumerate(vector_results, start=1):
        key = _key(doc)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in doc_map:
            doc_map[key] = {**doc, "bm25_score": None}

    # Score from BM25 results
    for rank, doc in enumerate(bm25_results, start=1):
        key = _key(doc)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in doc_map:
            doc_map[key] = {**doc, "vector_score": None}
        else:
            # Merge BM25 score into existing entry
            doc_map[key]["bm25_score"] = doc.get("bm25_score")

    # Sort by RRF score
    sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

    reranked = []
    for final_rank, key in enumerate(sorted_keys[:top_k], start=1):
        entry = {**doc_map[key]}
        entry["rrf_score"]  = round(rrf_scores[key], 6)
        entry["final_rank"] = final_rank
        reranked.append(entry)

    return reranked


# ── Context builder ───────────────────────────────────────────────────────────

def _build_context(docs: List[Dict], max_chars: int = 3000) -> str:
    """Format retrieved docs into a context string for the LLM."""
    parts = []
    total = 0
    for doc in docs:
        header = (f"[Source: {doc['source_file']} | Page {doc['page_num']} | "
                  f"Relevance rank: {doc.get('final_rank', '?')}]")
        snippet = doc["chunk_text"][:500]
        entry = f"{header}\n{snippet}"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)
    return "\n\n---\n\n".join(parts)


# ── Conversation memory ───────────────────────────────────────────────────────

class ConversationMemory:
    """Simple in-memory conversation history (last N turns)."""
    def __init__(self, max_turns: int = 6):
        self.history: List[Dict[str, str]] = []
        self.max_turns = max_turns

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        # Keep only last max_turns pairs
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def get_formatted(self) -> str:
        if not self.history:
            return "No previous conversation."
        lines = []
        for turn in self.history[-6:]:   # last 3 pairs
            prefix = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {turn['content'][:300]}")
        return "\n".join(lines)


# ── The Agentic RAG Pipeline ──────────────────────────────────────────────────

class AgenticRAGPipeline:
    """
    Full agentic RAG pipeline with logging to normalized SQLite DB.

    Usage:
        pipeline = AgenticRAGPipeline()
        response = pipeline.run(
            session_id="session-123",
            user_query="What diseases affect wheat crops in Central Asia?"
        )
    """

    def __init__(self):
        print("\n[PIPELINE] Initializing Agentic RAG Pipeline...")
        self.llm     = get_llm_client()
        self.bm25    = BM25Index()
        self.memory  = ConversationMemory()
        db_schema.init_db()
        print("[PIPELINE] Ready.\n")

    def _ensure_bm25_built(self):
        if self.bm25._bm25 is None:
            self.bm25.build_from_collection()

    @staticmethod
    def _sanitize_rewrite(rewritten: str, original_query: str) -> str:
        """
        Safety net against small-model degeneration (repetition loops,
        run-on entity hallucination). If the rewrite looks broken, fall
        back to the original query rather than trust the model output.

        Heuristics:
          - Too long (small models looping rarely stay short)
          - Repeats the same word/phrase 3+ times (loop signature)
          - Contains multiple sentences (should be exactly one)
        """
        text = rewritten.strip().strip('"').strip()

        # Heuristic 1: hard length cap — a real rewrite of a short question
        # should never balloon past ~200 chars.
        if len(text) > 220:
            return original_query

        # Heuristic 2: repeated phrase detection (e.g. "sister organization"
        # appearing 3+ times is the classic degenerate-loop signature).
        words = text.lower().split()
        for n in (2, 3):
            grams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
            if grams:
                most_common = max(set(grams), key=grams.count)
                if grams.count(most_common) >= 3:
                    return original_query

        # Heuristic 3: should be one sentence — multiple '?' or run-on commas
        # past a threshold suggests the model kept generating instead of
        # stopping.
        if text.count("?") > 1 or text.count(",") > 6:
            return original_query

        if not text:
            return original_query

        return text

    # ── Agent 1: Query Rewriter ───────────────────────────────────────────────

    def _query_rewriter(
        self,
        query_id: str,
        original_query: str,
        conversation_history: str,
        evaluator_feedback: str = "",
        step_order: int = 1,
    ) -> str:
        """
        LLM Call 1: Rewrite the query using conversation context.
        If evaluator_feedback is provided, also incorporate that to improve
        the query for retry.
        """
        t0 = time.time()

        feedback_section = ""
        if evaluator_feedback:
            feedback_section = (
                f"\n\nIMPORTANT — Previous retrieval failed. "
                f"Evaluator feedback: {evaluator_feedback}\n"
                f"Rewrite the query to address this gap."
            )

        system = (
            "You are a query rewriting assistant for a retrieval system.\n\n"
            "STRICT RULES:\n"
            "1. Output ONE rewritten query only. One sentence. Under 25 words.\n"
            "2. NEVER invent, substitute, or add named entities, organizations, "
            "or acronyms that are not in the original query or conversation "
            "history. If the original says 'PARC', the rewrite must still say "
            "'PARC' — never replace it with another organization.\n"
            "3. Do NOT explain, list alternatives, or repeat phrases.\n"
            "4. If the query is already clear, return it unchanged.\n\n"
            "Examples:\n"
            "Original: What diseases did they study last year?\n"
            "History: User asked about PARC wheat research.\n"
            "Rewritten: What diseases did PARC study in wheat research last year?\n\n"
            "Original: Tell me about cotton pests.\n"
            "History: No previous conversation.\n"
            "Rewritten: Tell me about cotton pests."
        )
        user = (
            f"Conversation history:\n{conversation_history}\n\n"
            f"Original query: {original_query}"
            f"{feedback_section}\n\n"
            f"Rewritten query (one sentence, under 25 words, same entities only):"
        )

        rewritten, usage = self.llm.call(
            system, user, max_tokens=60, temperature=0.0
        )
        rewritten = self._sanitize_rewrite(rewritten, original_query)
        duration = (time.time() - t0) * 1000

        step_id = _log_step(
            query_id, "query_rewriter", step_order,
            input_text=original_query,
            output_text=rewritten,
            duration_ms=duration,
        )
        _log_llm_call(step_id, ACTIVE_MODEL_NAME, system, user, rewritten, usage)

        print(f"  [REWRITER] '{original_query}' → '{rewritten}'")
        return rewritten

    # ── Agent 2: Orchestrator ─────────────────────────────────────────────────

    def _orchestrator(
        self,
        query_id: str,
        rewritten_query: str,
        step_order: int = 2,
    ) -> bool:
        """
        LLM Call 2: Decide if RAG is needed.
        Returns True if RAG is needed, False if LLM can answer directly.
        """
        t0 = time.time()

        system = (
            "You are a routing assistant for an agricultural knowledge base. "
            "Decide if the question requires retrieving documents from the "
            "knowledge base (RAG) or if you can answer directly from general "
            "knowledge. Output ONLY one word: 'RAG' or 'DIRECT'."
        )
        user = (
            f"Question: {rewritten_query}\n\n"
            f"The knowledge base contains: FAO guidelines for monitoring crop "
            f"diseases/pests/weeds, PARC Pakistan agricultural research "
            f"annual report 2023-24, Punjab Agriculture Department service "
            f"rules 2007.\n\n"
            f"Decision (RAG or DIRECT):"
        )

        decision_text, usage = self.llm.call(system, user, max_tokens=10)
        duration = (time.time() - t0) * 1000

        needs_rag = "RAG" in decision_text.upper()

        step_id = _log_step(
            query_id, "orchestrator", step_order,
            input_text=rewritten_query,
            output_text=f"Decision: {'RAG' if needs_rag else 'DIRECT'}",
            duration_ms=duration,
        )
        _log_llm_call(step_id, ACTIVE_MODEL_NAME, system, user, decision_text, usage)

        print(f"  [ORCHESTRATOR] Decision: {'RAG' if needs_rag else 'DIRECT'}")
        return needs_rag

    # ── Step 3: Retrieve (ChromaDB + BM25) ───────────────────────────────────

    def _retrieve(
        self,
        query_id: str,
        rewritten_query: str,
        step_order: int = 3,
    ) -> List[Dict]:
        """
        ChromaDB vector search + BM25 search.
        NOT an LLM call — pure retrieval.
        """
        t0 = time.time()

        vector_results = vector_store.similarity_search(
            rewritten_query, top_k=TOP_K_VECTOR
        )
        bm25_results = self.bm25.search(rewritten_query, top_k=TOP_K_BM25)

        duration = (time.time() - t0) * 1000

        summary = (
            f"Vector: {len(vector_results)} docs, "
            f"BM25: {len(bm25_results)} docs"
        )
        step_id = _log_step(
            query_id, "retrieval", step_order,
            input_text=rewritten_query,
            output_text=summary,
            duration_ms=duration,
        )
        # Log raw retrieval results
        for doc in vector_results:
            doc["bm25_score"] = None
        _log_retrieved_docs(query_id, step_id, vector_results + bm25_results)

        print(f"  [RETRIEVAL] {summary}")
        return vector_results, bm25_results

    # ── Step 4: RRF Reranking ─────────────────────────────────────────────────

    def _rerank(
        self,
        query_id: str,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        step_order: int = 4,
    ) -> List[Dict]:
        """
        RRF algorithm to merge vector + BM25 results.
        NOT an LLM call.
        """
        t0 = time.time()

        reranked = rrf_rerank(vector_results, bm25_results, top_k=TOP_K_FINAL)
        duration = (time.time() - t0) * 1000

        summary = f"RRF merged → top {len(reranked)} docs"
        step_id = _log_step(
            query_id, "reranking", step_order,
            input_text=f"vector:{len(vector_results)} bm25:{len(bm25_results)}",
            output_text=summary,
            duration_ms=duration,
        )
        _log_retrieved_docs(query_id, step_id, reranked)

        print(f"  [RERANK] {summary}")
        for d in reranked:
            print(f"    rank={d['final_rank']} rrf={d['rrf_score']:.5f} "
                  f"src={d['source_file']} p.{d['page_num']}")
        return reranked

    # ── Agent 3: Document Relevance Evaluator ─────────────────────────────────

    def _evaluator(
        self,
        query_id: str,
        original_query: str,
        rewritten_query: str,
        docs: List[Dict],
        step_order: int = 5,
    ) -> Tuple[bool, str]:
        """
        LLM Call 3: Judge if retrieved docs are relevant and sufficient.
        Returns (is_relevant: bool, feedback: str, verdict: str).
        verdict is one of "sufficient", "partial", "none".
        """
        t0 = time.time()

        context = _build_context(docs, max_chars=1500)

        system = (
            "You are a document relevance evaluator for a RAG system.\n\n"
            "Given a user question and retrieved document excerpts, classify "
            "the match into exactly one of three categories:\n\n"
            '  "sufficient"  — the documents directly answer the question.\n'
            '  "partial"     — the documents are topically related and contain '
            "useful context (e.g. related experiments, trials, programs) but "
            "do not state the exact answer.\n"
            '  "none"        — the documents are unrelated to the question.\n\n'
            "Output ONLY a JSON object:\n"
            '  "verdict": "sufficient" | "partial" | "none"\n'
            '  "feedback": brief explanation (max 50 words), and if "none" or '
            '"partial", what specific information is missing.'
        )
        user = (
            f"Original question: {original_query}\n"
            f"Rewritten question: {rewritten_query}\n\n"
            f"Retrieved documents:\n{context}\n\n"
            f"Evaluation (JSON only):"
        )

        eval_text, usage = self.llm.call(system, user, max_tokens=150)
        duration = (time.time() - t0) * 1000

        # Parse JSON response
        verdict = "none"
        feedback = eval_text
        try:
            import re
            json_match = re.search(r'\{.*?\}', eval_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                verdict = str(parsed.get("verdict", "none")).lower()
                feedback = parsed.get("feedback", eval_text)
        except (json.JSONDecodeError, AttributeError):
            low = eval_text.lower()
            verdict = "sufficient" if "sufficient" in low else (
                "partial" if "partial" in low else "none"
            )

        # "sufficient" or "partial" both count as relevant enough to proceed
        # to generation; "none" triggers the retry loop. The verdict string
        # itself is passed through so generation can adjust its confidence.
        is_relevant = verdict in ("sufficient", "partial")

        step_id = _log_step(
            query_id, "relevance_evaluator", step_order,
            input_text=rewritten_query,
            output_text=f"verdict={verdict} | {feedback}",
            duration_ms=duration,
        )
        _log_llm_call(step_id, ACTIVE_MODEL_NAME, system, user, eval_text, usage)

        icon = {"sufficient": "✓", "partial": "~", "none": "✗"}.get(verdict, "?")
        print(f"  [EVALUATOR] {verdict.upper()} {icon} | {feedback[:80]}")
        return is_relevant, feedback, verdict

    # ── Main LLM Call (RAG path) ──────────────────────────────────────────────

    def _generate_grounded(
        self,
        query_id: str,
        original_query: str,
        rewritten_query: str,
        docs: List[Dict],
        conversation_history: str,
        verdict: str = "sufficient",
        step_order: int = 6,
    ) -> str:
        """Main LLM Call — generate answer grounded in retrieved documents."""
        t0 = time.time()
        context = _build_context(docs, max_chars=2500)

        if verdict == "partial":
            confidence_instruction = (
                "The retrieved documents are only PARTIALLY relevant — they "
                "provide related context but do not directly state the answer. "
                "Use them to give the most useful partial answer you can, but "
                "explicitly flag which parts of your answer are directly "
                "supported by the documents versus inferred from related context."
            )
        else:
            confidence_instruction = (
                "The retrieved documents directly support answering this "
                "question. Answer using them with full confidence."
            )

        system = (
            "You are an expert agricultural research assistant. "
            "Answer the user's question using ONLY the provided document "
            "excerpts. Cite sources as [Source: filename, Page X]. "
            f"{confidence_instruction} "
            "Be concise but complete."
        )
        user = (
            f"Conversation history:\n{conversation_history}\n\n"
            f"Retrieved documents:\n{context}\n\n"
            f"Original question: {original_query}\n"
            f"Clarified question: {rewritten_query}\n\n"
            f"Answer (cite sources):"
        )

        answer, usage = self.llm.call(
            system, user, max_tokens=600, temperature=0.2
        )
        duration = (time.time() - t0) * 1000

        step_id = _log_step(
            query_id, "main_llm_grounded", step_order,
            input_text=rewritten_query,
            output_text=answer,
            duration_ms=duration,
        )
        _log_llm_call(step_id, ACTIVE_MODEL_NAME, system, user, answer, usage)
        return answer

    # ── Main LLM Call (Direct path) ───────────────────────────────────────────

    def _generate_direct(
        self,
        query_id: str,
        original_query: str,
        rewritten_query: str,
        conversation_history: str,
        step_order: int = 3,
    ) -> str:
        """Main LLM Call — answer directly without retrieval."""
        t0 = time.time()

        system = (
            "You are a helpful agricultural research assistant. "
            "Answer the user's question directly and concisely."
        )
        user = (
            f"Conversation history:\n{conversation_history}\n\n"
            f"Question: {rewritten_query}\n\nAnswer:"
        )

        answer, usage = self.llm.call(
            system, user, max_tokens=400, temperature=0.3
        )
        duration = (time.time() - t0) * 1000

        step_id = _log_step(
            query_id, "main_llm_direct", step_order,
            input_text=rewritten_query,
            output_text=answer,
            duration_ms=duration,
        )
        _log_llm_call(step_id, ACTIVE_MODEL_NAME, system, user, answer, usage)
        return answer

    # ── Main pipeline entry point ─────────────────────────────────────────────

    def run(self, session_id: str, user_query: str) -> str:
        """
        Execute the full agentic RAG pipeline for one user query.

        Args:
            session_id:  Unique session identifier.
            user_query:  The user's raw question.

        Returns:
            Final response string.
        """
        print(f"\n{'='*60}")
        print(f"[PIPELINE] Query: {user_query}")
        print(f"{'='*60}")

        # ── Create session and query records ──────────────────────────────────
        conn = db_schema.get_connection()
        # Upsert session
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)",
            (session_id,)
        )
        query_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO queries (query_id, session_id, original_query) "
            "VALUES (?,?,?)",
            (query_id, session_id, user_query)
        )
        conn.commit()
        conn.close()

        conversation_history = self.memory.get_formatted()
        retry_count = 0
        step_counter = 1

        # ── Step 1: Query Rewriter ────────────────────────────────────────────
        rewritten_query = self._query_rewriter(
            query_id, user_query, conversation_history,
            step_order=step_counter
        )
        step_counter += 1

        # ── Step 2: Orchestrator ──────────────────────────────────────────────
        needs_rag = self._orchestrator(
            query_id, rewritten_query, step_order=step_counter
        )
        step_counter += 1

        # ── Direct path (no RAG) ──────────────────────────────────────────────
        if not needs_rag:
            answer = self._generate_direct(
                query_id, user_query, rewritten_query,
                conversation_history, step_order=step_counter
            )
            _log_response(query_id, answer, used_rag=False, retry_count=0)
            self.memory.add("user", user_query)
            self.memory.add("assistant", answer)
            print(f"\n[PIPELINE] DIRECT response generated.")
            return answer

        # ── RAG path ──────────────────────────────────────────────────────────
        self._ensure_bm25_built()
        evaluator_feedback = ""

        while retry_count <= MAX_RETRIES:
            # Retry: rewrite with evaluator feedback
            if retry_count > 0:
                print(f"\n  [RETRY {retry_count}/{MAX_RETRIES}] "
                      f"Rewriting with evaluator feedback...")
                rewritten_query = self._query_rewriter(
                    query_id, user_query, conversation_history,
                    evaluator_feedback=evaluator_feedback,
                    step_order=step_counter,
                )
                step_counter += 1

                # Log retry count update
                conn = db_schema.get_connection()
                conn.execute(
                    "UPDATE pipeline_steps SET output_text = output_text || "
                    f"' [retry={retry_count}]' WHERE query_id = ?",
                    (query_id,)
                )
                conn.commit()
                conn.close()

            # ── Step 3: Retrieve ──────────────────────────────────────────────
            vector_results, bm25_results = self._retrieve(
                query_id, rewritten_query, step_order=step_counter
            )
            step_counter += 1

            # ── Step 4: RRF Rerank ────────────────────────────────────────────
            reranked_docs = self._rerank(
                query_id, vector_results, bm25_results,
                step_order=step_counter
            )
            step_counter += 1

            # ── Step 5: Relevance Evaluator ───────────────────────────────────
            is_relevant, evaluator_feedback, verdict = self._evaluator(
                query_id, user_query, rewritten_query,
                reranked_docs, step_order=step_counter
            )
            step_counter += 1

            if is_relevant:
                break  # Good docs found → proceed to generation

            retry_count += 1
            if retry_count > MAX_RETRIES:
                # Safe response — not enough info found
                safe_msg = (
                    "I couldn't find sufficient relevant information in the "
                    "knowledge base to answer your question accurately. "
                    f"The documents searched include the PARC Annual Report "
                    f"2023-24, FAO Guidelines for Monitoring Crop Diseases, "
                    f"and Punjab Agriculture Department rules. "
                    f"Evaluator note: {evaluator_feedback}"
                )
                _log_step(
                    query_id, "safe_response", step_counter,
                    input_text=rewritten_query,
                    output_text=safe_msg,
                    status="safe_fallback",
                )
                _log_response(
                    query_id, safe_msg, used_rag=True,
                    retry_count=retry_count
                )
                self.memory.add("user", user_query)
                self.memory.add("assistant", safe_msg)
                print(f"\n[PIPELINE] SAFE RESPONSE returned after "
                      f"{MAX_RETRIES} retries.")
                return safe_msg

        # ── Step 6: Generate grounded answer ──────────────────────────────────
        answer = self._generate_grounded(
            query_id, user_query, rewritten_query,
            reranked_docs, conversation_history, verdict=verdict,
            step_order=step_counter
        )

        _log_response(query_id, answer, used_rag=True, retry_count=retry_count)
        self.memory.add("user", user_query)
        self.memory.add("assistant", answer)

        print(f"\n[PIPELINE] RAG response generated (retries={retry_count}).")
        return answer


# ── DB Query helpers for inspection ──────────────────────────────────────────

def inspect_last_query():
    """Print a summary of the most recent pipeline execution from the DB."""
    conn = db_schema.get_connection()

    query = conn.execute(
        "SELECT * FROM queries ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()
    if not query:
        print("No queries logged yet.")
        return

    print(f"\n{'='*60}")
    print(f"Query: {query['original_query']}")
    print(f"ID:    {query['query_id']}")

    steps = conn.execute(
        "SELECT * FROM pipeline_steps WHERE query_id = ? ORDER BY step_order",
        (query['query_id'],)
    ).fetchall()
    print(f"\nPipeline steps ({len(steps)}):")
    for s in steps:
        print(f"  [{s['step_order']}] {s['step_name']:25s} "
              f"{s['duration_ms']:6.0f}ms  {s['status']}")

    llm_calls = conn.execute(
        """SELECT lc.model_name, lc.total_tokens, ps.step_name
           FROM llm_calls lc
           JOIN pipeline_steps ps ON lc.step_id = ps.step_id
           WHERE ps.query_id = ?""",
        (query['query_id'],)
    ).fetchall()
    total_tokens = sum(r['total_tokens'] or 0 for r in llm_calls)
    print(f"\nLLM calls: {len(llm_calls)} | Total tokens: {total_tokens}")

    docs = conn.execute(
        """SELECT rd.source_file, rd.page_num, rd.rrf_score, rd.final_rank
           FROM retrieved_docs rd
           WHERE rd.query_id = ?
             AND rd.rrf_score IS NOT NULL
             AND rd.step_id = (
                 SELECT step_id FROM pipeline_steps
                 WHERE query_id = ? AND step_name = 'reranking'
                 ORDER BY step_order DESC LIMIT 1
             )
           ORDER BY rd.final_rank""",
        (query['query_id'], query['query_id'])
    ).fetchall()
    if docs:
        print(f"\nTop retrieved docs:")
        for d in docs[:5]:
            print(f"  rank={d['final_rank']} rrf={d['rrf_score']:.5f} "
                  f"{d['source_file']} p.{d['page_num']}")

    response = conn.execute(
        "SELECT * FROM responses WHERE query_id = ?",
        (query['query_id'],)
    ).fetchone()
    if response:
        print(f"\nFinal response (used_rag={bool(response['used_rag'])}, "
              f"retries={response['retry_count']}):")
        print(response['final_response'][:400])

    conn.close()
