"""
Vector Store Module
===================
Uses ChromaDB (local, persistent) with sentence-transformers embeddings
(all-MiniLM-L6-v2) -- a genuinely semantic, CPU-friendly embedding model,
replacing the earlier TF-IDF approach.

Why this model:
- all-MiniLM-L6-v2 is small (~80MB), runs comfortably on CPU, no GPU needed
- Produces dense semantic vectors -- understands meaning/synonyms, not
  just literal word overlap (this is what TF-IDF could not do: e.g.
  "onion thrips" vs "Thrips tabaci" now map to similar vectors)
- One-time download from Hugging Face on first use, then cached locally

IMPORTANT: switching embedding functions requires re-indexing from
scratch (--reset). TF-IDF vectors and sentence-transformers vectors are
different mathematical spaces and cannot be mixed in the same
collection.

The original TFIDFEmbeddingFunction class is kept below (unused, not
wired into _get_ef()) for reference/comparison -- it's what this project
used before this swap, and demonstrates the reliability/semantic-quality
tradeoff discussed in the project documentation.
"""

import os
import pickle
import numpy as np
from typing import List, Tuple, Dict, Any

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from chromadb.utils import embedding_functions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) \
                   if os.path.basename(os.path.dirname(os.path.abspath(__file__))) == "src" \
                   else os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR      = os.path.join(BASE_DIR, "chroma_db")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
COLLECTION_NAME = "agriculture_docs"
TFIDF_DIM       = 2048   # vocabulary size for TF-IDF features (legacy, unused)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # current active embedding model

# ── Singleton instances ───────────────────────────────────────────────────────
_client:     chromadb.PersistentClient = None
_collection  = None
_ef = None


# ── Custom embedding function (offline TF-IDF) ────────────────────────────────

class TFIDFEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB-compatible embedding function using TF-IDF.
    Fitted lazily on first use and persisted to disk.
    """

    def __init__(self, dim: int = TFIDF_DIM, vectorizer_path: str = VECTORIZER_PATH):
        self.dim = dim
        self.vectorizer_path = vectorizer_path
        self.vectorizer: TfidfVectorizer = None
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(self.vectorizer_path):
            with open(self.vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            print(f"[VECTOR] Loaded TF-IDF vectorizer "
                  f"(vocab={len(self.vectorizer.vocabulary_)})")
        else:
            self.vectorizer = TfidfVectorizer(
                max_features=self.dim,
                ngram_range=(1, 2),   # unigrams + bigrams
                sublinear_tf=True,    # log normalization of TF
                min_df=1,
                strip_accents="unicode",
                analyzer="word",
            )
            print("[VECTOR] New TF-IDF vectorizer (will fit on first batch)")

    def fit(self, texts: List[str]):
        """Fit the vectorizer on a corpus of texts and save to disk."""
        self.vectorizer.fit(texts)
        with open(self.vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        print(f"[VECTOR] TF-IDF vectorizer fitted and saved "
              f"(vocab={len(self.vectorizer.vocabulary_)})")

    def is_fitted(self) -> bool:
        return hasattr(self.vectorizer, "vocabulary_")

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to L2-normalized TF-IDF vectors."""
        if not self.is_fitted():
            raise RuntimeError("Vectorizer not fitted. Call fit() first.")
        matrix = self.vectorizer.transform(texts).toarray().astype(np.float32)
        return normalize(matrix, norm="l2")

    def __call__(self, input: Documents) -> Embeddings:
        """ChromaDB calls this during add() and query()."""
        if not self.is_fitted():
            # Auto-fit on the first batch (happens during indexing)
            self.fit(input)
        return self.transform(list(input)).tolist()


# ── Embedding function (semantic, via sentence-transformers) ─────────────────

def _get_ef():
    """
    Returns ChromaDB's built-in SentenceTransformerEmbeddingFunction,
    using all-MiniLM-L6-v2. Downloads the model from Hugging Face on
    first use (~80MB), then caches it locally for subsequent runs --
    no download needed after the first time.
    """
    global _ef
    if _ef is None:
        print(f"[VECTOR] Loading embedding model '{EMBEDDING_MODEL}' "
              f"(downloads on first use, cached after)...")
        _ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        print(f"[VECTOR] Embedding model ready.")
    return _ef


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        print(f"[VECTOR] ChromaDB initialized at {CHROMA_DIR}")
    return _client


def _get_collection():
    """Collection uses the semantic embedding function from _get_ef()
    above -- no fitting step needed, unlike the old TF-IDF approach,
    since this model is already pretrained."""
    global _collection
    if _collection is None:
        client = _get_client()
        ef     = _get_ef()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[VECTOR] Collection '{COLLECTION_NAME}' ready "
              f"({_collection.count()} documents)")
    return _collection


# ── Public API ────────────────────────────────────────────────────────────────

def index_chunks(
    chunks: List[Tuple[str, str, int]],
    batch_size: int = 128,
    verbose: bool = True,
) -> int:
    """
    Add text chunks to ChromaDB. The embedding model is pretrained
    (no fitting step needed, unlike the previous TF-IDF approach) --
    ChromaDB calls the embedding function automatically on each batch
    added.

    Args:
        chunks:     List of (chunk_text, source_file, page_num).
        batch_size: Insert this many at a time.
        verbose:    Print progress.

    Returns:
        Number of chunks added.
    """
    import uuid

    collection = _get_collection()

    texts, ids, metadatas = [], [], []
    for chunk_text, source_file, page_num in chunks:
        texts.append(chunk_text)
        ids.append(str(uuid.uuid4()))
        metadatas.append({"source_file": source_file, "page_num": page_num})

    added = 0
    for i in range(0, len(texts), batch_size):
        b_texts = texts[i : i + batch_size]
        b_ids   = ids[i : i + batch_size]
        b_meta  = metadatas[i : i + batch_size]
        collection.add(documents=b_texts, ids=b_ids, metadatas=b_meta)
        added += len(b_texts)
        if verbose:
            print(f"  [VECTOR] Batch {i//batch_size + 1}: "
                  f"{added}/{len(texts)} indexed")

    if verbose:
        print(f"[VECTOR] Collection total: {collection.count()} docs")
    return added


def similarity_search(
    query: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search ChromaDB for top-k chunks most similar to query.

    Returns list of dicts:
      chunk_text, source_file, page_num, vector_score, doc_id
    """
    collection = _get_collection()
    if collection.count() == 0:
        print("[VECTOR] WARNING: Empty collection.")
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist, did in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0],
    ):
        output.append({
            "doc_id":       did,
            "chunk_text":   doc,
            "source_file":  meta.get("source_file", "unknown"),
            "page_num":     meta.get("page_num", 0),
            "vector_score": round(1.0 - dist, 4),
        })
    return output


def collection_size() -> int:
    try:
        return _get_collection().count()
    except Exception:
        return 0


def reset_collection():
    """Delete and recreate the collection. Required when switching
    embedding functions (e.g. TF-IDF -> sentence-transformers), since
    vectors from different embedding spaces cannot coexist."""
    client = _get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[VECTOR] Deleted collection '{COLLECTION_NAME}'")
    except Exception:
        pass
    # Clean up the legacy TF-IDF vectorizer file if it exists from a
    # previous run -- harmless no-op if it doesn't.
    if os.path.exists(VECTORIZER_PATH):
        os.remove(VECTORIZER_PATH)
        print("[VECTOR] Deleted legacy TF-IDF vectorizer file")
    global _collection, _ef
    _collection = None
    _ef = None


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Collection size:", collection_size())
    if collection_size() > 0:
        results = similarity_search("wheat rust disease monitoring", top_k=3)
        for r in results:
            print(f"\n[{r['source_file']} p.{r['page_num']}] "
                  f"score={r['vector_score']}")
            print(r["chunk_text"][:200])
