"""Dense pattern matching via FAISS, step 4a: embedding-only top-k retrieval.
Query text is built from thought+emotion+behavior slot content only -- situation
is excluded here and used solely at the summary stage. This is a tentative
candidate pool for the rerank stage (step 4b), not the final gate signal."""
import hashlib
from pathlib import Path

import faiss
import numpy as np

from data import PATTERN_IDS, PATTERN_QUERY_TEXTS
from embeddings import embed

CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "embeddings"
EMBEDDING_TOP_K = 5


def _corpus_digest(texts: list[str]) -> str:
    joined = "\x1f".join(texts).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]


def _load_or_compute_doc_vecs() -> np.ndarray:
    digest = _corpus_digest(PATTERN_QUERY_TEXTS)
    cache_path = CACHE_DIR / f"pattern_{digest}.npy"

    if cache_path.exists():
        return np.load(cache_path)

    doc_vecs = embed(PATTERN_QUERY_TEXTS)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for stale in CACHE_DIR.glob("pattern_*.npy"):
        stale.unlink()
    np.save(cache_path, doc_vecs)
    return doc_vecs


_index = None


def _get_index() -> faiss.IndexFlatIP:
    global _index
    if _index is None:
        doc_vecs = _load_or_compute_doc_vecs().astype("float32")
        _index = faiss.IndexFlatIP(doc_vecs.shape[1])
        _index.add(doc_vecs)
    return _index


def top_k_candidates(query: str, k: int = EMBEDDING_TOP_K) -> list[dict]:
    """Returns up to k {"pattern_id", "embedding_score"} dicts ranked by cosine
    similarity. No score-floor filtering here -- filtering/ranking-for-gate happens
    after the text-based rerank (step 4b/4c), not at this stage."""
    index = _get_index()
    query_vec = embed([query]).astype("float32")
    scores, indices = index.search(query_vec, k)

    return [
        {"pattern_id": PATTERN_IDS[i], "embedding_score": float(s)}
        for s, i in zip(scores[0], indices[0]) if i != -1
    ]
