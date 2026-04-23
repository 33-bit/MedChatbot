"""
retriever.py
------------
Hybrid retrieval: Dense (Qdrant) + Sparse (BM25) → RRF fusion → Cross-Encoder re-rank.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.config import (
    DISEASES_COLLECTION,
    DRUGS_COLLECTION,
    EMBED_MODEL,
    HYBRID_CANDIDATE_K,
    QDRANT_API_KEY,
    QDRANT_URL,
    RAG_TOP_K,
    RERANK_TOP_K,
)

E5_QUERY_PREFIX = "query: "
RRF_K = 60  # constant for Reciprocal Rank Fusion


@dataclass
class Hit:
    text: str
    score: float
    source_type: str        # "disease" | "drug"
    source_name: str        # disease name or drug name
    heading_path: str       # section heading path
    source_slug: str = ""
    chunk_id: str = ""
    metadata: dict | None = None


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)


def _embed(query: str) -> list[float]:
    is_e5 = "e5" in EMBED_MODEL.lower()
    text = (E5_QUERY_PREFIX + query) if is_e5 else query
    vec = _model().encode([text], normalize_embeddings=True)[0]
    return vec.tolist()


def _to_hit(point) -> Hit:
    p = point.payload or {}
    return Hit(
        text=p.get("text", ""),
        score=point.score,
        source_type=p.get("source_type", "disease"),
        source_name=p.get("source_name", ""),
        heading_path=p.get("heading_path", ""),
        source_slug=p.get("source_slug", ""),
        chunk_id=p.get("chunk_id", ""),
        metadata=p.get("metadata"),
    )


def dense_search(query: str, top_k: int = HYBRID_CANDIDATE_K) -> list[Hit]:
    """Search cả 2 collection, merge & sort theo score giảm dần."""
    qvec = _embed(query)
    client = _client()
    results: list[Hit] = []

    for collection in (DISEASES_COLLECTION, DRUGS_COLLECTION):
        if not client.collection_exists(collection):
            continue
        response = client.query_points(
            collection_name=collection,
            query=qvec,
            limit=top_k,
            with_payload=True,
        )
        results.extend(_to_hit(p) for p in response.points)

    results.sort(key=lambda h: h.score, reverse=True)
    return results[:top_k]


# Keep old name as alias for backwards compat
search = dense_search


def _hit_key(h: Hit) -> str:
    return h.chunk_id or h.text[:100]


def rrf_merge(
    dense_hits: list[Hit],
    sparse_hits: list[Hit],
    top_k: int = HYBRID_CANDIDATE_K,
) -> list[Hit]:
    """Reciprocal Rank Fusion of two ranked lists."""
    scores: dict[str, float] = {}
    hit_map: dict[str, Hit] = {}

    for rank, h in enumerate(dense_hits):
        key = _hit_key(h)
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
        hit_map[key] = h

    for rank, h in enumerate(sparse_hits):
        key = _hit_key(h)
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
        if key not in hit_map:
            hit_map[key] = h

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        Hit(
            text=hit_map[key].text,
            score=score,
            source_type=hit_map[key].source_type,
            source_name=hit_map[key].source_name,
            heading_path=hit_map[key].heading_path,
            source_slug=hit_map[key].source_slug,
            chunk_id=hit_map[key].chunk_id,
            metadata=hit_map[key].metadata,
        )
        for key, score in ranked
    ]


def hybrid_search(query: str, top_k: int = RERANK_TOP_K) -> list[Hit]:
    """Dense + BM25 → RRF fusion → Cross-Encoder re-rank."""
    from src.chat.bm25_index import bm25_search
    from src.chat.reranker import rerank

    dense_hits = dense_search(query, top_k=HYBRID_CANDIDATE_K)
    sparse_hits = bm25_search(query, top_k=HYBRID_CANDIDATE_K)

    fused = rrf_merge(dense_hits, sparse_hits, top_k=HYBRID_CANDIDATE_K)

    return rerank(query, fused, top_k=top_k)
