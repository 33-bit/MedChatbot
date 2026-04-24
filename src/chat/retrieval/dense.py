"""
dense.py
--------
Dense vector retrieval over Qdrant + hybrid search orchestrator.

Exports:
  - Hit: the retrieval result dataclass used by all retrieval modules.
  - dense_search: Qdrant vector search over disease + drug collections.
  - rrf_merge: Reciprocal Rank Fusion of two ranked lists.
  - hybrid_search: dense + BM25 → RRF → Cross-Encoder rerank.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.chat.retrieval.rerank import rerank
from src.chat.retrieval.sparse import bm25_search
from src.chat.retrieval.types import Hit
from src.config import (
    DISEASES_COLLECTION,
    DRUGS_COLLECTION,
    E5_QUERY_PREFIX,
    EMBED_MODEL,
    HYBRID_CANDIDATE_K,
    QDRANT_API_KEY,
    QDRANT_URL,
    RERANK_TOP_K,
    RRF_K,
)


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _qdrant() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)


def _embed(query: str) -> list[float]:
    text = (E5_QUERY_PREFIX + query) if "e5" in EMBED_MODEL.lower() else query
    vec = _embedder().encode([text], normalize_embeddings=True)[0]
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
    """Vector search both collections, merge and sort by score desc."""
    qvec = _embed(query)
    client = _qdrant()
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

    for ranked_list in (dense_hits, sparse_hits):
        for rank, h in enumerate(ranked_list):
            key = _hit_key(h)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            hit_map.setdefault(key, h)

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [replace(hit_map[key], score=score) for key, score in top]


def hybrid_search(query: str, top_k: int = RERANK_TOP_K) -> list[Hit]:
    """Dense + BM25 → RRF fusion → Cross-Encoder re-rank."""
    dense_hits = dense_search(query, top_k=HYBRID_CANDIDATE_K)
    sparse_hits = bm25_search(query, top_k=HYBRID_CANDIDATE_K)
    fused = rrf_merge(dense_hits, sparse_hits, top_k=HYBRID_CANDIDATE_K)
    return rerank(query, fused, top_k=top_k)
