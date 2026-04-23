"""
reranker.py
-----------
Cross-Encoder re-ranker using BAAI/bge-reranker-v2-m3.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.chat.retriever import Hit
from src.config import RERANKER_MODEL, RERANK_TOP_K


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


def rerank(query: str, hits: list[Hit], top_k: int = RERANK_TOP_K) -> list[Hit]:
    """Re-rank hits using cross-encoder, return top_k."""
    if not hits:
        return []
    pairs = [[query, h.text] for h in hits]
    scores = _reranker().predict(pairs)
    ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
    results = []
    for h, s in ranked[:top_k]:
        results.append(Hit(
            text=h.text,
            score=float(s),
            source_type=h.source_type,
            source_name=h.source_name,
            heading_path=h.heading_path,
            source_slug=h.source_slug,
            chunk_id=h.chunk_id,
            metadata=h.metadata,
        ))
    return results
