from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.chat.retrieval.types import Hit
from src.config import RERANK_BATCH_SIZE, RERANKER_DEVICE, RERANKER_MODEL, RERANK_TOP_K


@lru_cache(maxsize=1)
def reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL, device=RERANKER_DEVICE)


def preload_reranker() -> None:
    reranker()


def rerank(query: str, hits: list[Hit], top_k: int = RERANK_TOP_K) -> list[Hit]:
    """Re-rank hits using cross-encoder, return top_k."""
    if not hits:
        return []
    pairs = [[query, h.text] for h in hits]
    scores = reranker().predict(pairs, batch_size=RERANK_BATCH_SIZE)
    ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
    return [replace(h, score=float(s)) for h, s in ranked[:top_k]]
