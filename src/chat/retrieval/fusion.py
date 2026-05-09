from __future__ import annotations

from dataclasses import replace

from src.chat.retrieval.types import Hit
from src.config import HYBRID_CANDIDATE_K, RRF_K


def hit_key(h: Hit) -> str:
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
            key = hit_key(h)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            hit_map.setdefault(key, h)

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [replace(hit_map[key], score=score) for key, score in top]
