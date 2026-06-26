from __future__ import annotations

import re
import unicodedata
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


_HEADING_BOOST_TERMS = (
    "lieu dung",
    "cach dung",
    "thoi ky mang thai",
    "chong chi dinh",
    "trieu chung",
    "dieu tri",
)


def _normalize(text: str) -> str:
    text = (text or "").casefold().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _rerank_bonus(query: str, hit: Hit) -> float:
    query_norm = _normalize(query)
    source_name = _normalize(hit.source_name)
    heading = _normalize(hit.heading_path)
    bonus = 0.0

    if source_name and source_name in query_norm:
        bonus += 0.25
    elif source_name:
        source_tokens = {
            token for token in re.split(r"\W+", source_name)
            if len(token) >= 4
        }
        if source_tokens and any(token in query_norm for token in source_tokens):
            bonus += 0.08

    for term in _HEADING_BOOST_TERMS:
        if term in query_norm and term in heading:
            bonus += 0.10
            break
    return bonus


def rerank(query: str, hits: list[Hit], top_k: int = RERANK_TOP_K) -> list[Hit]:
    """Re-rank hits using cross-encoder, return top_k."""
    if not hits:
        return []
    pairs = [[query, h.text] for h in hits]
    scores = reranker().predict(pairs, batch_size=RERANK_BATCH_SIZE)
    adjusted = [
        (hit, float(score) + _rerank_bonus(query, hit))
        for hit, score in zip(hits, scores)
    ]
    ranked = sorted(adjusted, key=lambda x: x[1], reverse=True)
    return [replace(h, score=float(s)) for h, s in ranked[:top_k]]
