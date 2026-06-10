from __future__ import annotations

import logging

from src.chat.errors import QdrantUnavailable
from src.chat.retrieval.fusion import rrf_merge
from src.chat.retrieval.sparse import bm25_search
from src.chat.retrieval.types import Hit
from src.chat.timing import elapsed_ms
from src.config import HYBRID_CANDIDATE_K, RERANK_TOP_K

log = logging.getLogger(__name__)
_TEXT_PREVIEW_MAX_CHARS = 160


def _run_hybrid_search(
    query: str,
    top_k: int,
) -> tuple[list[Hit], list[Hit], list[Hit], list[Hit]]:
    import time

    from src.chat.retrieval.dense import dense_search
    from src.chat.retrieval.rerank import rerank

    total_start = time.perf_counter()
    stage_start = time.perf_counter()
    try:
        dense_hits = dense_search(query, top_k=HYBRID_CANDIDATE_K)
    except QdrantUnavailable:
        raise
    except Exception as e:
        raise QdrantUnavailable("Dense retrieval failed") from e
    log.info("retrieval timing stage=dense_total ms=%.1f hits=%d",
             elapsed_ms(stage_start), len(dense_hits))

    stage_start = time.perf_counter()
    try:
        sparse_hits = bm25_search(query, top_k=HYBRID_CANDIDATE_K)
    except Exception as e:
        raise QdrantUnavailable("Sparse retrieval failed") from e
    log.info("retrieval timing stage=sparse_total ms=%.1f hits=%d",
             elapsed_ms(stage_start), len(sparse_hits))

    stage_start = time.perf_counter()
    fused = rrf_merge(dense_hits, sparse_hits, top_k=HYBRID_CANDIDATE_K)
    log.info("retrieval timing stage=rrf_merge ms=%.1f hits=%d",
             elapsed_ms(stage_start), len(fused))

    stage_start = time.perf_counter()
    try:
        reranked = rerank(query, fused, top_k=top_k)
    except Exception as e:
        raise QdrantUnavailable("Rerank failed") from e
    log.info("retrieval timing stage=rerank_total ms=%.1f hits=%d",
             elapsed_ms(stage_start), len(reranked))
    log.info("retrieval timing stage=hybrid_total ms=%.1f hits=%d",
             elapsed_ms(total_start), len(reranked))
    return dense_hits, sparse_hits, fused, reranked


def _text_preview(text: str) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= _TEXT_PREVIEW_MAX_CHARS:
        return normalized
    return normalized[:_TEXT_PREVIEW_MAX_CHARS - 3].rstrip() + "..."


def _serialize_hits(hits: list[Hit], stage: str) -> list[dict]:
    return [
        {
            "rank": rank,
            "stage": stage,
            "chunk_id": hit.chunk_id,
            "source_type": hit.source_type,
            "source_slug": hit.source_slug,
            "source_name": hit.source_name,
            "heading_path": hit.heading_path,
            "score": float(hit.score),
            "text_preview": _text_preview(hit.text),
            "metadata": hit.metadata,
        }
        for rank, hit in enumerate(hits, start=1)
    ]


def hybrid_search(query: str, top_k: int = RERANK_TOP_K) -> list[Hit]:
    """Dense + BM25 → RRF fusion → Cross-Encoder re-rank."""
    return _run_hybrid_search(query, top_k)[-1]


def hybrid_search_with_debug(
    query: str,
    top_k: int = RERANK_TOP_K,
) -> tuple[list[Hit], dict]:
    dense_hits, sparse_hits, fused_hits, reranked_hits = _run_hybrid_search(
        query,
        top_k,
    )
    debug = {
        "query": query,
        "dense_hits": _serialize_hits(dense_hits, "dense"),
        "sparse_hits": _serialize_hits(sparse_hits, "sparse"),
        "fused_hits": _serialize_hits(fused_hits, "fused"),
        "reranked_hits": _serialize_hits(reranked_hits, "reranked"),
    }
    return reranked_hits, debug
