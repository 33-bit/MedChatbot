from __future__ import annotations

import logging

from src.chat.errors import QdrantUnavailable
from src.chat.retrieval.fusion import rrf_merge
from src.chat.retrieval.sparse import bm25_search
from src.chat.retrieval.types import Hit
from src.chat.timing import elapsed_ms
from src.config import HYBRID_CANDIDATE_K, RERANK_TOP_K

log = logging.getLogger(__name__)


def hybrid_search(query: str, top_k: int = RERANK_TOP_K) -> list[Hit]:
    """Dense + BM25 → RRF fusion → Cross-Encoder re-rank."""
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
        log.warning("Sparse search failed: %s", e)
        sparse_hits = []
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
        log.warning("Rerank failed: %s", e)
        reranked = fused[:top_k]
    log.info("retrieval timing stage=rerank_total ms=%.1f hits=%d",
             elapsed_ms(stage_start), len(reranked))
    log.info("retrieval timing stage=hybrid_total ms=%.1f hits=%d",
             elapsed_ms(total_start), len(reranked))
    return reranked
