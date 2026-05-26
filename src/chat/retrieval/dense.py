"""
dense.py
--------
Dense vector retrieval over Qdrant.

Exports:
  - Hit: the retrieval result dataclass used by all retrieval modules.
  - dense_search: Qdrant vector search over disease + drug collections.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from threading import Lock

from qdrant_client import QdrantClient
from src.config import (
    DISEASES_COLLECTION,
    DRUGS_COLLECTION,
    E5_QUERY_PREFIX,
    EMBED_MODEL,
    HYBRID_CANDIDATE_K,
    QDRANT_API_KEY,
    QDRANT_URL,
)
from sentence_transformers import SentenceTransformer

from src.chat.errors import QdrantUnavailable
from src.chat.retrieval.rerank import preload_reranker
from src.chat.retrieval.types import Hit
from src.chat.timing import log_stage_timing

log = logging.getLogger(__name__)
_COLLECTIONS = (DISEASES_COLLECTION, DRUGS_COLLECTION)
_COLLECTION_CACHE_TTL_SECONDS = 300
_collection_cache: tuple[float, tuple[str, ...]] | None = None
_collection_cache_lock = Lock()


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _qdrant() -> QdrantClient:
    if not QDRANT_URL:
        raise QdrantUnavailable("QDRANT_URL not configured")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)


def preload_models() -> None:
    """Load embedding and reranker models into process memory."""
    _embedder()
    preload_reranker()


def _available_collections() -> tuple[str, ...]:
    """Cache Qdrant collection existence checks briefly to avoid per-turn HEAD calls."""
    global _collection_cache

    now = time.monotonic()
    with _collection_cache_lock:
        if _collection_cache and now - _collection_cache[0] < _COLLECTION_CACHE_TTL_SECONDS:
            return _collection_cache[1]

    client = _qdrant()
    available: list[str] = []
    for collection in _COLLECTIONS:
        stage_start = time.perf_counter()
        try:
            exists = client.collection_exists(collection)
        except Exception as e:
            raise QdrantUnavailable(
                f"Qdrant collection check failed for {collection}"
            ) from e
        log_stage_timing(
            log,
            "retrieval",
            "qdrant_exists",
            stage_start,
            collection=collection,
            exists=exists,
        )
        if exists:
            available.append(collection)

    missing = [collection for collection in _COLLECTIONS if collection not in available]
    if missing:
        raise QdrantUnavailable(
            "Missing configured Qdrant collections: " + ", ".join(missing)
        )
    collections = tuple(available)
    with _collection_cache_lock:
        _collection_cache = (now, collections)
    return collections


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
    stage_start = time.perf_counter()
    qvec = _embed(query)
    log_stage_timing(log, "retrieval", "embed", stage_start)

    client = _qdrant()
    results: list[Hit] = []
    for collection in _available_collections():
        stage_start = time.perf_counter()
        try:
            response = client.query_points(
                collection_name=collection,
                query=qvec,
                limit=top_k,
                with_payload=True,
            )
        except Exception as e:
            raise QdrantUnavailable(
                f"Qdrant query failed for {collection}"
            ) from e
        log_stage_timing(
            log,
            "retrieval",
            "qdrant_query",
            stage_start,
            collection=collection,
            hits=len(response.points),
        )
        results.extend(_to_hit(p) for p in response.points)
    results.sort(key=lambda h: h.score, reverse=True)
    return results[:top_k]
