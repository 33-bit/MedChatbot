"""
cache.py
--------
Semantic cache for informational queries only (drug info, disease overview).
Does NOT cache diagnostic/personal-context queries — risk of returning stale
answer for a different patient.

Backed by GPTCache with FAISS (local, separate from medical Qdrant).
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from src.config import OUTPUT_DIR

CACHE_DIR = OUTPUT_DIR / "semantic_cache"
SIMILARITY_THRESHOLD = 0.85


@lru_cache(maxsize=1)
def _cache():
    """Lazy-init GPTCache. Return None if unavailable (graceful degrade)."""
    try:
        from gptcache import Cache
        from gptcache.adapter.api import init_similar_cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = Cache()
        init_similar_cache(
            data_dir=str(CACHE_DIR),
            cache_obj=cache,
        )
        return cache
    except Exception:
        return None


def _key(q: str) -> str:
    return hashlib.sha256(q.strip().lower().encode()).hexdigest()[:16]


def cache_get(query: str) -> str | None:
    cache = _cache()
    if cache is None:
        return None
    try:
        from gptcache.adapter.api import get
        result = get(query, cache_obj=cache)
        return result
    except Exception:
        return None


def cache_put(query: str, answer: str) -> None:
    cache = _cache()
    if cache is None:
        return
    try:
        from gptcache.adapter.api import put
        put(query, answer, cache_obj=cache)
    except Exception:
        pass
