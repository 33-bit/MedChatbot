"""
cache.py
--------
Semantic cache for informational queries only (drug info, disease overview).
Does NOT cache diagnostic / personal-context queries.
Backed by GPTCache with FAISS (local, separate from the medical Qdrant index).
"""

from __future__ import annotations

import logging
from functools import lru_cache

from src.config import OUTPUT_DIR, SEMANTIC_CACHE_THRESHOLD  # noqa: F401

log = logging.getLogger(__name__)

CACHE_DIR = OUTPUT_DIR / "semantic_cache"


@lru_cache(maxsize=1)
def _cache():
    """Lazy-init GPTCache. Return None if unavailable (graceful degrade)."""
    try:
        from gptcache import Cache
        from gptcache.adapter.api import init_similar_cache
    except ImportError as e:
        log.warning("GPTCache not installed; semantic cache disabled: %s", e)
        return None
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = Cache()
        init_similar_cache(data_dir=str(CACHE_DIR), cache_obj=cache)
        return cache
    except Exception as e:
        log.warning("GPTCache init failed; semantic cache disabled: %s", e)
        return None


def cache_get(query: str) -> str | None:
    cache = _cache()
    if cache is None:
        return None
    try:
        from gptcache.adapter.api import get
        return get(query, cache_obj=cache)
    except Exception as e:
        log.debug("cache_get failed: %s", e)
        return None


def cache_put(query: str, answer: str) -> None:
    cache = _cache()
    if cache is None:
        return
    try:
        from gptcache.adapter.api import put
        put(query, answer, cache_obj=cache)
    except Exception as e:
        log.debug("cache_put failed: %s", e)
