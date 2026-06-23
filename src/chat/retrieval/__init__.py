"""Public facade of the retrieval subpackage."""

from src.chat.retrieval.dense import dense_search
from src.chat.retrieval.fusion import rrf_merge
from src.chat.retrieval.kg import (
    KGContext,
    ensure_fulltext_indexes,
    format_kg_context,
    kg_search,
)
from src.chat.retrieval.rerank import rerank
from src.chat.retrieval.service import hybrid_search, hybrid_search_with_debug
from src.chat.retrieval.sparse import bm25_search
from src.chat.retrieval.types import Hit, RetrievalScope

__all__ = [
    "Hit",
    "KGContext",
    "RetrievalScope",
    "bm25_search",
    "dense_search",
    "ensure_fulltext_indexes",
    "format_kg_context",
    "hybrid_search",
    "hybrid_search_with_debug",
    "kg_search",
    "rerank",
    "rrf_merge",
]
