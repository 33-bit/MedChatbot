"""Public facade of the retrieval subpackage."""

from src.chat.retrieval.dense import dense_search, hybrid_search, rrf_merge
from src.chat.retrieval.kg import (
    KGContext,
    ensure_fulltext_indexes,
    format_kg_context,
    kg_search,
)
from src.chat.retrieval.rerank import rerank
from src.chat.retrieval.sparse import bm25_search
from src.chat.retrieval.types import Hit

__all__ = [
    "Hit",
    "KGContext",
    "bm25_search",
    "dense_search",
    "ensure_fulltext_indexes",
    "format_kg_context",
    "hybrid_search",
    "kg_search",
    "rerank",
    "rrf_merge",
]
