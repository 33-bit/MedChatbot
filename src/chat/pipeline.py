"""
pipeline.py
-----------
Orchestration: query → KG lookup + hybrid retrieval → re-rank → generate.
"""

from __future__ import annotations

from src.chat.generator import generate
from src.chat.kg_retriever import format_kg_context, kg_search
from src.chat.retriever import hybrid_search
from src.config import RERANK_TOP_K


def answer(question: str, top_k: int = RERANK_TOP_K) -> str:
    question = (question or "").strip()
    if not question:
        return "Bạn hãy đặt câu hỏi cụ thể nhé."

    # 1. KG lookup (structured facts)
    kg_ctx = kg_search(question)
    kg_text = format_kg_context(kg_ctx)

    # 2. Hybrid retrieval (dense + BM25 → RRF → cross-encoder re-rank)
    hits = hybrid_search(question, top_k=top_k)

    # 3. Generate answer
    return generate(question, hits, kg_text=kg_text)
