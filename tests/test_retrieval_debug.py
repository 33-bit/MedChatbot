from __future__ import annotations

import importlib
import logging

import pytest

from src.chat import retrieval
from src.chat.errors import QdrantUnavailable
from src.chat.retrieval import dense, service
from src.chat.retrieval.types import Hit

rerank_module = importlib.import_module("src.chat.retrieval.rerank")


def _hit(label: str, score: float) -> Hit:
    return Hit(
        text=f"{label}\n" + "word " * 80,
        score=score,
        source_type="disease",
        source_name=f"Source {label}",
        heading_path=f"Section > {label}",
        source_slug=f"source-{label}",
        chunk_id=f"chunk-{label}",
        metadata={"label": label},
    )


def test_drug_usage_query_promotes_same_drug_usage_heading():
    base = Hit(
        text="Thông tin chung về Almagate.",
        score=0.9,
        source_type="drug",
        source_name="Almagate",
        heading_path="Đại cương",
        source_slug="almagate",
        chunk_id="drug:almagate:dai-cuong",
        id="uuid-base",
    )
    disease = Hit(
        text="Đau dạ dày có nhiều nguyên nhân.",
        score=0.8,
        source_type="disease",
        source_name="Đau dạ dày",
        heading_path="Triệu chứng",
        source_slug="dau-da-day",
        chunk_id="disease:dau-da-day:trieu-chung",
        id="uuid-disease",
    )
    usage = Hit(
        text="Liều dùng: người lớn uống 1-2 gói/lần, ngày 3 lần.",
        score=0.4,
        source_type="drug",
        source_name="Almagate",
        heading_path="Liều dùng và cách dùng",
        source_slug="almagate",
        chunk_id="drug:almagate:lieu-dung-va-cach-dung",
        id="uuid-usage",
    )

    hits = service._ensure_drug_usage_context(
        "Almagate liều dùng thế nào?",
        [base, disease],
        [usage],
        top_k=2,
    )

    assert [hit.id for hit in hits] == ["uuid-base", "uuid-usage"]


def test_hybrid_search_with_debug_serializes_every_stage_candidate(
    monkeypatch,
    caplog,
):
    dense_hits = [_hit(f"dense-{index}", 10.0 - index) for index in range(3)]
    sparse_hits = [_hit(f"sparse-{index}", 20.0 - index) for index in range(2)]
    fused_hits = [_hit(f"fused-{index}", 0.3 - index / 100) for index in range(3)]
    reranked_hits = [_hit(f"reranked-{index}", 0.9 - index / 10) for index in range(2)]
    calls: list[tuple[str, str, int]] = []

    def fake_dense_search(query: str, top_k: int) -> list[Hit]:
        calls.append(("dense", query, top_k))
        return dense_hits

    def fake_sparse_search(query: str, top_k: int) -> list[Hit]:
        calls.append(("sparse", query, top_k))
        return sparse_hits

    def fake_rrf_merge(
        actual_dense_hits: list[Hit],
        actual_sparse_hits: list[Hit],
        top_k: int,
    ) -> list[Hit]:
        assert actual_dense_hits is dense_hits
        assert actual_sparse_hits is sparse_hits
        calls.append(("fused", "", top_k))
        return fused_hits

    def fake_rerank(
        query: str,
        actual_fused_hits: list[Hit],
        top_k: int,
    ) -> list[Hit]:
        assert actual_fused_hits is fused_hits
        calls.append(("reranked", query, top_k))
        return reranked_hits

    monkeypatch.setattr(service, "HYBRID_CANDIDATE_K", 3)
    monkeypatch.setattr(dense, "dense_search", fake_dense_search)
    monkeypatch.setattr(service, "bm25_search", fake_sparse_search)
    monkeypatch.setattr(service, "rrf_merge", fake_rrf_merge)
    monkeypatch.setattr(rerank_module, "rerank", fake_rerank)

    with caplog.at_level(logging.INFO, logger=service.__name__):
        hits, debug = service.hybrid_search_with_debug("test query", top_k=2)

    assert hits is reranked_hits
    assert calls == [
        ("dense", "test query", 3),
        ("sparse", "test query", 3),
        ("fused", "", 3),
        ("reranked", "test query", 2),
    ]
    assert debug["query"] == "test query"
    assert set(debug["timings_ms"]) == {
        "dense_search",
        "sparse_search",
        "fusion",
        "rerank",
        "hybrid_total",
    }
    assert all(
        isinstance(ms, float) and ms >= 0
        for ms in debug["timings_ms"].values()
    )

    expected_stages = [
        ("dense_hits", "dense", dense_hits),
        ("sparse_hits", "sparse", sparse_hits),
        ("fused_hits", "fused", fused_hits),
        ("reranked_hits", "reranked", reranked_hits),
    ]
    expected_fields = {
        "rank",
        "stage",
        "id",
        "chunk_id",
        "source_type",
        "source_slug",
        "source_name",
        "heading_path",
        "score",
        "text_preview",
        "metadata",
    }
    for key, stage, original_hits in expected_stages:
        assert len(debug[key]) == len(original_hits)
        for rank, (serialized, original) in enumerate(
            zip(debug[key], original_hits),
            start=1,
        ):
            normalized_text = " ".join(original.text.split())
            assert set(serialized) == expected_fields
            assert serialized == {
                "rank": rank,
                "stage": stage,
                "id": original.id,
                "chunk_id": original.chunk_id,
                "source_type": original.source_type,
                "source_slug": original.source_slug,
                "source_name": original.source_name,
                "heading_path": original.heading_path,
                "score": original.score,
                "text_preview": normalized_text[:157].rstrip() + "...",
                "metadata": original.metadata,
            }
            assert "text" not in serialized

    messages = "\n".join(caplog.messages)
    for stage in (
        "dense_total",
        "sparse_total",
        "rrf_merge",
        "rerank_total",
        "hybrid_total",
    ):
        assert f"stage={stage}" in messages


def test_hybrid_search_with_debug_preserves_sparse_error_mapping(monkeypatch):
    dense_hits = [_hit("dense", 1.0)]
    monkeypatch.setattr(dense, "dense_search", lambda *args, **kwargs: dense_hits)

    def fail_sparse(*args, **kwargs):
        raise RuntimeError("bm25 down")

    monkeypatch.setattr(service, "bm25_search", fail_sparse)

    with pytest.raises(QdrantUnavailable, match="Sparse retrieval failed") as exc_info:
        service.hybrid_search_with_debug("test query")

    debug = exc_info.value.debug
    assert debug["query"] == "test query"
    assert debug["error_stage"] == "sparse_search"
    assert debug["dense_hits"] == service._serialize_hits(dense_hits, "dense")
    assert "sparse_hits" not in debug
    assert set(debug["timings_ms"]) == {
        "dense_search",
        "sparse_search",
        "hybrid_total",
    }
    assert all(
        isinstance(ms, float) and ms >= 0
        for ms in debug["timings_ms"].values()
    )


def test_hybrid_search_with_debug_invokes_substage_callback_in_order(monkeypatch):
    monkeypatch.setattr(service, "HYBRID_CANDIDATE_K", 3)
    monkeypatch.setattr(dense, "dense_search", lambda query, top_k: [_hit("d", 1.0)])
    monkeypatch.setattr(service, "bm25_search", lambda query, top_k: [_hit("s", 1.0)])
    monkeypatch.setattr(service, "rrf_merge", lambda d, s, top_k: [_hit("f", 1.0)])
    monkeypatch.setattr(rerank_module, "rerank", lambda query, fused, top_k: [_hit("r", 1.0)])

    calls: list[tuple[str, str]] = []
    service.hybrid_search_with_debug(
        "q", top_k=2, on_stage=lambda stage, status, ms: calls.append((stage, status)),
    )
    assert [c[0] for c in calls] == ["dense", "sparse", "fusion", "rerank"]
    assert all(c[1] == "ok" for c in calls)


def test_hybrid_search_with_debug_without_callback_unchanged(monkeypatch):
    monkeypatch.setattr(service, "HYBRID_CANDIDATE_K", 3)
    monkeypatch.setattr(dense, "dense_search", lambda query, top_k: [_hit("d", 1.0)])
    monkeypatch.setattr(service, "bm25_search", lambda query, top_k: [_hit("s", 1.0)])
    monkeypatch.setattr(service, "rrf_merge", lambda d, s, top_k: [_hit("f", 1.0)])
    monkeypatch.setattr(rerank_module, "rerank", lambda query, fused, top_k: [_hit("r", 1.0)])

    hits, debug = service.hybrid_search_with_debug("q", top_k=2)
    assert "timings_ms" in debug
    assert set(debug["timings_ms"]) >= {"dense_search", "sparse_search", "fusion", "rerank"}


def test_hybrid_search_with_debug_emits_error_status_on_stage_failure(monkeypatch):
    monkeypatch.setattr(service, "HYBRID_CANDIDATE_K", 3)
    monkeypatch.setattr(dense, "dense_search", lambda query, top_k: [_hit("d", 1.0)])
    monkeypatch.setattr(service, "bm25_search", lambda query, top_k: [_hit("s", 1.0)])
    monkeypatch.setattr(service, "rrf_merge", lambda d, s, top_k: [_hit("f", 1.0)])

    def fail_rerank(query, fused, top_k):
        raise RuntimeError("rerank down")

    monkeypatch.setattr(rerank_module, "rerank", fail_rerank)

    calls: list[tuple[str, str]] = []
    with pytest.raises(QdrantUnavailable):
        service.hybrid_search_with_debug(
            "q", top_k=2, on_stage=lambda stage, status, ms: calls.append((stage, status)),
        )
    assert calls[-1] == ("rerank", "error")
    assert [c[0] for c in calls] == ["dense", "sparse", "fusion", "rerank"]


def test_retrieval_facade_exports_hybrid_search_with_debug():
    assert retrieval.hybrid_search_with_debug is service.hybrid_search_with_debug
