from __future__ import annotations

import importlib
from pathlib import Path

from src.chat.errors import QdrantUnavailable
from src.chat import retrieval
from src.chat.retrieval import dense
from src.chat.retrieval import fusion
from src.chat.retrieval import kg
from src.chat.retrieval import service
from src.chat.retrieval.types import Hit

rerank_module = importlib.import_module("src.chat.retrieval.rerank")


class FakeQdrant:
    def __init__(self, exists: bool = True, fail: bool = False) -> None:
        self.exists = exists
        self.fail = fail
        self.checked: list[str] = []

    def collection_exists(self, collection: str) -> bool:
        self.checked.append(collection)
        if self.fail:
            raise RuntimeError("qdrant down")
        return self.exists


def test_qdrant_collection_exists_is_cached(monkeypatch):
    fake_client = FakeQdrant()
    monkeypatch.setattr(dense, "_qdrant", lambda: fake_client)
    monkeypatch.setattr(dense, "_collection_cache", None)

    first = dense._available_collections()
    second = dense._available_collections()

    assert first == second == (dense.DISEASES_COLLECTION, dense.DRUGS_COLLECTION)
    assert fake_client.checked == [dense.DISEASES_COLLECTION, dense.DRUGS_COLLECTION]


def test_qdrant_collection_failure_raises_dependency_error(monkeypatch):
    monkeypatch.setattr(dense, "_qdrant", lambda: FakeQdrant(fail=True))
    monkeypatch.setattr(dense, "_collection_cache", None)

    try:
        dense._available_collections()
    except QdrantUnavailable as exc:
        assert "Qdrant collection check failed" in str(exc)
    else:
        raise AssertionError("expected QdrantUnavailable")


def test_missing_qdrant_collection_raises_dependency_error(monkeypatch):
    monkeypatch.setattr(dense, "_qdrant", lambda: FakeQdrant(exists=False))
    monkeypatch.setattr(dense, "_collection_cache", None)

    try:
        dense._available_collections()
    except QdrantUnavailable as exc:
        assert "Missing configured Qdrant collections" in str(exc)
    else:
        raise AssertionError("expected QdrantUnavailable")


def test_retrieval_facade_exports_canonical_modules():
    dense_source = Path("src/chat/retrieval/dense.py").read_text(encoding="utf-8")

    assert "src.chat.retrieval.service import hybrid_search" not in dense_source
    assert not hasattr(dense, "hybrid_search")
    assert retrieval.hybrid_search is service.hybrid_search
    assert retrieval.rrf_merge is fusion.rrf_merge
    assert retrieval.rerank is rerank_module.rerank


def test_hit_documents_score_semantics():
    assert "score" in (Hit.__doc__ or "").lower()
    assert "retrieval stage" in (Hit.__doc__ or "").lower()


def test_kg_fulltext_search_sanitizes_lucene_query_syntax():
    captured: dict[str, object] = {}

    class Tx:
        def run(self, query: str, **params):
            captured["query"] = query
            captured["params"] = params
            return []

    result = kg.fulltext_search(
        Tx(),
        "disease_name",
        "buồn nôn/nôn foo: đau [bụng]",
        3,
    )

    assert result == []
    assert captured["params"]["q"] == "buồn nôn nôn foo đau bụng"


def test_kg_drug_context_formats_adverse_reactions():
    ctx = kg.KGContext()

    kg._ingest_drug(
        ctx,
        "Demo",
        {
            "treats": [],
            "relieves": [],
            "contraindicated_for": [],
            "interactions": [],
            "adverse_reactions": [
                {"id": "symptom:S_nausea", "name": "Buồn nôn", "frequency": "common"},
                {
                    "id": "symptom:S_anaphylaxis",
                    "name": "Sốc phản vệ",
                    "frequency": "rare_serious",
                },
            ],
        },
    )

    assert "Tác dụng không mong muốn: Buồn nôn, Sốc phản vệ" in kg.format_kg_context(ctx)


def _hit(text: str) -> Hit:
    return Hit(
        text=text,
        score=1.0,
        source_type="disease",
        source_name="Nguồn kiểm thử",
        heading_path="",
        source_slug="nguon-kiem-thu",
    )


def test_hybrid_search_fails_closed_when_sparse_search_fails(monkeypatch):
    monkeypatch.setattr(dense, "dense_search", lambda *args, **kwargs: [_hit("dense")])

    def fail_sparse(*args, **kwargs):
        raise RuntimeError("bm25 down")

    monkeypatch.setattr(service, "bm25_search", fail_sparse)

    try:
        service.hybrid_search("Tôi bị ho")
    except QdrantUnavailable as exc:
        assert "Sparse retrieval failed" in str(exc)
    else:
        raise AssertionError("expected QdrantUnavailable")


def test_hybrid_search_fails_closed_when_rerank_fails(monkeypatch):
    monkeypatch.setattr(dense, "dense_search", lambda *args, **kwargs: [_hit("dense")])
    monkeypatch.setattr(service, "bm25_search", lambda *args, **kwargs: [_hit("sparse")])
    monkeypatch.setattr(service, "rrf_merge", lambda *args, **kwargs: [_hit("merged")])

    def fail_rerank(*args, **kwargs):
        raise RuntimeError("rerank down")

    monkeypatch.setattr(rerank_module, "rerank", fail_rerank)

    try:
        service.hybrid_search("Tôi bị ho")
    except QdrantUnavailable as exc:
        assert "Rerank failed" in str(exc)
    else:
        raise AssertionError("expected QdrantUnavailable")


def test_reranker_uses_cpu_and_small_prediction_batches(monkeypatch):
    captured: dict[str, object] = {}

    class FakeCrossEncoder:
        def __init__(self, model_name: str, *, device: str | None = None) -> None:
            captured["model_name"] = model_name
            captured["device"] = device

        def predict(self, pairs, batch_size: int = 32):
            captured["pairs"] = pairs
            captured["batch_size"] = batch_size
            return [0.2, 0.9]

    rerank_module.reranker.cache_clear()
    monkeypatch.setattr(rerank_module, "CrossEncoder", FakeCrossEncoder)

    result = rerank_module.rerank("đau bụng", [_hit("a"), _hit("b")], top_k=1)

    assert captured["device"] == "cpu"
    assert captured["batch_size"] == 4
    assert captured["pairs"] == [["đau bụng", "a"], ["đau bụng", "b"]]
    assert result[0].text == "b"
    rerank_module.reranker.cache_clear()
