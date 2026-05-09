from __future__ import annotations

from src.chat.errors import QdrantUnavailable
from src.chat.retrieval import dense


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
