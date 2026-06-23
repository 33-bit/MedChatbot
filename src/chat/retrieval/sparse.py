"""
sparse.py
---------
In-memory BM25Okapi index over disease + drug chunks for sparse retrieval.
Loads from JSONL on first call, then cached.
"""

from __future__ import annotations

import json
import re
import uuid
from functools import lru_cache

from rank_bm25 import BM25Okapi

from src.chat.retrieval.types import Hit, RetrievalScope
from src.config import OUTPUT_DIR

CHUNKS_DIR = OUTPUT_DIR / "chunks"
DISEASE_CHUNKS = CHUNKS_DIR / "disease_chunks.jsonl"
DRUG_CHUNKS = CHUNKS_DIR / "drug_chunks.jsonl"
HEALTH_INSURANCE_CHUNKS = CHUNKS_DIR / "health_insurance_chunks.jsonl"
_CHUNKS_BY_SCOPE: dict[RetrievalScope, tuple] = {
    "medical": (DISEASE_CHUNKS, DRUG_CHUNKS),
    "health_insurance": (HEALTH_INSURANCE_CHUNKS,),
}

_TOKEN_RE = re.compile(r"[a-zA-Zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _load_chunks(scope: RetrievalScope = "medical") -> list[dict]:
    chunks = []
    for path in _CHUNKS_BY_SCOPE[scope]:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                if "id" not in c:
                    c["id"] = str(uuid.uuid4())
                chunks.append(c)
    return chunks


class BM25Index:
    def __init__(self, chunks: list[dict]):
        self._chunks = chunks
        corpus = [_tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 20) -> list[Hit]:
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        top_idx = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_idx:
            if scores[idx] <= 0:
                break
            c = self._chunks[idx]
            results.append(Hit(
                text=c.get("text", ""),
                score=float(scores[idx]),
                source_type=c.get("source_type", "disease"),
                source_name=c.get("source_name", ""),
                heading_path=c.get("heading_path", ""),
                source_slug=c.get("source_slug", ""),
                chunk_id=c.get("chunk_id", ""),
                metadata=c.get("metadata"),
            ))
        return results


@lru_cache(maxsize=2)
def _index(scope: RetrievalScope = "medical") -> BM25Index:
    chunks = _load_chunks(scope)
    return BM25Index(chunks)


def bm25_search(
    query: str,
    top_k: int = 20,
    scope: RetrievalScope = "medical",
) -> list[Hit]:
    return _index(scope).search(query, top_k)
