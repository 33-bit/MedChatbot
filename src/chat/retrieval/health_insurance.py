"""Health-insurance specific retrieval helpers."""

from __future__ import annotations

import re
from functools import lru_cache

from src.chat.health_insurance import expand_health_insurance_query
from src.chat.health_insurance import _normalize as _normalize_query
from src.chat.retrieval.sparse import _load_chunks
from src.chat.retrieval.types import Hit


_ARTICLE_REF_RE = re.compile(
    r"(?:(?:các\s+)?khoản\s+"
    r"((?:\d+[a-z]?\s*(?:,|\s+và\s+|\s+và\s+khoản\s+)?\s*){1,8}))?"
    r"\s*Điều\s+(\d+[a-z]?)",
    re.IGNORECASE,
)
_FOOTNOTE_RE = re.compile(r"^\d{1,3}\s+(Khoản|Điểm|Điều|Luật số)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+[a-z]?", re.IGNORECASE)


def _body_without_footnotes(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if _FOOTNOTE_RE.match(line.strip()):
            continue
        lines.append(line)
    return "\n".join(lines)


def _article_refs(text: str) -> list[tuple[str, tuple[str, ...]]]:
    refs: list[tuple[str, tuple[str, ...]]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for match in _ARTICLE_REF_RE.finditer(_body_without_footnotes(text)):
        clause_text = match.group(1) or ""
        article = match.group(2).casefold()
        clauses = tuple(num.casefold() for num in _NUMBER_RE.findall(clause_text))
        key = (article, clauses)
        if key in seen:
            continue
        seen.add(key)
        refs.append(key)
    return refs


@lru_cache(maxsize=1)
def _health_chunks() -> tuple[dict, ...]:
    return tuple(_load_chunks("health_insurance"))


def _hit_from_chunk(chunk: dict, score: float) -> Hit:
    return Hit(
        text=chunk.get("text", ""),
        score=score,
        source_type=chunk.get("source_type", "health_insurance"),
        source_name=chunk.get("source_name", ""),
        heading_path=chunk.get("heading_path", ""),
        source_slug=chunk.get("source_slug", ""),
        chunk_id=chunk.get("chunk_id", ""),
        metadata=chunk.get("metadata"),
        id=str(chunk.get("id") or ""),
    )


def _referenced_hits(article: str, clauses: tuple[str, ...], score: float) -> list[Hit]:
    hits: list[Hit] = []
    clause_set = set(clauses)
    for chunk in _health_chunks():
        metadata = chunk.get("metadata") or {}
        if str(metadata.get("article_number", "")).casefold() != article:
            continue
        if clause_set:
            chunk_clause = str(metadata.get("clause_number", "")).casefold()
            if chunk_clause not in clause_set:
                continue
        hits.append(_hit_from_chunk(chunk, score))
    return hits


def _ref_priority(query: str, article: str, clauses: tuple[str, ...], order: int) -> tuple[int, int]:
    text = _normalize_query(query)
    clause_set = set(clauses)
    if article == "12" and "5" in clause_set and "ho gia dinh" in text:
        return (0, order)
    if article == "13" and "1" in clause_set and (
        "nguoi lao dong" in text or "nguoi su dung lao dong" in text
    ):
        return (0, order)
    if article == "23" and ("khong duoc huong" in text or "khong duoc bao hiem" in text):
        return (0, order)
    return (1, order)


def expand_health_insurance_hits(
    hits: list[Hit],
    *,
    query: str = "",
    max_added: int = 4,
) -> list[Hit]:
    """Append law chunks referenced by retrieved chunks.

    This handles legal answers that depend on a cross-reference, e.g. Article
    16 referring to clause 5 Article 12 for household participation.
    """
    expanded: list[Hit] = []
    seen: set[str] = set()
    added = 0
    refs_by_hit: dict[int, list[tuple[int, int, Hit, str, tuple[str, ...]]]] = {}
    deferred_refs: list[tuple[int, int, Hit, str, tuple[str, ...]]] = []
    ref_order = 0
    for hit_index, hit in enumerate(hits):
        if hit.source_type != "health_insurance":
            continue
        current_article = str((hit.metadata or {}).get("article_number", "")).casefold()
        for article, clauses in _article_refs(hit.text):
            if article == current_article:
                continue
            priority, order = _ref_priority(query, article, clauses, ref_order)
            ref_order += 1
            item = (priority, order, hit, article, clauses)
            if priority == 0:
                refs_by_hit.setdefault(hit_index, []).append(item)
            else:
                deferred_refs.append(item)

    def add_ref(hit: Hit, article: str, clauses: tuple[str, ...]) -> bool:
        nonlocal added
        for referenced in _referenced_hits(article, clauses, score=max(hit.score * 0.9, 0.01)):
            key = referenced.id or referenced.chunk_id
            if key in seen:
                continue
            expanded.append(referenced)
            seen.add(key)
            added += 1
            if added >= max_added:
                return False
        return True

    for hit_index, hit in enumerate(hits):
        key = hit.id or hit.chunk_id
        if key not in seen:
            expanded.append(hit)
            if key:
                seen.add(key)
        for _priority, _order, source_hit, article, clauses in sorted(refs_by_hit.get(hit_index, [])):
            if not add_ref(source_hit, article, clauses):
                return expanded

    for _priority, _order, source_hit, article, clauses in sorted(deferred_refs):
        if not add_ref(source_hit, article, clauses):
            return expanded
    return expanded


__all__ = [
    "expand_health_insurance_hits",
    "expand_health_insurance_query",
]
