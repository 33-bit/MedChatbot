from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from src.chat.errors import QdrantUnavailable
from src.chat.evidence_plan import (
    plan_requires_drug_usage_detail,
    plan_source_type,
    plan_targets,
    structured_text_match,
)
from src.chat.health_insurance import expand_health_insurance_query
from src.chat.retrieval.fusion import rrf_merge
from src.chat.retrieval.health_insurance import expand_health_insurance_hits
from src.chat.retrieval.sparse import DRUG_CHUNKS, bm25_search
from src.chat.retrieval.types import Hit, RetrievalScope
from src.chat.timing import elapsed_ms
from src.config import HYBRID_CANDIDATE_K, RERANK_TOP_K

log = logging.getLogger(__name__)
_TEXT_PREVIEW_MAX_CHARS = 160
_DRUG_USAGE_QUERY_TERMS = (
    "lieu",
    "lieu dung",
    "cach dung",
    "duong dung",
    "dung the nao",
    "dung nhu the nao",
    "uong",
    "boi",
    "tiem",
    "ngay may lan",
    "bao nhieu",
    "chi dinh",
    "chong chi dinh",
)
_DRUG_USAGE_HEADING_TERMS = (
    "lieu dung",
    "cach dung",
    "chi dinh",
    "chong chi dinh",
)
_DRUG_DOSAGE_HEADING_TERMS = (
    "lieu",
    "lieu dung",
    "lieu luong",
    "cach dung",
    "duong dung",
)
_DRUG_DOSE_HEADING_TERMS = (
    "lieu",
    "lieu dung",
    "lieu luong",
)
_DRUG_ADMIN_HEADING_TERMS = (
    "cach dung",
    "duong dung",
)
_DOSE_TOKEN_RE = re.compile(
    r"\b\d+(?:[,.]\d+)?(?:\s*[-–]\s*\d+(?:[,.]\d+)?)?\s*"
    r"(?:mg|g|mcg|ug|ml|%|lần(?:/ngày)?|ngày|tuần|tháng|năm|viên|giọt|ống)\b",
    flags=re.IGNORECASE,
)


@dataclass
class _HybridSearchDebug:
    query: str
    evidence_plan: dict[str, Any] | None = None
    dense_hits: list[Hit] | None = None
    sparse_hits: list[Hit] | None = None
    fused_hits: list[Hit] | None = None
    reranked_hits: list[Hit] | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    error_stage: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": self.query,
            "timings_ms": dict(self.timings_ms),
        }
        if self.evidence_plan is not None:
            payload["evidence_plan"] = self.evidence_plan
        for attr, key, stage in (
            ("dense_hits", "dense_hits", "dense"),
            ("sparse_hits", "sparse_hits", "sparse"),
            ("fused_hits", "fused_hits", "fused"),
            ("reranked_hits", "reranked_hits", "reranked"),
        ):
            hits = getattr(self, attr)
            if hits is not None:
                payload[key] = _serialize_hits(hits, stage)
        if self.error_stage:
            payload["error_stage"] = self.error_stage
        return payload


class _HybridSearchUnavailable(QdrantUnavailable):
    def __init__(self, message: str, debug: _HybridSearchDebug) -> None:
        super().__init__(message)
        self.debug = debug.as_dict()
        self.error_stage = debug.error_stage


def _normalize(text: str) -> str:
    text = (text or "").casefold().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_drug_usage_query(query: str) -> bool:
    normalized = _normalize(query)
    return any(term in normalized for term in _DRUG_USAGE_QUERY_TERMS)


def _hit_key(hit: Hit) -> tuple[str, str]:
    return (hit.id or "", hit.chunk_id or "")


def _has_drug_usage_heading(hit: Hit) -> bool:
    heading = _normalize(hit.heading_path)
    return any(term in heading for term in _DRUG_USAGE_HEADING_TERMS)


def _has_drug_dosage_heading(hit: Hit) -> bool:
    heading = _normalize(hit.heading_path)
    return any(term in heading for term in _DRUG_DOSAGE_HEADING_TERMS)


def _has_drug_dose_heading(hit: Hit) -> bool:
    segments = [
        _normalize(segment)
        for segment in (hit.heading_path or "").split(">")
        if segment.strip()
    ]
    for segment in segments:
        has_dose = any(term in segment for term in _DRUG_DOSE_HEADING_TERMS)
        has_admin_only = any(term in segment for term in _DRUG_ADMIN_HEADING_TERMS)
        if has_dose and not has_admin_only:
            return True
        if "lieu luong" in segment and _DOSE_TOKEN_RE.search(hit.text or ""):
            return True
    return False


@lru_cache(maxsize=1)
def _local_drug_chunks() -> tuple[Hit, ...]:
    if not DRUG_CHUNKS.exists():
        return ()
    hits: list[Hit] = []
    with open(DRUG_CHUNKS, encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            if chunk.get("source_type") != "drug":
                continue
            hits.append(Hit(
                text=chunk.get("text", ""),
                score=0.0,
                source_type="drug",
                source_name=chunk.get("source_name", ""),
                heading_path=chunk.get("heading_path", ""),
                source_slug=chunk.get("source_slug", ""),
                chunk_id=chunk.get("chunk_id", ""),
                metadata=chunk.get("metadata"),
                id=str(chunk.get("id") or ""),
            ))
    return tuple(hits)


def _local_drug_usage_candidates(slug: str) -> list[Hit]:
    same_drug = [
        hit for hit in _local_drug_chunks()
        if hit.source_slug == slug
    ]
    dose = [hit for hit in same_drug if _has_drug_dose_heading(hit)]
    usage = [
        hit for hit in same_drug
        if _has_drug_usage_heading(hit) and hit not in dose
    ]
    return dose + usage


def _ensure_drug_usage_context(
    query: str,
    hits: list[Hit],
    candidates: list[Hit],
    top_k: int,
) -> list[Hit]:
    if not _is_drug_usage_query(query):
        return hits
    drug_slugs = [
        hit.source_slug
        for hit in hits
        if hit.source_type == "drug" and hit.source_slug
    ]
    if not drug_slugs:
        return hits
    slug_set = set(drug_slugs)
    has_usage_heading = any(
        hit.source_type == "drug"
        and hit.source_slug in slug_set
        and _has_drug_usage_heading(hit)
        for hit in hits
    )
    has_dose_heading = any(
        hit.source_type == "drug"
        and hit.source_slug in slug_set
        and _has_drug_dose_heading(hit)
        for hit in hits
    )
    if has_usage_heading and has_dose_heading:
        return hits

    seen = {_hit_key(hit) for hit in hits}
    for slug in drug_slugs:
        replacement = next(
            (
                hit for hit in candidates
                if hit.source_type == "drug"
                and hit.source_slug == slug
                and _has_drug_dose_heading(hit)
                and _hit_key(hit) not in seen
            ),
            None,
        )
        if replacement is None:
            replacement = next(
                (
                    hit for hit in candidates
                    if hit.source_type == "drug"
                    and hit.source_slug == slug
                    and _has_drug_usage_heading(hit)
                    and _hit_key(hit) not in seen
                ),
                None,
            )
        if replacement is None:
            replacement = next(
                (
                    hit for hit in _local_drug_usage_candidates(slug)
                    if _hit_key(hit) not in seen
                ),
                None,
            )
        if replacement is None and not has_usage_heading:
            replacement = next(
                (
                    hit for hit in _local_drug_usage_candidates(slug)
                    if _has_drug_usage_heading(hit)
                    and _hit_key(hit) not in seen
                ),
                None,
            )
        if replacement is None:
            continue
        insert_at = next(
            (
                index + 1 for index, hit in enumerate(hits)
                if hit.source_type == "drug" and hit.source_slug == slug
            ),
            len(hits),
        )
        updated = list(hits)
        updated.insert(insert_at, replacement)
        while len(updated) > top_k:
            drop_at = next(
                (
                    index for index in range(len(updated) - 1, -1, -1)
                    if index != insert_at and updated[index].source_type != "drug"
                ),
                None,
            )
            if drop_at is None:
                drop_at = len(updated) - 1
                if drop_at == insert_at:
                    break
            updated.pop(drop_at)
        return updated
    return hits


def _dedupe_candidates(candidates: list[Hit]) -> list[Hit]:
    seen: set[tuple[str, str]] = set()
    unique: list[Hit] = []
    for hit in candidates:
        key = _hit_key(hit)
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)
    return unique


def _matches_plan_source(hit: Hit, source_type: str) -> bool:
    return not source_type or source_type == "medical" or hit.source_type == source_type


def _matches_plan_entity(hit: Hit, entity: object) -> bool:
    if not entity:
        return False
    return (
        structured_text_match(hit.source_name, entity)
        or structured_text_match(hit.source_slug, entity)
    )


def _matches_plan_heading(hit: Hit, targets: list[str]) -> bool:
    if not targets:
        return False
    return any(structured_text_match(hit.heading_path, target) for target in targets)


def _evidence_plan_candidate_score(
    hit: Hit,
    *,
    source_type: str,
    entity: object,
    targets: list[str],
    anchor_slugs: set[str],
) -> float:
    if not _matches_plan_source(hit, source_type):
        return -1.0
    score = 0.0
    if anchor_slugs and hit.source_slug in anchor_slugs:
        score += 3.0
    if _matches_plan_entity(hit, entity):
        score += 2.0
    if _matches_plan_heading(hit, targets):
        score += 4.0
    if source_type and source_type != "medical" and hit.source_type == source_type:
        score += 1.0
    return score


def _apply_evidence_plan_context(
    hits: list[Hit],
    candidates: list[Hit],
    top_k: int,
    evidence_plan: dict[str, Any] | None,
) -> list[Hit]:
    if not evidence_plan or top_k <= 0 or not hits:
        return hits
    source_type = plan_source_type(evidence_plan)
    targets = plan_targets(evidence_plan)
    entity = evidence_plan.get("entity")
    requires_drug_usage = plan_requires_drug_usage_detail(evidence_plan)
    if not source_type and not targets and not entity:
        return hits

    anchor_slugs = {
        hit.source_slug
        for hit in hits
        if hit.source_slug
        and _matches_plan_source(hit, source_type)
        and (not entity or _matches_plan_entity(hit, entity))
    }
    if not anchor_slugs and entity:
        anchor_slugs = {
            hit.source_slug
            for hit in candidates
            if hit.source_slug
            and _matches_plan_source(hit, source_type)
            and _matches_plan_entity(hit, entity)
        }

    seen = {_hit_key(hit) for hit in hits}
    scored: list[tuple[float, Hit]] = []
    for hit in _dedupe_candidates(candidates):
        if _hit_key(hit) in seen:
            continue
        if anchor_slugs and hit.source_slug not in anchor_slugs:
            continue
        score = _evidence_plan_candidate_score(
            hit,
            source_type=source_type,
            entity=entity,
            targets=targets,
            anchor_slugs=anchor_slugs,
        )
        if score <= 0:
            continue
        if requires_drug_usage and _has_drug_dosage_heading(hit):
            score += 5.0
        if targets and not _matches_plan_heading(hit, targets):
            if not (
                source_type == "drug"
                and requires_drug_usage
                and _has_drug_dosage_heading(hit)
            ):
                continue
        if (
            source_type == "drug"
            and requires_drug_usage
            and not targets
            and not _has_drug_dosage_heading(hit)
        ):
            continue
        scored.append((score, hit))
    if not scored:
        return hits

    scored.sort(key=lambda item: (item[0], item[1].score), reverse=True)
    updated = list(hits)
    insert_after = 0
    protected_keys: set[tuple[str, str]] = set()
    if anchor_slugs:
        insert_after = next(
            (
                index + 1
                for index, hit in enumerate(updated)
                if hit.source_slug in anchor_slugs
            ),
            1,
        )
        protected_keys.update(
            _hit_key(hit)
            for hit in updated[:insert_after]
            if hit.source_slug in anchor_slugs
        )
    inserted = 0
    max_inserted = 3 if source_type == "drug" and requires_drug_usage else 2
    for _score, candidate in scored[:max_inserted]:
        updated.insert(min(insert_after + inserted, len(updated)), candidate)
        protected_keys.add(_hit_key(candidate))
        inserted += 1

    while len(updated) > top_k:
        drop_at = next(
            (
                index
                for index in range(len(updated) - 1, -1, -1)
                if _hit_key(updated[index]) not in protected_keys
            ),
            len(updated) - 1,
        )
        if _hit_key(updated[drop_at]) in protected_keys:
            break
        updated.pop(drop_at)
    return updated


def _record_debug_timing(
    debug: _HybridSearchDebug,
    stage: str,
    stage_start: float,
) -> float:
    ms = elapsed_ms(stage_start)
    debug.timings_ms[stage] = round(ms, 2)
    return ms


def _record_debug_failure(
    debug: _HybridSearchDebug,
    stage: str,
    stage_start: float,
    total_start: float,
) -> None:
    debug.error_stage = stage
    _record_debug_timing(debug, stage, stage_start)
    debug.timings_ms["hybrid_total"] = round(elapsed_ms(total_start), 2)


def _attach_debug(
    exc: QdrantUnavailable,
    debug: _HybridSearchDebug,
) -> None:
    exc.debug = debug.as_dict()
    exc.error_stage = debug.error_stage


def _run_hybrid_search(
    query: str,
    top_k: int,
    scope: RetrievalScope = "medical",
    on_stage=None,
    evidence_plan: dict[str, Any] | None = None,
) -> tuple[list[Hit], _HybridSearchDebug]:
    import time

    from src.chat.retrieval.dense import dense_search
    from src.chat.retrieval.rerank import rerank

    debug = _HybridSearchDebug(query=query, evidence_plan=evidence_plan)
    retrieval_query = (
        expand_health_insurance_query(query)
        if scope == "health_insurance"
        else query
    )
    total_start = time.perf_counter()
    stage_start = time.perf_counter()
    try:
        if scope == "medical":
            dense_hits = dense_search(retrieval_query, top_k=HYBRID_CANDIDATE_K)
        else:
            dense_hits = dense_search(
                retrieval_query,
                top_k=HYBRID_CANDIDATE_K,
                scope=scope,
            )
    except QdrantUnavailable as exc:
        _record_debug_failure(
            debug, "dense_search", stage_start, total_start,
        )
        if on_stage is not None:
            on_stage("dense", "error", debug.timings_ms["dense_search"])
        _attach_debug(exc, debug)
        raise
    except Exception as e:
        _record_debug_failure(
            debug, "dense_search", stage_start, total_start,
        )
        if on_stage is not None:
            on_stage("dense", "error", debug.timings_ms["dense_search"])
        raise _HybridSearchUnavailable("Dense retrieval failed", debug) from e
    debug.dense_hits = dense_hits
    dense_ms = _record_debug_timing(debug, "dense_search", stage_start)
    if on_stage is not None:
        on_stage("dense", "ok", dense_ms)
    log.info("retrieval timing stage=dense_total ms=%.1f hits=%d",
             dense_ms, len(dense_hits))

    stage_start = time.perf_counter()
    try:
        if scope == "medical":
            sparse_hits = bm25_search(retrieval_query, top_k=HYBRID_CANDIDATE_K)
        else:
            sparse_hits = bm25_search(
                retrieval_query,
                top_k=HYBRID_CANDIDATE_K,
                scope=scope,
            )
    except Exception as e:
        _record_debug_failure(
            debug, "sparse_search", stage_start, total_start,
        )
        if on_stage is not None:
            on_stage("sparse", "error", debug.timings_ms["sparse_search"])
        raise _HybridSearchUnavailable("Sparse retrieval failed", debug) from e
    debug.sparse_hits = sparse_hits
    sparse_ms = _record_debug_timing(debug, "sparse_search", stage_start)
    if on_stage is not None:
        on_stage("sparse", "ok", sparse_ms)
    log.info("retrieval timing stage=sparse_total ms=%.1f hits=%d",
             sparse_ms, len(sparse_hits))

    stage_start = time.perf_counter()
    try:
        fused = rrf_merge(dense_hits, sparse_hits, top_k=HYBRID_CANDIDATE_K)
    except Exception as e:
        _record_debug_failure(
            debug, "fusion", stage_start, total_start,
        )
        if on_stage is not None:
            on_stage("fusion", "error", debug.timings_ms["fusion"])
        raise _HybridSearchUnavailable("Fusion failed", debug) from e
    debug.fused_hits = fused
    fusion_ms = _record_debug_timing(debug, "fusion", stage_start)
    if on_stage is not None:
        on_stage("fusion", "ok", fusion_ms)
    log.info("retrieval timing stage=rrf_merge ms=%.1f hits=%d",
             fusion_ms, len(fused))

    stage_start = time.perf_counter()
    try:
        reranked = rerank(query, fused, top_k=top_k)
    except Exception as e:
        _record_debug_failure(
            debug, "rerank", stage_start, total_start,
        )
        if on_stage is not None:
            on_stage("rerank", "error", debug.timings_ms["rerank"])
        raise _HybridSearchUnavailable("Rerank failed", debug) from e
    debug.reranked_hits = reranked
    if scope == "medical":
        candidates = fused + dense_hits + sparse_hits
        if evidence_plan:
            reranked = _apply_evidence_plan_context(
                reranked,
                candidates,
                top_k,
                evidence_plan,
            )
        reranked = _ensure_drug_usage_context(
            query,
            reranked,
            candidates,
            top_k,
        )
        debug.reranked_hits = reranked
    if scope == "health_insurance":
        reranked = expand_health_insurance_hits(reranked, query=query)
        debug.reranked_hits = reranked
    rerank_ms = _record_debug_timing(debug, "rerank", stage_start)
    if on_stage is not None:
        on_stage("rerank", "ok", rerank_ms)
    log.info("retrieval timing stage=rerank_total ms=%.1f hits=%d",
             rerank_ms, len(reranked))
    hybrid_ms = elapsed_ms(total_start)
    debug.timings_ms["hybrid_total"] = round(hybrid_ms, 2)
    log.info("retrieval timing stage=hybrid_total ms=%.1f hits=%d",
             hybrid_ms, len(reranked))
    return reranked, debug


def _text_preview(text: str) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= _TEXT_PREVIEW_MAX_CHARS:
        return normalized
    return normalized[:_TEXT_PREVIEW_MAX_CHARS - 3].rstrip() + "..."


def _serialize_hits(hits: list[Hit], stage: str) -> list[dict]:
    return [
        {
            "rank": rank,
            "stage": stage,
            "id": hit.id,
            "chunk_id": hit.chunk_id,
            "source_type": hit.source_type,
            "source_slug": hit.source_slug,
            "source_name": hit.source_name,
            "heading_path": hit.heading_path,
            "score": float(hit.score),
            "text_preview": _text_preview(hit.text),
            "metadata": hit.metadata,
        }
        for rank, hit in enumerate(hits, start=1)
    ]


def hybrid_search(
    query: str,
    top_k: int = RERANK_TOP_K,
    scope: RetrievalScope = "medical",
    evidence_plan: dict[str, Any] | None = None,
) -> list[Hit]:
    """Dense + BM25 → RRF fusion → Cross-Encoder re-rank."""
    hits, _ = _run_hybrid_search(
        query,
        top_k,
        scope=scope,
        evidence_plan=evidence_plan,
    )
    return hits


def hybrid_search_with_debug(
    query: str,
    top_k: int = RERANK_TOP_K,
    scope: RetrievalScope = "medical",
    on_stage=None,
    evidence_plan: dict[str, Any] | None = None,
) -> tuple[list[Hit], dict]:
    hits, debug = _run_hybrid_search(
        query,
        top_k,
        scope=scope,
        on_stage=on_stage,
        evidence_plan=evidence_plan,
    )
    return hits, debug.as_dict()
