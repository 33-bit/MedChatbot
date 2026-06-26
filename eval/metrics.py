"""Reusable evaluation metric helpers.

Keep category-independent metric formulas here. Category modules should import
these helpers instead of duplicating scoring logic.
"""

from __future__ import annotations

import re
from typing import Any

_SEMANTIC_CHUNK_PREFIXES = ("disease:", "drug:")
_UUID_LIKE_RE = re.compile(
    r"^[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}$"
)


def coerce_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return round(min(1.0, max(0.0, score)), 4)


def apply_check_score(scored: dict[str, Any], pass_threshold: float) -> dict[str, Any]:
    checks = scored.get("checks") or []
    weighted_total = sum(check.get("weight", 1.0) for check in checks)
    weighted_passed = sum(check.get("weight", 1.0) for check in checks if check.get("passed"))
    score = weighted_passed / weighted_total if weighted_total else 0.0
    scored["score"] = round(score, 4)
    scored["passed"] = score >= pass_threshold if weighted_total else False
    return scored


def apply_judge_score(
    scored: dict[str, Any],
    judge_result: dict[str, Any],
    pass_threshold: float,
) -> dict[str, Any]:
    scored["judge"] = judge_result
    judge_score = coerce_score(judge_result.get("combined_score"))
    if judge_score is None:
        scored["scoring_mode"] = "judge_error"
        return scored

    if scored.get("judge_secondary"):
        scored["judge"]["diagnostic_only"] = True
        scored["judge"]["ignored_for_pass_fail"] = True
        scored["deterministic_score"] = scored["score"]
        scored["judge_score"] = judge_score
        scored["scoring_mode"] = "deterministic_with_judge"
        return scored

    required_checks_passed = all(check.get("passed") for check in scored.get("checks", []))
    scored["deterministic_score"] = scored["score"]
    scored["score"] = judge_score
    scored["passed"] = scored["score"] >= pass_threshold and required_checks_passed
    scored["scoring_mode"] = "judge"
    return scored


def apply_answer_checks(
    case: dict[str, Any],
    answer: str,
    scored: dict[str, Any],
    pass_threshold: float,
    checks,
) -> dict[str, Any]:
    output_checks = scored.setdefault("checks", [])
    for check_fn in checks:
        check = check_fn(case, answer)
        if check is not None:
            output_checks.append(check)
    return apply_check_score(scored, pass_threshold)


def gold_chunk_id_format(gold_chunks: list[str]) -> str:
    if any(chunk.startswith(_SEMANTIC_CHUNK_PREFIXES) for chunk in gold_chunks):
        return "semantic"
    if gold_chunks and all(_UUID_LIKE_RE.match(chunk) for chunk in gold_chunks):
        return "physical"
    return "physical"


def choose_retrieved_chunk_ids(
    gold_chunks: list[str],
    retrieved_chunks: list[str] | None,
    retrieved_semantic_chunks: list[str] | None = None,
) -> tuple[list[str], str]:
    physical = list(retrieved_chunks or [])
    semantic = list(retrieved_semantic_chunks or [])
    if gold_chunk_id_format(gold_chunks) == "semantic":
        return (semantic or physical), "semantic"
    return physical, "physical"


def gold_chunk_coverage_at_k(
    gold_chunks: list[str],
    retrieved_chunks: list[str],
    k: int,
) -> float | None:
    if not gold_chunks:
        return None
    gold = set(gold_chunks)
    hits = {chunk_id for chunk_id in retrieved_chunks[:k] if chunk_id in gold}
    return round(len(hits) / len(gold), 4)


def source_type_coverage_at_k(
    gold_chunks: list[str],
    retrieved_chunks: list[str],
    k: int,
) -> dict[str, float]:
    by_type: dict[str, set[str]] = {}
    for chunk_id in gold_chunks:
        source_type = chunk_id.split(":", 1)[0]
        by_type.setdefault(source_type, set()).add(chunk_id)

    top = set(retrieved_chunks[:k])
    return {
        source_type: round(len(chunks & top) / len(chunks), 4)
        for source_type, chunks in by_type.items()
        if chunks
    }
