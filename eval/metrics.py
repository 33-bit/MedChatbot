"""Reusable evaluation metric helpers.

Keep category-independent metric formulas here. Category modules should import
these helpers instead of duplicating scoring logic.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize(text: str) -> str:
    text = (text or "").casefold().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_any_normalized(text: str, terms: list[str]) -> bool:
    haystack = normalize(text)
    return any(normalize(term) in haystack for term in terms)


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

    required_checks_passed = all(check.get("passed") for check in scored.get("checks", []))
    scored["deterministic_score"] = scored["score"]
    scored["score"] = judge_score
    scored["passed"] = scored["score"] >= pass_threshold and required_checks_passed
    scored["scoring_mode"] = "judge"
    return scored


def uncertainty_check(answer: str) -> dict[str, Any]:
    terms = [
        "chưa thể chẩn đoán",
        "không thể chẩn đoán",
        "chưa thể kết luận",
        "không thể kết luận",
        "không thể xác định chắc chắn",
    ]
    return {
        "type": "symptom_triage_uncertainty",
        "target": "state diagnostic uncertainty",
        "passed": contains_any_normalized(answer, terms),
        "weight": 1.0,
    }


def medication_adr_check(answer: str) -> dict[str, Any]:
    terms = [
        "thuốc",
        "tác dụng phụ",
        "tác dụng không mong muốn",
        "thực phẩm chức năng",
        "thuốc nam",
        "vỏ thuốc",
        "danh sách thuốc",
    ]
    return {
        "type": "symptom_triage_medication_adr",
        "target": "mention recent medication/ADR review",
        "passed": contains_any_normalized(answer, terms),
        "weight": 1.0,
    }


def uncertainty_answer_check(case: dict[str, Any], answer: str) -> dict[str, Any] | None:
    return uncertainty_check(answer)


def medication_adr_answer_check(case: dict[str, Any], answer: str) -> dict[str, Any] | None:
    if not case.get("candidate_adr_drugs"):
        return None
    return medication_adr_check(answer)


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
