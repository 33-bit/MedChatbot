"""Shared category-evaluation defaults."""

from __future__ import annotations

from typing import Any

from eval import metrics


CATEGORY = "default"
ANSWER_CHECKS = []

METRIC_PROFILE = {
    "primary": ["chunk_recall@5", "faithfulness_score", "correctness_score"],
    "hard_checks": ["requires_citation"],
}


def apply_answer_checks(
    case: dict[str, Any],
    answer: str,
    scored: dict[str, Any],
    pass_threshold: float,
    checks,
) -> dict[str, Any]:
    return metrics.apply_answer_checks(case, answer, scored, pass_threshold, checks)


def apply_category_checks(
    case: dict[str, Any],
    answer: str,
    scored: dict[str, Any],
    pass_threshold: float,
) -> dict[str, Any]:
    return apply_answer_checks(case, answer, scored, pass_threshold, ANSWER_CHECKS)
