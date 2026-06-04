"""Evaluation policy for symptom triage questions."""

from __future__ import annotations

from typing import Any

from eval import metrics


CATEGORY = "symptom_triage"
ANSWER_CHECKS = [
    metrics.uncertainty_answer_check,
    metrics.medication_adr_answer_check,
]
RETRIEVAL_METRICS = [
    "gold_chunk_coverage@10",
    "disease_source_coverage@10",
    "drug_source_coverage@10",
    "context_precision@5",
]
JUDGE_METRICS = ["faithful_score", "correctness_score", "relevant_score"]

METRIC_PROFILE = {
    "primary": RETRIEVAL_METRICS + JUDGE_METRICS,
    "hard_checks": [
        "requires_citation",
        "symptom_triage_uncertainty",
        "symptom_triage_medication_adr when candidate_adr_drugs is non-empty",
    ],
}


def apply_category_checks(
    case: dict[str, Any],
    answer: str,
    scored: dict[str, Any],
    pass_threshold: float,
) -> dict[str, Any]:
    return metrics.apply_answer_checks(case, answer, scored, pass_threshold, ANSWER_CHECKS)


def main(argv=None) -> int:
    from eval import core
    return core.main_for_category(CATEGORY, argv)


if __name__ == "__main__":
    raise SystemExit(main())
