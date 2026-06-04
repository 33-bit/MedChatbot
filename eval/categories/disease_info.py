"""Evaluation policy for disease information questions."""

from __future__ import annotations

from eval.categories import base


CATEGORY = "disease_info"
ANSWER_CHECKS = []
RETRIEVAL_METRICS = ["chunk_recall@5", "context_precision@5"]
JUDGE_METRICS = ["faithful_score", "correctness_score", "relevant_score"]

METRIC_PROFILE = {
    "primary": RETRIEVAL_METRICS + JUDGE_METRICS,
    "hard_checks": ["requires_citation"],
}


def apply_category_checks(case, answer, scored, pass_threshold):
    return base.apply_answer_checks(case, answer, scored, pass_threshold, ANSWER_CHECKS)


def main(argv=None) -> int:
    from eval import core
    return core.main_for_category(CATEGORY, argv)


if __name__ == "__main__":
    raise SystemExit(main())
