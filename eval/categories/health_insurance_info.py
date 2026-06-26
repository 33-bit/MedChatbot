"""Evaluation policy for health-insurance information questions."""

from __future__ import annotations

import re

from eval.categories import base


CATEGORY = "health_insurance_info"


_OFF_SCOPE_RE = re.compile(r"Tôi chỉ hỗ trợ các câu hỏi về sức khỏe, bệnh lý và thuốc", re.IGNORECASE)


def _not_off_scope_refusal(case, answer):
    return {
        "type": "health_insurance_not_off_scope_refusal",
        "target": "no off-scope refusal",
        "passed": not _OFF_SCOPE_RE.search(answer or ""),
        "weight": 2.0,
    }


def _has_health_insurance_source(case, answer):
    return {
        "type": "health_insurance_source_citation",
        "target": "Luật Bảo hiểm y tế source",
        "passed": "Luật Bảo hiểm y tế" in (answer or ""),
        "weight": 2.0,
    }


def _no_medical_corpus_source(case, answer):
    text = answer or ""
    return {
        "type": "health_insurance_no_medical_corpus_source",
        "target": "no disease/drug source labels",
        "passed": (
            "Hướng dẫn chẩn đoán và điều trị" not in text
            and "Dược thư Quốc gia" not in text
        ),
        "weight": 2.0,
    }


ANSWER_CHECKS = [
    _not_off_scope_refusal,
    _has_health_insurance_source,
    _no_medical_corpus_source,
]
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
