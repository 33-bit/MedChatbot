"""Evaluation policy for drug information questions."""

from __future__ import annotations

import re

from eval.categories import base


CATEGORY = "drug_info"

_OFF_SCOPE_RE = re.compile(
    r"Tôi chỉ hỗ trợ các câu hỏi về sức khỏe, bệnh lý và thuốc",
    re.IGNORECASE,
)
_GENERIC_EMERGENCY_RE = re.compile(
    r"(gọi 115|gọi cấp cứu|cấp cứu ngay|sốt cao\s*[>≥]\s*39[,.]5)",
    re.IGNORECASE,
)
_DOSE_OR_ROUTE_RE = re.compile(
    r"(liều|liều dùng|cách dùng|đường dùng|uống|bôi|tiêm|xịt|nhỏ|mg|ml|%|lần/ngày)",
    re.IGNORECASE,
)


def _not_off_scope_refusal(case, answer):
    return {
        "type": "drug_not_off_scope_refusal",
        "target": "no off-scope refusal",
        "passed": not _OFF_SCOPE_RE.search(answer or ""),
        "weight": 2.0,
    }


def _no_generic_emergency_boilerplate(case, answer):
    return {
        "type": "drug_no_generic_emergency_boilerplate",
        "target": "no generic emergency warning",
        "passed": not _GENERIC_EMERGENCY_RE.search(answer or ""),
        "weight": 2.0,
    }


def _has_drug_source(case, answer):
    return {
        "type": "drug_source_citation",
        "target": "Dược thư Quốc gia drug source",
        "passed": "Dược thư Quốc gia" in (answer or ""),
        "weight": 2.0,
    }


def _no_disease_only_source_for_dose_or_route(case, answer):
    asks_dose_or_route = bool(_DOSE_OR_ROUTE_RE.search(case.get("question") or ""))
    text = answer or ""
    has_drug_source = "Dược thư Quốc gia" in text
    has_disease_source = "Hướng dẫn chẩn đoán và điều trị" in text
    return {
        "type": "drug_no_disease_only_source_for_dose_or_route",
        "target": "dose/route answers cite drug source",
        "passed": not asks_dose_or_route or has_drug_source or not has_disease_source,
        "weight": 2.0,
    }


ANSWER_CHECKS = [
    _not_off_scope_refusal,
    _no_generic_emergency_boilerplate,
    _has_drug_source,
    _no_disease_only_source_for_dose_or_route,
]
RETRIEVAL_METRICS = ["chunk_recall@5", "context_precision@5"]
JUDGE_METRICS = ["faithful_score", "correctness_score", "relevant_score"]

METRIC_PROFILE = {
    "primary": RETRIEVAL_METRICS + JUDGE_METRICS,
    "hard_checks": ["requires_citation", "faithfulness_score == 1.0"],
}


def apply_category_checks(case, answer, scored, pass_threshold):
    return base.apply_answer_checks(case, answer, scored, pass_threshold, ANSWER_CHECKS)


def main(argv=None) -> int:
    from eval import core
    return core.main_for_category(CATEGORY, argv)


if __name__ == "__main__":
    raise SystemExit(main())
