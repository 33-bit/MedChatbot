"""Evaluation policy for disease information questions."""

from __future__ import annotations

import re

from eval.categories import base


CATEGORY = "disease_info"

_OFF_SCOPE_RE = re.compile(
    r"Tôi chỉ hỗ trợ các câu hỏi về sức khỏe, bệnh lý và thuốc",
    re.IGNORECASE,
)
_GENERIC_TRIAGE_RE = re.compile(
    r"(chưa thể chẩn đoán chắc chắn qua chat|"
    r"khi nào cần đi khám ngay|"
    r"nhận định sơ bộ|"
    r"gọi cấp cứu|"
    r"gọi 115)",
    re.IGNORECASE,
)


def _not_off_scope_refusal(case, answer):
    return {
        "type": "disease_not_off_scope_refusal",
        "target": "no off-scope refusal",
        "passed": not _OFF_SCOPE_RE.search(answer or ""),
        "weight": 2.0,
    }


def _no_generic_triage_boilerplate(case, answer):
    question = (case.get("question") or "").casefold()
    allow_triage = any(
        term in question
        for term in (
            "dấu hiệu nguy hiểm",
            "nguy hiểm",
            "đi khám",
            "cấp cứu",
            "điều trị",
            "xử trí",
            "nghiêm trọng",
            "nặng hơn",
            "cảnh báo",
            "chuyển biến xấu",
        )
    )
    return {
        "type": "disease_no_generic_triage_boilerplate",
        "target": "no generic triage sections",
        "passed": allow_triage or not _GENERIC_TRIAGE_RE.search(answer or ""),
        "weight": 2.0,
    }


def _has_disease_source(case, answer):
    return {
        "type": "disease_source_citation",
        "target": "Bạch Mai disease source",
        "passed": "Hướng dẫn chẩn đoán và điều trị" in (answer or ""),
        "weight": 1.5,
    }


ANSWER_CHECKS = [
    _not_off_scope_refusal,
    _no_generic_triage_boilerplate,
    _has_disease_source,
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
