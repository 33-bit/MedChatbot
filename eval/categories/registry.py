"""Registry for category-specific evaluators."""

from __future__ import annotations

from typing import Any

from eval.categories import base, disease_info, drug_info, symptom_triage


_EVALUATORS = {
    disease_info.CATEGORY: disease_info,
    drug_info.CATEGORY: drug_info,
    symptom_triage.CATEGORY: symptom_triage,
}


def get_category_module(category: str | None):
    return _EVALUATORS.get(category or "", base)


def apply_category_checks(
    case: dict[str, Any],
    answer: str,
    scored: dict[str, Any],
    pass_threshold: float,
) -> dict[str, Any]:
    module = get_category_module(case.get("category"))
    return module.apply_category_checks(case, answer, scored, pass_threshold)


def metric_profile(category: str | None) -> dict[str, list[str]]:
    module = get_category_module(category)
    return dict(getattr(module, "METRIC_PROFILE", base.METRIC_PROFILE))
