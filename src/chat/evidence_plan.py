"""Semantic evidence planning helpers for retrieval and generation."""

from __future__ import annotations

from typing import Any

ANSWER_DOMAINS = {
    "symptom_or_care",
    "disease_info",
    "drug_info",
    "health_insurance_info",
}
SOURCE_TYPES = {"medical", "disease", "drug", "health_insurance"}
SAFETY_MODES = {"factual_info", "patient_action", "emergency_action"}
ANSWER_STYLES = {"direct_yes_no", "exact_list", "short_explanation", "stepwise"}
DRUG_USAGE_SLOTS = {"dose", "route", "duration", "administration"}


def default_evidence_plan(domain: str = "symptom_or_care") -> dict[str, Any]:
    if domain not in ANSWER_DOMAINS:
        domain = "symptom_or_care"
    return {
        "domain": domain,
        "source_type": _default_source_type(domain),
        "entity": None,
        "answer_slot": "general",
        "safety_mode": "factual_info",
        "target_heading_paths": [],
        "required_facts": [],
        "answer_style": "short_explanation",
        "confidence": 0.5,
        "needs_fallback": False,
    }


def normalize_evidence_plan(
    raw: object,
    *,
    fallback_domain: str = "symptom_or_care",
) -> dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    domain = _choice(data.get("domain"), ANSWER_DOMAINS, fallback_domain)
    source_type = _choice(
        data.get("source_type"),
        SOURCE_TYPES,
        _default_source_type(domain),
    )
    safety_mode = _choice(data.get("safety_mode"), SAFETY_MODES, "factual_info")
    answer_style = _choice(
        data.get("answer_style"),
        ANSWER_STYLES,
        "short_explanation",
    )
    confidence = _confidence(data.get("confidence"), default=0.5)
    return {
        "domain": domain,
        "source_type": source_type,
        "entity": _optional_text(data.get("entity")),
        "answer_slot": _optional_text(data.get("answer_slot")) or "general",
        "safety_mode": safety_mode,
        "target_heading_paths": _text_list(data.get("target_heading_paths"), limit=6),
        "required_facts": _text_list(data.get("required_facts"), limit=8),
        "answer_style": answer_style,
        "confidence": confidence,
        "needs_fallback": bool(data.get("needs_fallback")) and confidence < 0.55,
    }


def fallback_domain_for_intent(intent: str, label: str = "") -> str:
    if intent == "health_insurance_info":
        return "health_insurance_info"
    if intent == "contextual_drug_info":
        return "drug_info"
    if label == "diagnostic" or intent in {"symptom_triage", "care_seeking_advice", "emergency"}:
        return "symptom_or_care"
    if intent in {"pure_info", "condition_management_info"}:
        return "disease_info"
    return "symptom_or_care"


def should_run_evidence_planner(plan: dict[str, Any] | None) -> bool:
    return bool(
        isinstance(plan, dict)
        and plan.get("needs_fallback")
        and _confidence(plan.get("confidence"), default=0.0) < 0.55
    )


def plan_answer_domain(
    plan: dict[str, Any] | None,
    fallback: str = "symptom_or_care",
) -> str:
    if not isinstance(plan, dict):
        return fallback
    return _choice(plan.get("domain"), ANSWER_DOMAINS, fallback)


def plan_source_type(plan: dict[str, Any] | None) -> str:
    if not isinstance(plan, dict):
        return ""
    return _choice(plan.get("source_type"), SOURCE_TYPES, "")


def plan_requires_drug_usage_detail(plan: dict[str, Any] | None) -> bool:
    if not isinstance(plan, dict):
        return False
    return str(plan.get("answer_slot") or "").strip().casefold() in DRUG_USAGE_SLOTS


def plan_targets(plan: dict[str, Any] | None) -> list[str]:
    if not isinstance(plan, dict):
        return []
    return _text_list(plan.get("target_heading_paths"), limit=6)


def plan_required_facts(plan: dict[str, Any] | None) -> list[str]:
    if not isinstance(plan, dict):
        return []
    return _text_list(plan.get("required_facts"), limit=8)


def compact_key(value: object) -> str:
    text = str(value or "").casefold().replace("đ", "d")
    return " ".join(text.split())


def structured_text_match(left: object, right: object) -> bool:
    left_key = compact_key(left)
    right_key = compact_key(right)
    if not left_key or not right_key:
        return False
    return left_key == right_key or left_key in right_key or right_key in left_key


def _default_source_type(domain: str) -> str:
    if domain == "drug_info":
        return "drug"
    if domain == "disease_info":
        return "disease"
    if domain == "health_insurance_info":
        return "health_insurance"
    return "medical"


def _choice(value: object, allowed: set[str], default: str) -> str:
    text = str(value or "").strip()
    return text if text in allowed else default


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _text_list(value: object, *, limit: int) -> list[str]:
    items = value if isinstance(value, list) else []
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        key = compact_key(text)
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _confidence(value: object, *, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default
