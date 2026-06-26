"""
analyzer.py
-----------
One-shot input analysis for guardrail, routing, and entity extraction.
"""

from __future__ import annotations

import json

from src.chat.evidence_plan import fallback_domain_for_intent, normalize_evidence_plan
from src.chat.guards.guardrail import VALID_VERDICTS
from src.chat.llm.mini import call_mini
from src.chat.mode_policy import normalize_intent
from src.chat.prompts import TURN_ANALYSIS_SYSTEM
from src.config import GUARDRAIL_MAX_TOKENS, GUARDRAIL_MODEL

VALID_LABELS = ("diagnostic", "informational", "clarification_answer", "greeting_other")
ANALYZABLE_LABELS = {"diagnostic", "informational", "clarification_answer"}
VALID_URGENCY_LEVELS = {"routine", "urgent", "emergency"}
VALID_CONTEXT_RELATIONS = {
    "continue",
    "switch_subject",
    "resume_subject",
    "new_entity",
    "off_topic",
    "uncertain",
}
PERSONALIZED_INTENTS = {
    "condition_management_info",
    "contextual_drug_info",
    "symptom_triage",
    "care_seeking_advice",
    "emergency",
    "clarification_answer",
}


def _dict_field(data: dict, key: str) -> dict:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def _list_field(data: dict, key: str) -> list:
    value = data.get(key)
    return value if isinstance(value, list) else []


def _bool_field(data: dict, key: str, default: bool = False) -> bool:
    value = data.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def analyze_turn(
    user_message: str,
    last_bot_message: str = "",
    history: list[dict] | None = None,
    session_context: dict | None = None,
) -> dict:
    """Return combined guardrail, turn classification, and raw entity extraction."""
    payload = {
        "history": (history or [])[-10:],
        "last_bot_message": last_bot_message,
        "session_context": session_context or {},
        "user_message": user_message,
    }
    result = call_mini(
        TURN_ANALYSIS_SYSTEM,
        json.dumps(payload, ensure_ascii=False),
        model=GUARDRAIL_MODEL,
        max_tokens=GUARDRAIL_MAX_TOKENS,
        stage="turn_analysis",
    )
    if not isinstance(result, dict):
        return _fallback(user_message)

    guardrail = _dict_field(result, "guardrail")
    verdict = guardrail.get("verdict", "allow")
    if verdict not in VALID_VERDICTS:
        verdict = "allow"

    turn = _dict_field(result, "turn")
    label = turn.get("label", "informational")
    if label not in VALID_LABELS:
        label = "informational"
    intent = normalize_intent(turn.get("intent"), label)

    triage = _dict_field(result, "triage")
    urgency = str(triage.get("urgency") or "routine").strip().casefold()
    if urgency not in VALID_URGENCY_LEVELS:
        urgency = "routine"
    red_flags = [
        str(flag).strip()
        for flag in _list_field(triage, "red_flags")
        if str(flag).strip()
    ]

    entities = _dict_field(result, "entities")
    symptoms = _list_field(entities, "symptoms")
    medications = _list_field(entities, "medications")

    rewrite = _dict_field(result, "rewrite")
    rewritten = rewrite.get("rewritten") or user_message
    clarification = rewrite.get("clarification") or ""

    if verdict != "allow" or label not in ANALYZABLE_LABELS:
        symptoms = []
        medications = []
        rewritten = user_message
        clarification = ""
        intent = normalize_intent(None, label)
        urgency = "routine"
        red_flags = []
    elif intent == "emergency" or urgency == "emergency":
        urgency = "emergency"
        intent = "emergency"

    context = _normalize_context(_dict_field(result, "context"))
    if _requires_medical_profile(intent, context):
        context["needs_medical_profile"] = True
    profile_candidates = _normalize_profile_candidates(
        _list_field(result, "profile_candidates")
    )
    if verdict != "allow" or label not in ANALYZABLE_LABELS:
        context = _empty_context(relation="off_topic" if verdict == "off_topic" else "uncertain")
        profile_candidates = []
    evidence_plan = normalize_evidence_plan(
        _dict_field(result, "evidence_plan"),
        fallback_domain=fallback_domain_for_intent(intent, label),
    )
    if verdict != "allow" or label not in ANALYZABLE_LABELS:
        evidence_plan = normalize_evidence_plan(
            None,
            fallback_domain=fallback_domain_for_intent(intent, label),
        )

    return {
        "analysis_succeeded": True,
        "guardrail": {"verdict": verdict, "reason": guardrail.get("reason", "")},
        "turn": {
            "label": label,
            "intent": intent,
            "direct_answer_requested": _bool_field(turn, "direct_answer_requested"),
        },
        "triage": {
            "urgency": urgency,
            "red_flags": red_flags,
            "reason": str(triage.get("reason") or ""),
        },
        "rewrite": {
            "rewritten": rewritten,
            "confident": _bool_field(rewrite, "confident", default=True),
            "clarification": clarification,
        },
        "entities": {"symptoms": symptoms, "medications": medications},
        "context": context,
        "evidence_plan": evidence_plan,
        "profile_candidates": profile_candidates,
    }


def _fallback(user_message: str) -> dict:
    return {
        "analysis_succeeded": False,
        "guardrail": {"verdict": "allow", "reason": ""},
        "turn": {
            "label": "informational",
            "intent": "pure_info",
            "direct_answer_requested": False,
        },
        "triage": {"urgency": "routine", "red_flags": [], "reason": ""},
        "rewrite": {"rewritten": user_message, "confident": True, "clarification": ""},
        "entities": {"symptoms": [], "medications": []},
        "context": _empty_context(),
        "evidence_plan": normalize_evidence_plan(
            None,
            fallback_domain="symptom_or_care",
        ),
        "profile_candidates": [],
    }


def _normalize_context(context: dict) -> dict:
    subject = _dict_field(context, "subject")
    subject_id = subject.get("id")
    relation = context.get("relation", "uncertain")
    if relation not in VALID_CONTEXT_RELATIONS:
        relation = "uncertain"
    references = []
    for raw in _list_field(context, "references"):
        if not isinstance(raw, dict):
            continue
        reference_type = raw.get("type") or raw.get("reference_type")
        reference_id = raw.get("id") or raw.get("reference_id")
        if not reference_type or not reference_id:
            continue
        references.append({
            "type": str(reference_type),
            "id": str(reference_id),
            "source": str(raw.get("source") or "inferred"),
            "confidence": _confidence(raw.get("confidence")),
        })
    return {
        "subject": {
            "id": str(subject_id) if subject_id else None,
            "relationship": str(subject.get("relationship") or ""),
            "display_name": str(subject.get("display_name") or ""),
            "source": str(subject.get("source") or ""),
            "confidence": _confidence(subject.get("confidence")),
        },
        "references": references,
        "relation": relation,
        "needs_medical_profile": _bool_field(context, "needs_medical_profile"),
        "ambiguous": _bool_field(context, "ambiguous"),
        "clarification": str(context.get("clarification") or ""),
        "active_subject_confidence": _confidence(context.get("active_subject_confidence")),
    }


def _empty_context(relation: str = "uncertain") -> dict:
    return {
        "subject": {
            "id": None,
            "relationship": "",
            "display_name": "",
            "source": "",
            "confidence": 0.0,
        },
        "references": [],
        "relation": relation,
        "needs_medical_profile": False,
        "ambiguous": False,
        "clarification": "",
        "active_subject_confidence": 0.0,
    }


def _confidence(value: object) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _normalize_profile_candidates(candidates: list) -> list[dict]:
    normalized: list[dict] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        value = candidate.get("value")
        if value is None:
            normalized.append(dict(candidate))
            continue
        item = dict(candidate)
        if not isinstance(value, dict):
            item["value"] = {"value": value}
        normalized.append(item)
    return normalized


def _requires_medical_profile(intent: str, context: dict) -> bool:
    subject = context.get("subject")
    return bool(
        intent in PERSONALIZED_INTENTS
        and isinstance(subject, dict)
        and subject.get("id")
        and _confidence(subject.get("confidence")) >= 0.8
        and not context.get("ambiguous")
    )
