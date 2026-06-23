"""Select profile facts relevant to the current question.

The selector is what the pipeline calls to assemble the safe medical-profile
projection. It excludes inactive / refuted / entered-in_error / superseded
facts and ranks confirmed > unconfirmed. Conflicting current rows are
folded into a `conflicting_facts` excluded reason that the answer path
surfaces as a clarification prompt.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from src.chat.context.domain import (
    ClinicalCase,
    ConversationContextBundle,
    SessionState,
)
from src.chat.context.resolver import SubjectResolution
from src.chat.profile.domain import ProfileFact

_SAFETY_FACT_TYPES = {
    "age",
    "sex",
    "allergy",
    "chronic_disease",
    "medication_use",
    "pregnancy_status",
}
_MEDICALLY_RELEVANT_FACT_TYPES = _SAFETY_FACT_TYPES | {
    "diagnosis",
    "symptom_state",
    "symptom_history",
}


@dataclass(frozen=True)
class _RankedFact:
    fact: ProfileFact
    score: float
    reasons: list[str]


def build_context_bundle(
    *,
    analysis: dict,
    state: SessionState,
    resolution: SubjectResolution,
    facts: list[ProfileFact],
    active_case: ClinicalCase | None,
    now: float | None = None,
    fact_limit: int = 12,
    turn_limit: int = 5,
) -> ConversationContextBundle:
    context = analysis.get("context") if isinstance(analysis.get("context"), dict) else {}
    excluded_reason = _excluded_reason(analysis, context, resolution)
    if excluded_reason:
        return ConversationContextBundle(
            subject=None,
            safety_profile=[],
            relevant_facts=[],
            active_case=None,
            reference_turns=[],
            excluded_reason=excluded_reason,
        )

    subject_id = resolution.subject_id
    current_time = time.time() if now is None else now
    entity_refs = {
        (str(ref.get("type") or ref.get("reference_type")), str(ref.get("id") or ref.get("reference_id")))
        for ref in context.get("references", [])
        if isinstance(ref, dict) and (ref.get("id") or ref.get("reference_id"))
    }
    eligible = [
        fact for fact in facts
        if fact.owner_id == state.owner_id
        and fact.subject_id == subject_id
        and fact.superseded_by is None
        and not _expired(fact, current_time)
        and not _is_excluded(fact)
    ]
    eligible, has_conflicts = _remove_conflicts(eligible)
    if has_conflicts:
        return ConversationContextBundle(
            subject={"id": subject_id, "relationship": subject_id},
            safety_profile=[],
            relevant_facts=[],
            active_case=None,
            reference_turns=[],
            excluded_reason="conflicting_facts",
        )

    ranked: list[_RankedFact] = []
    for fact in eligible:
        score = 100.0
        reasons = ["subject_match"]
        if getattr(fact, "verification_status", "unconfirmed") == "confirmed":
            score += 60
            reasons.append("confirmed")
        else:
            score -= 5
            reasons.append("unconfirmed")
        if active_case and fact.subject_id == active_case.subject_id:
            score += 40
            reasons.append("active_case")
        if (fact.entity_type, fact.entity_id) in entity_refs:
            score += 30
            reasons.append("entity_match")
        if fact.fact_type in _MEDICALLY_RELEVANT_FACT_TYPES:
            score += 20
            reasons.append("medically_relevant")
        age_days = max(0.0, current_time - fact.updated_at) / 86400
        if age_days <= 30:
            score += 10
            reasons.append("recent")
        if fact.temporal_status in {"historical", "resolved"}:
            score -= 50
            reasons.append("historical_or_resolved")
        ranked.append(_RankedFact(fact, score, reasons))

    ranked.sort(key=lambda item: (item.score, item.fact.updated_at), reverse=True)
    selected = ranked[:max(0, fact_limit)]
    safety = [item.fact for item in selected if item.fact.fact_type in _SAFETY_FACT_TYPES]
    relevant = [item.fact for item in selected if item.fact.fact_type not in _SAFETY_FACT_TYPES]
    reasons = {item.fact.profile_fact_id: item.reasons for item in selected}
    turns = [
        {
            "turn_id": turn.turn_id,
            "role": turn.role,
            "content": turn.content,
            "created_at": turn.created_at,
        }
        for turn in [
            turn for turn in state.recent_turns
            if turn.subject_id == subject_id
        ][-max(0, turn_limit):]
    ]
    return ConversationContextBundle(
        subject={"id": subject_id, "relationship": subject_id},
        safety_profile=safety,
        relevant_facts=relevant,
        active_case=(
            active_case
            if active_case
            and active_case.subject_id == subject_id
            and active_case.status == "active"
            else None
        ),
        reference_turns=turns,
        selection_reasons=reasons,
    )


def _excluded_reason(
    analysis: dict,
    context: dict,
    resolution: SubjectResolution,
) -> str | None:
    if not analysis.get("analysis_succeeded", True):
        return "analyzer_failure"
    if (
        analysis.get("guardrail", {}).get("verdict") != "allow"
        or context.get("relation") == "off_topic"
        or resolution.ambiguous
    ):
        return "out_of_scope"
    return None


def _expired(fact: ProfileFact, now: float) -> bool:
    if fact.valid_until is not None and fact.valid_until <= now:
        return True
    return fact.fact_type == "symptom_state" and fact.temporal_status == "resolved"


def _is_excluded(fact: ProfileFact) -> bool:
    if getattr(fact, "inactive", False):
        return True
    if getattr(fact, "verification_status", "unconfirmed") in {"refuted", "entered_in_error"}:
        return True
    return False


def _remove_conflicts(facts: list[ProfileFact]) -> tuple[list[ProfileFact], bool]:
    groups: dict[tuple, list[ProfileFact]] = {}
    for fact in facts:
        key = (fact.fact_type, fact.entity_type, fact.entity_id, fact.attribute)
        groups.setdefault(key, []).append(fact)
    result: list[ProfileFact] = []
    has_conflicts = False
    for group in groups.values():
        current_values = {
            json.dumps(fact.value, sort_keys=True, ensure_ascii=False)
            for fact in group
            if fact.temporal_status == "current"
        }
        if len(current_values) > 1:
            has_conflicts = True
            continue
        result.extend(group)
    return result, has_conflicts
