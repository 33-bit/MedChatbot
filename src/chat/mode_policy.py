"""Answer mode policy for a single chatbot turn."""

from __future__ import annotations

from dataclasses import dataclass

VALID_MODES = ("auto", "information", "diagnostic")
VALID_INTENTS = (
    "pure_info",
    "condition_management_info",
    "contextual_drug_info",
    "symptom_triage",
    "care_seeking_advice",
    "emergency",
    "clarification_answer",
    "off_scope",
)

INFO_INTENTS = {
    "pure_info",
    "condition_management_info",
    "contextual_drug_info",
}
DIAGNOSTIC_INTENTS = {"symptom_triage", "care_seeking_advice"}

SUGGEST_INFORMATION_REPLY = (
    "Câu hỏi này phù hợp với chế độ Thông tin hơn. "
    "Bạn muốn trả lời ở chế độ Thông tin không?"
)
OFF_SCOPE_REPLY = "Tôi chỉ hỗ trợ các câu hỏi về sức khỏe, bệnh lý và thuốc."


@dataclass(frozen=True)
class ModeDecision:
    allow: bool
    route_label: str | None = None
    force_answer: bool | None = None
    suggest_mode: str | None = None
    reply: str | None = None
    use_patient_context: bool | None = None


def normalize_mode(mode: str | None) -> str:
    value = str(mode or "auto").strip().casefold()
    return value if value in VALID_MODES else "auto"


def intent_from_label(label: str) -> str:
    if label == "diagnostic":
        return "symptom_triage"
    if label == "clarification_answer":
        return "clarification_answer"
    if label == "greeting_other":
        return "off_scope"
    return "pure_info"


def normalize_intent(intent: str | None, label: str) -> str:
    value = str(intent or "").strip()
    return value if value in VALID_INTENTS else intent_from_label(label)


def mode_label(mode: str) -> str:
    return {
        "auto": "Auto",
        "information": "Thông tin",
        "diagnostic": "Chẩn đoán",
    }.get(normalize_mode(mode), "Auto")


def apply_mode_policy(mode: str, intent: str, active_flow: bool = False) -> ModeDecision:
    mode = normalize_mode(mode)
    intent = normalize_intent(intent, "informational")

    if intent == "off_scope":
        return ModeDecision(False, reply=OFF_SCOPE_REPLY)

    if intent == "emergency":
        return ModeDecision(True, route_label="diagnostic", force_answer=True)

    if intent == "clarification_answer":
        return ModeDecision(True, route_label="clarification_answer")

    if mode == "auto":
        if intent in INFO_INTENTS:
            return ModeDecision(True, route_label="informational")
        if intent == "care_seeking_advice":
            return ModeDecision(True, route_label="diagnostic", force_answer=True)
        return ModeDecision(True, route_label="diagnostic")

    if mode == "information":
        if intent in INFO_INTENTS:
            return ModeDecision(True, route_label="informational")
        if intent in DIAGNOSTIC_INTENTS:
            return ModeDecision(True, route_label="diagnostic", force_answer=True)
        if active_flow:
            return ModeDecision(True, route_label="clarification_answer")
        return ModeDecision(True)

    if intent == "pure_info":
        return ModeDecision(
            False,
            suggest_mode="information",
            reply=SUGGEST_INFORMATION_REPLY,
        )
    if intent in INFO_INTENTS:
        return ModeDecision(
            True,
            route_label="informational",
            use_patient_context=True,
        )
    if intent == "care_seeking_advice":
        return ModeDecision(True, route_label="diagnostic", force_answer=True)
    return ModeDecision(True, route_label="diagnostic")
