"""
analyzer.py
-----------
One-shot input analysis for guardrail, routing, and entity extraction.
"""

from __future__ import annotations

import json

from src.chat.guards.guardrail import VALID_VERDICTS
from src.chat.llm.mini import call_mini
from src.chat.mode_policy import normalize_intent
from src.chat.prompts import TURN_ANALYSIS_SYSTEM
from src.config import GUARDRAIL_MAX_TOKENS, GUARDRAIL_MODEL

VALID_LABELS = ("diagnostic", "informational", "clarification_answer", "greeting_other")
ANALYZABLE_LABELS = {"diagnostic", "informational", "clarification_answer"}


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
) -> dict:
    """Return combined guardrail, turn classification, and raw entity extraction."""
    payload = {
        "history": (history or [])[-10:],
        "last_bot_message": last_bot_message,
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

    return {
        "guardrail": {"verdict": verdict, "reason": guardrail.get("reason", "")},
        "turn": {
            "label": label,
            "intent": intent,
            "direct_answer_requested": _bool_field(turn, "direct_answer_requested"),
        },
        "rewrite": {
            "rewritten": rewritten,
            "confident": _bool_field(rewrite, "confident", default=True),
            "clarification": clarification,
        },
        "entities": {"symptoms": symptoms, "medications": medications},
    }


def _fallback(user_message: str) -> dict:
    return {
        "guardrail": {"verdict": "allow", "reason": ""},
        "turn": {
            "label": "informational",
            "intent": "pure_info",
            "direct_answer_requested": False,
        },
        "rewrite": {"rewritten": user_message, "confident": True, "clarification": ""},
        "entities": {"symptoms": [], "medications": []},
    }
