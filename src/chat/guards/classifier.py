"""
classifier.py
-------------
Classify each user turn to route it correctly:
  - diagnostic          : user describes symptoms, wants help narrowing
  - informational       : factual question about a disease/drug
  - clarification_answer: replying to a previous clarification batch
  - greeting_other      : greetings, thanks, off-topic
"""

from __future__ import annotations

import json

from src.chat.llm.mini import call_mini
from src.chat.prompts import CLASSIFIER_SYSTEM

VALID_LABELS = ("diagnostic", "informational", "clarification_answer", "greeting_other")


def classify(user_message: str, last_bot_message: str = "") -> dict:
    """Returns {label, cacheable}."""
    payload = {"last_bot_message": last_bot_message, "user_message": user_message}
    result = call_mini(CLASSIFIER_SYSTEM, json.dumps(payload, ensure_ascii=False))
    if not isinstance(result, dict):
        return {"label": "informational", "cacheable": False}
    label = result.get("label", "informational")
    if label not in VALID_LABELS:
        label = "informational"
    return {"label": label, "cacheable": bool(result.get("cacheable", False))}
