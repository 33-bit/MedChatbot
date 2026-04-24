"""
rewriter.py
-----------
Rewrite follow-up queries using conversation history.
If the LLM is not confident, returns a clarification prompt instead.
"""

from __future__ import annotations

import json

from src.chat.llm.mini import call_mini
from src.chat.prompts import REWRITER_SYSTEM


def rewrite(question: str, history: list[dict]) -> dict:
    """Returns {rewritten, confident, clarification}."""
    if not history:
        return {"rewritten": question, "confident": True, "clarification": ""}

    payload = {"history": history[-10:], "question": question}
    result = call_mini(REWRITER_SYSTEM, json.dumps(payload, ensure_ascii=False))
    if not isinstance(result, dict):
        return {"rewritten": question, "confident": True, "clarification": ""}
    return {
        "rewritten": result.get("rewritten", question),
        "confident": bool(result.get("confident", True)),
        "clarification": result.get("clarification", ""),
    }
