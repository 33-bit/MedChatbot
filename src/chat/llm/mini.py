"""
mini.py
-------
Light-weight wrapper around the fast xAI model for:
  - query rewriting
  - entity extraction
  - turn classification
  - clarification answer parsing
  - guardrail classification

Always expects JSON output. `parse_json` is exported for other modules
(e.g. guardrail) that call the LLM directly.
"""

from __future__ import annotations

import json
import re

from xai_sdk.chat import system, user

from src.chat.clients import get_xai
from src.config import FAST_MODEL

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json(text: str):
    """Best-effort JSON extraction from an LLM response (handles code fences)."""
    text = _JSON_FENCE.sub("", text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for open_c, close_c in [("{", "}"), ("[", "]")]:
            start = text.find(open_c)
            end = text.rfind(close_c)
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
    return None


def call_mini(system_prompt: str, user_prompt: str) -> dict | list | None:
    chat = get_xai().chat.create(model=FAST_MODEL)
    chat.append(system(system_prompt))
    chat.append(user(user_prompt))
    response = chat.sample()
    return parse_json(response.content or "")
