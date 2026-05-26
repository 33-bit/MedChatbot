"""
mini.py
-------
Light-weight wrapper around the fast OpenAI model for:
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
import logging
import re
import time

from src.chat.clients import get_openai
from src.chat.timing import elapsed_ms
from src.config import FAST_MODEL, FAST_MODEL_MAX_TOKENS

log = logging.getLogger(__name__)

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


def _get_field(obj, key: str):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def message_text(message) -> str:
    """Extract assistant text from an OpenAI Chat Completions response."""
    choices = _get_field(message, "choices") or []
    if not choices:
        return ""
    first = choices[0]
    msg = _get_field(first, "message") or {}
    content = _get_field(msg, "content")
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content or []:
        if isinstance(block, dict):
            if block.get("type") in {"text", "output_text"} and block.get("text"):
                parts.append(block["text"])
            continue
        if getattr(block, "type", None) in {"text", "output_text"} and getattr(block, "text", None):
            parts.append(block.text)
    return "".join(parts)


def call_mini(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    stage: str = "mini",
) -> dict | list | None:
    resolved_model = model or FAST_MODEL
    resolved_max_tokens = max_tokens or FAST_MODEL_MAX_TOKENS
    start = time.perf_counter()
    try:
        response = get_openai().chat.completions.create(
            model=resolved_model,
            max_tokens=resolved_max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
    except Exception as e:
        log.warning("Mini LLM call failed: %s", e)
        return None
    finally:
        log.info("llm timing stage=%s model=%s ms=%.1f",
                 stage, resolved_model, elapsed_ms(start))
    return parse_json(message_text(response))
