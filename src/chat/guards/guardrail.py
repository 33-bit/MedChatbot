"""
guardrail.py
------------
First line of defense before any expensive LLM call.

Two tiers:
  0. Regex prefilter (free)       — trivial/greeting/abuse patterns
  1. Cheap LLM check (grok-3-mini)— spam, off-topic, gibberish, prompt injection

Verdicts:
  allow | greeting | trivial | off_topic | injection | abuse
"""

from __future__ import annotations

import logging
import re

from xai_sdk.chat import system, user

from src.chat.clients import get_xai
from src.chat.llm.mini import parse_json
from src.chat.prompts import GUARDRAIL_SYSTEM
from src.config import GUARDRAIL_MAX_LEN, GUARDRAIL_MIN_LEN, GUARDRAIL_MODEL

log = logging.getLogger(__name__)

_GREETING_RE = re.compile(
    r"^(hi|hello|hey|chào|xin chào|alo|ok|thanks?|cảm ơn|hihi|haha|bye|tạm biệt)[\s\.\!\?😀-🙏]*$",
    re.IGNORECASE,
)
_NON_WORD_RE = re.compile(r"[^\w\sÀ-ỹ]", re.UNICODE)

VALID_VERDICTS = ("allow", "greeting", "off_topic", "injection", "abuse")

VERDICT_REPLIES = {
    "greeting": "Xin chào! Tôi là trợ lý y tế. Bạn có thể mô tả triệu chứng hoặc hỏi về bệnh/thuốc.",
    "trivial": "Bạn hãy đặt câu hỏi cụ thể hơn nhé.",
    "off_topic": "Tôi chỉ hỗ trợ các câu hỏi về sức khỏe, bệnh lý và thuốc.",
    "injection": "Tôi không thể thực hiện yêu cầu này. Vui lòng hỏi về vấn đề y tế.",
    "abuse": "",
}


def _regex_verdict(text: str) -> str | None:
    t = text.strip()
    if len(t) > GUARDRAIL_MAX_LEN:
        return "abuse"
    if _GREETING_RE.match(t):
        return "greeting"
    if len(t) < GUARDRAIL_MIN_LEN:
        return "trivial"
    cleaned = _NON_WORD_RE.sub("", t)
    if len(t) >= 5 and len(cleaned) / len(t) < 0.3:
        return "trivial"
    return None


def _llm_verdict(text: str) -> dict:
    """Ask the cheap guardrail model. Fail open (verdict=allow) on error."""
    try:
        chat = get_xai().chat.create(model=GUARDRAIL_MODEL)
        chat.append(system(GUARDRAIL_SYSTEM))
        chat.append(user(text[:GUARDRAIL_MAX_LEN]))
        response = chat.sample()
    except Exception as e:
        # xAI SDK doesn't expose a single exception base; we log and fail open.
        log.warning("Guardrail LLM failed, failing open: %s", e)
        return {"verdict": "allow", "reason": ""}

    data = parse_json(response.content or "")
    if not isinstance(data, dict):
        return {"verdict": "allow", "reason": ""}
    verdict = data.get("verdict", "allow")
    if verdict not in VALID_VERDICTS:
        verdict = "allow"
    return {"verdict": verdict, "reason": data.get("reason", "")}


def check(text: str) -> dict:
    """
    Returns {verdict, reply}.
      - reply == "" with verdict=allow → pass through to pipeline
      - otherwise reply is the canned response to send directly
    """
    rv = _regex_verdict(text)
    if rv is not None:
        return {"verdict": rv, "reply": VERDICT_REPLIES.get(rv, "")}

    result = _llm_verdict(text)
    verdict = result["verdict"]
    if verdict == "allow":
        return {"verdict": "allow", "reply": ""}
    return {"verdict": verdict, "reply": VERDICT_REPLIES.get(verdict, "")}
