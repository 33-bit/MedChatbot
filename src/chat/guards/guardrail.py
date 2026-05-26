"""
guardrail.py
------------
Free preflight checks before any expensive LLM call.

Verdicts:
  allow | greeting | trivial | off_topic | injection | abuse
"""

from __future__ import annotations

import re

from src.config import GUARDRAIL_MAX_LEN, GUARDRAIL_MIN_LEN

_GREETING_RE = re.compile(
    r"^(hi|hello|hey|chào|xin chào|alo|ok|thanks?|cảm ơn|hihi|haha|bye|tạm biệt)[\s\.\!\?😀-🙏]*$",
    re.IGNORECASE,
)
_NON_WORD_RE = re.compile(r"[^\w\sÀ-ỹ]", re.UNICODE)

VALID_VERDICTS = (
    "allow",
    "greeting",
    "trivial",
    "off_topic",
    "injection",
    "abuse",
)

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


def regex_check(text: str) -> dict | None:
    """Cheap guardrail check without an LLM call."""
    verdict = _regex_verdict(text)
    if verdict is None:
        return None
    return {"verdict": verdict, "reply": VERDICT_REPLIES.get(verdict, "")}
