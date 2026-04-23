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

from src.chat.llm_mini import call_mini

CLASSIFY_SYSTEM = """Phân loại lời nhắn của người dùng thành 1 trong 4 nhãn:
- "diagnostic": mô tả triệu chứng, muốn được chẩn đoán/tư vấn
- "informational": hỏi thông tin về một bệnh/thuốc cụ thể (không kèm triệu chứng cá nhân)
- "clarification_answer": đang trả lời câu hỏi làm rõ của bot (có đánh số, yes/no, mô tả mức độ...)
- "greeting_other": chào hỏi, cảm ơn, lạc đề

Input: {last_bot_message: str, user_message: str}

Trả về JSON: {"label": "...", "cacheable": true/false}
"cacheable" chỉ true với informational câu hỏi chung chung (không có thông tin cá nhân).

CHỈ trả JSON."""


def classify(user_message: str, last_bot_message: str = "") -> dict:
    """Returns {label, cacheable}."""
    import json
    payload = {"last_bot_message": last_bot_message, "user_message": user_message}
    result = call_mini(CLASSIFY_SYSTEM, json.dumps(payload, ensure_ascii=False))
    if not isinstance(result, dict):
        return {"label": "informational", "cacheable": False}
    label = result.get("label", "informational")
    if label not in ("diagnostic", "informational", "clarification_answer", "greeting_other"):
        label = "informational"
    return {"label": label, "cacheable": bool(result.get("cacheable", False))}
