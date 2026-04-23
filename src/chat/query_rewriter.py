"""
query_rewriter.py
-----------------
Rewrite follow-up queries using conversation history.
If the LLM is not confident, returns a clarification prompt instead.
"""

from __future__ import annotations

from src.chat.llm_mini import call_mini

REWRITE_SYSTEM = """Bạn là trợ lý y tế. Viết lại câu hỏi của người bệnh thành câu hỏi độc lập,
đầy đủ ngữ cảnh để có thể tra cứu tài liệu mà không cần lịch sử hội thoại.

Input: lịch sử hội thoại (JSON array) + câu hỏi mới nhất.

Trả về JSON:
{
  "rewritten": "câu hỏi đầy đủ ngữ cảnh",
  "confident": true/false,
  "clarification": "câu hỏi làm rõ (chỉ khi confident=false)"
}

Quy tắc:
- Nếu câu hỏi đã rõ ràng (không cần ngữ cảnh), "rewritten" = câu hỏi gốc, "confident" = true.
- Nếu có đại từ tham chiếu (nó, thuốc này, bệnh đó) → thay bằng tên cụ thể từ lịch sử.
- Nếu quá mơ hồ, không thể viết lại → "confident" = false và đặt "clarification".
- CHỈ trả JSON."""


def rewrite(question: str, history: list[dict]) -> dict:
    """Returns {rewritten, confident, clarification}."""
    if not history:
        return {"rewritten": question, "confident": True, "clarification": ""}

    payload = {"history": history[-10:], "question": question}
    import json
    result = call_mini(REWRITE_SYSTEM, json.dumps(payload, ensure_ascii=False))
    if not isinstance(result, dict):
        return {"rewritten": question, "confident": True, "clarification": ""}
    return {
        "rewritten": result.get("rewritten", question),
        "confident": bool(result.get("confident", True)),
        "clarification": result.get("clarification", ""),
    }
