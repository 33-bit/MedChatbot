"""LLM evidence curation for factual disease/drug answers."""

from __future__ import annotations

import json
from typing import Any

from src.chat.llm.mini import call_mini
from src.chat.retrieval.types import Hit
from src.config import FAST_MODEL_MAX_TOKENS

_SUPPORTED_DOMAINS = {"disease_info", "drug_info"}

EVIDENCE_BRIEF_SYSTEM = """
Bạn là bộ chọn bằng chứng cho chatbot y tế.
Đọc câu hỏi và các đoạn tài liệu đã truy xuất, rồi chọn TỐI THIỂU các đoạn
trực tiếp cần để trả lời đúng câu hỏi.

Quy tắc:
- Không trả lời người dùng; chỉ lập brief bằng chứng.
- Ưu tiên đoạn tổng quan/định nghĩa nếu đoạn đó đã trả lời trực tiếp câu hỏi.
- Không chọn các đoạn chẩn đoán, điều trị, biến chứng, tiên lượng hoặc cảnh báo
  nếu người dùng không hỏi trực tiếp các phần đó.
- Với câu hỏi bệnh học tổng quan, giữ các ý định nghĩa, bản chất bệnh, nhóm hay
  gặp và ảnh hưởng chính nếu tài liệu nêu rõ.
- Với câu hỏi thuốc, chỉ chọn đúng phần thông tin được hỏi như chỉ định, liều,
  cách dùng, chống chỉ định, thận trọng hoặc tác dụng không mong muốn.
- Nếu evidence_plan.answer_slot là dose, route, duration hoặc administration,
  phải chọn các đoạn "Liều lượng và cách dùng", "Liều dùng", "Cách dùng" hoặc
  tương đương khi có trong danh sách tài liệu; không được kết luận thiếu liều/cách
  dùng nếu một đoạn được cung cấp có số liệu liều, tần suất, thời gian hoặc đường dùng.
- Nếu nhiều đoạn cùng liên quan, chọn số đoạn ít nhất vẫn đủ bao phủ câu hỏi.
- Đoạn [1] là đoạn được truy xuất xếp hạng cao nhất. Nếu đoạn này cùng thực
  thể và trả lời trực tiếp câu hỏi, phải chọn nó và đưa các ý cốt lõi của nó
  vào must_include_facts.
- Sắp xếp selected_indexes theo thứ tự xuất hiện trong danh sách tài liệu.

Trả về JSON hợp lệ:
{
  "selected_indexes": [1, 2],
  "must_include_facts": ["..."],
  "avoid_topics": ["..."],
  "brief": "..."
}
""".strip()


def build_evidence_brief(
    *,
    question: str,
    hits: list[Hit],
    evidence_plan: dict | None,
    answer_domain: str,
) -> dict[str, Any] | None:
    if answer_domain not in _SUPPORTED_DOMAINS or len(hits) < 2:
        return None
    candidates = "\n\n---\n\n".join(
        _format_candidate(index, hit)
        for index, hit in enumerate(hits, start=1)
    )
    user_prompt = "\n\n".join(
        (
            f"Câu hỏi:\n{question}",
            "Kế hoạch bằng chứng hiện có:\n"
            + json.dumps(evidence_plan or {}, ensure_ascii=False, indent=2),
            f"Các đoạn tài liệu:\n{candidates}",
        )
    )
    result = call_mini(
        EVIDENCE_BRIEF_SYSTEM,
        user_prompt,
        max_tokens=max(FAST_MODEL_MAX_TOKENS, 1536),
        stage="evidence_brief",
    )
    if isinstance(result, dict) and isinstance(result.get("evidence_brief"), dict):
        result = result["evidence_brief"]
    if not isinstance(result, dict):
        return None

    selected = _int_list(result.get("selected_indexes"), max_value=len(hits))
    if not selected:
        return None
    return {
        "selected_indexes": selected,
        "must_include_facts": _text_list(result.get("must_include_facts"), limit=10),
        "avoid_topics": _text_list(result.get("avoid_topics"), limit=8),
        "brief": str(result.get("brief") or "").strip(),
    }


def _format_candidate(index: int, hit: Hit) -> str:
    heading = f" — {hit.heading_path}" if hit.heading_path else ""
    text = (hit.text or "").strip()
    if len(text) > 2200:
        text = text[:2200].rstrip() + "..."
    return f"[{index}] {hit.source_name}{heading}\n{text}"


def _int_list(value: object, *, max_value: int) -> list[int]:
    raw_items = value if isinstance(value, list) else []
    result: list[int] = []
    seen: set[int] = set()
    for item in raw_items:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number < 1 or number > max_value or number in seen:
            continue
        seen.add(number)
        result.append(number)
    return sorted(result)


def _text_list(value: object, *, limit: int) -> list[str]:
    raw_items = value if isinstance(value, list) else []
    result: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = str(item or "").strip()
        if not text:
            continue
        key = " ".join(text.casefold().split())
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result
