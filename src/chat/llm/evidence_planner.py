"""Fallback mini-LLM evidence planner.

The primary turn analyzer normally produces the evidence plan. This module is
only used when that analyzer explicitly marks the plan as uncertain.
"""

from __future__ import annotations

import json
from typing import Any

from src.chat.evidence_plan import normalize_evidence_plan
from src.chat.llm.mini import call_mini
from src.config import FAST_MODEL, FAST_MODEL_MAX_TOKENS

EVIDENCE_PLANNER_SYSTEM = """Bạn là bộ lập kế hoạch bằng chứng cho chatbot y tế.

Nhiệm vụ: đọc câu hỏi và phân tích lượt hội thoại đã có, rồi chọn loại tài liệu
và loại thông tin cần truy xuất/trả lời. Không trả lời người dùng.

Trả về JSON đúng cấu trúc:
{
  "evidence_plan": {
    "domain": "symptom_or_care | disease_info | drug_info | health_insurance_info",
    "source_type": "medical | disease | drug | health_insurance",
    "entity": "tên bệnh/thuốc/chủ đề luật chuẩn hóa hoặc null",
    "answer_slot": "definition | symptoms | cause | transmission | diagnosis | treatment | prognosis | dose | route | duration | administration | contraindication | adverse_effect | interaction | pregnancy | pediatric | insurance_rule | first_aid | general",
    "safety_mode": "factual_info | patient_action | emergency_action",
    "target_heading_paths": ["các tiêu đề/nhóm nội dung tài liệu có khả năng cần"],
    "required_facts": ["các ý bắt buộc cần được kiểm tra trong tài liệu"],
    "answer_style": "direct_yes_no | exact_list | short_explanation | stepwise",
    "confidence": 0.0,
    "needs_fallback": false
  }
}

Quy tắc:
- Lập kế hoạch theo ý nghĩa câu hỏi, không dựa vào danh sách từ khóa.
- Câu hỏi về liều, cách dùng, đường dùng hoặc thời gian dùng thuốc dùng domain="drug_info", source_type="drug".
- Câu hỏi về triệu chứng, nguyên nhân, lây truyền, chẩn đoán, biến chứng hoặc điều trị bệnh dùng domain="disease_info", source_type="disease", trừ khi người dùng đang mô tả ca hiện tại cần phân luồng.
- Câu hỏi bảo hiểm y tế dùng domain="health_insurance_info", source_type="health_insurance".
- safety_mode="emergency_action" chỉ khi có người đang trong tình huống cần hành động cấp cứu ngay.
- Nếu không chắc, vẫn chọn kế hoạch tốt nhất và đặt confidence thấp; không đặt needs_fallback=true trong fallback planner.
- CHỈ trả JSON, không markdown, không giải thích."""


def plan_evidence(
    question: str,
    *,
    analysis: dict[str, Any] | None = None,
    fallback_domain: str = "symptom_or_care",
) -> dict[str, Any]:
    payload = {
        "question": question,
        "analysis": analysis or {},
    }
    result = call_mini(
        EVIDENCE_PLANNER_SYSTEM,
        json.dumps(payload, ensure_ascii=False),
        model=FAST_MODEL,
        max_tokens=FAST_MODEL_MAX_TOKENS,
        stage="evidence_plan_fallback",
    )
    if isinstance(result, dict) and isinstance(result.get("evidence_plan"), dict):
        raw = result["evidence_plan"]
    else:
        raw = result
    plan = normalize_evidence_plan(raw, fallback_domain=fallback_domain)
    plan["needs_fallback"] = False
    return plan
