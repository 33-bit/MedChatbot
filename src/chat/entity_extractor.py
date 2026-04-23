"""
entity_extractor.py
-------------------
Extract symptoms + drugs (with slot values) from user message using
LLM mini, then canonicalize names to IDs via Neo4j fulltext search.
"""

from __future__ import annotations

from src.chat.kg_retriever import _driver, _fulltext_search
from src.chat.llm_mini import call_mini

EXTRACTION_SYSTEM = """Bạn là trợ lý y tế. Trích xuất thông tin từ lời nói của người bệnh.

Trả về JSON với cấu trúc:
{
  "symptoms": [
    {"name": "tên triệu chứng", "onset": "khi nào/bao lâu", "severity": "mức độ",
     "pattern": "đặc điểm", "associated": "triệu chứng kèm"}
  ],
  "medications": ["tên thuốc 1", "tên thuốc 2"]
}

Quy tắc:
- Nếu thông tin nào không có, bỏ qua key đó (đừng điền null).
- Tên triệu chứng/thuốc để nguyên tiếng Việt như người bệnh nói.
- CHỈ trả JSON, không giải thích."""


def _canonicalize(name: str, entity_type: str) -> str | None:
    """Match free-text name to canonical ID via Neo4j fulltext."""
    idx_map = {"symptom": "symptom_name", "drug": "drug_name"}
    idx = idx_map.get(entity_type)
    if not idx or not name.strip():
        return None
    try:
        with _driver().session() as session:
            results = session.execute_read(_fulltext_search, idx, name, 1)
            if results and results[0]["score"] > 1.0:
                return results[0]["props"].get("id")
    except Exception:
        pass
    return None


def extract_entities(text: str) -> dict:
    """Returns {symptoms: [{symptom_id, name, onset, ...}], medications: [drug_id]}."""
    parsed = call_mini(EXTRACTION_SYSTEM, text) or {}
    if not isinstance(parsed, dict):
        return {"symptoms": [], "medications": []}

    result_symptoms = []
    for s in parsed.get("symptoms", []) or []:
        name = s.get("name", "")
        sid = _canonicalize(name, "symptom")
        entry = {"symptom_id": sid or f"raw:{name}", "name": name}
        for k in ("onset", "severity", "pattern", "associated"):
            if s.get(k):
                entry[k] = s[k]
        result_symptoms.append(entry)

    result_drugs = []
    for d_name in parsed.get("medications", []) or []:
        did = _canonicalize(d_name, "drug")
        if did:
            result_drugs.append(did)

    return {"symptoms": result_symptoms, "medications": result_drugs}
