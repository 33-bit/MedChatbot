"""
differential.py
---------------
Diagnostic narrowing via Neo4j KG + symptom catalog.

Flow:
  1. Rank candidate diseases by symptom overlap
  2. Find discriminative symptoms (present in some candidates but not all)
  3. Build clarification batch (3-5 questions at once) using
     the symptom catalog's slot templates (onset/severity/pattern/associated)
  4. Parse user answer back into slot updates via LLM mini
"""

from __future__ import annotations

import json
from pathlib import Path

from src.chat.kg_retriever import _driver
from src.chat.llm_mini import call_mini
from src.config import OUTPUT_DIR

SYMPTOM_DIR = OUTPUT_DIR / "entities" / "symptoms"

BATCH_SIZE = 4
MIN_CANDIDATES_TO_STOP = 2


def _load_symptom_catalog() -> dict[str, dict]:
    """Load symptom JSONs for clarification_questions lookup."""
    catalog = {}
    if not SYMPTOM_DIR.exists():
        return catalog
    for f in SYMPTOM_DIR.glob("*.json"):
        if f.name.startswith("_"):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            sid = data.get("symptom_id")
            if sid:
                catalog[sid] = data
        except Exception:
            continue
    return catalog


_SYMPTOM_CATALOG: dict[str, dict] | None = None


def symptom_catalog() -> dict[str, dict]:
    global _SYMPTOM_CATALOG
    if _SYMPTOM_CATALOG is None:
        _SYMPTOM_CATALOG = _load_symptom_catalog()
    return _SYMPTOM_CATALOG


def rank_candidates(symptom_ids: list[str], limit: int = 10) -> list[dict]:
    """Rank diseases by number of matching symptoms."""
    if not symptom_ids:
        return []
    clean_ids = [s for s in symptom_ids if s and not s.startswith("raw:")]
    if not clean_ids:
        return []
    try:
        with _driver().session() as session:
            rows = session.run(
                "MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom) "
                "WHERE s.id IN $sids "
                "RETURN d.id AS id, d.name_vi AS name, count(s) AS overlap "
                "ORDER BY overlap DESC LIMIT $limit",
                sids=clean_ids, limit=limit,
            ).data()
        return [{"disease_id": r["id"], "name": r["name"], "overlap": r["overlap"]}
                for r in rows]
    except Exception:
        return []


def discriminative_symptoms(
    candidate_disease_ids: list[str],
    known_symptom_ids: list[str],
    limit: int = BATCH_SIZE,
) -> list[str]:
    """
    Find symptoms that appear in some candidates but not all.
    These best split the differential.
    """
    if len(candidate_disease_ids) < 2:
        return []
    try:
        with _driver().session() as session:
            rows = session.run(
                "MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom) "
                "WHERE d.id IN $dids AND NOT s.id IN $known "
                "WITH s.id AS sid, count(DISTINCT d) AS freq "
                "WHERE freq > 0 AND freq < $total "
                "RETURN sid, freq, abs(freq - ($total / 2.0)) AS dist "
                "ORDER BY dist ASC LIMIT $limit",
                dids=candidate_disease_ids,
                known=known_symptom_ids,
                total=len(candidate_disease_ids),
                limit=limit,
            ).data()
        return [r["sid"] for r in rows]
    except Exception:
        return []


def build_clarification(symptom_ids: list[str]) -> str:
    """Build a single user-facing question covering multiple symptoms + slots."""
    catalog = symptom_catalog()
    lines = ["Để thu hẹp chẩn đoán, bạn có thể cho biết thêm:"]
    for i, sid in enumerate(symptom_ids, 1):
        entry = catalog.get(sid, {})
        name = entry.get("name_vi", sid.replace("symptom:S_", "").replace("_", " "))
        cq = entry.get("clarification_questions", {}) or {}

        parts = [f"{i}. Bạn có bị **{name}** không?"]
        for key in ("onset", "severity", "pattern", "associated"):
            q = cq.get(key)
            if q:
                parts.append(f"   - {q}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


PARSE_ANSWER_SYSTEM = """Bạn là trợ lý y tế. Phân tích câu trả lời của người bệnh cho một bộ câu hỏi
làm rõ triệu chứng. Với mỗi triệu chứng được hỏi, trả về trạng thái (có/không/không rõ)
và các slot thông tin nếu có.

Input: JSON {asked_symptoms: [{symptom_id, name}], user_answer: "..."}

Trả về JSON:
{
  "results": [
    {
      "symptom_id": "...",
      "present": "yes" | "no" | "unknown",
      "onset": "...",        // optional
      "severity": "...",     // optional
      "pattern": "...",      // optional
      "associated": "..."    // optional
    }
  ]
}

CHỈ trả JSON."""


def parse_clarification_answer(
    asked_symptoms: list[dict],
    user_answer: str,
) -> list[dict]:
    """Parse free-text answer into per-symptom slot values."""
    payload = {"asked_symptoms": asked_symptoms, "user_answer": user_answer}
    result = call_mini(PARSE_ANSWER_SYSTEM, json.dumps(payload, ensure_ascii=False))
    if not isinstance(result, dict):
        return []
    return result.get("results", []) or []


def should_ask_clarification(candidates: list[dict]) -> bool:
    """Decide if we should narrow further vs just answer."""
    return len(candidates) > MIN_CANDIDATES_TO_STOP
