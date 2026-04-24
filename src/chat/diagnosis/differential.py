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
import logging
from functools import lru_cache

from neo4j.exceptions import Neo4jError, ServiceUnavailable

from src.chat.clients import get_neo4j
from src.chat.llm.mini import call_mini
from src.chat.prompts import CLARIFICATION_PARSE_SYSTEM
from src.config import (
    CLARIFICATION_BATCH_SIZE,
    MIN_CANDIDATES_TO_STOP,
    OUTPUT_DIR,
)

log = logging.getLogger(__name__)

SYMPTOM_DIR = OUTPUT_DIR / "entities" / "symptoms"


@lru_cache(maxsize=1)
def symptom_catalog() -> dict[str, dict]:
    """Load symptom JSONs for clarification_questions lookup (cached)."""
    catalog: dict[str, dict] = {}
    if not SYMPTOM_DIR.exists():
        return catalog
    for f in SYMPTOM_DIR.glob("*.json"):
        if f.name.startswith("_"):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log.debug("Skipping %s: %s", f.name, e)
            continue
        sid = data.get("symptom_id")
        if sid:
            catalog[sid] = data
    return catalog


def rank_candidates(symptom_ids: list[str], limit: int = 10) -> list[dict]:
    """Rank diseases by number of matching symptoms."""
    if not symptom_ids:
        return []
    clean_ids = [s for s in symptom_ids if s and not s.startswith("raw:")]
    if not clean_ids:
        return []
    try:
        with get_neo4j().session() as session:
            rows = session.run(
                "MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom) "
                "WHERE s.id IN $sids "
                "RETURN d.id AS id, d.name_vi AS name, count(s) AS overlap "
                "ORDER BY overlap DESC LIMIT $limit",
                sids=clean_ids, limit=limit,
            ).data()
    except (Neo4jError, ServiceUnavailable) as e:
        log.warning("rank_candidates failed: %s", e)
        return []
    return [{"disease_id": r["id"], "name": r["name"], "overlap": r["overlap"]}
            for r in rows]


def discriminative_symptoms(
    candidate_disease_ids: list[str],
    known_symptom_ids: list[str],
    limit: int = CLARIFICATION_BATCH_SIZE,
) -> list[str]:
    """Find symptoms that appear in some candidates but not all."""
    if len(candidate_disease_ids) < 2:
        return []
    try:
        with get_neo4j().session() as session:
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
    except (Neo4jError, ServiceUnavailable) as e:
        log.warning("discriminative_symptoms failed: %s", e)
        return []
    return [r["sid"] for r in rows]


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


def parse_clarification_answer(
    asked_symptoms: list[dict],
    user_answer: str,
) -> list[dict]:
    """Parse free-text answer into per-symptom slot values."""
    payload = {"asked_symptoms": asked_symptoms, "user_answer": user_answer}
    result = call_mini(CLARIFICATION_PARSE_SYSTEM, json.dumps(payload, ensure_ascii=False))
    if not isinstance(result, dict):
        return []
    return result.get("results", []) or []


def should_ask_clarification(candidates: list[dict]) -> bool:
    """Decide if we should narrow further vs just answer."""
    return len(candidates) > MIN_CANDIDATES_TO_STOP
