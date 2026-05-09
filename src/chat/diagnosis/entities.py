"""
entities.py
-----------
Extract symptoms + drugs (with slot values) from user message using
LLM mini, then canonicalize names to IDs via Neo4j fulltext search.
"""

from __future__ import annotations

from neo4j.exceptions import Neo4jError, ServiceUnavailable

from src.chat.clients import get_neo4j
from src.chat.errors import Neo4jUnavailable
from src.chat.llm.mini import call_mini
from src.chat.prompts import ENTITY_EXTRACTION_SYSTEM
from src.chat.retrieval.kg import fulltext_search

_INDEX_FOR_TYPE = {"symptom": "symptom_name", "drug": "drug_name"}


def _canonicalize(name: str, entity_type: str) -> str | None:
    """Match free-text name to canonical ID via Neo4j fulltext."""
    idx = _INDEX_FOR_TYPE.get(entity_type)
    if not idx or not name.strip():
        return None
    try:
        with get_neo4j().session() as session:
            results = session.execute_read(fulltext_search, idx, name, 1)
    except (Neo4jError, ServiceUnavailable) as e:
        raise Neo4jUnavailable(
            f"Neo4j canonicalization failed for {entity_type}:{name}"
        ) from e
    if results and results[0]["score"] > 1.0:
        return results[0]["props"].get("id")
    return None


def extract_entities(text: str) -> dict:
    """Returns {symptoms: [{symptom_id, name, onset, ...}], medications: [drug_id]}."""
    parsed = call_mini(ENTITY_EXTRACTION_SYSTEM, text) or {}
    return normalize_entities(parsed)


def normalize_entities(parsed: dict) -> dict:
    """Canonicalize already-extracted symptom/drug names without another LLM call."""
    if not isinstance(parsed, dict):
        return {"symptoms": [], "medications": []}

    result_symptoms = []
    for s in parsed.get("symptoms", []) or []:
        if not isinstance(s, dict):
            continue
        name = str(s.get("name", "")).strip()
        if not name:
            continue
        sid = _canonicalize(name, "symptom")
        entry = {"symptom_id": sid or f"raw:{name}", "name": name}
        for k in ("onset", "severity", "pattern", "associated"):
            if s.get(k):
                entry[k] = s[k]
        result_symptoms.append(entry)

    result_drugs = []
    for d_name in parsed.get("medications", []) or []:
        d_name = str(d_name).strip()
        if not d_name:
            continue
        did = _canonicalize(d_name, "drug")
        if did:
            result_drugs.append(did)

    return {"symptoms": result_symptoms, "medications": result_drugs}
