"""
entities.py
-----------
Extract symptoms + drugs (with slot values) from user message using
LLM mini, then canonicalize names to IDs via Neo4j fulltext search.
"""

from __future__ import annotations

import logging

from neo4j.exceptions import Neo4jError, ServiceUnavailable

from src.chat.clients import get_neo4j
from src.chat.llm.mini import call_mini
from src.chat.prompts import ENTITY_EXTRACTION_SYSTEM
from src.chat.retrieval.kg import fulltext_search

log = logging.getLogger(__name__)

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
        log.debug("canonicalize(%s, %s) failed: %s", name, entity_type, e)
        return None
    if results and results[0]["score"] > 1.0:
        return results[0]["props"].get("id")
    return None


def extract_entities(text: str) -> dict:
    """Returns {symptoms: [{symptom_id, name, onset, ...}], medications: [drug_id]}."""
    parsed = call_mini(ENTITY_EXTRACTION_SYSTEM, text) or {}
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
