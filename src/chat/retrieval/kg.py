"""
kg.py
-----
Knowledge Graph retrieval from Neo4j.
Match entities via fulltext index, then traverse relationships
to gather structured medical context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from neo4j.exceptions import Neo4jError, ServiceUnavailable

from src.chat.clients import get_neo4j

log = logging.getLogger(__name__)


@dataclass
class KGContext:
    """Structured facts extracted from the Knowledge Graph."""
    matched_entities: list[dict] = field(default_factory=list)
    related_diseases: list[dict] = field(default_factory=list)
    related_drugs: list[dict] = field(default_factory=list)
    related_symptoms: list[dict] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.matched_entities or self.related_diseases
                    or self.related_drugs or self.related_symptoms
                    or self.relationships)


# ---------- Index management ----------

_FULLTEXT_INDEXES = [
    ("disease_name", "Disease", ["n.name_vi", "n.name_en"]),
    ("drug_name", "Drug", ["n.name_vi", "n.generic_name_vi"]),
    ("symptom_name", "Symptom", ["n.name_vi", "n.name_en"]),
]


def ensure_fulltext_indexes() -> None:
    """Create fulltext indexes if they don't exist (idempotent)."""
    with get_neo4j().session() as session:
        existing = {r["name"] for r in session.run("SHOW INDEXES YIELD name")}
        for name, label, fields in _FULLTEXT_INDEXES:
            if name not in existing:
                session.run(
                    f"CREATE FULLTEXT INDEX {name} FOR (n:{label}) "
                    f"ON EACH [{', '.join(fields)}]"
                )


# ---------- Low-level primitives ----------

def fulltext_search(tx, index_name: str, query: str, limit: int = 5) -> list[dict]:
    """Run fulltext search, return list of {props, score}."""
    result = tx.run(
        "CALL db.index.fulltext.queryNodes($idx, $q) "
        "YIELD node, score WHERE score > 0.5 "
        "RETURN node {.*, _labels: labels(node)} AS props, score "
        "ORDER BY score DESC LIMIT $limit",
        idx=index_name, q=query, limit=limit,
    )
    return [{"props": r["props"], "score": r["score"]} for r in result]


# ---------- Per-entity traversals ----------

def _traverse_disease(tx, disease_id: str) -> dict:
    symptoms = tx.run(
        "MATCH (d {id: $did})-[r:HAS_SYMPTOM]->(s:Symptom) "
        "RETURN s.id AS id, s.name_vi AS name, r.is_red_flag AS red_flag LIMIT 20",
        did=disease_id,
    ).data()
    drugs = tx.run(
        "MATCH (dr:Drug)-[:TREATS]->(d {id: $did}) "
        "RETURN dr.id AS id, dr.name_vi AS name LIMIT 10",
        did=disease_id,
    ).data()
    comorbidities = tx.run(
        "MATCH (d {id: $did})-[:COMORBID_RISK]-(other:Disease) "
        "RETURN other.id AS id, other.name_vi AS name LIMIT 5",
        did=disease_id,
    ).data()
    return {"symptoms": symptoms, "drugs": drugs, "comorbidities": comorbidities}


def _traverse_drug(tx, drug_id: str) -> dict:
    treats = tx.run(
        "MATCH (dr {id: $did})-[:TREATS]->(d:Disease) "
        "RETURN d.id AS id, d.name_vi AS name LIMIT 10",
        did=drug_id,
    ).data()
    relieves = tx.run(
        "MATCH (dr {id: $did})-[:RELIEVES]->(s:Symptom) "
        "RETURN s.id AS id, s.name_vi AS name LIMIT 10",
        did=drug_id,
    ).data()
    contraindicated = tx.run(
        "MATCH (dr {id: $did})-[:CONTRAINDICATED_FOR]->(d:Disease) "
        "RETURN d.id AS id, d.name_vi AS name LIMIT 10",
        did=drug_id,
    ).data()
    interactions = tx.run(
        "MATCH (dr {id: $did})-[:INTERACTS_WITH]-(other:Drug) "
        "RETURN other.id AS id, other.name_vi AS name LIMIT 10",
        did=drug_id,
    ).data()
    return {
        "treats": treats,
        "relieves": relieves,
        "contraindicated_for": contraindicated,
        "interactions": interactions,
    }


def _traverse_symptom(tx, symptom_id: str) -> dict:
    diseases = tx.run(
        "MATCH (d:Disease)-[:HAS_SYMPTOM]->(s {id: $sid}) "
        "RETURN d.id AS id, d.name_vi AS name LIMIT 10",
        sid=symptom_id,
    ).data()
    red_flag_for = tx.run(
        "MATCH (s {id: $sid})-[:RED_FLAG_FOR]->(d:Disease) "
        "RETURN d.id AS id, d.name_vi AS name LIMIT 5",
        sid=symptom_id,
    ).data()
    return {"diseases": diseases, "red_flag_for": red_flag_for}


# ---------- KG search: merge fulltext matches + traversals ----------

def _ingest_disease(ctx: KGContext, entity_name: str, info: dict) -> None:
    for s in info["symptoms"]:
        if not s.get("name"):
            continue
        ctx.related_symptoms.append(s)
        if s.get("red_flag"):
            ctx.relationships.append(f"Dấu hiệu cảnh báo của {entity_name}: {s['name']}")
    for d in info["drugs"]:
        if d.get("name"):
            ctx.related_drugs.append(d)
    for c in info["comorbidities"]:
        if c.get("name"):
            ctx.relationships.append(f"{entity_name} liên quan bệnh đồng mắc: {c['name']}")


def _ingest_drug(ctx: KGContext, entity_name: str, info: dict) -> None:
    for d in info["treats"]:
        if d.get("name"):
            ctx.related_diseases.append(d)
    for s in info["relieves"]:
        if s.get("name"):
            ctx.related_symptoms.append(s)
    for c in info["contraindicated_for"]:
        if c.get("name"):
            ctx.relationships.append(f"{entity_name} chống chỉ định với: {c['name']}")
    for i in info["interactions"]:
        if i.get("name"):
            ctx.relationships.append(f"{entity_name} tương tác với: {i['name']}")


def _ingest_symptom(ctx: KGContext, entity_name: str, info: dict) -> None:
    for d in info["diseases"]:
        if d.get("name"):
            ctx.related_diseases.append(d)
    for d in info["red_flag_for"]:
        if d.get("name"):
            ctx.relationships.append(
                f"Triệu chứng {entity_name} là dấu hiệu cảnh báo của: {d['name']}"
            )


_ENTITY_HANDLERS = {
    "disease": (_traverse_disease, _ingest_disease),
    "drug":    (_traverse_drug,    _ingest_drug),
    "symptom": (_traverse_symptom, _ingest_symptom),
}

_INDEX_TO_TYPE = {
    "disease_name": "disease",
    "drug_name":    "drug",
    "symptom_name": "symptom",
}


def kg_search(query: str) -> KGContext:
    """Search KG for entities matching query, traverse their relationships."""
    ctx = KGContext()
    try:
        with get_neo4j().session() as session:
            for idx_name, entity_type in _INDEX_TO_TYPE.items():
                try:
                    matches = session.execute_read(fulltext_search, idx_name, query, 3)
                except Neo4jError as e:
                    log.debug("fulltext %s failed: %s", idx_name, e)
                    continue

                traverse, ingest = _ENTITY_HANDLERS[entity_type]
                for m in matches:
                    props = m["props"]
                    entity_id = props.get("id", "")
                    name = props.get("name_vi", "") or props.get("name_en", "")
                    ctx.matched_entities.append({
                        "id": entity_id, "type": entity_type,
                        "name": name, "score": m["score"],
                    })
                    info = session.execute_read(traverse, entity_id)
                    ingest(ctx, name, info)
    except ServiceUnavailable as e:
        log.warning("Neo4j unavailable: %s", e)
    except Neo4jError as e:
        log.warning("Neo4j error during KG search: %s", e)
    return ctx


# ---------- Formatting ----------

def _dedupe_names(items: list[dict], limit: int = 10) -> list[str]:
    seen, out = set(), []
    for it in items:
        name = it.get("name", "")
        if name and name not in seen:
            seen.add(name)
            out.append(name)
            if len(out) >= limit:
                break
    return out


def format_kg_context(ctx: KGContext) -> str:
    """Format KGContext into a text block for the LLM prompt."""
    if ctx.is_empty:
        return ""

    parts = []
    names = [e["name"] for e in ctx.matched_entities if e["name"]]
    if names:
        parts.append(f"Thực thể liên quan: {', '.join(names)}")

    if (diseases := _dedupe_names(ctx.related_diseases)):
        parts.append(f"Bệnh liên quan: {', '.join(diseases)}")

    if (drugs := _dedupe_names(ctx.related_drugs)):
        parts.append(f"Thuốc liên quan: {', '.join(drugs)}")

    if ctx.relationships:
        parts.append("Quan hệ y khoa:")
        parts.extend(f"  - {r}" for r in ctx.relationships[:10])

    return "\n".join(parts)
