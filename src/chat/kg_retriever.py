"""
kg_retriever.py
---------------
Knowledge Graph retrieval from Neo4j.
Matches entities in user query via fulltext index, then traverses
relationships to gather structured medical context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from neo4j import GraphDatabase

from src.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER


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


@lru_cache(maxsize=1)
def _driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def ensure_fulltext_indexes() -> None:
    """Create fulltext indexes if they don't exist (idempotent)."""
    with _driver().session() as session:
        existing = {r["name"] for r in session.run("SHOW INDEXES YIELD name")}
        if "disease_name" not in existing:
            session.run(
                "CREATE FULLTEXT INDEX disease_name FOR (n:Disease) "
                "ON EACH [n.name_vi, n.name_en]"
            )
        if "drug_name" not in existing:
            session.run(
                "CREATE FULLTEXT INDEX drug_name FOR (n:Drug) "
                "ON EACH [n.name_vi, n.generic_name_vi]"
            )
        if "symptom_name" not in existing:
            session.run(
                "CREATE FULLTEXT INDEX symptom_name FOR (n:Symptom) "
                "ON EACH [n.name_vi, n.name_en]"
            )


def _fulltext_search(tx, index_name: str, query: str, limit: int = 5) -> list[dict]:
    """Run fulltext search, return list of node properties."""
    safe_query = query.replace('"', '\\"')
    result = tx.run(
        f"CALL db.index.fulltext.queryNodes($idx, $q) "
        f"YIELD node, score WHERE score > 0.5 "
        f"RETURN node {{.*, _labels: labels(node)}} AS props, score "
        f"ORDER BY score DESC LIMIT $limit",
        idx=index_name, q=safe_query, limit=limit,
    )
    return [{"props": r["props"], "score": r["score"]} for r in result]


def _traverse_disease(tx, disease_id: str) -> dict:
    """Get symptoms, drugs, and key relationships for a disease."""
    symptoms = tx.run(
        "MATCH (d {id: $did})-[r:HAS_SYMPTOM]->(s:Symptom) "
        "RETURN s.id AS id, s.name_vi AS name, r.is_red_flag AS red_flag "
        "LIMIT 20",
        did=disease_id,
    ).data()

    drugs = tx.run(
        "MATCH (dr:Drug)-[:TREATS]->(d {id: $did}) "
        "RETURN dr.id AS id, dr.name_vi AS name "
        "LIMIT 10",
        did=disease_id,
    ).data()

    comorbidities = tx.run(
        "MATCH (d {id: $did})-[:COMORBID_RISK]-(other:Disease) "
        "RETURN other.id AS id, other.name_vi AS name "
        "LIMIT 5",
        did=disease_id,
    ).data()

    return {"symptoms": symptoms, "drugs": drugs, "comorbidities": comorbidities}


def _traverse_drug(tx, drug_id: str) -> dict:
    """Get indications, contraindications, and interactions for a drug."""
    treats = tx.run(
        "MATCH (dr {id: $did})-[:TREATS]->(d:Disease) "
        "RETURN d.id AS id, d.name_vi AS name "
        "LIMIT 10",
        did=drug_id,
    ).data()

    relieves = tx.run(
        "MATCH (dr {id: $did})-[:RELIEVES]->(s:Symptom) "
        "RETURN s.id AS id, s.name_vi AS name "
        "LIMIT 10",
        did=drug_id,
    ).data()

    contraindicated = tx.run(
        "MATCH (dr {id: $did})-[:CONTRAINDICATED_FOR]->(d:Disease) "
        "RETURN d.id AS id, d.name_vi AS name "
        "LIMIT 10",
        did=drug_id,
    ).data()

    interactions = tx.run(
        "MATCH (dr {id: $did})-[:INTERACTS_WITH]-(other:Drug) "
        "RETURN other.id AS id, other.name_vi AS name "
        "LIMIT 10",
        did=drug_id,
    ).data()

    return {
        "treats": treats,
        "relieves": relieves,
        "contraindicated_for": contraindicated,
        "interactions": interactions,
    }


def _traverse_symptom(tx, symptom_id: str) -> dict:
    """Get diseases associated with a symptom."""
    diseases = tx.run(
        "MATCH (d:Disease)-[:HAS_SYMPTOM]->(s {id: $sid}) "
        "RETURN d.id AS id, d.name_vi AS name "
        "LIMIT 10",
        sid=symptom_id,
    ).data()

    red_flag_for = tx.run(
        "MATCH (s {id: $sid})-[:RED_FLAG_FOR]->(d:Disease) "
        "RETURN d.id AS id, d.name_vi AS name "
        "LIMIT 5",
        sid=symptom_id,
    ).data()

    return {"diseases": diseases, "red_flag_for": red_flag_for}


def kg_search(query: str) -> KGContext:
    """Search KG for entities matching query, traverse their relationships."""
    ctx = KGContext()

    try:
        with _driver().session() as session:
            # Search across all entity types
            for idx_name, entity_type in [
                ("disease_name", "disease"),
                ("drug_name", "drug"),
                ("symptom_name", "symptom"),
            ]:
                try:
                    matches = session.execute_read(
                        _fulltext_search, idx_name, query, limit=3
                    )
                except Exception:
                    continue

                for m in matches:
                    props = m["props"]
                    entity_id = props.get("id", "")
                    ctx.matched_entities.append({
                        "id": entity_id,
                        "type": entity_type,
                        "name": props.get("name_vi", "") or props.get("name_en", ""),
                        "score": m["score"],
                    })

                    # Traverse based on entity type
                    if entity_type == "disease":
                        info = session.execute_read(_traverse_disease, entity_id)
                        for s in info["symptoms"]:
                            if s.get("name"):
                                ctx.related_symptoms.append(s)
                                if s.get("red_flag"):
                                    ctx.relationships.append(
                                        f"Dấu hiệu cảnh báo của {props.get('name_vi', '')}: {s['name']}"
                                    )
                        for d in info["drugs"]:
                            if d.get("name"):
                                ctx.related_drugs.append(d)
                        for c in info["comorbidities"]:
                            if c.get("name"):
                                ctx.relationships.append(
                                    f"{props.get('name_vi', '')} liên quan bệnh đồng mắc: {c['name']}"
                                )

                    elif entity_type == "drug":
                        info = session.execute_read(_traverse_drug, entity_id)
                        for d in info["treats"]:
                            if d.get("name"):
                                ctx.related_diseases.append(d)
                        for s in info["relieves"]:
                            if s.get("name"):
                                ctx.related_symptoms.append(s)
                        for c in info["contraindicated_for"]:
                            if c.get("name"):
                                ctx.relationships.append(
                                    f"{props.get('name_vi', '')} chống chỉ định với: {c['name']}"
                                )
                        for i in info["interactions"]:
                            if i.get("name"):
                                ctx.relationships.append(
                                    f"{props.get('name_vi', '')} tương tác với: {i['name']}"
                                )

                    elif entity_type == "symptom":
                        info = session.execute_read(_traverse_symptom, entity_id)
                        for d in info["diseases"]:
                            if d.get("name"):
                                ctx.related_diseases.append(d)
                        for d in info["red_flag_for"]:
                            if d.get("name"):
                                ctx.relationships.append(
                                    f"Triệu chứng {props.get('name_vi', '')} là dấu hiệu cảnh báo của: {d['name']}"
                                )
    except Exception:
        # KG unavailable — gracefully return empty context
        pass

    return ctx


def format_kg_context(ctx: KGContext) -> str:
    """Format KGContext into a text block for the LLM prompt."""
    if ctx.is_empty:
        return ""

    parts = []

    if ctx.matched_entities:
        names = [e["name"] for e in ctx.matched_entities if e["name"]]
        if names:
            parts.append(f"Thực thể liên quan: {', '.join(names)}")

    if ctx.related_diseases:
        seen = set()
        names = []
        for d in ctx.related_diseases:
            if d["name"] and d["name"] not in seen:
                seen.add(d["name"])
                names.append(d["name"])
        if names:
            parts.append(f"Bệnh liên quan: {', '.join(names[:10])}")

    if ctx.related_drugs:
        seen = set()
        names = []
        for d in ctx.related_drugs:
            if d["name"] and d["name"] not in seen:
                seen.add(d["name"])
                names.append(d["name"])
        if names:
            parts.append(f"Thuốc liên quan: {', '.join(names[:10])}")

    if ctx.relationships:
        parts.append("Quan hệ y khoa:")
        for r in ctx.relationships[:10]:
            parts.append(f"  - {r}")

    return "\n".join(parts)
