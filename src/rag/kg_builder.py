"""
kg_builder.py
-------------
Build a Knowledge Graph in Neo4j from disease, drug, and symptom entities.

Nodes: Disease, Drug, Symptom, ICD10, Chapter
Edges: HAS_SYMPTOM, TREATS, CONTRAINDICATED_FOR, INTERACTS_WITH,
       BELONGS_TO_CHAPTER, HAS_ICD10, RED_FLAG_FOR, RELIEVES, etc.

Usage:
    python -m src.rag.kg_builder          # load all into Neo4j
    python -m src.rag.kg_builder --clear   # wipe DB first
    python -m src.rag.kg_builder --dry-run  # just print stats
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, OUTPUT_DIR

DISEASE_ENTITY_DIR = OUTPUT_DIR / "entities" / "diseases"
DRUG_ENTITY_DIR = OUTPUT_DIR / "entities" / "drugs"
SYMPTOM_ENTITY_DIR = OUTPUT_DIR / "entities" / "symptoms"

ALIAS_MAP_PATH = SYMPTOM_ENTITY_DIR / "_alias_map.json"


def load_alias_map() -> dict[str, str]:
    if ALIAS_MAP_PATH.exists():
        return json.loads(ALIAS_MAP_PATH.read_text(encoding="utf-8"))
    return {}


def canon(sid: str, alias_map: dict[str, str]) -> str:
    return alias_map.get(sid, sid)


def collect_graph(alias_map: dict[str, str]) -> tuple[dict[str, dict], list[dict]]:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    # --- Disease entities ---
    for f in sorted(DISEASE_ENTITY_DIR.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        did = d.get("disease_id", "")
        if not did:
            continue
        slug = d.get("disease_slug", f.stem)

        nodes[did] = {
            "id": did, "label": "Disease",
            "name_vi": d.get("name_vi", ""),
            "name_en": d.get("name_en", ""),
            "slug": slug,
            "icd10": d.get("coding", {}).get("icd10", ""),
            "body_system": d.get("classification", {}).get("body_system", ""),
            "care_level": d.get("classification", {}).get("care_level", ""),
            "urgency": d.get("classification", {}).get("urgency", ""),
            "when_to_see_doctor": d.get("treatment", {}).get("when_to_see_doctor", ""),
            "home_care_summary_vi": d.get("treatment", {}).get("home_care_summary_vi", ""),
        }

        body_sys = d.get("classification", {}).get("body_system", "")
        if body_sys:
            ch_id = f"chapter:{body_sys}"
            if ch_id not in nodes:
                nodes[ch_id] = {"id": ch_id, "label": "Chapter", "name_vi": body_sys}
            edges.append({"src": did, "tgt": ch_id, "type": "BELONGS_TO_CHAPTER"})

        for sid in d.get("symptoms", {}).get("primary_ids", []):
            edges.append({"src": did, "tgt": canon(sid, alias_map), "type": "HAS_SYMPTOM"})

        for sid in d.get("symptoms", {}).get("red_flag_ids", []):
            edges.append({"src": did, "tgt": canon(sid, alias_map), "type": "RED_FLAG_FOR"})

        for drug_id in d.get("treatment", {}).get("first_line_drug_ids", []):
            edges.append({"src": drug_id, "tgt": did, "type": "TREATS"})

        for drug_id in d.get("treatment", {}).get("contraindicated_drug_ids", []):
            edges.append({"src": drug_id, "tgt": did, "type": "CONTRAINDICATED_FOR"})

        for phase in d.get("phases", []):
            for sid in phase.get("symptom_ids", []):
                edges.append({
                    "src": did, "tgt": canon(sid, alias_map), "type": "HAS_SYMPTOM",
                    "phase": phase.get("name", ""),
                })

        for cid in d.get("hospitalization_criteria", {}).get("comorbid_disease_ids", []):
            edges.append({"src": did, "tgt": cid, "type": "COMORBID_RISK"})

    # --- Drug entities ---
    for f in sorted(DRUG_ENTITY_DIR.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        drug_id = d.get("drug_id", "")
        if not drug_id:
            continue
        slug = d.get("drug_slug", f.stem)

        nodes[drug_id] = {
            "id": drug_id, "label": "Drug",
            "generic_name_vi": d.get("generic_name_vi", ""),
            "generic_name_en": d.get("generic_name_en", ""),
            "slug": slug,
            "drug_class_vi": d.get("drug_class_vi", ""),
            "atc_code": d.get("coding", {}).get("atc_code", ""),
            "dosage_adults": d.get("dosage", {}).get("adults", ""),
            "dosage_children": d.get("dosage", {}).get("children", ""),
            "pregnancy_category": d.get("pregnancy_category", ""),
            "breastfeeding": d.get("breastfeeding", ""),
            "adr_common": json.dumps(d.get("adr_summary", {}).get("common", []), ensure_ascii=False),
            "adr_rare_serious": json.dumps(d.get("adr_summary", {}).get("rare_serious", []), ensure_ascii=False),
        }

        for sid in d.get("indicated_for", {}).get("symptom_ids", []):
            edges.append({"src": drug_id, "tgt": canon(sid, alias_map), "type": "RELIEVES"})

        for icd_id in d.get("indicated_for", {}).get("disease_ids", []):
            edges.append({"src": drug_id, "tgt": icd_id, "type": "TREATS"})

        for icd_id in d.get("contraindicated_in", {}).get("disease_ids", []):
            edges.append({"src": drug_id, "tgt": icd_id, "type": "CONTRAINDICATED_FOR"})

        for inter in d.get("drug_interactions", []):
            other = inter.get("with_drug_id", "")
            if other:
                edges.append({
                    "src": drug_id, "tgt": other, "type": "INTERACTS_WITH",
                    "severity": inter.get("severity", ""),
                    "mechanism_vi": inter.get("mechanism_vi", ""),
                    "clinical_effect_vi": inter.get("clinical_effect_vi", ""),
                    "management_vi": inter.get("management_vi", ""),
                })

        for di in d.get("disease_interactions", []):
            target = di.get("disease_id", "")
            if target:
                edges.append({
                    "src": drug_id, "tgt": target,
                    "type": "DISEASE_INTERACTION",
                    "effect_vi": di.get("effect_vi", ""),
                    "management_vi": di.get("management_vi", ""),
                })

    # --- Symptom entities ---
    for f in sorted(SYMPTOM_ENTITY_DIR.glob("*.json")):
        if f.name.startswith("_"):
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        sid = d.get("symptom_id", "")
        if not sid:
            continue
        if sid not in nodes:
            nodes[sid] = {
                "id": sid, "label": "Symptom",
                "name_vi": d.get("name_vi", ""),
                "name_en": d.get("name_en", ""),
                "body_system": d.get("body_system", ""),
                "symptom_type": d.get("type", ""),
                "q_onset": d.get("clarification_questions", {}).get("onset", ""),
                "q_severity": d.get("clarification_questions", {}).get("severity", ""),
                "q_pattern": d.get("clarification_questions", {}).get("pattern", ""),
                "q_associated": d.get("clarification_questions", {}).get("associated", ""),
            }

    return nodes, edges


# ── Neo4j loading ──────────────────────────────────────────────────────

BATCH_SIZE = 500

CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Disease) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Drug) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Symptom) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Chapter) REQUIRE n.id IS UNIQUE",
]


def create_constraints(tx):
    for c in CONSTRAINTS:
        tx.run(c)


def upsert_nodes(tx, batch: list[dict], label: str):
    tx.run(
        f"UNWIND $batch AS row "
        f"MERGE (n:{label} {{id: row.id}}) "
        f"SET n += row",
        batch=batch,
    )


def upsert_edges(tx, batch: list[dict], rel_type: str):
    props = set()
    for e in batch:
        props.update(k for k in e if k not in ("src", "tgt", "type"))
    set_clause = ", ".join(f"r.{k} = row.{k}" for k in sorted(props))
    if set_clause:
        set_clause = "SET " + set_clause

    tx.run(
        f"UNWIND $batch AS row "
        f"MERGE (a {{id: row.src}}) "
        f"MERGE (b {{id: row.tgt}}) "
        f"MERGE (a)-[r:{rel_type}]->(b) "
        f"{set_clause}",
        batch=batch,
    )


LABEL_BY_PREFIX = {
    "drug:": "Drug",
    "symptom:": "Symptom",
    "ICD10:": "Disease",
    "chapter:": "Chapter",
}


def label_stubs(tx):
    """Assign labels to stub nodes created by MERGE in edges."""
    for prefix, label in LABEL_BY_PREFIX.items():
        tx.run(
            f"MATCH (n) WHERE n.id STARTS WITH $prefix "
            f"AND NOT n:{label} "
            f"SET n:{label}",
            prefix=prefix,
        )


def load_to_neo4j(nodes: dict[str, dict], edges: list[dict], clear: bool = False):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("Connected to Neo4j")

    with driver.session() as session:
        if clear:
            session.run("MATCH (n) DETACH DELETE n")
            print("Cleared existing data")

        session.execute_write(create_constraints)
        print("Constraints created")

        # Group nodes by label
        by_label: dict[str, list[dict]] = defaultdict(list)
        for n in nodes.values():
            label = n.pop("label")
            by_label[label].append(n)

        for label, node_list in by_label.items():
            for i in range(0, len(node_list), BATCH_SIZE):
                batch = node_list[i : i + BATCH_SIZE]
                session.execute_write(upsert_nodes, batch, label)
            print(f"  {label}: {len(node_list)} nodes")

        # Group edges by type
        by_type: dict[str, list[dict]] = defaultdict(list)
        for e in edges:
            by_type[e["type"]].append(e)

        for rel_type, edge_list in by_type.items():
            for i in range(0, len(edge_list), BATCH_SIZE):
                batch = edge_list[i : i + BATCH_SIZE]
                session.execute_write(upsert_edges, batch, rel_type)
            print(f"  {rel_type}: {len(edge_list)} edges")

        session.execute_write(label_stubs)
        print("  Labeled stub nodes")

    driver.close()


# ── Stats & main ───────────────────────────────────────────────────────

def print_stats(nodes: dict[str, dict], edges: list[dict]) -> None:
    node_types = defaultdict(int)
    for n in nodes.values():
        node_types[n.get("label", "?")] += 1
    edge_types = defaultdict(int)
    for e in edges:
        edge_types[e["type"]] += 1

    print("\nNodes:")
    for t, c in sorted(node_types.items()):
        print(f"  {t}: {c}")
    print(f"  Total: {len(nodes)}")

    print("\nEdges:")
    for t, c in sorted(edge_types.items()):
        print(f"  {t}: {c}")
    print(f"  Total: {len(edges)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true", help="Wipe Neo4j DB before loading")
    parser.add_argument("--dry-run", action="store_true", help="Only print stats, don't load")
    args = parser.parse_args()

    alias_map = load_alias_map()
    nodes, edges = collect_graph(alias_map)
    print_stats(nodes, edges)

    if args.dry_run:
        print("\n(dry-run — not loading to Neo4j)")
        return

    load_to_neo4j(nodes, edges, clear=args.clear)
    print("\nDone — KG loaded to Neo4j")


if __name__ == "__main__":
    main()
