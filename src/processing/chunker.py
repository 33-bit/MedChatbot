"""
chunker.py
----------
Per-subsection chunker for disease and drug documents.
Produces chunks.jsonl ready for embedding + Qdrant upload.

Usage:
    python -m src.processing.chunker
    python -m src.processing.chunker --source diseases
    python -m src.processing.chunker --source drugs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR

DISEASE_FINAL_DIR = OUTPUT_DIR / "bachmai" / "final"
DISEASE_ENTITY_DIR = OUTPUT_DIR / "entities" / "diseases"
DRUG_FINAL_DIR = OUTPUT_DIR / "otc_drugs" / "final_json"
DRUG_ENTITY_DIR = OUTPUT_DIR / "entities" / "drugs"
CHUNKS_OUT = OUTPUT_DIR / "chunks"

SKIP_HEADINGS = {"TÀI LIỆU THAM KHẢO"}
MIN_CHUNK_CHARS = 30


def flatten_to_chunks(
    sections: list[dict],
    source_type: str,
    source_slug: str,
    source_name: str,
    extra_meta: dict,
) -> list[dict]:
    chunks = []

    def walk(node: dict, parent_path: str, depth: int) -> None:
        heading = node.get("heading", "")
        heading_clean = heading.strip()

        if any(skip in heading_clean.upper() for skip in SKIP_HEADINGS):
            return

        path = f"{parent_path} > {heading_clean}" if parent_path else heading_clean
        content = node.get("content", "").strip()
        subs = node.get("subsections", [])

        if subs:
            if content and len(content) >= MIN_CHUNK_CHARS:
                chunks.append(_make_chunk(
                    source_type, source_slug, source_name,
                    path, content, extra_meta,
                ))
            for sub in subs:
                walk(sub, path, depth + 1)
        else:
            if content and len(content) >= MIN_CHUNK_CHARS:
                chunks.append(_make_chunk(
                    source_type, source_slug, source_name,
                    path, content, extra_meta,
                ))

    for section in sections:
        walk(section, "", 0)

    return chunks


def _make_chunk(
    source_type: str,
    source_slug: str,
    source_name: str,
    heading_path: str,
    content: str,
    extra_meta: dict,
) -> dict:
    chunk_id = f"{source_type}:{source_slug}:{_slugify(heading_path)}"
    text = f"{source_name}\n{heading_path}\n\n{content}"
    return {
        "chunk_id": chunk_id,
        "source_type": source_type,
        "source_slug": source_slug,
        "source_name": source_name,
        "heading_path": heading_path,
        "text": text,
        "metadata": extra_meta,
    }


def _slugify(text: str) -> str:
    import re
    text = text.lower()
    text = re.sub(r"[^a-z0-9_\s]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:80]


def chunk_diseases() -> list[dict]:
    all_chunks = []
    for f in sorted(DISEASE_FINAL_DIR.glob("*.json")):
        slug = f.stem
        doc = json.loads(f.read_text(encoding="utf-8"))

        entity_path = DISEASE_ENTITY_DIR / f"{slug}.json"
        entity = json.loads(entity_path.read_text(encoding="utf-8")) if entity_path.exists() else {}

        meta = {
            "disease_id": entity.get("disease_id", ""),
            "chapter": doc.get("chapter", ""),
            "icd10": entity.get("coding", {}).get("icd10", ""),
            "body_system": entity.get("classification", {}).get("body_system", ""),
        }

        chunks = flatten_to_chunks(
            doc.get("sections", []),
            source_type="disease",
            source_slug=slug,
            source_name=doc.get("disease", slug),
            extra_meta=meta,
        )
        all_chunks.extend(chunks)

    return all_chunks


def chunk_drugs() -> list[dict]:
    all_chunks = []
    for f in sorted(DRUG_FINAL_DIR.glob("*.json")):
        slug = f.stem
        doc = json.loads(f.read_text(encoding="utf-8"))

        entity_path = DRUG_ENTITY_DIR / f"{slug}.json"
        entity = json.loads(entity_path.read_text(encoding="utf-8")) if entity_path.exists() else {}

        meta = {
            "drug_id": entity.get("drug_id", ""),
            "drug_class": doc.get("drug_class", ""),
            "atc_code": entity.get("coding", {}).get("atc_code", ""),
            "international_name": doc.get("international_name", ""),
        }

        chunks = flatten_to_chunks(
            doc.get("sections", []),
            source_type="drug",
            source_slug=slug,
            source_name=doc.get("name", slug),
            extra_meta=meta,
        )
        all_chunks.extend(chunks)

    return all_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["diseases", "drugs", "all"], default="all")
    args = parser.parse_args()

    CHUNKS_OUT.mkdir(parents=True, exist_ok=True)

    total = 0
    if args.source in ("diseases", "all"):
        disease_chunks = chunk_diseases()
        out = CHUNKS_OUT / "disease_chunks.jsonl"
        with open(out, "w", encoding="utf-8") as fh:
            for c in disease_chunks:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"Disease chunks: {len(disease_chunks)} → {out}")
        total += len(disease_chunks)

    if args.source in ("drugs", "all"):
        drug_chunks = chunk_drugs()
        out = CHUNKS_OUT / "drug_chunks.jsonl"
        with open(out, "w", encoding="utf-8") as fh:
            for c in drug_chunks:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"Drug chunks: {len(drug_chunks)} → {out}")
        total += len(drug_chunks)

    print(f"\nTotal: {total} chunks")


if __name__ == "__main__":
    main()
