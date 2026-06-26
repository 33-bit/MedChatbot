"""
chunker.py
----------
Per-subsection chunker for disease, drug, and health-insurance documents.
Produces chunks.jsonl ready for embedding + Qdrant upload.

Usage:
    python -m src.rag.chunker
    python -m src.rag.chunker --source diseases
    python -m src.rag.chunker --source drugs
    python -m src.rag.chunker --source health_insurance
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR

DISEASE_FINAL_DIR = OUTPUT_DIR / "bachmai" / "final"
DISEASE_ENTITY_DIR = OUTPUT_DIR / "entities" / "diseases"
DRUG_FINAL_DIR = OUTPUT_DIR / "otc_drugs" / "final_json"
DRUG_ENTITY_DIR = OUTPUT_DIR / "entities" / "drugs"
HEALTH_INSURANCE_DOCUMENT = OUTPUT_DIR / "health_insurance" / "22-vbhn-vpqh.json"
CHUNKS_OUT = OUTPUT_DIR / "chunks"

SKIP_HEADINGS = {"TÀI LIỆU THAM KHẢO"}
MIN_CHUNK_CHARS = 30
HEALTH_INSURANCE_MAX_CHARS = 1200
HEALTH_INSURANCE_OVERLAP_CHARS = 150
HEALTH_INSURANCE_SOURCE_SLUG = "22-vbhn-vpqh"
_CLAUSE_RE = re.compile(r"^(\d+)\.\s+(.*)$")


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
    import hashlib
    import re
    text = text.lower()
    # Preserve dots (a., b., 1.1., ...) so sub-section letter prefixes are kept,
    # otherwise e.g. "a. Giai đoạn sốt" and "b. Giai đoạn nguy hiểm" collapse
    # to the same chunk_id. Convert runs of dots to underscores for stability.
    sanitized = re.sub(r"[^a-z0-9_\s.]", "", text)
    sanitized = re.sub(r"\.+", "_", sanitized)
    sanitized = re.sub(r"\s+", "_", sanitized.strip())
    sanitized = re.sub(r"_+", "_", sanitized)
    # Long heading paths (deep sub-sub-subsections) get truncated; to keep
    # uniqueness we append a short hash of the full path. We use a 6-char
    # truncated sha1 over the original (case-folded) heading_path so sibling
    # sub-sections get distinct IDs.
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:6]
    return f"{sanitized[:120]}_{h}"


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


def _split_long_legal_text(text: str) -> list[str]:
    text = text.strip()
    if len(text) <= HEALTH_INSURANCE_MAX_CHARS:
        return [text] if text else []

    parts: list[str] = []
    start = 0
    while start < len(text):
        limit = min(len(text), start + HEALTH_INSURANCE_MAX_CHARS)
        end = limit
        if limit < len(text):
            candidates = [
                text.rfind("\n", start, limit),
                text.rfind(". ", start, limit),
                text.rfind("; ", start, limit),
            ]
            boundary = max(candidates)
            if boundary > start + HEALTH_INSURANCE_MAX_CHARS // 2:
                end = boundary + (2 if text[boundary:boundary + 2] in {". ", "; "} else 1)
        part = text[start:end].strip()
        if part:
            parts.append(part)
        if end >= len(text):
            break
        start = max(end - HEALTH_INSURANCE_OVERLAP_CHARS, start + 1)
    return parts


def _article_units(body: str) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    current_clause = ""
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        text = "\n".join(current_lines).strip()
        if text:
            units.append((current_clause, text))
        current_lines = []

    for line in body.splitlines():
        match = _CLAUSE_RE.match(line.strip())
        if match:
            flush()
            current_clause = match.group(1)
        current_lines.append(line)
    flush()
    return units


def chunk_health_insurance() -> list[dict]:
    if not HEALTH_INSURANCE_DOCUMENT.exists():
        raise FileNotFoundError(
            f"Missing {HEALTH_INSURANCE_DOCUMENT}. "
            "Run `python -m src.processing.health_insurance.parse` first."
        )

    document = json.loads(HEALTH_INSURANCE_DOCUMENT.read_text(encoding="utf-8"))
    chunks: list[dict] = []
    source_name = f"{document['document_title']} ({document['document_number']})"
    for article in document.get("articles", []):
        article_number = str(article["article_number"])
        article_heading = f"Điều {article_number}. {article['article_title']}".strip()
        chapter_heading = (
            f"Chương {article['chapter_number']}. {article['chapter_title']}"
        )
        body = article.get("body", "").strip()
        units = (
            [("", body)]
            if len(body) <= HEALTH_INSURANCE_MAX_CHARS
            else _article_units(body)
        )
        for unit_number, (clause_number, unit_text) in enumerate(units, start=1):
            for part_number, content in enumerate(_split_long_legal_text(unit_text), start=1):
                clause_suffix = f":clause:{clause_number}" if clause_number else ""
                chunk_id = (
                    f"health_insurance:{HEALTH_INSURANCE_SOURCE_SLUG}:"
                    f"article:{article_number}{clause_suffix}:unit:{unit_number}:"
                    f"part:{part_number}"
                )
                heading_path = f"{chapter_heading} > {article_heading}"
                if clause_number:
                    heading_path += f" > Khoản {clause_number}"
                text = f"{source_name}\n{heading_path}\n\n{content}"
                chunks.append({
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)),
                    "chunk_id": chunk_id,
                    "source_type": "health_insurance",
                    "source_slug": HEALTH_INSURANCE_SOURCE_SLUG,
                    "source_name": source_name,
                    "heading_path": heading_path,
                    "text": text,
                    "metadata": {
                        "document_number": document["document_number"],
                        "issued_date": document["issued_date"],
                        "chapter_number": article["chapter_number"],
                        "chapter_title": article["chapter_title"],
                        "article_number": article_number,
                        "article_title": article["article_title"],
                        "clause_number": clause_number,
                        "unit_number": unit_number,
                        "part_number": part_number,
                        "page_start": article["page_start"],
                        "page_end": article["page_end"],
                    },
                })
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["diseases", "drugs", "health_insurance", "all"],
        default="all",
    )
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

    if args.source == "health_insurance" or (
        args.source == "all" and HEALTH_INSURANCE_DOCUMENT.exists()
    ):
        health_insurance_chunks = chunk_health_insurance()
        out = CHUNKS_OUT / "health_insurance_chunks.jsonl"
        with open(out, "w", encoding="utf-8") as fh:
            for chunk in health_insurance_chunks:
                fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        print(
            f"Health-insurance chunks: {len(health_insurance_chunks)} → {out}"
        )
        total += len(health_insurance_chunks)

    print(f"\nTotal: {total} chunks")


if __name__ == "__main__":
    main()
