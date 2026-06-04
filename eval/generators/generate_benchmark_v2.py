#!/usr/bin/env python3
"""Generate v2 medical QA benchmark cases.

This generator intentionally separates category prompts:
- disease_info: one question per disease document, explicit disease name.
- drug_info: one question per drug document, explicit drug name.
- symptom_triage: symptom-only triage questions from Neo4j disease + ADR evidence.

Usage:
    PYTHONPATH=. python3 eval/generators/generate_benchmark_v2.py
    PYTHONPATH=. python3 eval/generators/generate_benchmark_v2.py --symptom-triage-target 250
    PYTHONPATH=. python3 eval/generators/generate_benchmark_v2.py --out eval/datasets/medical_qa_benchmark_v2.jsonl
    PYTHONPATH=. python3 eval/generators/generate_benchmark_v2.py --category symptom_triage --append
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.chat.clients import get_neo4j, get_openai
from src.config import MODEL

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DEFAULT_OUT = PROJECT_ROOT / "eval" / "datasets" / "medical_qa_benchmark_v2.jsonl"
DISEASE_DIR = PROJECT_ROOT / "outputs" / "bachmai" / "final"
DRUG_DIR = PROJECT_ROOT / "outputs" / "otc_drugs" / "final_json"

CONTENT_CHAR_BUDGET = 12000
TRIAGE_DOC_CHAR_BUDGET = 3500
TRIAGE_SECTION_CHAR_BUDGET = 900
MAX_CONCURRENCY = 1
LLM_RESPONSE_PREVIEW_CHARS = 500
DEFAULT_SYMPTOM_TRIAGE_TARGET = 100

Category = Literal["disease_info", "drug_info", "symptom_triage"]

SYSTEM_PROMPTS: dict[Category, str] = {
    "disease_info": """Bạn là chuyên gia QA y tế tạo benchmark cho chatbot RAG.

Nhiệm vụ: tạo đúng 1 câu hỏi category `disease_info` từ một tài liệu bệnh.

Quy tắc bắt buộc:
- Câu hỏi phải nêu rõ tên bệnh hoặc biệt danh rất rõ của bệnh trong tài liệu.
- Người hỏi là bệnh nhân hoặc người nhà, ngôn ngữ tự nhiên.
- Không tạo câu hỏi chẩn đoán mơ hồ chỉ liệt kê triệu chứng mà không nêu tên bệnh.
- `reference_answer` chỉ dùng thông tin trong tài liệu.
- `supporting_heading_paths` chọn 1-3 heading_path y nguyên từ marker trong prompt.
- Trả JSON hợp lệ, không markdown.

Output schema:
{"test_case":{"question":str,"reference_answer":str,"supporting_heading_paths":[str]}}""",
    "drug_info": """Bạn là chuyên gia QA y tế tạo benchmark cho chatbot RAG.

Nhiệm vụ: tạo đúng 1 câu hỏi category `drug_info` từ một tài liệu thuốc.

Quy tắc bắt buộc:
- Câu hỏi phải nêu rõ tên thuốc trong tài liệu.
- Hỏi về chỉ định, liều dùng, cách dùng, chống chỉ định, thận trọng, tương tác hoặc tác dụng không mong muốn.
- Người hỏi là bệnh nhân hoặc người nhà, ngôn ngữ tự nhiên.
- `reference_answer` chỉ dùng thông tin trong tài liệu, không kê đơn ngoài tài liệu.
- `supporting_heading_paths` chọn 1-3 heading_path y nguyên từ marker trong prompt.
- Trả JSON hợp lệ, không markdown.

Output schema:
{"test_case":{"question":str,"reference_answer":str,"supporting_heading_paths":[str]}}""",
    "symptom_triage": """Bạn là chuyên gia QA y tế tạo benchmark cho chatbot RAG.

Nhiệm vụ: tạo đúng 1 câu hỏi category `symptom_triage` từ một cụm triệu chứng và các nguồn bệnh/thuốc liên quan.

Quy tắc bắt buộc:
- Câu hỏi chỉ nêu triệu chứng và câu hỏi triage như "có nguy hiểm không", "nên làm gì", "có cần đi khám không".
- Không nêu tên bệnh hoặc tên thuốc trong câu hỏi.
- `reference_answer` phải nói chưa thể chẩn đoán chắc chắn qua chat.
- `reference_answer` phải nêu các nguyên nhân bệnh hợp lý từ nguồn bệnh nếu có.
- `reference_answer` phải nêu thuốc/thực phẩm chức năng/thuốc nam hoặc tác dụng không mong muốn của thuốc có thể là nguyên nhân nếu nguồn thuốc / ADR được cung cấp.
- `reference_answer` phải hỏi người dùng có dùng thuốc gì gần đây không, và khuyên mang danh sách/vỏ thuốc khi đi khám.
- `reference_answer` phải nêu khi nào cần đi khám/cấp cứu nếu nguồn có dấu hiệu nguy hiểm.
- `supporting_refs` chọn 2-6 nguồn thật sự hỗ trợ `reference_answer`; mỗi item phải copy y nguyên `doc_key` và `heading_path` từ marker trong prompt.
- Nếu `reference_answer` nêu nguyên nhân do thuốc / ADR, `supporting_refs` phải có ít nhất 1 nguồn thuốc / ADR.
- Trả JSON hợp lệ, không markdown.

Output schema:
{"test_case":{"question":str,"reference_answer":str,"supporting_refs":[{"doc_key":str,"heading_path":str}]}}""",
}


@dataclass(frozen=True)
class SourceDoc:
    title: str
    path: str
    source_type: Literal["disease", "drug"]
    source_slug: str
    flat_rows: list[dict]


@dataclass(frozen=True)
class EvidenceDoc:
    title: str
    path: str
    source_type: Literal["disease", "drug"]
    source_slug: str
    role: str
    evidence: list[str] = field(default_factory=list)
    flat_rows: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class TriageBundle:
    seed_symptoms: list[dict]
    disease_docs: list[EvidenceDoc]
    adr_drug_docs: list[EvidenceDoc]
    red_flag_docs: list[EvidenceDoc]

    def all_docs(self) -> list[EvidenceDoc]:
        docs: list[EvidenceDoc] = []
        seen: set[tuple[str, str]] = set()
        for doc in self.disease_docs + self.adr_drug_docs + self.red_flag_docs:
            key = (doc.source_type, doc.source_slug)
            if key in seen:
                continue
            seen.add(key)
            docs.append(doc)
        return docs


class SingleDocCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    reference_answer: str
    supporting_heading_paths: list[str] = Field(default_factory=list)


class SingleDocOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_case: SingleDocCase


class SupportingRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_key: str
    heading_path: str


class SymptomTriageCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    reference_answer: str
    supporting_refs: list[SupportingRef] = Field(default_factory=list)


class SymptomTriageOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_case: SymptomTriageCase


def _norm(text: str) -> str:
    text = (text or "").casefold().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def slugify(text: str) -> str:
    norm = _norm(text)
    norm = re.sub(r"[^a-z0-9]+", "-", norm).strip("-")
    return norm or "case"


def chunker_slugify(text: str) -> str:
    """Mirror src.rag.chunker._slugify exactly."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9_\s]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:80]


def chunk_id_for(source_type: str, source_slug: str, heading_path: str) -> str:
    return f"{source_type}:{source_slug}:{chunker_slugify(heading_path)}"


def doc_key_for(source_type: str, source_slug: str) -> str:
    return f"{source_type}:{source_slug}"


def stable_case_id(prefix: str, question: str, index: int) -> str:
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:6]
    return f"V2-{slugify(prefix)[:24]}-{digest}-{index}"


def load_doc_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def flatten_sections(sections: list[dict], parents: tuple[str, ...] = ()) -> list[dict]:
    rows: list[dict] = []
    for section in sections or []:
        heading = str(section.get("heading") or "").strip()
        content = str(section.get("content") or "").strip()
        path = parents + ((heading,) if heading else ())
        rows.append({"heading": heading, "path": " > ".join(path), "content": content})
        rows.extend(flatten_sections(section.get("subsections") or [], path))
    return rows


def format_content_for_prompt(flat_rows: list[dict], char_budget: int) -> str:
    blocks: list[str] = []
    used = 0
    for row in flat_rows:
        if not row.get("content"):
            continue
        block = f"[heading_path: {row['path']}]\n{row['content']}"
        if used + len(block) + 4 > char_budget and blocks:
            break
        blocks.append(block)
        used += len(block) + 4
    return "\n\n".join(blocks)


def _source_doc_entry(doc: EvidenceDoc | SourceDoc) -> dict[str, str]:
    return {"title": doc.title, "path": doc.path}


def _response_content(response) -> str:
    try:
        return str(getattr(response.choices[0].message, "content", "") or "")
    except Exception:
        return ""


def _preview(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:LLM_RESPONSE_PREVIEW_CHARS]


def _decode_first_json_object(content: str) -> dict:
    decoder = json.JSONDecoder()
    stripped = content.strip()
    try:
        value, _ = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        if start < 0:
            raise
        value, _ = decoder.raw_decode(stripped[start:])
    if not isinstance(value, dict):
        raise ValueError("LLM response JSON root is not an object")
    return value


def parse_llm_json(response, label: str) -> dict | None:
    content = _response_content(response)
    if not content.strip():
        log.warning("Empty LLM response for %s", label)
        return None
    try:
        return _decode_first_json_object(content)
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("Invalid JSON from LLM for %s: %s; preview=%r", label, exc, _preview(content))
        return None


async def ask_json(
    sem: asyncio.Semaphore,
    category: Category,
    user_prompt: str,
    *,
    temperature: float = 0.4,
) -> dict | None:
    client = get_openai()

    def call_llm():
        return client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[category]},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )

    loop = asyncio.get_event_loop()
    async with sem:
        response = await loop.run_in_executor(None, call_llm)
    return parse_llm_json(response, category)


def single_doc_user_prompt(doc: SourceDoc, category: Literal["disease_info", "drug_info"], char_budget: int) -> str:
    label = "Tên bệnh" if category == "disease_info" else "Tên thuốc"
    return (
        f"{label}: {doc.title}\n"
        f"Đường dẫn tài liệu: {doc.path}\n\n"
        "Nội dung tài liệu, mỗi đoạn có marker [heading_path: ...]:\n"
        f"{format_content_for_prompt(doc.flat_rows, char_budget)}"
    )


def build_single_doc_case(
    parsed: SingleDocCase,
    doc: SourceDoc,
    *,
    category: Literal["disease_info", "drug_info"],
    index: int,
) -> dict:
    valid_paths = {row["path"] for row in doc.flat_rows if row.get("path")}
    gold_paths = [path for path in parsed.supporting_heading_paths if path in valid_paths]
    return {
        "id": stable_case_id(f"{category}-{doc.source_slug}", parsed.question, index),
        "category": category,
        "priority": "medium",
        "question": parsed.question,
        "reference_answer": parsed.reference_answer,
        "source_docs": [_source_doc_entry(doc)],
        "acceptable_source_docs": [_source_doc_entry(doc)],
        "gold_chunks": [chunk_id_for(doc.source_type, doc.source_slug, path) for path in gold_paths],
        "gold_heading_paths": gold_paths,
        "requires_citation": True,
        "generated": True,
        "llm_generated": True,
        "generator_version": "v2",
    }


async def generate_single_doc_case(
    sem: asyncio.Semaphore,
    doc: SourceDoc,
    category: Literal["disease_info", "drug_info"],
    index: int,
    content_char_budget: int,
) -> dict | None:
    raw = await ask_json(
        sem,
        category,
        single_doc_user_prompt(doc, category, content_char_budget),
        temperature=0.35,
    )
    if raw is None:
        return None
    try:
        parsed = SingleDocOutput.model_validate(raw).test_case
    except ValidationError as exc:
        log.warning("Validation failed for %s %s: %s", category, doc.title, exc)
        return None
    return build_single_doc_case(parsed, doc, category=category, index=index)


def _title_from_doc(doc: dict, path: Path, source_type: str) -> str:
    if source_type == "disease":
        return str(doc.get("disease") or doc.get("title") or doc.get("name") or path.stem)
    return str(doc.get("title") or doc.get("name") or doc.get("generic_name_vi") or path.stem)


def collect_source_docs() -> tuple[list[SourceDoc], list[SourceDoc]]:
    diseases: list[SourceDoc] = []
    drugs: list[SourceDoc] = []

    for path in sorted(DISEASE_DIR.glob("*.json")):
        doc = load_doc_json(path)
        if not doc:
            continue
        diseases.append(SourceDoc(
            title=_title_from_doc(doc, path, "disease"),
            path=str(path),
            source_type="disease",
            source_slug=path.stem,
            flat_rows=flatten_sections(doc.get("sections") or []),
        ))

    for path in sorted(DRUG_DIR.glob("*.json")):
        doc = load_doc_json(path)
        if not doc:
            continue
        drugs.append(SourceDoc(
            title=_title_from_doc(doc, path, "drug"),
            path=str(path),
            source_type="drug",
            source_slug=path.stem,
            flat_rows=flatten_sections(doc.get("sections") or []),
        ))

    return diseases, drugs


def build_doc_index(diseases: list[SourceDoc], drugs: list[SourceDoc]) -> dict[tuple[str, str], SourceDoc]:
    index: dict[tuple[str, str], SourceDoc] = {}
    for doc in diseases + drugs:
        keys = {
            doc.source_slug,
            slugify(doc.title),
            doc.source_slug.replace("_", "-"),
            doc.source_slug.replace("-", "_"),
        }
        for key in keys:
            index[(doc.source_type, key)] = doc
    return index


def _evidence_doc(
    row: dict,
    source_type: Literal["disease", "drug"],
    role: str,
    doc_index: dict[tuple[str, str], SourceDoc],
    evidence: list[str],
) -> EvidenceDoc | None:
    title = str(row.get("name") or row.get("title") or row.get("id") or "")
    slug = str(row.get("slug") or slugify(title))
    doc = doc_index.get((source_type, slug))
    if doc is None:
        doc = doc_index.get((source_type, slug.replace("_", "-")))
    if doc is None:
        doc = doc_index.get((source_type, slug.replace("-", "_")))
    if doc is None:
        log.debug("No %s document found for Neo4j slug=%s title=%s", source_type, slug, title)
        return None
    return EvidenceDoc(
        title=doc.title,
        path=doc.path,
        source_type=doc.source_type,
        source_slug=doc.source_slug,
        role=role,
        evidence=[item for item in evidence if item],
        flat_rows=doc.flat_rows,
    )


def _query_rows(session, query: str, **params) -> list[dict]:
    return session.run(query, **params).data()


def collect_seed_symptoms(min_diseases: int = 3, max_diseases: int = 12) -> list[dict]:
    query = """
    MATCH (s:Symptom)
    OPTIONAL MATCH (d:Disease)-[:HAS_SYMPTOM]->(s)
    WITH s, count(DISTINCT d) AS disease_count
    OPTIONAL MATCH (dr:Drug)-[:MAY_CAUSE_ADR]->(s)
    WITH s, disease_count, count(DISTINCT dr) AS adr_count
    OPTIONAL MATCH (rd:Disease)-[:RED_FLAG_FOR]->(s)
    WITH s, disease_count, adr_count, count(DISTINCT rd) AS redflag_count
    WHERE disease_count >= $min_diseases
      AND disease_count <= $max_diseases
      AND (adr_count > 0 OR redflag_count > 0)
    RETURN s.id AS id, coalesce(s.name_vi, s.id) AS name,
           disease_count, adr_count, redflag_count
    ORDER BY (adr_count + redflag_count) DESC, disease_count DESC
    """
    with get_neo4j().session() as session:
        return _query_rows(
            session,
            query,
            min_diseases=min_diseases,
            max_diseases=max_diseases,
        )


def collect_co_symptoms(seed_id: str, limit: int = 3) -> list[dict]:
    query = """
    MATCH (d:Disease)-[:HAS_SYMPTOM]->(:Symptom {id: $seed_id})
    WITH collect(DISTINCT d)[..5] AS diseases
    UNWIND diseases AS d
    MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
    WHERE s.id <> $seed_id
    WITH s, count(DISTINCT d) AS freq
    WHERE freq >= 2
    RETURN s.id AS id, coalesce(s.name_vi, s.id) AS name, freq
    ORDER BY freq DESC, id
    LIMIT $limit
    """
    with get_neo4j().session() as session:
        return _query_rows(session, query, seed_id=seed_id, limit=limit)


def collect_triage_bundle(
    seed: dict,
    doc_index: dict[tuple[str, str], SourceDoc],
) -> TriageBundle | None:
    co_symptoms = collect_co_symptoms(seed["id"])
    symptoms = [{"id": seed["id"], "name": seed["name"]}] + [
        {"id": row["id"], "name": row["name"]} for row in co_symptoms
    ]
    symptom_ids = [row["id"] for row in symptoms]
    symptom_names = [row["name"] for row in symptoms]

    disease_query = """
    MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
    WHERE s.id IN $symptom_ids
    RETURN d.id AS id, d.slug AS slug, coalesce(d.name_vi, d.id) AS name,
           collect(DISTINCT coalesce(s.name_vi, s.id)) AS evidence,
           count(DISTINCT s) AS overlap
    ORDER BY overlap DESC, name
    LIMIT 5
    """
    adr_query = """
    MATCH (dr:Drug)-[r:MAY_CAUSE_ADR]->(s:Symptom)
    WHERE s.id IN $symptom_ids
    RETURN dr.id AS id, dr.slug AS slug,
           coalesce(dr.generic_name_vi, dr.name_vi, dr.id) AS name,
           collect(DISTINCT coalesce(r.text, s.name_vi, s.id)) AS evidence,
           count(DISTINCT s) AS overlap
    ORDER BY overlap DESC, name
    LIMIT 4
    """
    red_flag_query = """
    MATCH (d:Disease)-[:RED_FLAG_FOR]->(s:Symptom)
    WHERE s.id IN $symptom_ids
    RETURN d.id AS id, d.slug AS slug, coalesce(d.name_vi, d.id) AS name,
           collect(DISTINCT coalesce(s.name_vi, s.id)) AS evidence
    ORDER BY name
    LIMIT 3
    """
    with get_neo4j().session() as session:
        disease_rows = _query_rows(session, disease_query, symptom_ids=symptom_ids)
        adr_rows = _query_rows(session, adr_query, symptom_ids=symptom_ids)
        red_flag_rows = _query_rows(session, red_flag_query, symptom_ids=symptom_ids)

    disease_docs = [
        doc for row in disease_rows
        if (doc := _evidence_doc(
            row,
            "disease",
            "disease_candidate",
            doc_index,
            list(row.get("evidence") or []) + symptom_names,
        ))
    ]
    adr_drug_docs = [
        doc for row in adr_rows
        if (doc := _evidence_doc(
            row,
            "drug",
            "adr_candidate",
            doc_index,
            list(row.get("evidence") or []) + symptom_names + ["tác dụng không mong muốn"],
        ))
    ]
    red_flag_docs = [
        doc for row in red_flag_rows
        if (doc := _evidence_doc(
            row,
            "disease",
            "red_flag",
            doc_index,
            list(row.get("evidence") or []) + symptom_names,
        ))
    ]

    if not disease_docs and not adr_drug_docs:
        return None
    return TriageBundle(symptoms, disease_docs, adr_drug_docs, red_flag_docs)


def _prompt_rows_for_doc(doc: EvidenceDoc, char_budget: int = TRIAGE_DOC_CHAR_BUDGET) -> list[dict]:
    rows: list[dict] = []
    used = 0
    for row in doc.flat_rows:
        heading_path = str(row.get("path") or "").strip()
        content = str(row.get("content") or "").strip()
        if not heading_path or not content:
            continue
        excerpt = content[:TRIAGE_SECTION_CHAR_BUDGET]
        block_len = len(heading_path) + len(excerpt) + 64
        if rows and used + block_len > char_budget:
            break
        rows.append({"heading_path": heading_path, "content": excerpt})
        used += block_len
    return rows


def triage_prompt_rows_by_doc(bundle: TriageBundle) -> dict[str, list[dict]]:
    return {
        doc_key_for(doc.source_type, doc.source_slug): _prompt_rows_for_doc(doc)
        for doc in bundle.all_docs()
    }


def _format_evidence_docs(title: str, docs: list[EvidenceDoc], prompt_rows: dict[str, list[dict]]) -> str:
    if not docs:
        return f"{title}: không có"
    blocks = [title + ":"]
    for i, doc in enumerate(docs, start=1):
        doc_key = doc_key_for(doc.source_type, doc.source_slug)
        evidence = "; ".join(doc.evidence[:6]) or "không có evidence text"
        sections = []
        for row in prompt_rows.get(doc_key, []):
            sections.append(
                f"[doc_key: {doc_key} | heading_path: {row['heading_path']}]\n"
                f"{row['content']}"
            )
        section_text = "\n\n".join(sections) or "Không có đoạn tài liệu trong giới hạn prompt."
        blocks.append(
            f"{i}. {doc.title} ({doc.source_type}, {doc.role}, doc_key={doc_key})\n"
            f"Evidence KG: {evidence}\n"
            f"Các đoạn tài liệu có thể chọn làm supporting_refs:\n{section_text}"
        )
    return "\n\n".join(blocks)


def format_triage_bundle_for_prompt(bundle: TriageBundle) -> str:
    symptoms = ", ".join(f"{s['name']} ({s['id']})" for s in bundle.seed_symptoms)
    prompt_rows = triage_prompt_rows_by_doc(bundle)
    parts = [
        f"Triệu chứng đầu vào: {symptoms}",
        "Chỉ được đưa vào `supporting_refs` các cặp doc_key + heading_path xuất hiện trong marker [doc_key: ... | heading_path: ...].",
        _format_evidence_docs("Nguồn bệnh", bundle.disease_docs, prompt_rows),
        _format_evidence_docs("Nguồn thuốc / ADR", bundle.adr_drug_docs, prompt_rows),
        _format_evidence_docs("Nguồn dấu hiệu nguy hiểm", bundle.red_flag_docs, prompt_rows),
    ]
    return "\n\n".join(parts)


def validated_supporting_refs(parsed: SymptomTriageCase, bundle: TriageBundle) -> list[tuple[EvidenceDoc, str]]:
    docs_by_key = {
        doc_key_for(doc.source_type, doc.source_slug): doc
        for doc in bundle.all_docs()
    }
    prompt_paths = {
        key: {row["heading_path"] for row in rows}
        for key, rows in triage_prompt_rows_by_doc(bundle).items()
    }
    refs: list[tuple[EvidenceDoc, str]] = []
    seen: set[tuple[str, str]] = set()
    for ref in parsed.supporting_refs:
        key = (ref.doc_key, ref.heading_path)
        if key in seen:
            continue
        doc = docs_by_key.get(ref.doc_key)
        if doc is None or ref.heading_path not in prompt_paths.get(ref.doc_key, set()):
            continue
        seen.add(key)
        refs.append((doc, ref.heading_path))
    return refs


def build_symptom_triage_case(
    parsed: SymptomTriageCase,
    bundle: TriageBundle,
    *,
    index: int,
) -> dict | None:
    docs = bundle.all_docs()
    source_docs = [_source_doc_entry(doc) for doc in docs]
    supporting_refs = validated_supporting_refs(parsed, bundle)
    if not supporting_refs:
        return None
    gold_chunks = [
        chunk_id_for(doc.source_type, doc.source_slug, heading_path)
        for doc, heading_path in supporting_refs
    ]
    gold_heading_paths = [heading_path for _, heading_path in supporting_refs]
    return {
        "id": stable_case_id("symptom-triage", parsed.question, index),
        "category": "symptom_triage",
        "priority": "high" if bundle.red_flag_docs else "medium",
        "question": parsed.question,
        "reference_answer": parsed.reference_answer,
        "source_docs": source_docs,
        "acceptable_source_docs": source_docs,
        "seed_symptoms": bundle.seed_symptoms,
        "candidate_diseases": [doc.title for doc in bundle.disease_docs],
        "candidate_adr_drugs": [doc.title for doc in bundle.adr_drug_docs],
        "red_flag_sources": [doc.title for doc in bundle.red_flag_docs],
        "gold_chunks": gold_chunks,
        "gold_heading_paths": gold_heading_paths,
        "gold_supporting_refs": [
            {
                "doc_key": doc_key_for(doc.source_type, doc.source_slug),
                "heading_path": heading_path,
                "chunk_id": chunk_id,
            }
            for (doc, heading_path), chunk_id in zip(supporting_refs, gold_chunks)
        ],
        "requires_citation": True,
        "expected_behavior": [
            "state uncertainty",
            "mention plausible disease causes",
            "mention medication/ADR as possible cause",
            "ask about recent drugs, supplements, herbal medicine, and OTC medicine",
            "recommend appropriate care",
            "give urgent red flags when present",
            "avoid definitive diagnosis",
        ],
        "generated": True,
        "llm_generated": True,
        "generator_version": "v2",
    }


async def generate_symptom_triage_case(
    sem: asyncio.Semaphore,
    bundle: TriageBundle,
    index: int,
) -> dict | None:
    raw = await ask_json(
        sem,
        "symptom_triage",
        format_triage_bundle_for_prompt(bundle),
        temperature=0.55,
    )
    if raw is None:
        return None
    try:
        parsed = SymptomTriageOutput.model_validate(raw).test_case
    except ValidationError as exc:
        log.warning("Validation failed for symptom_triage bundle %d: %s", index, exc)
        return None
    return build_symptom_triage_case(parsed, bundle, index=index)


async def run(
    out_path: Path,
    seed: int,
    concurrency: int,
    content_char_budget: int,
    symptom_triage_target: int,
    disease_limit: int | None,
    drug_limit: int | None,
    categories: tuple[Category, ...] = ("disease_info", "drug_info", "symptom_triage"),
    append: bool = False,
) -> int:
    diseases, drugs = collect_source_docs()
    if disease_limit is not None:
        diseases = diseases[:disease_limit]
    if drug_limit is not None:
        drugs = drugs[:drug_limit]
    if not diseases and not drugs:
        log.error("No source documents found under %s or %s", DISEASE_DIR, DRUG_DIR)
        return 2

    if concurrency != 1:
        log.info("Ignoring --concurrency=%d; benchmark generation sends LLM requests one by one.", concurrency)

    sem = asyncio.Semaphore(1)
    rows: list[dict] = []
    next_index = count_jsonl_rows(out_path) if append else 0

    if "disease_info" in categories:
        for doc in diseases:
            row = await generate_single_doc_case(
                sem,
                doc,
                "disease_info",
                next_index,
                content_char_budget,
            )
            next_index += 1
            if row:
                rows.append(row)

    if "drug_info" in categories:
        for doc in drugs:
            row = await generate_single_doc_case(
                sem,
                doc,
                "drug_info",
                next_index,
                content_char_budget,
            )
            next_index += 1
            if row:
                rows.append(row)

    doc_index = build_doc_index(diseases, drugs)
    triage_bundles: list[TriageBundle] = []
    if "symptom_triage" in categories and symptom_triage_target > 0:
        rng = random.Random(seed)
        seed_symptoms = collect_seed_symptoms()
        rng.shuffle(seed_symptoms)
        for symptom in seed_symptoms:
            bundle = collect_triage_bundle(symptom, doc_index)
            if bundle is None:
                continue
            triage_bundles.append(bundle)
            if len(triage_bundles) >= symptom_triage_target:
                break

        for bundle in triage_bundles:
            row = await generate_symptom_triage_case(sem, bundle, next_index)
            next_index += 1
            if row:
                rows.append(row)

    if append:
        append_jsonl(out_path, rows)
    else:
        write_jsonl(out_path, rows)
    log.info(
        "%s %d cases to %s (categories=%s, disease=%d, drug=%d, symptom_triage=%d requested/%d bundled)",
        "Appended" if append else "Wrote",
        len(rows),
        out_path,
        ",".join(categories),
        len(diseases),
        len(drugs),
        symptom_triage_target,
        len(triage_bundles),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate v2 benchmark JSONL.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=MAX_CONCURRENCY,
        help="Deprecated compatibility option; LLM requests are always sent one by one.",
    )
    parser.add_argument("--content-budget", type=int, default=CONTENT_CHAR_BUDGET)
    parser.add_argument("--symptom-triage-target", type=int, default=DEFAULT_SYMPTOM_TRIAGE_TARGET)
    parser.add_argument("--disease-limit", type=int, default=None)
    parser.add_argument("--drug-limit", type=int, default=None)
    parser.add_argument(
        "--category",
        action="append",
        choices=["disease_info", "drug_info", "symptom_triage"],
        help="Generate only this category. Repeat for multiple categories. Default: all categories.",
    )
    parser.add_argument("--append", action="store_true", help="Append rows to --out instead of overwriting it.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(
        run(
            out_path=args.out,
            seed=args.seed,
            concurrency=args.concurrency,
            content_char_budget=args.content_budget,
            symptom_triage_target=args.symptom_triage_target,
            disease_limit=args.disease_limit,
            drug_limit=args.drug_limit,
            categories=tuple(args.category) if args.category else ("disease_info", "drug_info", "symptom_triage"),
            append=args.append,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
