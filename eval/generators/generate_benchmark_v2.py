#!/usr/bin/env python3
"""Generate v2 medical QA benchmark cases.

This generator intentionally separates category prompts:
- disease_info: one question per disease document, explicit disease name.
- drug_info: one question per drug document, explicit drug name.

Usage:
    PYTHONPATH=. python3 eval/generators/generate_benchmark_v2.py
    PYTHONPATH=. python3 eval/generators/generate_benchmark_v2.py --out eval/datasets/medical_qa_benchmark_v2.jsonl
    PYTHONPATH=. python3 eval/generators/generate_benchmark_v2.py --category drug_info --append
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.chat.clients import get_openai
from src.config import MODEL

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DEFAULT_OUT = PROJECT_ROOT / "eval" / "datasets" / "medical_qa_benchmark_v2.jsonl"
DISEASE_DIR = PROJECT_ROOT / "outputs" / "bachmai" / "final"
DRUG_DIR = PROJECT_ROOT / "outputs" / "otc_drugs" / "final_json"

CONTENT_CHAR_BUDGET = 12000
MAX_CONCURRENCY = 1
LLM_RESPONSE_PREVIEW_CHARS = 500

Category = Literal["disease_info", "drug_info"]

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
}


@dataclass(frozen=True)
class SourceDoc:
    title: str
    path: str
    source_type: Literal["disease", "drug"]
    source_slug: str
    flat_rows: list[dict]


class SingleDocCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    reference_answer: str
    supporting_heading_paths: list[str] = Field(default_factory=list)


class SingleDocOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_case: SingleDocCase


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


def _source_doc_entry(doc: SourceDoc) -> dict[str, str]:
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


async def run(
    out_path: Path,
    concurrency: int,
    content_char_budget: int,
    disease_limit: int | None,
    drug_limit: int | None,
    categories: tuple[Category, ...] = ("disease_info", "drug_info"),
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

    if append:
        append_jsonl(out_path, rows)
    else:
        write_jsonl(out_path, rows)
    log.info(
        "%s %d cases to %s (categories=%s, disease=%d, drug=%d)",
        "Appended" if append else "Wrote",
        len(rows),
        out_path,
        ",".join(categories),
        len(diseases),
        len(drugs),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate v2 benchmark JSONL.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=MAX_CONCURRENCY,
        help="Deprecated compatibility option; LLM requests are always sent one by one.",
    )
    parser.add_argument("--content-budget", type=int, default=CONTENT_CHAR_BUDGET)
    parser.add_argument("--disease-limit", type=int, default=None)
    parser.add_argument("--drug-limit", type=int, default=None)
    parser.add_argument(
        "--category",
        action="append",
        choices=["disease_info", "drug_info"],
        help="Generate only this category. Repeat for multiple categories. Default: all categories.",
    )
    parser.add_argument("--append", action="store_true", help="Append rows to --out instead of overwriting it.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(
        run(
            out_path=args.out,
            concurrency=args.concurrency,
            content_char_budget=args.content_budget,
            disease_limit=args.disease_limit,
            drug_limit=args.drug_limit,
            categories=tuple(args.category) if args.category else ("disease_info", "drug_info"),
            append=args.append,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
