"""
generator.py
------------
Build prompt from context + call OpenAI for the final answer.
"""

from __future__ import annotations

import logging
import re
import time

from src.chat.clients import get_openai
from src.chat.llm.mini import message_text
from src.chat.prompts import GENERATOR_SYSTEM
from src.chat.replies import TECHNICAL_ERROR_REPLY
from src.chat.retrieval.types import Hit
from src.chat.timing import elapsed_ms
from src.config import MODEL, MODEL_MAX_TOKENS

log = logging.getLogger(__name__)

DRUG_URL_TEMPLATE = "https://trungtamthuoc.com/hoat-chat/{slug}"
NO_DATA_REPLY = ("Tôi không tìm thấy thông tin phù hợp trong tài liệu. "
                 "Bạn vui lòng hỏi cụ thể hơn hoặc tham khảo ý kiến bác sĩ.")
_CITATION_RE = re.compile(r"\[([0-9][0-9,\-\s]*)\]")


def _source_key(h: Hit) -> tuple:
    return (h.source_type, h.source_slug)


def _dedupe(hits: list[Hit]) -> tuple[list[Hit], list[int]]:
    seen: dict[tuple, int] = {}
    unique: list[Hit] = []
    cite_idx: list[int] = []
    for h in hits:
        k = _source_key(h)
        if k not in seen:
            unique.append(h)
            seen[k] = len(unique)
        cite_idx.append(seen[k])
    return unique, cite_idx


def _drug_label(h: Hit) -> str:
    url = DRUG_URL_TEMPLATE.format(slug=h.source_slug)
    return f"Dược thư Quốc gia 2022 - [{h.source_name}]({url})"


def _disease_label(h: Hit) -> str:
    chapter = (h.metadata or {}).get("chapter", "")
    return (
        "Hướng dẫn chẩn đoán và điều trị - Bệnh viện Bạch Mai - "
        f"{chapter} - {h.source_name}".rstrip(" -")
    )


def _format_context(hits: list[Hit], cite_idx: list[int]) -> str:
    blocks = []
    for i, h in enumerate(hits):
        header = f"[{cite_idx[i]}] {h.source_name}"
        if h.heading_path:
            header += f" — {h.heading_path}"
        blocks.append(f"{header}\n{h.text}")
    return "\n\n---\n\n".join(blocks)


def _cited_source_numbers(text: str, max_index: int) -> list[int]:
    cited: list[int] = []
    seen: set[int] = set()

    def add(num: int) -> None:
        if 1 <= num <= max_index and num not in seen:
            seen.add(num)
            cited.append(num)

    for match in _CITATION_RE.finditer(text):
        for part in match.group(1).replace(" ", "").split(","):
            if "-" in part:
                start, end = part.split("-", 1)
                if start.isdigit() and end.isdigit():
                    for num in range(int(start), int(end) + 1):
                        add(num)
            elif part.isdigit():
                add(int(part))
    return cited


def _format_sources(unique: list[Hit], source_numbers: list[int] | None = None) -> str:
    lines = []
    numbers = source_numbers or list(range(1, len(unique) + 1))
    for i in numbers:
        h = unique[i - 1]
        label = _drug_label(h) if h.source_type == "drug" else _disease_label(h)
        lines.append(f"[{i}] {label}")
    return "\n".join(lines)


def _format_patient(patient: dict | None) -> str:
    """Format patient session state for prompt."""
    if not patient:
        return ""
    parts = []
    if patient.get("symptoms"):
        lines = []
        for s in patient["symptoms"]:
            name = s.get("name", "")
            slots = [f"{k}: {s[k]}" for k in ("onset", "severity", "pattern", "associated") if s.get(k)]
            line = f"- {name}"
            if slots:
                line += f" ({'; '.join(slots)})"
            lines.append(line)
        parts.append("Triệu chứng người bệnh:\n" + "\n".join(lines))
    if patient.get("medications"):
        parts.append("Thuốc đang dùng: " + ", ".join(patient["medications"]))
    if patient.get("candidate_diseases"):
        names = [d.get("name", "") for d in patient["candidate_diseases"][:5]]
        parts.append("Bệnh nghi ngờ (shortlist): " + ", ".join(filter(None, names)))
    return "\n\n".join(parts)


def generate(
    question: str,
    hits: list[Hit],
    kg_text: str = "",
    patient: dict | None = None,
) -> str:
    if not hits and not kg_text:
        return NO_DATA_REPLY

    unique, cite_idx = _dedupe(hits)
    context = _format_context(hits, cite_idx)
    patient_text = _format_patient(patient)

    prompt_parts = [f"Câu hỏi: {question}\n"]
    if patient_text:
        prompt_parts.append(f"Thông tin người bệnh đã cung cấp:\n{patient_text}\n")
    if kg_text:
        prompt_parts.append(f"Thông tin từ Knowledge Graph:\n{kg_text}\n")
    if context:
        prompt_parts.append(f"Tài liệu tham khảo:\n{context}\n")
    prompt_parts.append("Hãy trả lời câu hỏi trên dựa vào tài liệu và thông tin người bệnh.")
    prompt_user = "\n".join(prompt_parts)

    start = time.perf_counter()
    try:
        response = get_openai().chat.completions.create(
            model=MODEL,
            max_tokens=MODEL_MAX_TOKENS,
            messages=[
                {"role": "system", "content": GENERATOR_SYSTEM},
                {"role": "user", "content": prompt_user},
            ],
            # reasoning_effort="high",
            extra_body={"thinking": {"type": "disabled"}},
        )
    except Exception as e:
        log.warning("Generator LLM call failed: %s", e)
        return TECHNICAL_ERROR_REPLY
    finally:
        log.info("llm timing stage=generator model=%s ms=%.1f",
                 MODEL, elapsed_ms(start))

    answer = message_text(response).strip()
    if not answer:
        log.warning("Generator LLM returned an empty response")
        return TECHNICAL_ERROR_REPLY
    if unique:
        source_numbers = _cited_source_numbers(answer, len(unique))
        return f"{answer}\n\nNguồn:\n{_format_sources(unique, source_numbers or None)}"
    return answer
