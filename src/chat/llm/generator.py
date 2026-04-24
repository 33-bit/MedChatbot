"""
generator.py
------------
Build prompt from context + call Grok (xai_sdk) for the final answer.
"""

from __future__ import annotations

from xai_sdk.chat import system, user

from src.chat.clients import get_xai
from src.chat.prompts import GENERATOR_SYSTEM
from src.chat.retrieval.types import Hit
from src.config import MODEL

DRUG_URL_TEMPLATE = "https://trungtamthuoc.com/hoat-chat/{slug}"


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


def _format_sources(unique: list[Hit]) -> str:
    lines = []
    for i, h in enumerate(unique, 1):
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
    if not hits and not kg_text and not patient:
        return ("Tôi không tìm thấy thông tin phù hợp trong tài liệu. "
                "Bạn vui lòng hỏi cụ thể hơn hoặc tham khảo ý kiến bác sĩ.")

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

    chat = get_xai().chat.create(model=MODEL)
    chat.append(system(GENERATOR_SYSTEM))
    chat.append(user(prompt_user))
    response = chat.sample()

    answer = (response.content or "").strip()
    if unique:
        return f"{answer}\n\nNguồn:\n{_format_sources(unique)}"
    return answer
