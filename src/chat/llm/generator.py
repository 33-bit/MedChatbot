"""
generator.py
------------
Build prompt from context + call OpenAI for the final answer.
"""

from __future__ import annotations

import json
import logging
import re
import time

from src.chat.clients import get_openai
from src.chat.llm.mini import chat_completion_extra_kwargs, message_text
from src.chat.prompts import GENERATOR_SYSTEM
from src.chat.replies import TECHNICAL_ERROR_REPLY
from src.chat.retrieval.types import Hit
from src.chat.storage.seed_doctors import SPECIALTIES
from src.chat.timing import elapsed_ms
from src.config import BASE_URL, MODEL, MODEL_MAX_TOKENS

log = logging.getLogger(__name__)

DRUG_URL_TEMPLATE = "https://trungtamthuoc.com/hoat-chat/{slug}"
NO_DATA_REPLY = ("Tôi không tìm thấy thông tin phù hợp trong tài liệu. "
                 "Bạn vui lòng hỏi cụ thể hơn hoặc tham khảo ý kiến bác sĩ.")
_CITATION_RE = re.compile(r"\[([0-9][0-9,\-\s]*)\]")
_HORIZONTAL_RULE_RE = re.compile(r"(?m)^\s*-{3,}\s*$")
_DOCTOR_SPECIALTY_OPTIONS = ", ".join(SPECIALTIES)


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


def _extract_usage(response) -> dict | None:
    usage = getattr(response, "usage", None)
    if not usage:
        return None
    pt = getattr(usage, "prompt_tokens", None)
    ct = getattr(usage, "completion_tokens", None)
    tt = getattr(usage, "total_tokens", None)
    if pt is None and ct is None and tt is None:
        return None
    return {
        "prompt_tokens": pt or 0,
        "completion_tokens": ct or 0,
        "total_tokens": tt if tt is not None else (pt or 0) + (ct or 0),
    }


def _clean_answer_format(text: str) -> str:
    text = _HORIZONTAL_RULE_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_payload_answer(answer: str) -> str:
    return answer.replace("\\n", "\n").strip()


def _parse_answer_payload(text: str) -> tuple[str, bool, str | None]:
    payload_text = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", payload_text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        payload_text = fence.group(1).strip()
    if payload_text.casefold().startswith("json\n"):
        payload_text = payload_text.split("\n", 1)[1].strip()
    try:
        payload = json.loads(payload_text)
    except Exception:
        # Some models return JSON-looking text with raw newlines inside the
        # answer string. Extract that shape so raw JSON never leaks to users.
        match = re.search(
            r'"answer"\s*:\s*"(.*)"\s*,\s*"doctor_handoff_recommended"\s*:\s*(true|false)',
            payload_text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match:
            specialty_match = re.search(
                r'"doctor_specialty"\s*:\s*(null|"(.*?)")',
                payload_text,
                flags=re.DOTALL | re.IGNORECASE,
            )
            specialty = (
                specialty_match.group(2).strip()
                if specialty_match and specialty_match.group(2)
                else None
            )
            return (
                _normalize_payload_answer(match.group(1)),
                match.group(2).casefold() == "true",
                specialty,
            )
        return text, False, None
    if not isinstance(payload, dict):
        return text, False, None
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return text, False, None
    specialty = payload.get("doctor_specialty")
    if not isinstance(specialty, str) or not specialty.strip():
        specialty = None
    else:
        specialty = specialty.strip()
    return (
        _normalize_payload_answer(answer),
        bool(payload.get("doctor_handoff_recommended")),
        specialty,
    )


def generate(
    question: str,
    hits: list[Hit],
    kg_text: str = "",
    patient: dict | None = None,
    *,
    return_meta: bool = False,
):
    if not hits and not kg_text:
        if return_meta:
            return NO_DATA_REPLY, {
                "usage": None,
                "model": MODEL,
                "no_data": True,
                "doctor_handoff_recommended": True,
                "doctor_specialty": None,
            }
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
    if return_meta:
        prompt_parts.append(
            "Hãy trả lời câu hỏi trên dựa vào tài liệu và thông tin người bệnh. "
            "Trả về JSON hợp lệ, không bọc markdown/code fence, với đúng 3 trường: "
            "answer (chuỗi trả lời cho người bệnh; escape xuống dòng bằng \\n), "
            "doctor_handoff_recommended (boolean), và doctor_specialty "
            f"(một trong: {_DOCTOR_SPECIALTY_OPTIONS}; hoặc null). "
            "Đặt doctor_handoff_recommended=true nếu câu trả lời còn thiếu chắc chắn, không đủ dữ kiện, "
            "không thể kết luận an toàn, hoặc cần bác sĩ khám trực tiếp để xác minh. "
            "Khi doctor_handoff_recommended=true, chọn doctor_specialty phù hợp nhất với tình huống; "
            "nếu false thì đặt doctor_specialty=null."
        )
    else:
        prompt_parts.append("Hãy trả lời câu hỏi trên dựa vào tài liệu và thông tin người bệnh.")
    prompt_user = "\n".join(prompt_parts)

    start = time.perf_counter()
    response = None
    try:
        response = get_openai().chat.completions.create(
            model=MODEL,
            max_tokens=MODEL_MAX_TOKENS,
            messages=[
                {"role": "system", "content": GENERATOR_SYSTEM},
                {"role": "user", "content": prompt_user},
            ],
            # reasoning_effort="high",
            **chat_completion_extra_kwargs(BASE_URL, model=MODEL),
        )
    except Exception as e:
        log.warning("Generator LLM call failed: %s", e)
        if return_meta:
            return TECHNICAL_ERROR_REPLY, {
                "usage": None,
                "model": MODEL,
                "error": repr(e),
                "doctor_specialty": None,
            }
        return TECHNICAL_ERROR_REPLY
    finally:
        log.info("llm timing stage=generator model=%s ms=%.1f",
                 MODEL, elapsed_ms(start))

    usage = _extract_usage(response)
    raw_answer = _clean_answer_format(message_text(response))
    answer, doctor_offer, doctor_specialty = (
        _parse_answer_payload(raw_answer)
        if return_meta
        else (raw_answer, False, None)
    )
    answer = _clean_answer_format(answer)
    if not answer:
        log.warning("Generator LLM returned an empty response")
        if return_meta:
            return TECHNICAL_ERROR_REPLY, {
                "usage": usage,
                "model": MODEL,
                "empty": True,
                "doctor_specialty": None,
            }
        return TECHNICAL_ERROR_REPLY
    if unique:
        source_numbers = _cited_source_numbers(answer, len(unique))
        rendered = f"{answer}\n\nNguồn:\n{_format_sources(unique, source_numbers or None)}"
    else:
        rendered = answer
    if return_meta:
        return rendered, {
            "usage": usage,
            "model": MODEL,
            "doctor_handoff_recommended": doctor_offer,
            "doctor_specialty": doctor_specialty,
        }
    return rendered
