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
import unicodedata
from typing import Literal
from urllib.parse import quote

from src.chat.clients import get_openai
from src.chat.llm.mini import chat_completion_extra_kwargs, message_text
from src.chat.context.resolver import format_subject_address
from src.chat.prompts import (
    DISEASE_INFO_INSTRUCTIONS,
    DRUG_INFO_INSTRUCTIONS,
    GENERATOR_SYSTEM,
    HEALTH_INSURANCE_INFO_INSTRUCTIONS,
    SYMPTOM_OR_CARE_INSTRUCTIONS,
)
from src.chat.replies import TECHNICAL_ERROR_REPLY
from src.chat.retrieval.types import Hit
from src.chat.storage.seed_doctors import SPECIALTIES
from src.chat.timing import elapsed_ms
from src.config import BASE_URL, MODEL, MODEL_MAX_TOKENS, PUBLIC_BASE_URL

log = logging.getLogger(__name__)

AnswerDomain = Literal[
    "symptom_or_care",
    "disease_info",
    "drug_info",
    "health_insurance_info",
]

DRUG_URL_TEMPLATE = "https://trungtamthuoc.com/hoat-chat/{slug}"
NO_DATA_REPLY = ("Tôi không tìm thấy thông tin phù hợp trong tài liệu. "
                 "Bạn vui lòng hỏi cụ thể hơn hoặc tham khảo ý kiến bác sĩ.")
_CITATION_RE = re.compile(r"\[([0-9][0-9,\-\s]*)\]")
_HORIZONTAL_RULE_RE = re.compile(r"(?m)^\s*-{3,}\s*$")
_DOCTOR_SPECIALTY_OPTIONS = ", ".join(SPECIALTIES)
_DOSE_TOKEN_RE = re.compile(
    r"\b\d+(?:[,.]\d+)?(?:\s*[-–]\s*\d+(?:[,.]\d+)?)?\s*"
    r"(?:mg|g|mcg|ug|ml|%|lần(?:/ngày)?|ngày|tuần|tháng|năm|viên|giọt|ống)\b",
    flags=re.IGNORECASE,
)
_ANSWER_DOMAIN_INSTRUCTIONS: dict[AnswerDomain, str] = {
    "symptom_or_care": SYMPTOM_OR_CARE_INSTRUCTIONS,
    "disease_info": DISEASE_INFO_INSTRUCTIONS,
    "drug_info": DRUG_INFO_INSTRUCTIONS,
    "health_insurance_info": HEALTH_INSURANCE_INFO_INSTRUCTIONS,
}
_DISEASE_CONTEXT_TERMS = (
    "benh",
    "trieu chung",
    "chan doan",
    "nguyen nhan",
    "bien chung",
    "giai doan",
    "phan loai",
    "co lay",
)
_TREATMENT_OR_MEDICINE_TERMS = (
    "dieu tri",
    "thuoc",
    "lieu",
    "lieu dung",
    "cach dung",
    "uong",
    "boi",
    "tiem",
    "phac do",
    "khang sinh",
)
_DANGER_OR_CARE_TERMS = (
    "nguy hiem",
    "dau hieu nang",
    "cap cuu",
    "goi 115",
    "di kham",
    "kham ngay",
    "dieu tri",
    "xu tri",
    "nghiem trong",
    "nang hon",
    "canh bao",
    "chuyen bien xau",
)
_GENERIC_DISEASE_BOILERPLATE = (
    "chua the chan doan chac chan qua chat",
    "khong the chan doan chac chan qua chat",
    "khong the ket luan chan doan qua chat",
)
_GENERIC_DRUG_WARNING_TERMS = (
    "phu nu co thai",
    "mang thai",
    "cho con bu",
    "benh gan",
    "benh than",
    "gan/than",
    "suy gan",
    "suy than",
)
_DRUG_EMERGENCY_TERMS = (
    "goi 115",
    "cap cuu",
    "sot cao >39.5",
    "sot cao > 39.5",
    "sot cao tren 39.5",
    "sot cao >39,5",
    "sot cao > 39,5",
)
_DRUG_USAGE_QUERY_TERMS = (
    "lieu",
    "lieu dung",
    "cach dung",
    "duong dung",
    "dung the nao",
    "uong",
    "boi",
    "tiem",
    "ngay may lan",
    "bao nhieu",
)
_DRUG_USAGE_LINE_TERMS = (
    "lieu",
    "lieu dung",
    "cach dung",
    "duong dung",
    "uong",
    "boi",
    "tiem",
    "ngay",
    "lan",
    "mg",
    "ml",
    "%",
)
_INSUFFICIENT_USAGE_TERMS = (
    "khong du thong tin",
    "khong co du thong tin",
    "tai lieu duoc cung cap khong du",
)


def _source_key(h: Hit) -> tuple:
    if h.source_type == "health_insurance":
        return (
            h.source_type,
            h.source_slug,
            (h.metadata or {}).get("article_number", ""),
        )
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
    label = (
        "Hướng dẫn chẩn đoán và điều trị - Bệnh viện Bạch Mai - "
        f"{chapter} - {h.source_name}".rstrip(" -")
    )
    if not h.source_slug:
        return label
    path = f"/sources/bachmai/{quote(h.source_slug, safe='')}.pdf"
    url = f"{PUBLIC_BASE_URL.rstrip('/')}{path}" if PUBLIC_BASE_URL else path
    return f"[{label}]({url})"


def _health_insurance_label(h: Hit) -> str:
    metadata = h.metadata or {}
    article_number = metadata.get("article_number", "")
    article_title = metadata.get("article_title", "")
    label = "Luật Bảo hiểm y tế - 22/VBHN-VPQH"
    if article_number:
        label += f" - Điều {article_number}"
    if article_title:
        label += f". {article_title}"
    path = "/sources/health-insurance/22-vbhn-vpqh.pdf"
    page = metadata.get("page_start")
    if page:
        path += f"#page={page}"
    url = f"{PUBLIC_BASE_URL.rstrip('/')}{path}" if PUBLIC_BASE_URL else path
    return f"[{label}]({url})"


def _format_context(hits: list[Hit], cite_idx: list[int]) -> str:
    blocks = []
    for i, h in enumerate(hits):
        header = f"[{cite_idx[i]}] {h.source_name}"
        if h.heading_path:
            header += f" — {h.heading_path}"
        blocks.append(f"{header}\n{h.text}")
    return "\n\n---\n\n".join(blocks)


def _normalize_text(text: str) -> str:
    text = (text or "").casefold().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    normalized = _normalize_text(text)
    return any(term in normalized for term in terms)


def _source_text(hits: list[Hit]) -> str:
    return "\n".join(h.text for h in hits if h.text)


def _source_contains_any(hits: list[Hit], terms: tuple[str, ...]) -> bool:
    return _contains_any(_source_text(hits), terms)


def _question_asks_drug_usage_detail(question: str) -> bool:
    return _contains_any(question, _DRUG_USAGE_QUERY_TERMS)


def _source_has_drug_usage_detail(hits: list[Hit]) -> bool:
    drug_text = "\n".join(hit.text for hit in hits if hit.source_type == "drug")
    if _DOSE_TOKEN_RE.search(drug_text):
        return True
    return _contains_any(drug_text, _DRUG_USAGE_LINE_TERMS)


def _question_asks_disease_context(question: str) -> bool:
    return _contains_any(question, _DISEASE_CONTEXT_TERMS)


def _question_asks_treatment_or_medicine(question: str) -> bool:
    return _contains_any(question, _TREATMENT_OR_MEDICINE_TERMS)


def _question_asks_danger_or_care(question: str) -> bool:
    return _contains_any(question, _DANGER_OR_CARE_TERMS)


def _filter_hits_for_answer_domain(
    question: str,
    hits: list[Hit],
    answer_domain: AnswerDomain,
) -> list[Hit]:
    if answer_domain == "health_insurance_info":
        scoped = [hit for hit in hits if hit.source_type == "health_insurance"]
        return scoped or hits

    if answer_domain == "drug_info":
        drug_hits = [hit for hit in hits if hit.source_type == "drug"]
        if not drug_hits:
            return hits
        if _question_asks_disease_context(question):
            disease_hits = [hit for hit in hits if hit.source_type == "disease"]
            other_hits = [
                hit for hit in hits
                if hit.source_type not in {"drug", "disease"}
            ]
            return drug_hits + disease_hits + other_hits
        return drug_hits

    if answer_domain == "disease_info":
        disease_hits = [hit for hit in hits if hit.source_type == "disease"]
        if not disease_hits:
            return hits
        if _question_asks_treatment_or_medicine(question):
            drug_hits = [hit for hit in hits if hit.source_type == "drug"]
            other_hits = [
                hit for hit in hits
                if hit.source_type not in {"drug", "disease"}
            ]
            return disease_hits + drug_hits + other_hits
        return disease_hits

    return hits


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
        if h.source_type == "drug":
            label = _drug_label(h)
        elif h.source_type == "health_insurance":
            label = _health_insurance_label(h)
        else:
            label = _disease_label(h)
        lines.append(f"[{i}] {label}")
    return "\n".join(lines)


def _ensure_health_insurance_article_citations(
    answer: str,
    unique: list[Hit],
    source_numbers: list[int],
) -> tuple[str, list[int]]:
    """Attach missing citations when a BHYT answer names a retrieved article.

    Legal answers often mention "Điều 12" while citing a different article
    that cross-references it. For eval and user trust, if the retrieved context
    contains that named article, cite it explicitly.
    """
    if not any(hit.source_type == "health_insurance" for hit in unique):
        return answer, source_numbers

    cited = set(source_numbers)
    updated = answer
    for index, hit in enumerate(unique, start=1):
        if hit.source_type != "health_insurance" or index in cited:
            continue
        article = str((hit.metadata or {}).get("article_number") or "").strip()
        if not article:
            continue
        pattern = re.compile(
            rf"(Điều\s+{re.escape(article)})(?!\d)(?!\s*\[[0-9,\-\s]+\])",
            flags=re.IGNORECASE,
        )
        updated, count = pattern.subn(rf"\1 [{index}]", updated, count=1)
        if count:
            cited.add(index)

    return updated, sorted(cited)


def _format_patient(patient: dict | None) -> str:
    """Format patient session state for prompt."""
    if not patient:
        return ""
    if "safety_profile" in patient or "relevant_facts" in patient:
        return _format_context_bundle(patient)
    # IPS-inspired medical profile: when present, prefer the cautious text
    # the projection emitted. Falls back to the legacy session formatter.
    if isinstance(patient, dict) and patient.get("medical_profile_text"):
        return str(patient["medical_profile_text"])
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


def _format_context_bundle(bundle: dict) -> str:
    parts: list[str] = []
    subject = bundle.get("subject")
    if isinstance(subject, dict) and subject.get("id"):
        subject_label = format_subject_address(subject)
        parts.append(
            f"Chủ thể y tế: {subject_label}. Luôn gọi đúng người này trong câu trả lời."
        )

    safety = bundle.get("safety_profile") or []
    if safety:
        parts.append("Hồ sơ an toàn:\n" + "\n".join(
            f"- {_format_profile_fact(fact)}" for fact in safety[:8]
        ))
    relevant = bundle.get("relevant_facts") or []
    if relevant:
        parts.append("Dữ kiện liên quan:\n" + "\n".join(
            f"- {_format_profile_fact(fact)}" for fact in relevant[:12]
        ))

    active_case = bundle.get("active_case")
    if isinstance(active_case, dict) and active_case.get("symptoms"):
        symptoms = [
            symptom.get("name") or symptom.get("symptom_id")
            for symptom in active_case["symptoms"]
            if isinstance(symptom, dict)
        ]
        if symptoms:
            parts.append("Triệu chứng trong ca hiện tại: " + ", ".join(symptoms))

    reference_turns = bundle.get("reference_turns") or []
    if reference_turns:
        lines = [
            f"- {turn.get('role', '')}: {turn.get('content', '')}"
            for turn in reference_turns[-5:]
            if isinstance(turn, dict) and turn.get("content")
        ]
        if lines:
            parts.append("Lượt hội thoại tham chiếu:\n" + "\n".join(lines))
    return "\n\n".join(parts)


def _format_profile_fact(fact: object) -> str:
    if not isinstance(fact, dict):
        return str(fact)
    entity = fact.get("entity_id") or fact.get("fact_type") or "dữ kiện"
    attribute = fact.get("attribute") or "value"
    value = fact.get("value")
    if isinstance(value, dict):
        value_text = ", ".join(f"{key}={item}" for key, item in value.items())
    else:
        value_text = str(value)
    temporal = fact.get("temporal_status")
    suffix = f" ({temporal})" if temporal else ""
    return f"{entity}: {attribute}={value_text}{suffix}"


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


def _strip_citation_markers(text: str) -> str:
    return _CITATION_RE.sub("", text)


def _compact_for_match(text: str) -> str:
    return re.sub(r"\s+", "", _normalize_text(text))


def _dose_tokens(text: str) -> list[str]:
    text = _strip_citation_markers(text)
    return [match.group(0) for match in _DOSE_TOKEN_RE.finditer(text)]


def _has_unsupported_dose_token(line: str, hits: list[Hit]) -> bool:
    tokens = _dose_tokens(line)
    if not tokens:
        return False
    source_compact = _compact_for_match(_source_text(hits))
    return any(_compact_for_match(token) not in source_compact for token in tokens)


def _line_has_generic_drug_warning(line: str, hits: list[Hit]) -> bool:
    normalized = _normalize_text(line)
    if any(term in normalized for term in _DRUG_EMERGENCY_TERMS):
        return not _source_contains_any(hits, _DRUG_EMERGENCY_TERMS)
    warning_terms = tuple(
        term for term in _GENERIC_DRUG_WARNING_TERMS
        if term in normalized
    )
    return bool(warning_terms) and not _source_contains_any(hits, warning_terms)


def _line_has_generic_disease_boilerplate(line: str) -> bool:
    normalized = _normalize_text(line)
    return any(term in normalized for term in _GENERIC_DISEASE_BOILERPLATE)


def _answer_says_insufficient_drug_usage(answer: str) -> bool:
    normalized = _normalize_text(answer)
    if not any(term in normalized for term in _INSUFFICIENT_USAGE_TERMS):
        return False
    return any(term in normalized for term in ("lieu", "cach dung", "duong dung", "dung"))


def _drug_usage_fallback_answer(hits: list[Hit], cite_idx: list[int]) -> str | None:
    lines: list[str] = []
    seen: set[str] = set()
    for hit, source_number in zip(hits, cite_idx):
        if hit.source_type != "drug":
            continue
        added_from_hit = 0
        for raw_line in hit.text.splitlines():
            line = raw_line.strip(" -*•\t")
            if len(line) < 8:
                continue
            normalized = _normalize_text(line)
            has_usage_term = any(term in normalized for term in _DRUG_USAGE_LINE_TERMS)
            has_dose_token = bool(_DOSE_TOKEN_RE.search(line))
            if not (has_usage_term or has_dose_token):
                continue
            if ">" in line:
                continue
            compact = _compact_for_match(line)
            if compact in seen:
                continue
            seen.add(compact)
            lines.append(f"- {line} [{source_number}]")
            added_from_hit += 1
            if added_from_hit >= 3 or len(lines) >= 7:
                break
        if len(lines) >= 7:
            break
    if not lines:
        return None
    return (
        "Theo chuyên luận thuốc được truy xuất:\n"
        + "\n".join(lines)
        + "\n\nLưu ý: dùng thuốc theo đơn hoặc hướng dẫn của bác sĩ/dược sĩ."
    )


def _strip_triage_section(answer: str) -> str:
    lines = answer.splitlines()
    kept: list[str] = []
    skipping = False
    for line in lines:
        normalized = _normalize_text(line.strip("*#:- "))
        starts_triage_section = (
            "khi nao can di kham ngay" in normalized
            or "dau hieu nguy hiem" in normalized
            or "can di kham ngay" in normalized
            or "goi cap cuu" in normalized
        )
        if starts_triage_section:
            skipping = True
            continue
        if skipping:
            stripped = line.strip()
            normalized_line = _normalize_text(stripped)
            looks_like_next_heading = (
                stripped
                and not stripped.startswith(("-", "*", "+"))
                and len(stripped) <= 90
                and not any(
                    term in normalized_line
                    for term in (
                        "goi 115",
                        "cap cuu",
                        "di kham",
                        "kham ngay",
                        "sot cao",
                        "kho tho",
                        "lo mo",
                    )
                )
            )
            if looks_like_next_heading:
                skipping = False
            else:
                continue
        if not _line_has_generic_disease_boilerplate(line):
            kept.append(line)
    return "\n".join(kept)


def enforce_info_answer_contract(
    answer: str,
    hits: list[Hit],
    answer_domain: AnswerDomain,
    question: str = "",
) -> str:
    if answer_domain == "drug_info":
        removed_dose_claim = False
        kept: list[str] = []
        for line in answer.splitlines():
            if _line_has_generic_drug_warning(line, hits):
                continue
            if _has_unsupported_dose_token(line, hits):
                removed_dose_claim = True
                continue
            kept.append(line)
        answer = "\n".join(kept)
        if removed_dose_claim:
            answer = _clean_answer_format(
                answer
                + "\n\nLưu ý: Tài liệu được cung cấp không đủ thông tin để xác nhận chi tiết liều hoặc cách dùng vừa hỏi."
            )
        return _clean_answer_format(answer)

    if answer_domain == "disease_info":
        kept = [
            line for line in answer.splitlines()
            if not _line_has_generic_disease_boilerplate(line)
        ]
        answer = "\n".join(kept)
        if not _question_asks_danger_or_care(question):
            answer = _strip_triage_section(answer)
        return _clean_answer_format(answer)

    return answer


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
    answer_domain: AnswerDomain = "symptom_or_care",
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

    context_hits = _filter_hits_for_answer_domain(question, hits, answer_domain)
    unique, cite_idx = _dedupe(context_hits)
    context = _format_context(context_hits, cite_idx)
    patient_text = _format_patient(patient)
    system_prompt = "\n\n".join(
        (
            GENERATOR_SYSTEM,
            _ANSWER_DOMAIN_INSTRUCTIONS.get(
                answer_domain,
                SYMPTOM_OR_CARE_INSTRUCTIONS,
            ),
        )
    )

    prompt_parts = [f"Câu hỏi: {question}\n"]
    if patient_text:
        prompt_parts.append(f"Thông tin người bệnh đã cung cấp:\n{patient_text}\n")
    if kg_text:
        prompt_parts.append(f"Thông tin từ Knowledge Graph:\n{kg_text}\n")
    if context:
        prompt_parts.append(f"Tài liệu tham khảo:\n{context}\n")
    if (
        answer_domain == "drug_info"
        and _question_asks_drug_usage_detail(question)
        and _source_has_drug_usage_detail(context_hits)
    ):
        prompt_parts.append(
            "Lưu ý riêng cho câu hỏi thuốc: tài liệu tham khảo đã có thông tin "
            "liều/cách dùng. Hãy nêu chính xác các số liệu, tần suất, thời gian "
            "hoặc đường dùng có trong tài liệu; không trả lời rằng tài liệu không "
            "đủ thông tin về liều/cách dùng."
        )
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
                {"role": "system", "content": system_prompt},
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
        answer = enforce_info_answer_contract(
            answer,
            context_hits,
            answer_domain,
            question,
        )
        if not answer:
            answer = "Tôi không có đủ thông tin trong tài liệu để trả lời chắc chắn."
        if (
            answer_domain == "drug_info"
            and _question_asks_drug_usage_detail(question)
            and _source_has_drug_usage_detail(context_hits)
            and _answer_says_insufficient_drug_usage(answer)
        ):
            fallback = _drug_usage_fallback_answer(context_hits, cite_idx)
            if fallback:
                answer = fallback
        source_numbers = _cited_source_numbers(answer, len(unique))
        answer, source_numbers = _ensure_health_insurance_article_citations(
            answer,
            unique,
            source_numbers,
        )
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
