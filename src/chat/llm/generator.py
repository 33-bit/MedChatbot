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
from src.chat.evidence_plan import (
    normalize_evidence_plan,
    plan_required_facts,
    plan_requires_drug_usage_detail,
    plan_source_type,
    plan_targets,
    structured_text_match,
)
from src.chat.llm.answer_verifier import repair_answer_with_evidence
from src.chat.llm.evidence_brief import build_evidence_brief
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
    r"(?:mg|g|mcg|ug|ml|%|lần(?:/ngày)?|ngày|giờ|phút|tuần|tháng|năm|viên|giọt|ống)\b",
    flags=re.IGNORECASE,
)
_INSUFFICIENT_USAGE_TERMS = (
    "khong du thong tin",
    "khong co du thong tin",
    "tai lieu duoc cung cap khong du",
    "tai lieu khong du",
    "khong tim thay thong tin",
    "khong co thong tin",
)
_ANSWER_DOMAIN_INSTRUCTIONS: dict[AnswerDomain, str] = {
    "symptom_or_care": SYMPTOM_OR_CARE_INSTRUCTIONS,
    "disease_info": DISEASE_INFO_INSTRUCTIONS,
    "drug_info": DRUG_INFO_INSTRUCTIONS,
    "health_insurance_info": HEALTH_INSURANCE_INFO_INSTRUCTIONS,
}
_EVIDENCE_REPAIR_DOMAINS = {"disease_info", "drug_info"}
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
_DISEASE_OVERVIEW_QUERY_TERMS = (
    "la gi",
    "dinh nghia",
    "tong quan",
    "anh huong",
)
_DISEASE_OVERVIEW_HEADING_TERMS = (
    "dai cuong",
    "tong quan",
)
_DISEASE_OVERVIEW_OFF_SCOPE_TERMS = (
    "chan doan:",
    "dieu tri:",
    "bien chung:",
    "dieu tri chuan:",
    "tien luong:",
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
    "dung nhu the nao",
    "ngay may lan",
    "bao nhieu",
    "bao lau",
    "may lan",
    "thoi gian dung",
)
_DRUG_USAGE_LINE_TERMS = (
    "lieu",
    "lieu dung",
    "lieu luong",
    "cach dung",
    "duong dung",
    "uong",
    "boi",
    "dan",
    "ngam",
    "nho",
    "xit",
    "hit",
    "tiem",
    "ngay",
    "lan",
    "mg",
    "ml",
    "%",
)
_DRUG_DOSE_LINE_TERMS = (
    "lieu",
    "lieu dung",
    "lieu luong",
    "nguoi lon",
    "thieu nien",
    "tre em",
    "uong",
    "nho",
    "boi",
    "dan",
    "ngam",
    "xit",
    "hit",
    "tiem",
    "moi lan",
    "khong vuot qua",
    "ngay dung",
)
_DRUG_ADMIN_LINE_TERMS = (
    "cach dung",
    "duong dung",
    "uong",
    "nho",
    "boi",
    "dan",
    "ngam",
    "xit",
    "hit",
    "tiem",
)
_DRUG_INDICATION_QUERY_TERMS = (
    "co the dung",
    "co dung duoc",
    "duoc khong",
    "dung de",
    "cong dung",
    "chi dinh",
)
_DRUG_INDICATION_HEADING_TERMS = (
    "cong dung",
    "chi dinh",
)
_DRUG_INDICATION_LINE_TERMS = (
    "duoc su dung",
    "duoc dung",
    "dung de",
    "dieu tri",
    "giam dau",
    "chi dinh",
)
_DRUG_ADULT_QUERY_TERMS = (
    "nguoi lon",
    "nguoi truong thanh",
)
_DRUG_PEDIATRIC_TERMS = (
    "tre",
    "tre em",
    "tre so sinh",
    "thieu nien",
)
_DRUG_ALLERGY_QUERY_TERMS = (
    "di ung",
    "man cam",
    "qua man",
)
_DRUG_ALLERGY_WARNING_TERMS = (
    "di ung",
    "man cam",
    "qua man",
)
_DRUG_CONTRAINDICATION_ACTION_TERMS = (
    "khong su dung",
    "khong dung",
    "khong nen",
    "chong chi dinh",
)
_UNSUPPORTED_SAFETY_MINIMIZATION_TERMS = (
    "hiem khi",
    "hiem gap",
    "khong ghi nhan",
    "chua ghi nhan",
    "khong thay",
    "khong co bang chung",
)
_DRUG_FOCUS_STOP_WORDS = {
    "toi",
    "minh",
    "ban",
    "bac",
    "si",
    "ke",
    "don",
    "thuoc",
    "can",
    "nen",
    "muon",
    "biet",
    "hoi",
    "duoc",
    "khong",
    "co",
    "the",
    "dung",
    "su",
    "uong",
    "boi",
    "nho",
    "dan",
    "ngam",
    "xit",
    "hit",
    "tiem",
    "nhu",
    "nao",
    "lieu",
    "luong",
    "cach",
    "duong",
    "bao",
    "nhieu",
    "lan",
    "trong",
    "khoang",
    "thoi",
    "gian",
    "la",
    "ve",
    "cho",
    "voi",
    "khi",
    "va",
    "hoac",
    "neu",
    "bi",
    "benh",
    "con",
    "vay",
    "phu",
    "hop",
}


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


def _cite_idx_for_unique(hits: list[Hit], unique: list[Hit]) -> list[int]:
    mapping = {
        _source_key(hit): index
        for index, hit in enumerate(unique, start=1)
    }
    return [mapping.get(_source_key(hit), 1) for hit in hits]


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


def _normalized_has_term(normalized: str, term: str) -> bool:
    if " " in term:
        return term in normalized
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", normalized))


def _source_text(hits: list[Hit]) -> str:
    return "\n".join(h.text for h in hits if h.text)


def _source_contains_any(hits: list[Hit], terms: tuple[str, ...]) -> bool:
    return _contains_any(_source_text(hits), terms)


def _question_asks_drug_usage_detail(question: str) -> bool:
    normalized = _normalize_text(question)
    if any(
        _normalized_has_term(normalized, term)
        for term in _DRUG_USAGE_QUERY_TERMS
    ):
        return True
    route_verbs = "uong|boi|nho|dan|ngam|xit|hit|tiem|dat|dung thuoc|su dung thuoc"
    usage_asks = "the nao|nhu the nao|bao lau|may lan|bao nhieu|lieu nao|luc nao|khi nao"
    return bool(
        re.search(rf"\b(?:{route_verbs})\b.{{0,40}}\b(?:{usage_asks})\b", normalized)
        or re.search(rf"\b(?:{usage_asks})\b.{{0,40}}\b(?:{route_verbs})\b", normalized)
    )


def _question_asks_drug_indication(question: str) -> bool:
    return _contains_any(question, _DRUG_INDICATION_QUERY_TERMS)


def _question_asks_adult_drug_use(question: str) -> bool:
    return _contains_any(question, _DRUG_ADULT_QUERY_TERMS)


def _source_has_drug_usage_detail(hits: list[Hit]) -> bool:
    drug_text = "\n".join(hit.text for hit in hits if hit.source_type == "drug")
    if _DOSE_TOKEN_RE.search(drug_text):
        return True
    return _contains_any(drug_text, _DRUG_USAGE_LINE_TERMS)


def _question_asks_disease_context(question: str) -> bool:
    return _contains_any(question, _DISEASE_CONTEXT_TERMS)


def _question_asks_disease_overview(question: str) -> bool:
    return (
        _contains_any(question, _DISEASE_OVERVIEW_QUERY_TERMS)
        and not _question_asks_treatment_or_medicine(question)
        and not _question_asks_danger_or_care(question)
    )


def _question_asks_treatment_or_medicine(question: str) -> bool:
    normalized = _normalize_text(question)
    return any(
        _normalized_has_term(normalized, term)
        for term in _TREATMENT_OR_MEDICINE_TERMS
    )


def _question_asks_danger_or_care(question: str) -> bool:
    return _contains_any(question, _DANGER_OR_CARE_TERMS)


def _filter_hits_for_answer_domain(
    question: str,
    hits: list[Hit],
    answer_domain: AnswerDomain,
    evidence_plan: dict | None = None,
) -> list[Hit]:
    plan_source = plan_source_type(evidence_plan)
    if plan_source and plan_source != "medical":
        scoped = [hit for hit in hits if hit.source_type == plan_source]
        if scoped:
            hits = scoped

    if answer_domain == "health_insurance_info":
        scoped = [hit for hit in hits if hit.source_type == "health_insurance"]
        return _prioritize_evidence_plan_hits(scoped or hits, evidence_plan)

    if answer_domain == "drug_info":
        drug_hits = [hit for hit in hits if hit.source_type == "drug"]
        if not drug_hits:
            return _prioritize_evidence_plan_hits(hits, evidence_plan)
        entity = (evidence_plan or {}).get("entity")
        entity_hits = [
            hit for hit in drug_hits
            if _hit_matches_evidence_entity(hit, entity)
        ]
        if entity_hits:
            drug_hits = entity_hits
        if _question_asks_disease_context(question):
            disease_hits = [hit for hit in hits if hit.source_type == "disease"]
            other_hits = [
                hit for hit in hits
                if hit.source_type not in {"drug", "disease"}
            ]
            return _prioritize_evidence_plan_hits(
                drug_hits + disease_hits + other_hits,
                evidence_plan,
            )
        return _prioritize_evidence_plan_hits(drug_hits, evidence_plan)

    if answer_domain == "disease_info":
        disease_hits = [hit for hit in hits if hit.source_type == "disease"]
        if not disease_hits:
            return _prioritize_evidence_plan_hits(hits, evidence_plan)
        entity = (evidence_plan or {}).get("entity")
        entity_hits = [
            hit for hit in disease_hits
            if _hit_matches_evidence_entity(hit, entity)
        ]
        if entity_hits:
            disease_hits = entity_hits
        if _question_asks_treatment_or_medicine(question):
            drug_hits = [hit for hit in hits if hit.source_type == "drug"]
            other_hits = [
                hit for hit in hits
                if hit.source_type not in {"drug", "disease"}
            ]
            return _prioritize_evidence_plan_hits(
                disease_hits + drug_hits + other_hits,
                evidence_plan,
            )
        return _prioritize_evidence_plan_hits(disease_hits, evidence_plan)

    return _prioritize_evidence_plan_hits(hits, evidence_plan)


def _hit_matches_evidence_target(hit: Hit, target: str) -> bool:
    return structured_text_match(hit.heading_path, target)


def _hit_matches_evidence_entity(hit: Hit, entity: str | None) -> bool:
    return bool(
        entity
        and (
            structured_text_match(hit.source_name, entity)
            or structured_text_match(hit.source_slug, entity)
        )
    )


def _prioritize_evidence_plan_hits(
    hits: list[Hit],
    evidence_plan: dict | None,
) -> list[Hit]:
    if not evidence_plan or not hits:
        return hits
    targets = plan_targets(evidence_plan)
    entity = evidence_plan.get("entity")
    if not targets and not entity:
        return hits

    def score(hit: Hit) -> int:
        value = 0
        if entity and (
            structured_text_match(hit.source_name, entity)
            or structured_text_match(hit.source_slug, entity)
        ):
            value += 2
        if any(structured_text_match(hit.heading_path, target) for target in targets):
            value += 3
        return value

    ranked = sorted(
        enumerate(hits),
        key=lambda item: (score(item[1]), -item[0]),
        reverse=True,
    )
    return [hit for _index, hit in ranked]


def _dedupe_hits_by_chunk(hits: list[Hit]) -> list[Hit]:
    seen: set[tuple] = set()
    result: list[Hit] = []
    for hit in hits:
        key = (
            hit.chunk_id or hit.id or "",
            hit.source_type,
            hit.source_slug,
            hit.heading_path,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(hit)
    return result


def _heading_label(hit: Hit) -> str:
    return hit.heading_path or hit.source_name or hit.chunk_id or hit.source_slug


def _selected_evidence_context(
    hits: list[Hit],
    evidence_plan: dict | None,
    answer_domain: AnswerDomain,
) -> tuple[list[Hit], dict]:
    """Prioritize disease/drug sections from the plan without dropping context."""
    pack = {
        "constrained": False,
        "target_headings": [],
        "selected_headings": [],
        "excluded_headings": [],
    }
    if answer_domain not in _EVIDENCE_REPAIR_DOMAINS or not evidence_plan or not hits:
        return hits, pack

    targets = plan_targets(evidence_plan)
    if not targets:
        return hits, pack

    entity = evidence_plan.get("entity")
    selected = [
        hit for hit in hits
        if any(_hit_matches_evidence_target(hit, target) for target in targets)
        and (not entity or _hit_matches_evidence_entity(hit, entity))
    ]
    if not selected:
        selected = [
            hit for hit in hits
            if any(_hit_matches_evidence_target(hit, target) for target in targets)
        ]
    if not selected:
        return hits, pack

    selected = _dedupe_hits_by_chunk(selected)
    selected_keys = {
        (
            hit.chunk_id or hit.id or "",
            hit.source_type,
            hit.source_slug,
            hit.heading_path,
        )
        for hit in selected
    }
    other = [
        _heading_label(hit)
        for hit in hits
        if (
            hit.chunk_id or hit.id or "",
            hit.source_type,
            hit.source_slug,
            hit.heading_path,
        )
        not in selected_keys
    ]
    pack = {
        "constrained": True,
        "target_headings": targets,
        "selected_headings": [_heading_label(hit) for hit in selected],
        "excluded_headings": list(dict.fromkeys(other))[:8],
    }
    remaining = [
        hit for hit in hits
        if (
            hit.chunk_id or hit.id or "",
            hit.source_type,
            hit.source_slug,
            hit.heading_path,
        )
        not in selected_keys
    ]
    return _dedupe_hits_by_chunk(selected + remaining), pack


def _apply_evidence_brief_selection(
    hits: list[Hit],
    evidence_brief: dict | None,
) -> list[Hit]:
    if not evidence_brief:
        return hits
    indexes = evidence_brief.get("selected_indexes")
    if not isinstance(indexes, list):
        return hits
    selected: list[Hit] = []
    seen: set[int] = set()
    for item in indexes:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if index < 1 or index > len(hits) or index in seen:
            continue
        seen.add(index)
        selected.append(hits[index - 1])
    if not selected:
        return hits
    selected_keys = {
        (
            hit.chunk_id or hit.id or "",
            hit.source_type,
            hit.source_slug,
            hit.heading_path,
        )
        for hit in selected
    }
    remaining = [
        hit for hit in hits
        if (
            hit.chunk_id or hit.id or "",
            hit.source_type,
            hit.source_slug,
            hit.heading_path,
        )
        not in selected_keys
    ]
    return _dedupe_hits_by_chunk(selected + remaining)


def _hit_has_drug_usage_detail(hit: Hit) -> bool:
    if hit.source_type != "drug":
        return False
    heading = _normalize_text(hit.heading_path)
    if any(term in heading for term in ("lieu", "cach dung", "duong dung")):
        return True
    for line in (hit.text or "").splitlines():
        normalized = _normalize_text(line)
        if not any(term in normalized for term in ("lieu", "cach dung", "duong dung", "uong", "boi", "tiem")):
            continue
        if _DOSE_TOKEN_RE.search(line):
            return True
    return False


def _hit_has_drug_dose_heading(hit: Hit) -> bool:
    if hit.source_type != "drug":
        return False
    segments = [
        _normalize_text(segment)
        for segment in (hit.heading_path or "").split(">")
        if segment.strip()
    ]
    for segment in segments:
        has_dose = any(term in segment for term in ("lieu", "lieu dung", "lieu luong"))
        has_admin_only = any(term in segment for term in ("cach dung", "duong dung"))
        if has_dose and not has_admin_only:
            return True
        if "lieu luong" in segment and _DOSE_TOKEN_RE.search(hit.text or ""):
            return True
    return False


def _hit_has_drug_admin_heading(hit: Hit) -> bool:
    if hit.source_type != "drug":
        return False
    heading = _normalize_text(hit.heading_path)
    return any(term in heading for term in ("cach dung", "duong dung"))


def _hit_has_drug_indication_heading(hit: Hit) -> bool:
    if hit.source_type != "drug":
        return False
    heading = _normalize_text(hit.heading_path)
    return any(term in heading for term in _DRUG_INDICATION_HEADING_TERMS)


def _hit_has_disease_overview_heading(hit: Hit) -> bool:
    if hit.source_type != "disease":
        return False
    return _contains_any(hit.heading_path, _DISEASE_OVERVIEW_HEADING_TERMS)


def _is_pediatric_text_without_adult(text: str) -> bool:
    normalized = _normalize_text(text)
    has_pediatric = any(
        _normalized_has_term(normalized, term)
        for term in _DRUG_PEDIATRIC_TERMS
    )
    has_adult = any(
        _normalized_has_term(normalized, term)
        for term in _DRUG_ADULT_QUERY_TERMS
    )
    return has_pediatric and not has_adult


def _meaningful_drug_focus_tokens(text: str) -> list[str]:
    normalized = _normalize_text(text)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return [
        token
        for token in tokens
        if len(token) >= 3
        and token not in _DRUG_FOCUS_STOP_WORDS
        and not token.isdigit()
    ]


def _drug_usage_focus_terms(question: str, hits: list[Hit]) -> set[str]:
    normalized = _normalize_text(question)
    for hit in hits:
        if hit.source_type != "drug":
            continue
        for name in (hit.source_name, hit.source_slug):
            name_normalized = _normalize_text(name)
            if name_normalized:
                normalized = normalized.replace(name_normalized, " ")

    tokens = _meaningful_drug_focus_tokens(normalized)
    terms: set[str] = set(tokens)
    for size in (2, 3):
        for index in range(0, len(tokens) - size + 1):
            terms.add(" ".join(tokens[index:index + size]))
    return terms


def _drug_usage_focus_score(text: str, focus_terms: set[str]) -> int:
    if not focus_terms:
        return 0
    normalized = _normalize_text(text)
    score = 0
    for term in focus_terms:
        if _normalized_has_term(normalized, term):
            score += 2 if " " in term else 1
    return score


def _drug_usage_focus_threshold(focus_terms: set[str]) -> int:
    return 2 if any(" " in term for term in focus_terms) else 1


def _question_age_years(question: str) -> int | None:
    normalized = _normalize_text(question)
    match = re.search(r"\b(\d{1,3})\s*tuoi\b", normalized)
    if not match:
        return None
    age = int(match.group(1))
    return age if 0 <= age <= 120 else None


def _age_line_applies(line: str, age_years: int | None) -> bool:
    if age_years is None:
        return True
    normalized = _normalize_text(line)
    if _normalized_has_term(normalized, "nguoi cao tuoi") or _normalized_has_term(
        normalized,
        "cao tuoi",
    ):
        return age_years >= 60
    if _normalized_has_term(normalized, "nguoi lon"):
        return age_years >= 18
    if _normalized_has_term(normalized, "tre em") and "tuoi" not in normalized:
        return age_years < 18
    if "tuoi" not in normalized:
        return True

    range_match = re.search(r"\b(\d{1,3})\s*[-–]\s*(\d{1,3})\s*tuoi\b", normalized)
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        return start <= age_years <= end

    lower_bound_match = re.search(
        r"(?:tren|tu|>=|≥)\s*(\d{1,3})\s*tuoi",
        normalized,
    )
    if lower_bound_match:
        return age_years >= int(lower_bound_match.group(1))

    upper_bound_match = re.search(r"(?:duoi|<)\s*(\d{1,3})\s*tuoi", normalized)
    if upper_bound_match:
        return age_years < int(upper_bound_match.group(1))

    exact_match = re.search(r"\b(\d{1,3})\s*tuoi\b", normalized)
    if exact_match:
        return age_years == int(exact_match.group(1))

    return True


def _fallback_hit_pairs(
    hits: list[Hit],
    cite_idx: list[int],
) -> list[tuple[Hit, int]]:
    pairs: list[tuple[Hit, int]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for hit, source_number in zip(hits, cite_idx):
        key = (
            hit.source_type,
            hit.source_slug,
            hit.heading_path,
            _compact_for_match(hit.text),
        )
        if key in seen:
            continue
        seen.add(key)
        pairs.append((hit, source_number))
    return pairs


def _ensure_drug_usage_after_brief(
    selected_hits: list[Hit],
    available_hits: list[Hit],
    *,
    required: bool,
) -> list[Hit]:
    if not required:
        return selected_hits
    if any(_hit_has_drug_usage_detail(hit) for hit in selected_hits):
        return selected_hits
    for hit in available_hits:
        if _hit_has_drug_usage_detail(hit):
            return _dedupe_hits_by_chunk(selected_hits + [hit])
    return selected_hits


def _ensure_disease_overview_after_brief(
    selected_hits: list[Hit],
    available_hits: list[Hit],
    *,
    required: bool,
) -> list[Hit]:
    if not required:
        return selected_hits
    if any(_hit_has_disease_overview_heading(hit) for hit in selected_hits):
        return selected_hits
    for hit in available_hits:
        if _hit_has_disease_overview_heading(hit):
            return _dedupe_hits_by_chunk([hit] + selected_hits)
    return selected_hits


def _format_evidence_context_policy(pack: dict) -> str:
    if not pack.get("constrained"):
        return ""
    lines = [
        "Ưu tiên tài liệu theo kế hoạch bằng chứng:",
        "- Trả lời trước từ các mục được chọn; nếu câu hỏi có nhiều ý, dùng thêm mục tham khảo còn lại khi chúng trực tiếp trả lời phần chưa được bao phủ.",
        "- Không lan sang chẩn đoán, điều trị, biến chứng hoặc cảnh báo nếu người dùng không hỏi phần đó.",
    ]
    selected = pack.get("selected_headings") or []
    if selected:
        lines.append("- Mục được chọn: " + "; ".join(selected))
    excluded = pack.get("excluded_headings") or []
    if excluded:
        lines.append("- Mục còn lại chỉ dùng khi trực tiếp liên quan: " + "; ".join(excluded))
    return "\n".join(lines)


def _format_evidence_brief(evidence_brief: dict | None) -> str:
    if not evidence_brief:
        return ""
    lines = ["Brief bằng chứng đã chọn:"]
    brief = str(evidence_brief.get("brief") or "").strip()
    if brief:
        lines.append(f"- Tóm tắt phạm vi: {brief}")
    facts = evidence_brief.get("must_include_facts") or []
    if facts:
        lines.append("- Ý bắt buộc phải bao phủ: " + "; ".join(map(str, facts)))
    avoid = evidence_brief.get("avoid_topics") or []
    if avoid:
        lines.append("- Không lan sang các phần: " + "; ".join(map(str, avoid)))
    lines.append(
        "Ưu tiên brief; nếu brief thiếu một phần của câu hỏi, dùng tài liệu tham khảo còn lại khi trực tiếp liên quan."
    )
    return "\n".join(lines)


def _should_repair_with_evidence(
    answer_domain: AnswerDomain,
    evidence_plan: dict | None,
    context_hits: list[Hit],
) -> bool:
    if answer_domain not in _EVIDENCE_REPAIR_DOMAINS:
        return False
    return bool(context_hits)


def _format_evidence_plan(evidence_plan: dict | None) -> str:
    if not evidence_plan:
        return ""
    lines = [
        f"- Miền trả lời: {evidence_plan.get('domain')}",
        f"- Loại nguồn cần ưu tiên: {evidence_plan.get('source_type')}",
        f"- Thực thể chính: {evidence_plan.get('entity') or 'không rõ'}",
        f"- Loại thông tin cần trả lời: {evidence_plan.get('answer_slot')}",
        f"- Chế độ an toàn: {evidence_plan.get('safety_mode')}",
        f"- Kiểu trả lời: {evidence_plan.get('answer_style')}",
    ]
    targets = plan_targets(evidence_plan)
    if targets:
        lines.append("- Vùng tài liệu cần ưu tiên: " + "; ".join(targets))
    required_facts = plan_required_facts(evidence_plan)
    if required_facts:
        lines.append("- Ý bắt buộc cần bao phủ: " + "; ".join(required_facts))
    return "\n".join(lines)


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


def _line_has_unsupported_safety_minimization(line: str, hits: list[Hit]) -> bool:
    normalized = _normalize_text(line)
    matched = [
        term for term in _UNSUPPORTED_SAFETY_MINIMIZATION_TERMS
        if term in normalized
    ]
    if not matched:
        return False
    source_normalized = _normalize_text(_source_text(hits))
    return any(term not in source_normalized for term in matched)


def _question_asks_drug_allergy_safety(question: str) -> bool:
    return _contains_any(question, _DRUG_ALLERGY_QUERY_TERMS)


def _extract_allergy_warning_phrase(line: str) -> str:
    stripped = line.strip(" -*•\t")
    match = re.search(
        r"((?:quá mẫn cảm|quá mẫn|mẫn cảm|dị ứng)\s+với\s+[^.;,\n]+)",
        stripped,
        flags=re.IGNORECASE,
    )
    if match:
        phrase = match.group(1).strip()
        phrase = phrase[:1].lower() + phrase[1:]
        return f"Không sử dụng nếu bạn {phrase}."
    return stripped


def _source_drug_allergy_warning(hits: list[Hit]) -> str:
    for hit in hits:
        if hit.source_type != "drug":
            continue
        heading = _normalize_text(hit.heading_path)
        heading_is_contraindication = "chong chi dinh" in heading
        for raw_line in (hit.text or "").splitlines():
            line = raw_line.strip(" -*•\t")
            if len(line) < 6:
                continue
            normalized = _normalize_text(line)
            if not any(term in normalized for term in _DRUG_ALLERGY_WARNING_TERMS):
                continue
            has_action = any(
                term in normalized for term in _DRUG_CONTRAINDICATION_ACTION_TERMS
            )
            if has_action or heading_is_contraindication:
                return _extract_allergy_warning_phrase(line)
    return ""


def _answer_has_allergy_contraindication(answer: str) -> bool:
    normalized = _normalize_text(answer)
    return (
        any(term in normalized for term in _DRUG_ALLERGY_WARNING_TERMS)
        and any(term in normalized for term in _DRUG_CONTRAINDICATION_ACTION_TERMS)
    )


def _line_has_generic_disease_boilerplate(line: str) -> bool:
    normalized = _normalize_text(line)
    return any(term in normalized for term in _GENERIC_DISEASE_BOILERPLATE)


def _answer_says_insufficient_drug_usage(answer: str) -> bool:
    normalized = _normalize_text(answer)
    if not any(term in normalized for term in _INSUFFICIENT_USAGE_TERMS):
        return False
    return any(
        _normalized_has_term(normalized, term)
        for term in (
            "lieu",
            "cach dung",
            "duong dung",
            "uong",
            "boi",
            "nho",
            "dan",
            "ngam",
            "xit",
            "hit",
            "tiem",
        )
    )


def _answer_has_drug_usage_detail(answer: str) -> bool:
    if _DOSE_TOKEN_RE.search(_strip_citation_markers(answer)):
        return True
    normalized = _normalize_text(answer)
    route_terms = (
        "duong dung",
        "uong",
        "boi",
        "nho",
        "dan",
        "ngam",
        "xit",
        "hit",
        "tiem",
        "dat",
    )
    return bool(_CITATION_RE.search(answer)) and any(
        _normalized_has_term(normalized, term)
        for term in route_terms
    )


def _drug_usage_fallback_answer(
    hits: list[Hit],
    cite_idx: list[int],
    *,
    question: str = "",
    include_indication: bool = False,
    adult_only: bool = False,
) -> str | None:
    lines: list[str] = []
    seen: set[str] = set()
    hit_pairs = _fallback_hit_pairs(hits, cite_idx)
    primary_slug = next(
        (
            hit.source_slug
            for hit, _source_number in hit_pairs
            if hit.source_type == "drug" and hit.source_slug
        ),
        "",
    )
    focus_terms = _drug_usage_focus_terms(question, hits)
    focus_threshold = _drug_usage_focus_threshold(focus_terms)
    question_age_years = _question_age_years(question)
    has_focus_match = any(
        _drug_usage_focus_score(hit.heading_path + "\n" + hit.text, focus_terms)
        >= focus_threshold
        for hit, _source_number in hit_pairs
        if hit.source_type == "drug"
    )
    dose_keys: set[tuple[str, str, str, str]] = set()
    admin_keys: set[tuple[str, str, str, str]] = set()
    usage_keys: set[tuple[str, str, str, str]] = set()
    for hit, _source_number in hit_pairs:
        if primary_slug and hit.source_type == "drug" and hit.source_slug != primary_slug:
            continue
        if adult_only and _is_pediatric_text_without_adult(hit.heading_path):
            continue
        hit_key = (
            hit.chunk_id or hit.id or "",
            hit.source_type,
            hit.source_slug,
            hit.heading_path,
        )
        if _hit_has_drug_dose_heading(hit):
            dose_keys.add(hit_key)
        if _hit_has_drug_admin_heading(hit):
            admin_keys.add(hit_key)
        if _hit_has_drug_usage_detail(hit):
            usage_keys.add(hit_key)
    preferred_keys = (dose_keys | admin_keys) if dose_keys else usage_keys
    if include_indication:
        indication_candidates: list[tuple[int, int, str, int]] = []
        order = 0
        for hit, source_number in hit_pairs:
            if primary_slug and hit.source_type == "drug" and hit.source_slug != primary_slug:
                continue
            if adult_only and _is_pediatric_text_without_adult(hit.heading_path):
                continue
            if not _hit_has_drug_indication_heading(hit):
                continue
            heading_compacts = {
                _compact_for_match(part)
                for part in (hit.heading_path or "").split(">")
                if part.strip()
            }
            added_from_hit = 0
            for raw_line in hit.text.splitlines():
                line = raw_line.strip(" -*•\t")
                if len(line) < 8 or ">" in line:
                    continue
                normalized = _normalize_text(line)
                if normalized.startswith(("bang:", "bang ", "hinh:", "hinh ")):
                    continue
                if normalized == _normalize_text(hit.source_name):
                    continue
                if normalized.startswith("chi dinh cua "):
                    continue
                compact = _compact_for_match(line)
                if compact in heading_compacts:
                    continue
                if not any(term in normalized for term in _DRUG_INDICATION_LINE_TERMS):
                    continue
                score = 1
                if any(
                    phrase in normalized
                    for phrase in ("dieu tri", "duoc su dung", "duoc dung", "dung de")
                ):
                    score += 3
                if "giam dau" in normalized:
                    score += 2
                if compact in seen:
                    continue
                indication_candidates.append((score, order, line, source_number))
                order += 1
                added_from_hit += 1
                if added_from_hit >= 4:
                    break
        indication_candidates.sort(key=lambda item: (-item[0], item[1]))
        for _score, _order, line, source_number in indication_candidates[:2]:
            compact = _compact_for_match(line)
            if compact in seen:
                continue
            seen.add(compact)
            if not lines:
                line = (
                    "Có thể dùng trong phạm vi công dụng/chỉ định được nêu: "
                    + line
                )
            lines.append(f"- {line} [{source_number}]")
    for hit, source_number in hit_pairs:
        if hit.source_type != "drug":
            continue
        if primary_slug and hit.source_slug != primary_slug:
            continue
        if adult_only and _is_pediatric_text_without_adult(hit.heading_path):
            continue
        hit_key = (
            hit.chunk_id or hit.id or "",
            hit.source_type,
            hit.source_slug,
            hit.heading_path,
        )
        if preferred_keys and hit_key not in preferred_keys:
            continue
        dose_heading = hit_key in dose_keys
        admin_heading = hit_key in admin_keys
        added_from_hit = 0
        local_section_seen = False
        local_section_matches_focus = True
        for raw_line in hit.text.splitlines():
            line = raw_line.strip(" -*•\t")
            if len(line) < 8:
                continue
            normalized = _normalize_text(line)
            if normalized.startswith(("bang:", "bang ", "hinh:", "hinh ")):
                continue
            if not _age_line_applies(line, question_age_years):
                continue
            if adult_only and _is_pediatric_text_without_adult(line):
                continue
            has_dose_term = any(
                _normalized_has_term(normalized, term)
                for term in _DRUG_DOSE_LINE_TERMS
            )
            has_admin_term = any(
                _normalized_has_term(normalized, term)
                for term in _DRUG_ADMIN_LINE_TERMS
            )
            has_dose_token = bool(_DOSE_TOKEN_RE.search(line))
            looks_like_local_heading = (
                line.endswith(":")
                and not has_dose_token
                and len(line) <= 160
            )
            if looks_like_local_heading:
                local_section_seen = True
                local_section_matches_focus = (
                    not has_focus_match
                    or _drug_usage_focus_score(line, focus_terms) >= focus_threshold
                )
                continue
            if (
                local_section_seen
                and has_focus_match
                and not local_section_matches_focus
            ):
                continue
            if ">" in line:
                continue
            if line.endswith(":") and not has_dose_token:
                continue
            if dose_heading:
                if not (has_dose_term or has_dose_token):
                    continue
            elif admin_heading:
                if dose_keys:
                    if not has_admin_term:
                        continue
                elif not (has_dose_term or has_admin_term or has_dose_token):
                    continue
            elif not has_dose_token:
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


def _apply_drug_usage_fallback(
    answer: str,
    *,
    answer_domain: AnswerDomain,
    question: str,
    asks_drug_usage_detail: bool,
    include_indication: bool,
    context_hits: list[Hit],
    cite_idx: list[int],
) -> str:
    if not (
        answer_domain == "drug_info"
        and asks_drug_usage_detail
        and _source_has_drug_usage_detail(context_hits)
    ):
        return answer
    if answer.strip() and not _answer_says_insufficient_drug_usage(answer):
        if _answer_has_drug_usage_detail(answer):
            return answer
    fallback = _drug_usage_fallback_answer(
        context_hits,
        cite_idx,
        question=question,
        include_indication=include_indication or _question_asks_drug_indication(question),
        adult_only=_question_asks_adult_drug_use(question),
    )
    if not fallback:
        return answer
    return fallback


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


def _split_answer_units(answer: str) -> list[str]:
    units: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        units.extend(
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", stripped)
            if part.strip()
        )
    return units


def _filter_disease_overview_answer(answer: str, _hits: list[Hit]) -> str:
    kept: list[str] = []
    for unit in _split_answer_units(answer):
        normalized = _normalize_text(unit.strip(" -*•\t"))
        if any(term in normalized for term in _DISEASE_OVERVIEW_OFF_SCOPE_TERMS):
            continue
        if normalized.startswith(("chan doan", "dieu tri", "bien chung", "tien luong")):
            continue
        kept.append(unit)
    filtered = " ".join(kept)
    return filtered or answer


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
            if _line_has_unsupported_safety_minimization(line, hits):
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
        if (
            _question_asks_drug_allergy_safety(question)
            and not _answer_has_allergy_contraindication(answer)
        ):
            allergy_warning = _source_drug_allergy_warning(hits)
            if allergy_warning:
                answer = _clean_answer_format(answer + f"\n{allergy_warning} [1]")
        return _clean_answer_format(answer)

    if answer_domain == "disease_info":
        kept = [
            line for line in answer.splitlines()
            if not _line_has_generic_disease_boilerplate(line)
        ]
        answer = "\n".join(kept)
        if not _question_asks_danger_or_care(question):
            answer = _strip_triage_section(answer)
        if _question_asks_disease_overview(question):
            answer = _filter_disease_overview_answer(answer, hits)
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
    evidence_plan: dict | None = None,
    return_meta: bool = False,
):
    normalized_plan = (
        normalize_evidence_plan(evidence_plan, fallback_domain=answer_domain)
        if evidence_plan
        else None
    )
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

    domain_context_hits = _filter_hits_for_answer_domain(
        question,
        hits,
        answer_domain,
        normalized_plan,
    )
    context_hits = domain_context_hits
    context_hits, evidence_context_policy = _selected_evidence_context(
        context_hits,
        normalized_plan,
        answer_domain,
    )
    evidence_brief = build_evidence_brief(
        question=question,
        hits=context_hits,
        evidence_plan=normalized_plan,
        answer_domain=answer_domain,
    )
    pre_brief_context_hits = context_hits
    context_hits = _apply_evidence_brief_selection(context_hits, evidence_brief)
    context_hits = _ensure_disease_overview_after_brief(
        context_hits,
        domain_context_hits,
        required=answer_domain == "disease_info"
        and _question_asks_disease_overview(question),
    )
    asks_drug_usage_detail = (
        plan_requires_drug_usage_detail(normalized_plan)
        or _question_asks_drug_usage_detail(question)
    )
    include_drug_indication = (
        answer_domain == "drug_info"
        and (normalized_plan or {}).get("answer_slot") == "indication"
    )
    context_hits = _ensure_drug_usage_after_brief(
        context_hits,
        domain_context_hits,
        required=answer_domain == "drug_info" and asks_drug_usage_detail,
    )
    unique, cite_idx = _dedupe(context_hits)
    domain_cite_idx = _cite_idx_for_unique(domain_context_hits, unique)
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
    evidence_plan_text = _format_evidence_plan(normalized_plan)
    if evidence_plan_text:
        prompt_parts.append(
            "Kế hoạch bằng chứng cần bám sát:\n"
            f"{evidence_plan_text}\n"
            "Nếu tài liệu tham khảo không có đủ các ý bắt buộc, hãy nói rõ phần thiếu; "
            "không trả lời chung chung hoặc đổi sang loại thông tin khác.\n"
        )
    evidence_context_policy_text = _format_evidence_context_policy(
        evidence_context_policy
    )
    if evidence_context_policy_text:
        prompt_parts.append(f"{evidence_context_policy_text}\n")
    evidence_brief_text = _format_evidence_brief(evidence_brief)
    if evidence_brief_text:
        prompt_parts.append(f"{evidence_brief_text}\n")
    if context:
        prompt_parts.append(f"Tài liệu tham khảo:\n{context}\n")
    if answer_domain in _EVIDENCE_REPAIR_DOMAINS and context_hits:
        prompt_parts.append(
            "Ưu tiên bằng chứng: đoạn [1] là đoạn được xếp hạng cao nhất trong "
            "phạm vi tài liệu đã chọn. Nếu đoạn này trả lời trực tiếp câu hỏi, "
            "hãy bao phủ các ý định nghĩa, số liệu hoặc điều kiện chính của đoạn [1] "
            "trước khi thêm chi tiết từ đoạn khác."
        )
    if (
        answer_domain == "drug_info"
        and asks_drug_usage_detail
        and _source_has_drug_usage_detail(context_hits)
    ):
        prompt_parts.append(
            "Lưu ý riêng cho câu hỏi thuốc: tài liệu tham khảo đã có thông tin "
            "liều/cách dùng. Hãy nêu chính xác các số liệu, tần suất, thời gian "
            "hoặc đường dùng có trong tài liệu; không trả lời rằng tài liệu không "
            "đủ thông tin về liều/cách dùng. Nếu cùng một mục có nhiều phác đồ "
            "theo bệnh/đối tượng, chỉ nêu phác đồ khớp với tình trạng hoặc đối "
            "tượng trong câu hỏi, trừ khi người dùng hỏi rộng hơn. Không áp dụng "
            "phác đồ ghi riêng cho bệnh/tình trạng khác khi tên tình trạng không "
            "khớp rõ."
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
        answer = _apply_drug_usage_fallback(
            answer,
            answer_domain=answer_domain,
            question=question,
            asks_drug_usage_detail=asks_drug_usage_detail,
            include_indication=include_drug_indication,
            context_hits=domain_context_hits,
            cite_idx=domain_cite_idx,
        )
        if _should_repair_with_evidence(answer_domain, normalized_plan, context_hits):
            answer = repair_answer_with_evidence(
                question=question,
                answer=answer,
                evidence_text=context,
                evidence_plan=normalized_plan,
                answer_domain=answer_domain,
            )
            answer = _clean_answer_format(answer)
            if not answer:
                answer = "Tôi không có đủ thông tin trong tài liệu để trả lời chắc chắn."
            answer = enforce_info_answer_contract(
                answer,
                context_hits,
                answer_domain,
                question,
            )
            answer = _apply_drug_usage_fallback(
                answer,
                answer_domain=answer_domain,
                question=question,
                asks_drug_usage_detail=asks_drug_usage_detail,
                include_indication=include_drug_indication,
                context_hits=domain_context_hits,
                cite_idx=domain_cite_idx,
            )
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
