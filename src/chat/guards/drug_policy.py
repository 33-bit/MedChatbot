"""Deterministic OTC-only policy for drug-information questions."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

OTC_ONLY_REPLY = (
    "Tôi chỉ có thể cung cấp thông tin về thuốc không kê đơn (OTC) "
    "nằm trong danh mục được hỗ trợ. Với thuốc này, bạn vui lòng hỏi "
    "bác sĩ hoặc dược sĩ."
)

_WHITELIST_PATH = Path(__file__).with_name("otc_whitelist.json")
_DRUG_INTENTS = {"pure_info", "contextual_drug_info"}
_DRUG_REQUEST_PATTERNS = (
    re.compile(r"\bthong tin\s+(?:ve\s+)?thuoc\b"),
    re.compile(
        r"\bthuoc\b.{0,80}\b(?:la gi|cong dung|tac dung|chi dinh|"
        r"chong chi dinh|tuong tac|lieu|cach dung|dung duoc khong)\b"
    ),
    re.compile(
        r"\b(?:cong dung|tac dung|chi dinh|chong chi dinh|tuong tac|"
        r"lieu dung|cach dung)\b.{0,80}\bthuoc\b"
    ),
    re.compile(r"\b(?:lieu dung|uong bao nhieu|may vien|ngay uong)\b"),
)
_NON_DRUG_NEGATIONS = (
    "khong dung thuoc",
    "khong su dung thuoc",
    "khong can thuoc",
)
_ALIAS_QUALIFIERS_RE = re.compile(
    r"\b(?:don thanh phan|phoi hop|dang bao che|dang don chat|cac dang)\b"
)
_CANDIDATE_SPLIT_RE = re.compile(
    r"\s+(?:va|voi|hoac|va hoac)\s+|(?<!\d),(?!\d)|[;+]"
)
_CANDIDATE_FILLERS = {
    "thuoc",
    "hoat",
    "chat",
    "loai",
    "dang",
    "vien",
    "nen",
    "nang",
    "siro",
    "dung",
    "dich",
    "uong",
    "boi",
    "tiem",
    "truyen",
    "mg",
    "g",
    "mcg",
    "microgam",
    "ml",
    "iu",
    "ui",
    "don",
    "vi",
    "ham",
    "luong",
    "phoi",
    "hop",
}
_ROUTES = (
    ("tiem tinh mach", ("tiem tinh mach", "truyen tinh mach")),
    ("truyen tinh mach", ("truyen tinh mach",)),
    ("tiem bap", ("tiem bap",)),
    ("tiem duoi da", ("tiem duoi da",)),
    ("dat hau mon", ("dat hau mon",)),
    ("thuoc tra mat", ("thuoc tra mat", "tra mat")),
    ("nho mat", ("thuoc tra mat", "tra mat", "nho mat")),
    ("xit mui", ("xit mui",)),
    ("dung ngoai", ("dung ngoai",)),
    ("boi", ("dung ngoai", "boi")),
    ("tiem", ("tiem",)),
    ("uong", ("uong",)),
)
_STRENGTH_RE = re.compile(
    r"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>microgam|mcg|mg|g|iu|ui|ml)\b"
)
_LIMIT_RE = re.compile(
    r"(?P<op><=|>=|<|>|≤|≥)\s*(?P<value>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>microgam|mcg|mg|g|iu|ui|ml)\b"
)


@dataclass(frozen=True)
class DrugPolicyDecision:
    is_drug_question: bool
    allowed: bool
    reason: str
    matched_otc_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class _OtcEntry:
    name: str
    aliases: tuple[str, ...]
    route_constraint: str
    specific_condition: str


def _normalize(text: object) -> str:
    value = str(text or "").strip().casefold().replace("đ", "d")
    value = value.replace("µg", "mcg").replace("μg", "mcg")
    value = unicodedata.normalize("NFD", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = re.sub(r"[^a-z0-9<>=≤≥.,%]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _useful_aliases(row: dict) -> tuple[str, ...]:
    raw_values = {
        str(row.get("ten_normalized") or ""),
        str(row.get("ten_ngan") or ""),
    }
    full_name = str(row.get("ten_hoat_chat") or "")
    raw_values.add(full_name)
    raw_values.update(re.findall(r"\(([^()]*)\)", full_name))

    aliases: set[str] = set()
    for raw in raw_values:
        normalized = _normalize(raw)
        if not normalized:
            continue
        candidates = {normalized}
        candidates.update(part.strip() for part in normalized.split(","))
        candidates.update(part.strip() for part in re.split(r"\s+va\s+", normalized))
        candidates.add(_ALIAS_QUALIFIERS_RE.split(normalized, maxsplit=1)[0].strip())
        for candidate in candidates:
            if len(candidate) >= 4 and candidate not in {"thuoc", "vitamin"}:
                aliases.add(candidate)
    return tuple(sorted(aliases, key=lambda item: (-len(item), item)))


@lru_cache(maxsize=1)
def _otc_entries() -> tuple[_OtcEntry, ...]:
    rows = json.loads(_WHITELIST_PATH.read_text(encoding="utf-8"))
    entries = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("otc"):
            continue
        aliases = _useful_aliases(row)
        if not aliases:
            continue
        entries.append(
            _OtcEntry(
                name=str(row.get("ten_ngan") or row.get("ten_hoat_chat") or "").strip(),
                aliases=aliases,
                route_constraint=_normalize(row.get("duong_dung_dang_bao_che")),
                specific_condition=_normalize(row.get("dieu_kien_cu_the")),
            )
        )
    return tuple(entries)


def _field(data: object, key: str) -> object:
    return data.get(key) if isinstance(data, dict) else None


def _medication_candidates(analysis: dict) -> list[str]:
    candidates: list[str] = []
    entities = _field(analysis, "entities")
    medications = _field(entities, "medications")
    if isinstance(medications, list):
        for medication in medications:
            if isinstance(medication, dict):
                value = medication.get("name") or medication.get("drug_id")
            else:
                value = medication
            if value:
                candidates.append(str(value).removeprefix("drug:"))

    context = _field(analysis, "context")
    references = _field(context, "references")
    if isinstance(references, list):
        for reference in references:
            if not isinstance(reference, dict) or reference.get("type") != "drug":
                continue
            value = reference.get("id") or reference.get("name")
            if value:
                candidates.append(str(value).removeprefix("drug:"))

    return list(dict.fromkeys(item.strip() for item in candidates if item.strip()))


def _has_explicit_drug_request(question: str) -> bool:
    normalized = _normalize(question)
    if any(phrase in normalized for phrase in _NON_DRUG_NEGATIONS):
        return False
    return any(pattern.search(normalized) for pattern in _DRUG_REQUEST_PATTERNS)


def _is_drug_question(question: str, analysis: dict, candidates: list[str]) -> bool:
    turn = _field(analysis, "turn")
    intent = str(_field(turn, "intent") or "")
    label = str(_field(turn, "label") or "")
    if intent == "contextual_drug_info":
        return True
    if candidates and intent in _DRUG_INTENTS:
        return True
    if candidates and label == "informational":
        return True
    return _has_explicit_drug_request(question)


def _contains_alias(text: str, alias: str) -> bool:
    return re.search(
        rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
        text,
    ) is not None


def _candidate_entries(candidate: str) -> list[_OtcEntry]:
    normalized = _normalize(candidate)
    matches = [
        entry
        for entry in _otc_entries()
        if any(_contains_alias(normalized, alias) for alias in entry.aliases)
    ]
    if not matches:
        return []

    residual = normalized
    matched_aliases = sorted(
        {
            alias
            for entry in matches
            for alias in entry.aliases
            if _contains_alias(normalized, alias)
        },
        key=len,
        reverse=True,
    )
    for alias in matched_aliases:
        residual = re.sub(
            rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
            " ",
            residual,
        )
    residual = _STRENGTH_RE.sub(" ", residual)
    remaining_tokens = {
        token
        for token in re.findall(r"[a-z]+", residual)
        if token not in _CANDIDATE_FILLERS
    }
    return [] if remaining_tokens else matches


def _explicit_routes(question: str) -> tuple[tuple[str, ...], ...]:
    normalized = _normalize(question)
    routes = []
    for query_phrase, allowed_phrases in _ROUTES:
        if _contains_alias(normalized, query_phrase):
            routes.append(allowed_phrases)
    return tuple(routes)


def _normalized_strength(value: float, unit: str) -> tuple[float, str]:
    if unit == "g":
        return value * 1000, "mg"
    if unit in {"mcg", "microgam"}:
        return value / 1000, "mg"
    if unit == "ui":
        return value, "iu"
    return value, unit


def _strengths(text: str) -> tuple[tuple[float, str], ...]:
    values = []
    for match in _STRENGTH_RE.finditer(_normalize(text)):
        value = float(match.group("value").replace(",", "."))
        values.append(_normalized_strength(value, match.group("unit")))
    return tuple(values)


def _limits(entry: _OtcEntry) -> tuple[tuple[str, float, str], ...]:
    constraints = f"{entry.route_constraint} {entry.specific_condition}"
    limits = []
    for match in _LIMIT_RE.finditer(constraints):
        value = float(match.group("value").replace(",", "."))
        normalized_value, normalized_unit = _normalized_strength(
            value,
            match.group("unit"),
        )
        limits.append((match.group("op"), normalized_value, normalized_unit))
    return tuple(limits)


def _within_limit(value: float, operator: str, limit: float) -> bool:
    return {
        "<": value < limit,
        "<=": value <= limit,
        "≤": value <= limit,
        ">": value > limit,
        ">=": value >= limit,
        "≥": value >= limit,
    }[operator]


def _entry_matches_explicit_constraints(entry: _OtcEntry, question: str) -> bool:
    for allowed_phrases in _explicit_routes(question):
        if not any(_contains_alias(entry.route_constraint, phrase) for phrase in allowed_phrases):
            return False

    entry_limits = _limits(entry)
    for value, unit in _strengths(question):
        relevant_limits = [limit for limit in entry_limits if limit[2] == unit]
        if relevant_limits and not all(
            _within_limit(value, operator, limit)
            for operator, limit, _ in relevant_limits
        ):
            return False
        if entry_limits and not relevant_limits:
            return False
    return True


def evaluate_drug_policy(question: str, analysis: dict) -> DrugPolicyDecision:
    """Allow only resolved OTC drug questions and explicit in-range constraints."""
    candidates = _medication_candidates(analysis)
    if not _is_drug_question(question, analysis, candidates):
        return DrugPolicyDecision(False, True, "not_a_drug_question")

    if analysis.get("analysis_succeeded") is False and not candidates:
        return DrugPolicyDecision(True, False, "unresolved_drug")
    if not candidates:
        return DrugPolicyDecision(True, False, "unresolved_drug")

    matched_entries: list[_OtcEntry] = []
    for candidate in candidates:
        segments = [
            segment.strip()
            for segment in _CANDIDATE_SPLIT_RE.split(_normalize(candidate))
            if segment.strip()
        ]
        if not segments:
            return DrugPolicyDecision(True, False, "unresolved_drug")
        for segment in segments:
            entries = _candidate_entries(segment)
            if not entries:
                return DrugPolicyDecision(True, False, "not_in_otc_list")
            constrained_entries = [
                entry
                for entry in entries
                if _entry_matches_explicit_constraints(entry, question)
            ]
            if not constrained_entries:
                return DrugPolicyDecision(True, False, "outside_otc_constraints")
            matched_entries.extend(constrained_entries)

    names = tuple(sorted({entry.name for entry in matched_entries}))
    return DrugPolicyDecision(True, True, "otc_allowed", names)
