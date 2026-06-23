"""Subject resolution for the conversation context.

Resolves which clinical subject a turn refers to (self / family). Conservative:
ambiguous cases yield a clarification prompt instead of a guess.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from src.chat.context.domain import ContextReference, SessionState

_TRUSTED_SOURCES = {"explicit", "ui", "pronoun", "history"}
_SUBJECT_LABELS = {
    "self": "bạn",
    "toi": "bạn",
    "father": "bố bạn",
    "bo": "bố bạn",
    "cha": "bố bạn",
    "mother": "mẹ bạn",
    "me": "mẹ bạn",
    "grandfather": "ông bạn",
    "grandmother": "bà bạn",
    "husband": "chồng bạn",
    "wife": "vợ bạn",
    "spouse": "vợ/chồng bạn",
    "son": "con trai bạn",
    "daughter": "con gái bạn",
    "child": "con bạn",
    "brother": "anh/em trai của bạn",
    "sister": "chị/em gái của bạn",
}


@dataclass(frozen=True)
class SubjectResolution:
    subject_id: str | None
    source: str | None
    confidence: float
    ambiguous: bool
    clarification: str


def resolve_subject(context: dict, state: SessionState) -> SubjectResolution:
    """Resolve a subject conservatively before any durable-profile lookup."""
    if context.get("ambiguous"):
        return SubjectResolution(
            None,
            None,
            0.0,
            True,
            context.get("clarification") or "Thông tin này đang nói về ai?",
        )

    subject = context.get("subject") if isinstance(context.get("subject"), dict) else {}
    subject_id = subject.get("id")
    source = str(subject.get("source") or "").lower()
    confidence = _confidence(subject.get("confidence"))
    if subject_id and source in _TRUSTED_SOURCES and confidence >= 0.8:
        return SubjectResolution(str(subject_id), source, confidence, False, "")

    if state.active_subject_id and not subject_id:
        active_confidence = _confidence(context.get("active_subject_confidence"))
        if active_confidence >= 0.9:
            return SubjectResolution(
                state.active_subject_id,
                "history",
                active_confidence,
                False,
                "",
            )

    needs_profile = bool(context.get("needs_medical_profile"))
    if needs_profile:
        return SubjectResolution(
            None,
            None,
            confidence,
            True,
            context.get("clarification") or "Câu hỏi này đang nói về bạn hay người khác?",
        )
    return SubjectResolution(None, None, confidence, False, "")


def context_references(context: dict) -> list[ContextReference]:
    references: list[ContextReference] = []
    for raw in context.get("references", []):
        if not isinstance(raw, dict):
            continue
        reference_type = raw.get("type") or raw.get("reference_type")
        reference_id = raw.get("id") or raw.get("reference_id")
        if not reference_type or not reference_id:
            continue
        references.append(ContextReference(
            reference_type=str(reference_type),
            reference_id=str(reference_id),
            source=str(raw.get("source") or "inferred"),
            confidence=_confidence(raw.get("confidence")),
        ))
    return references


def format_subject_address(subject: dict | None) -> str:
    """Return a safe Vietnamese label for the medical subject."""
    if not isinstance(subject, dict):
        return "bạn"
    raw = _clean_label(
        subject.get("relationship")
        or subject.get("id")
        or subject.get("subject_id")
    )
    if raw:
        key = _label_key(raw)
        if key in _SUBJECT_LABELS:
            return _SUBJECT_LABELS[key]
    display_name = _clean_label(subject.get("display_name"))
    if display_name:
        return display_name
    if not raw:
        return "bạn"
    if raw.startswith("person:"):
        return raw.split(":", 1)[1].strip() or "người bệnh"
    return raw.replace(" của tôi", " của bạn").replace(" tôi", " bạn")


def _confidence(value: object) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _clean_label(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.replace("\n", " ").replace("\r", " ").split())[:80]


def _label_key(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.casefold())
    ascii_text = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    return " ".join(ascii_text.replace("_", " ").replace("-", " ").split())
