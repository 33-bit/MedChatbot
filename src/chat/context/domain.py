"""Conversation-context types: session-scoped, Redis-only.

Symptoms, the active diagnostic case, the active subject, recent turns, and
clarification state. Profile facts are not stored here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from src.chat.profile.domain import ProfileFact


@dataclass
class Turn:
    turn_id: str
    role: str
    content: str
    created_at: float
    subject_id: str | None = None


@dataclass
class ContextReference:
    reference_type: str
    reference_id: str
    source: str
    confidence: float


@dataclass
class ClarificationState:
    question: str
    subject_id: str | None = None
    created_at: float | None = None


@dataclass
class SessionState:
    session_id: str
    owner_id: str
    recent_turns: list[Turn] = field(default_factory=list)
    active_subject_id: str | None = None
    active_entity_refs: list[ContextReference] = field(default_factory=list)
    active_case_id: str | None = None
    pending_clarification: ClarificationState | None = None
    revision: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionState":
        return cls(
            session_id=str(data["session_id"]),
            owner_id=str(data["owner_id"]),
            recent_turns=[Turn(**turn) for turn in data.get("recent_turns", [])],
            active_subject_id=data.get("active_subject_id"),
            active_entity_refs=[
                ContextReference(**reference)
                for reference in data.get("active_entity_refs", [])
            ],
            active_case_id=data.get("active_case_id"),
            pending_clarification=(
                ClarificationState(**data["pending_clarification"])
                if data.get("pending_clarification")
                else None
            ),
            revision=int(data.get("revision", 0)),
        )


@dataclass
class ClinicalCase:
    case_id: str
    subject_id: str
    symptoms: list[dict] = field(default_factory=list)
    candidate_diseases: list[dict] = field(default_factory=list)
    answered_questions: list[str] = field(default_factory=list)
    clarification_queue: list[str] = field(default_factory=list)
    status: str = "active"
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class ConversationContextBundle:
    subject: dict | None
    safety_profile: list
    relevant_facts: list
    active_case: ClinicalCase | None
    reference_turns: list[dict]
    excluded_reason: str | None = None
    selection_reasons: dict[str, list[str]] = field(default_factory=dict)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "safety_profile": [fact.to_prompt_dict() for fact in self.safety_profile],
            "relevant_facts": [fact.to_prompt_dict() for fact in self.relevant_facts],
            "active_case": self.active_case.__dict__ if self.active_case else None,
            "reference_turns": self.reference_turns,
            "excluded_reason": self.excluded_reason,
        }
