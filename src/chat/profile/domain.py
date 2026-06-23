"""Domain model for the IPS-inspired medical profile subsystem.

Conversation context types (SessionState, Turn, ClinicalCase, etc.) live in
`src.chat.context.domain` to keep durable and session-only state separate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

ALL_SECTIONS: tuple[str, ...] = ("problems", "allergies", "medications", "pregnancy")

ALLERGY_TYPES: tuple[str, ...] = ("allergy", "intolerance")

# Section-aware fact_type -> section mapping. Anything outside this list is
# transient (symptoms, age) and excluded from the formal profile.
SECTION_FACT_TYPES: dict[str, frozenset[str]] = {
    "problems": frozenset({"chronic_disease", "diagnosis"}),
    "allergies": frozenset({"allergy"}),
    "medications": frozenset({"medication_use"}),
    "pregnancy": frozenset({"pregnancy_status"}),
}

VERIFICATION_STATUSES: tuple[str, ...] = (
    "unconfirmed", "confirmed", "refuted", "entered_in_error",
)
SOURCE_KINDS: tuple[str, ...] = ("chat_explicit", "profile_manual", "profile_edit")
REPORTER_ROLES: tuple[str, ...] = ("patient", "related_person")
SECTION_STATUSES: tuple[str, ...] = ("unknown", "none_known", "has_entries")


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


@dataclass
class ProfileSubject:
    subject_id: str
    owner_id: str
    relationship: str
    display_name: str | None = None
    birth_date: str | None = None
    gender: str | None = None
    created_at: float | None = None
    updated_at: float | None = None


@dataclass
class ProfileFact:
    profile_fact_id: str
    owner_id: str
    subject_id: str
    section: str
    fact_type: str
    entity_type: str | None
    entity_id: str | None
    attribute: str
    value: dict
    temporal_status: str
    confidence: float
    verification_status: str = "unconfirmed"
    source_kind: str = "chat_explicit"
    reporter_role: str | None = None
    valid_from: float | None = None
    valid_until: float | None = None
    superseded_by: str | None = None
    inactive: bool = False
    coding_system: str | None = None
    coding_code: str | None = None
    coding_display: str | None = None
    source_turn_id: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
    confirmed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_prompt_dict(self) -> dict[str, Any]:
        """Return clinical content without storage or identity metadata."""
        return {
            "fact_type": self.fact_type,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "attribute": self.attribute,
            "value": self.value,
            "temporal_status": self.temporal_status,
            "confidence": self.confidence,
            "verification_status": self.verification_status,
            "source_kind": self.source_kind,
        }


# ---------------------------------------------------------------------------
# IPS-aligned entry shapes used by the projection layer
# ---------------------------------------------------------------------------


@dataclass
class Demographics:
    display_name: str | None = None
    relationship: str | None = None
    birth_date: str | None = None
    gender: str | None = None
    age: int | None = None


@dataclass
class ProblemEntry:
    profile_fact_id: str
    condition: str
    clinical_status: str
    verification_status: str
    onset: str | None = None
    resolution: str | None = None
    severity: str | None = None
    coding_system: str | None = None
    coding_code: str | None = None
    coding_display: str | None = None
    source: str = "chat_explicit"
    confirmed: bool = False
    inactive: bool = False
    updated_at: float = 0.0


@dataclass
class AllergyEntry:
    profile_fact_id: str
    agent: str
    type: str
    clinical_status: str
    criticality: str
    reactions: list[str] = field(default_factory=list)
    reaction_severity: str | None = None
    coding_system: str | None = None
    coding_code: str | None = None
    coding_display: str | None = None
    source: str = "chat_explicit"
    confirmed: bool = False
    inactive: bool = False
    updated_at: float = 0.0


@dataclass
class MedicationEntry:
    profile_fact_id: str
    medication: str
    status: str
    effective_period_start: str | None = None
    effective_period_end: str | None = None
    dosage_text: str | None = None
    dose_value: str | None = None
    dose_unit: str | None = None
    frequency: str | None = None
    route: str | None = None
    coding_system: str | None = None
    coding_code: str | None = None
    coding_display: str | None = None
    source: str = "chat_explicit"
    confirmed: bool = False
    inactive: bool = False
    updated_at: float = 0.0


@dataclass
class PregnancyEntry:
    profile_fact_id: str
    status: str
    estimated_due_date: str | None = None
    source: str = "chat_explicit"
    confirmed: bool = False
    inactive: bool = False
    updated_at: float = 0.0


@dataclass
class SectionState:
    section: str
    status: str
    reviewed_at: float | None = None


@dataclass
class MedicalProfile:
    owner_id: str
    subject_id: str
    demographics: Demographics
    problems: list[ProblemEntry] = field(default_factory=list)
    allergies: list[AllergyEntry] = field(default_factory=list)
    medications: list[MedicationEntry] = field(default_factory=list)
    pregnancy: list[PregnancyEntry] = field(default_factory=list)
    section_states: dict[str, SectionState] = field(default_factory=dict)
    built_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner_id": self.owner_id,
            "subject_id": self.subject_id,
            "demographics": asdict(self.demographics),
            "problems": [asdict(p) for p in self.problems],
            "allergies": [asdict(a) for a in self.allergies],
            "medications": [asdict(m) for m in self.medications],
            "pregnancy": [asdict(p) for p in self.pregnancy],
            "section_states": {
                section: asdict(state) for section, state in self.section_states.items()
            },
            "built_at": self.built_at,
        }
