"""Build the user-facing and prompt-safe medical profile.

Transient symptoms live in conversation context (Redis) and never reach
these tables. Confirmed facts outrank unconfirmed ones; the formatter
describes unconfirmed entries as "previously reported" and tells the LLM
to request confirmation when the fact is safety-relevant.
"""

from __future__ import annotations

import time
from typing import Any

from src.chat.profile.domain import (
    ALL_SECTIONS,
    AllergyEntry,
    Demographics,
    MedicalProfile,
    MedicationEntry,
    PregnancyEntry,
    ProblemEntry,
    SectionState,
    SECTION_FACT_TYPES,
)
from src.chat.profile import repository as repo


def build_medical_profile(
    owner_id: str,
    subject_id: str,
    *,
    facts: list | None = None,
    now: float | None = None,
) -> MedicalProfile:
    timestamp = time.time() if now is None else now
    subject = repo.get_subject(owner_id, subject_id)
    birth_date = (subject or {}).get("birth_date")
    age = _age_from_birth_date(birth_date, now=timestamp)
    demographics = Demographics(
        display_name=(subject or {}).get("display_name"),
        relationship=(subject or {}).get("relationship"),
        birth_date=birth_date,
        gender=(subject or {}).get("gender"),
        age=age,
    )
    if facts is None:
        facts = repo.list_profile_facts(owner_id, subject_id)

    section_states: dict[str, SectionState] = {}
    for section, status in repo.get_all_section_states(owner_id, subject_id).items():
        if section not in ALL_SECTIONS:
            continue
        section_states[section] = SectionState(section=section, status=status)

    problems: list[ProblemEntry] = []
    allergies: list[AllergyEntry] = []
    medications: list[MedicationEntry] = []
    pregnancy: list[PregnancyEntry] = []
    for fact in facts:
        if getattr(fact, "inactive", False) or getattr(fact, "superseded_by", None) is not None:
            continue
        if getattr(fact, "temporal_status", "") in {"refuted", "entered_in_error"}:
            continue
        if fact.section == "problems" and fact.fact_type in SECTION_FACT_TYPES["problems"]:
            problems.append(_to_problem(fact))
        elif fact.section == "allergies" and fact.fact_type in SECTION_FACT_TYPES["allergies"]:
            allergies.append(_to_allergy(fact))
        elif fact.section == "medications" and fact.fact_type in SECTION_FACT_TYPES["medications"]:
            medications.append(_to_medication(fact))
        elif fact.section == "pregnancy" and fact.fact_type in SECTION_FACT_TYPES["pregnancy"]:
            pregnancy.append(_to_pregnancy(fact))

    def _sort_key(entry: Any) -> tuple:
        return (
            0 if getattr(entry, "confirmed", False) else 1,
            0 if getattr(entry, "inactive", False) else 1,
            -float(getattr(entry, "updated_at", 0.0) or 0.0),
            getattr(entry, "profile_fact_id", "") or "",
        )

    problems.sort(key=_sort_key)
    allergies.sort(key=_sort_key)
    medications.sort(key=_sort_key)
    pregnancy.sort(key=_sort_key)

    counts = {
        "problems": len(problems),
        "allergies": len(allergies),
        "medications": len(medications),
        "pregnancy": len(pregnancy),
    }
    for section in ALL_SECTIONS:
        if counts.get(section, 0) > 0:
            section_states[section] = SectionState(
                section=section, status="has_entries", reviewed_at=timestamp,
            )
        elif section not in section_states:
            section_states[section] = SectionState(section=section, status="unknown")

    return MedicalProfile(
        owner_id=owner_id,
        subject_id=subject_id,
        demographics=demographics,
        problems=problems,
        allergies=allergies,
        medications=medications,
        pregnancy=pregnancy,
        section_states=section_states,
        built_at=timestamp,
    )


def is_projection_fresh(profile: MedicalProfile, *, now: float | None = None) -> bool:
    timestamp = time.time() if now is None else now
    return (timestamp - profile.built_at) < 30.0


def _age_from_birth_date(birth_date: str | None, *, now: float) -> int | None:
    if not birth_date:
        return None
    try:
        year, month, day = (int(part) for part in birth_date.split("-", 2))
    except (ValueError, AttributeError):
        return None
    from datetime import datetime
    today = datetime.utcfromtimestamp(now)
    age = today.year - year - ((today.month, today.day) < (month, day))
    return age if age >= 0 else None


def _to_problem(fact) -> ProblemEntry:
    value = fact.value or {}
    return ProblemEntry(
        profile_fact_id=fact.profile_fact_id,
        condition=str(fact.entity_id or value.get("name") or "không rõ"),
        clinical_status=(
            value.get("clinical_status")
            or ("active" if fact.temporal_status == "current" else fact.temporal_status)
        ),
        verification_status=fact.verification_status,
        onset=value.get("onset"),
        resolution=value.get("resolution"),
        severity=value.get("severity"),
        coding_system=fact.coding_system,
        coding_code=fact.coding_code,
        coding_display=fact.coding_display,
        source=fact.source_kind,
        confirmed=fact.verification_status == "confirmed",
        inactive=fact.inactive,
        updated_at=fact.updated_at,
    )


def _to_allergy(fact) -> AllergyEntry:
    value = fact.value or {}
    reactions = value.get("reactions") or []
    if isinstance(reactions, str):
        reactions = [reactions]
    return AllergyEntry(
        profile_fact_id=fact.profile_fact_id,
        agent=str(fact.entity_id or value.get("agent") or "không rõ"),
        type=str(value.get("type") or "allergy"),
        clinical_status=str(value.get("clinical_status") or fact.temporal_status),
        criticality=str(value.get("criticality") or "low"),
        reactions=[str(reaction) for reaction in reactions],
        reaction_severity=value.get("reaction_severity"),
        coding_system=fact.coding_system,
        coding_code=fact.coding_code,
        coding_display=fact.coding_display,
        source=fact.source_kind,
        confirmed=fact.verification_status == "confirmed",
        inactive=fact.inactive,
        updated_at=fact.updated_at,
    )


def _to_medication(fact) -> MedicationEntry:
    value = fact.value or {}
    return MedicationEntry(
        profile_fact_id=fact.profile_fact_id,
        medication=str(fact.entity_id or value.get("medication") or "không rõ"),
        status=str(value.get("status") or fact.temporal_status or "unknown"),
        effective_period_start=value.get("effective_period_start"),
        effective_period_end=value.get("effective_period_end"),
        dosage_text=value.get("dosage_text"),
        dose_value=value.get("dose_value"),
        dose_unit=value.get("dose_unit"),
        frequency=value.get("frequency"),
        route=value.get("route"),
        coding_system=fact.coding_system,
        coding_code=fact.coding_code,
        coding_display=fact.coding_display,
        source=fact.source_kind,
        confirmed=fact.verification_status == "confirmed",
        inactive=fact.inactive,
        updated_at=fact.updated_at,
    )


def _to_pregnancy(fact) -> PregnancyEntry:
    value = fact.value or {}
    status = value.get("status")
    if status is None:
        status = value.get("value")
    return PregnancyEntry(
        profile_fact_id=fact.profile_fact_id,
        status=str(status if status is not None else fact.temporal_status or "unknown"),
        estimated_due_date=value.get("estimated_due_date"),
        source=fact.source_kind,
        confirmed=fact.verification_status == "confirmed",
        inactive=fact.inactive,
        updated_at=fact.updated_at,
    )


# ---------------------------------------------------------------------------
# Prompt formatter
# ---------------------------------------------------------------------------


def format_profile_for_prompt(profile: MedicalProfile) -> str:
    parts: list[str] = []
    demo = profile.demographics
    label_parts: list[str] = []
    if demo.display_name:
        label_parts.append(demo.display_name)
    if demo.relationship:
        label_parts.append(f"({demo.relationship})")
    if label_parts:
        parts.append("Chủ thể: " + " ".join(label_parts))
    if demo.age is not None:
        parts.append(f"Tuổi (tính từ ngày sinh): {demo.age}")
    if demo.birth_date:
        parts.append(f"Ngày sinh: {demo.birth_date}")
    if demo.gender:
        parts.append(f"Giới tính hành chính: {demo.gender}")

    def _block(heading: str, entries: list, none_known_text: str) -> None:
        state = profile.section_states.get(heading)
        if entries:
            lines = [_format_entry(heading, entry) for entry in entries]
            parts.append(f"{_heading(heading)}:\n" + "\n".join(f"- {line}" for line in lines))
        elif state and state.status == "none_known":
            parts.append(f"{_heading(heading)}: {none_known_text}")

    _block(
        "problems",
        profile.problems,
        "Người dùng đã xác nhận không có bệnh nền.",
    )
    _block(
        "allergies",
        profile.allergies,
        "Người dùng đã xác nhận không có dị ứng nào.",
    )
    _block(
        "medications",
        profile.medications,
        "Người dùng đã xác nhận không dùng thuốc thường xuyên.",
    )
    _block(
        "pregnancy",
        profile.pregnancy,
        "Người dùng đã xác nhận không liên quan tới thai kỳ.",
    )

    unconfirmed = sum(
        1 for entries in (
            profile.problems, profile.allergies,
            profile.medications, profile.pregnancy,
        )
        for entry in entries
        if not getattr(entry, "confirmed", False)
    )
    if unconfirmed:
        parts.append(
            "Lưu ý: Một số thông tin trong hồ sơ chưa được người dùng xác nhận. "
            "Hãy diễn đạt các thông tin đó là \"đã được ghi nhận trước đó\", "
            "tránh khẳng định chắc chắn và hỏi xác nhận khi chúng ảnh hưởng tới an toàn."
        )
    return "\n\n".join(parts)


def _heading(section: str) -> str:
    return {
        "problems": "Vấn đề sức khỏe (problems)",
        "allergies": "Dị ứng (allergies)",
        "medications": "Thuốc đang dùng (medications)",
        "pregnancy": "Thai kỳ (pregnancy)",
    }.get(section, section)


def _format_entry(section: str, entry) -> str:
    if section == "problems":
        status = "đã xác nhận" if entry.confirmed else "đã ghi nhận trước đó (chưa xác nhận)"
        line = f"{entry.condition} — trạng thái lâm sàng: {entry.clinical_status} — {status}"
        if entry.onset:
            line += f"; khởi phát: {entry.onset}"
        if entry.resolution:
            line += f"; hết: {entry.resolution}"
        if entry.severity:
            line += f"; mức độ: {entry.severity}"
        return line
    if section == "allergies":
        status = "đã xác nhận" if entry.confirmed else "đã ghi nhận trước đó (chưa xác nhận)"
        line = (
            f"{entry.agent} — loại: {entry.type} — mức quan trọng: {entry.criticality} — {status}"
        )
        if entry.reactions:
            line += f"; phản ứng: {', '.join(entry.reactions)}"
        if entry.reaction_severity:
            line += f"; mức độ phản ứng: {entry.reaction_severity}"
        return line
    if section == "medications":
        status = "đã xác nhận" if entry.confirmed else "đã ghi nhận trước đó (chưa xác nhận)"
        line = f"{entry.medication} — trạng thái: {entry.status} — {status}"
        if entry.dosage_text:
            line += f"; liều dùng: {entry.dosage_text}"
        elif entry.dose_value and entry.dose_unit:
            line += f"; liều: {entry.dose_value} {entry.dose_unit}"
        if entry.frequency:
            line += f"; tần suất: {entry.frequency}"
        if entry.route:
            line += f"; đường dùng: {entry.route}"
        return line
    if section == "pregnancy":
        status = "đã xác nhận" if entry.confirmed else "đã ghi nhận trước đó (chưa xác nhận)"
        line = f"thai kỳ: {entry.status} — {status}"
        if entry.estimated_due_date:
            line += f"; dự sinh: {entry.estimated_due_date}"
        return line
    return str(entry)
