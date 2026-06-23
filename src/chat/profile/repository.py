"""Repository for the medical profile subsystem.

Owner-scoped SQLite access for subjects, profile facts, section states, and
preferences. Redis handles only short-lived callback tokens and pending edits
(see `src/chat/profile/ui_state.py`).
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from threading import RLock

from src.chat.clients import get_sqlite
from src.chat.profile.domain import (
    ALL_SECTIONS,
    ProfileFact,
    ProfileSubject,
    REPORTER_ROLES,
    SECTION_FACT_TYPES,
    SECTION_STATUSES,
    SOURCE_KINDS,
    VERIFICATION_STATUSES,
)

from src import config as _config
from src.chat.security.identity import is_owner_key

log = __import__("logging").getLogger(__name__)
_SQLITE_LOCK = RLock()

_FACT_COLUMNS = (
    "profile_fact_id, owner_id, subject_id, section, fact_type, entity_type, "
    "entity_id, attribute, value_json, temporal_status, confidence, "
    "verification_status, source_kind, reporter_role, valid_from, valid_until, "
    "superseded_by, inactive, coding_system, coding_code, coding_display, "
    "source_turn_id, created_at, updated_at, confirmed_at"
)


# ---------------------------------------------------------------------------
# Owner-key guard
# ---------------------------------------------------------------------------


def _require_owner_key(owner_id: str) -> None:
    if not is_owner_key(owner_id):
        raise ValueError("Profile operations require a pseudonymous owner key")


# ---------------------------------------------------------------------------
# Subject operations
# ---------------------------------------------------------------------------


def ensure_subject(
    owner_id: str,
    subject_id: str,
    *,
    relationship: str | None = None,
    display_name: str | None = None,
    birth_date: str | None = None,
    gender: str | None = None,
    now: float | None = None,
) -> None:
    _require_owner_key(owner_id)
    timestamp = time.time() if now is None else now
    with _SQLITE_LOCK:
        conn = get_sqlite()
        conn.execute(
            "INSERT INTO medical_profile_subject "
            "(subject_id, owner_id, relationship, display_name, "
            "birth_date, gender, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(owner_id, subject_id) DO UPDATE SET "
            "relationship=excluded.relationship, "
            "display_name=COALESCE(excluded.display_name, medical_profile_subject.display_name), "
            "birth_date=COALESCE(excluded.birth_date, medical_profile_subject.birth_date), "
            "gender=COALESCE(excluded.gender, medical_profile_subject.gender), "
            "updated_at=excluded.updated_at",
            (
                subject_id,
                owner_id,
                relationship or subject_id,
                display_name,
                birth_date,
                gender,
                timestamp,
                timestamp,
            ),
        )
        conn.commit()


def list_subjects(owner_id: str) -> list[dict]:
    _require_owner_key(owner_id)
    with _SQLITE_LOCK:
        rows = get_sqlite().execute(
            "SELECT subject_id, relationship, display_name, "
            "birth_date, gender, created_at, updated_at "
            "FROM medical_profile_subject WHERE owner_id = ? "
            "ORDER BY updated_at DESC",
            (owner_id,),
        ).fetchall()
    return [
        {
            "subject_id": row[0],
            "relationship": row[1],
            "display_name": row[2],
            "birth_date": row[3],
            "gender": row[4],
            "created_at": row[5],
            "updated_at": row[6],
        }
        for row in rows
    ]


def get_subject(owner_id: str, subject_id: str) -> dict | None:
    _require_owner_key(owner_id)
    with _SQLITE_LOCK:
        row = get_sqlite().execute(
            "SELECT subject_id, owner_id, relationship, display_name, "
            "birth_date, gender, created_at, updated_at "
            "FROM medical_profile_subject "
            "WHERE owner_id = ? AND subject_id = ?",
            (owner_id, subject_id),
        ).fetchone()
    if not row:
        return None
    return {
        "subject_id": row[0],
        "owner_id": row[1],
        "relationship": row[2],
        "display_name": row[3],
        "birth_date": row[4],
        "gender": row[5],
        "created_at": row[6],
        "updated_at": row[7],
    }


def update_subject_demographics(
    owner_id: str,
    subject_id: str,
    *,
    display_name: str | None = None,
    birth_date: str | None = None,
    gender: str | None = None,
    expected_updated_at: float | None,
    now: float | None = None,
    clear_birth_date: bool = False,
    clear_gender: bool = False,
) -> dict | None:
    _require_owner_key(owner_id)
    timestamp = time.time() if now is None else now
    display_name_clean = display_name.strip() if isinstance(display_name, str) else None
    birth_date_clean = birth_date.strip() if isinstance(birth_date, str) else None
    gender_clean = gender.strip() if isinstance(gender, str) else None
    with _SQLITE_LOCK:
        conn = get_sqlite()
        try:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT updated_at FROM medical_profile_subject "
                "WHERE owner_id = ? AND subject_id = ?",
                (owner_id, subject_id),
            ).fetchone()
            if not current:
                conn.rollback()
                return None
            current_updated_at = current[0]
            if (
                expected_updated_at is not None
                and abs(float(current_updated_at) - float(expected_updated_at)) > 1e-6
            ):
                conn.rollback()
                return None
            set_clauses = ["updated_at = ?"]
            params: list[object] = [timestamp]
            if display_name is not None:
                set_clauses.append("display_name = ?")
                params.append(display_name_clean or None)
            if clear_birth_date:
                set_clauses.append("birth_date = NULL")
            elif birth_date is not None:
                set_clauses.append("birth_date = ?")
                params.append(birth_date_clean or None)
            if clear_gender:
                set_clauses.append("gender = NULL")
            elif gender is not None:
                set_clauses.append("gender = ?")
                params.append(gender_clean or None)
            params.extend([owner_id, subject_id])
            conn.execute(
                f"UPDATE medical_profile_subject SET {', '.join(set_clauses)} "
                "WHERE owner_id = ? AND subject_id = ?",
                params,
            )
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise
    return get_subject(owner_id, subject_id)


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------


def get_user_preference(owner_id: str, preference: str) -> bool | None:
    _require_owner_key(owner_id)
    if preference not in {"storage", "personalization"}:
        raise ValueError(f"Invalid preference: {preference}")
    with _SQLITE_LOCK:
        row = get_sqlite().execute(
            "SELECT enabled FROM medical_profile_preference "
            "WHERE owner_id = ? AND preference = ?",
            (owner_id, preference),
        ).fetchone()
    return bool(row[0]) if row else None


def set_user_preference(
    owner_id: str,
    preference: str,
    enabled: bool,
    *,
    now: float | None = None,
) -> None:
    _require_owner_key(owner_id)
    if preference not in {"storage", "personalization"}:
        raise ValueError(f"Invalid preference: {preference}")
    timestamp = time.time() if now is None else now
    consented_at = timestamp if enabled else None
    with _SQLITE_LOCK:
        conn = get_sqlite()
        conn.execute(
            "INSERT INTO medical_profile_preference "
            "(owner_id, preference, enabled, consented_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(owner_id, preference) DO UPDATE SET "
            "enabled=excluded.enabled, "
            "consented_at=CASE WHEN excluded.enabled = 1 "
            "THEN COALESCE(medical_profile_preference.consented_at, excluded.consented_at) "
            "ELSE NULL END, updated_at=excluded.updated_at",
            (owner_id, preference, int(enabled), consented_at, timestamp),
        )
        conn.commit()


def get_all_user_preferences(owner_id: str) -> dict[str, bool | None]:
    _require_owner_key(owner_id)
    with _SQLITE_LOCK:
        rows = get_sqlite().execute(
            "SELECT preference, enabled FROM medical_profile_preference "
            "WHERE owner_id = ?",
            (owner_id,),
        ).fetchall()
    return {row[0]: bool(row[1]) for row in rows}


# ---------------------------------------------------------------------------
# Section state
# ---------------------------------------------------------------------------


def get_section_state(
    owner_id: str,
    subject_id: str,
    section: str,
) -> str | None:
    _require_owner_key(owner_id)
    if section not in ALL_SECTIONS:
        return None
    with _SQLITE_LOCK:
        row = get_sqlite().execute(
            "SELECT status FROM medical_profile_section_state "
            "WHERE owner_id = ? AND subject_id = ? AND section = ?",
            (owner_id, subject_id, section),
        ).fetchone()
    return row[0] if row else None


def get_all_section_states(
    owner_id: str,
    subject_id: str,
) -> dict[str, str]:
    _require_owner_key(owner_id)
    with _SQLITE_LOCK:
        rows = get_sqlite().execute(
            "SELECT section, status FROM medical_profile_section_state "
            "WHERE owner_id = ? AND subject_id = ?",
            (owner_id, subject_id),
        ).fetchall()
    return {row[0]: row[1] for row in rows}


def set_section_status(
    owner_id: str,
    subject_id: str,
    section: str,
    status: str,
    *,
    now: float | None = None,
) -> None:
    _require_owner_key(owner_id)
    if section not in ALL_SECTIONS:
        raise ValueError(f"Invalid section: {section}")
    if status not in SECTION_STATUSES:
        raise ValueError(f"Invalid section status: {status}")
    timestamp = time.time() if now is None else now
    with _SQLITE_LOCK:
        conn = get_sqlite()
        conn.execute(
            "INSERT INTO medical_profile_section_state "
            "(owner_id, subject_id, section, status, reviewed_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(owner_id, subject_id, section) DO UPDATE SET "
            "status=excluded.status, reviewed_at=excluded.reviewed_at",
            (owner_id, subject_id, section, status, timestamp),
        )
        conn.commit()


def recompute_section_status(
    owner_id: str,
    subject_id: str,
    section: str,
    *,
    now: float | None = None,
) -> str:
    _require_owner_key(owner_id)
    timestamp = time.time() if now is None else now
    with _SQLITE_LOCK:
        row = get_sqlite().execute(
            "SELECT 1 FROM medical_profile_fact "
            "WHERE owner_id = ? AND subject_id = ? AND section = ? "
            "AND superseded_by IS NULL AND inactive = 0 LIMIT 1",
            (owner_id, subject_id, section),
        ).fetchone()
    new_status = "has_entries" if row else "unknown"
    set_section_status(owner_id, subject_id, section, new_status, now=timestamp)
    return new_status


# ---------------------------------------------------------------------------
# Fact lifecycle
# ---------------------------------------------------------------------------


def _fact_values(fact: ProfileFact) -> tuple:
    return (
        fact.profile_fact_id,
        fact.owner_id,
        fact.subject_id,
        fact.section,
        fact.fact_type,
        fact.entity_type,
        fact.entity_id,
        fact.attribute,
        json.dumps(fact.value, ensure_ascii=False, sort_keys=True),
        fact.temporal_status,
        fact.confidence,
        fact.verification_status,
        fact.source_kind,
        fact.reporter_role,
        fact.valid_from,
        fact.valid_until,
        fact.superseded_by,
        int(bool(fact.inactive)),
        fact.coding_system,
        fact.coding_code,
        fact.coding_display,
        fact.source_turn_id,
        fact.created_at,
        fact.updated_at,
        fact.confirmed_at,
    )


def _row_to_fact(row: tuple) -> ProfileFact:
    values = list(row)
    values[8] = json.loads(values[8])
    # `inactive` is stored as INTEGER (0/1); convert to bool for downstream use.
    values[17] = bool(values[17])
    return ProfileFact(*values)


def list_profile_facts(
    owner_id: str,
    subject_id: str | None = None,
    *,
    include_superseded: bool = False,
) -> list[ProfileFact]:
    _require_owner_key(owner_id)
    clauses = ["owner_id = ?"]
    params: list[object] = [owner_id]
    if subject_id is not None:
        clauses.append("subject_id = ?")
        params.append(subject_id)
    if not include_superseded:
        clauses.append("superseded_by IS NULL")
    with _SQLITE_LOCK:
        rows = get_sqlite().execute(
            f"SELECT {_FACT_COLUMNS} FROM medical_profile_fact "
            f"WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC",
            params,
        ).fetchall()
    return [_row_to_fact(row) for row in rows]


def list_facts_paginated(
    owner_id: str,
    subject_id: str,
    *,
    page: int = 0,
    page_size: int = 5,
    include_superseded: bool = False,
    include_inactive: bool = False,
) -> list[ProfileFact]:
    _require_owner_key(owner_id)
    if page < 0:
        page = 0
    if page_size <= 0:
        page_size = 5
    clauses = ["owner_id = ?", "subject_id = ?"]
    if not include_superseded:
        clauses.append("superseded_by IS NULL")
    if not include_inactive:
        clauses.append("inactive = 0")
    offset = page * page_size
    with _SQLITE_LOCK:
        rows = get_sqlite().execute(
            f"SELECT {_FACT_COLUMNS} FROM medical_profile_fact "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY updated_at DESC, profile_fact_id ASC "
            "LIMIT ? OFFSET ?",
            (owner_id, subject_id, page_size, offset),
        ).fetchall()
    return [_row_to_fact(row) for row in rows]


def count_active_facts(owner_id: str, subject_id: str) -> int:
    _require_owner_key(owner_id)
    with _SQLITE_LOCK:
        row = get_sqlite().execute(
            "SELECT COUNT(*) FROM medical_profile_fact "
            "WHERE owner_id = ? AND subject_id = ? "
            "AND superseded_by IS NULL AND inactive = 0",
            (owner_id, subject_id),
        ).fetchone()
    return int(row[0]) if row else 0


def count_facts(
    owner_id: str,
    subject_id: str,
    *,
    include_superseded: bool = False,
) -> int:
    _require_owner_key(owner_id)
    clauses = ["owner_id = ?", "subject_id = ?"]
    if not include_superseded:
        clauses.append("superseded_by IS NULL")
    with _SQLITE_LOCK:
        row = get_sqlite().execute(
            f"SELECT COUNT(*) FROM medical_profile_fact "
            f"WHERE {' AND '.join(clauses)}",
            (owner_id, subject_id),
        ).fetchone()
    return int(row[0]) if row else 0


def count_subjects(owner_id: str) -> int:
    _require_owner_key(owner_id)
    with _SQLITE_LOCK:
        row = get_sqlite().execute(
            "SELECT COUNT(*) FROM medical_profile_subject WHERE owner_id = ?",
            (owner_id,),
        ).fetchone()
    return int(row[0]) if row else 0


def get_fact(owner_id: str, profile_fact_id: str) -> ProfileFact | None:
    _require_owner_key(owner_id)
    with _SQLITE_LOCK:
        row = get_sqlite().execute(
            f"SELECT {_FACT_COLUMNS} FROM medical_profile_fact "
            "WHERE owner_id = ? AND profile_fact_id = ?",
            (owner_id, profile_fact_id),
        ).fetchone()
    return _row_to_fact(row) if row else None


def _ensure_subject_in_transaction(
    conn: sqlite3.Connection,
    subject_id: str,
    owner_id: str,
    timestamp: float,
) -> None:
    conn.execute(
        "INSERT INTO medical_profile_subject "
        "(subject_id, owner_id, relationship, display_name, "
        "birth_date, gender, created_at, updated_at) "
        "VALUES (?, ?, ?, NULL, NULL, NULL, ?, ?) "
        "ON CONFLICT(owner_id, subject_id) DO UPDATE SET updated_at=excluded.updated_at",
        (subject_id, owner_id, subject_id, timestamp, timestamp),
    )


def write_profile_candidates(
    *,
    owner_id: str,
    resolved_subject_id: str,
    profile_candidates: list[dict],
    source_turn_id: str,
    now: float | None = None,
) -> list[ProfileFact]:
    """Validate and transactionally upsert profile facts extracted from a turn.

    Only chat_explicit facts with sufficient confidence persist. Inferred
    facts are silently dropped (never become profile candidates).
    """
    _require_owner_key(owner_id)
    timestamp = time.time() if now is None else now
    prepared: list[ProfileFact] = []
    for candidate in profile_candidates:
        fact = _candidate_to_fact(
            candidate,
            owner_id=owner_id,
            resolved_subject_id=resolved_subject_id,
            source_turn_id=source_turn_id,
            timestamp=timestamp,
        )
        if fact is not None:
            prepared.append(fact)
    if not prepared:
        return []

    written: list[ProfileFact] = []
    with _SQLITE_LOCK:
        conn = get_sqlite()
        try:
            conn.execute("BEGIN IMMEDIATE")
            for fact in prepared:
                _ensure_subject_in_transaction(
                    conn, fact.subject_id, fact.owner_id, timestamp,
                )
                duplicate = _find_duplicate(conn, fact)
                if duplicate is not None:
                    written.append(duplicate)
                    continue
                conn.execute(
                    "INSERT INTO medical_profile_fact "
                    f"({_FACT_COLUMNS}) VALUES ({', '.join('?' for _ in range(25))})",
                    _fact_values(fact),
                )
                _supersede_matching(
                    conn, fact, timestamp=timestamp,
                )
                written.append(fact)
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise
    return written


def _candidate_to_fact(
    candidate: object,
    *,
    owner_id: str,
    resolved_subject_id: str,
    source_turn_id: str,
    timestamp: float,
) -> ProfileFact | None:
    if not isinstance(candidate, dict):
        return None
    fact_type = candidate.get("fact_type")
    entity_type = candidate.get("entity_type")
    temporal_status = candidate.get("temporal_status", "unknown")
    source = str(candidate.get("source") or "")
    confidence = _confidence(candidate.get("confidence"))
    if (
        fact_type not in _ALLOWED_FACT_TYPES
        or entity_type not in _ALLOWED_ENTITY_TYPES
        or temporal_status not in _ALLOWED_TEMPORAL_STATUS
        or source not in _PERSISTABLE_SOURCES
        or (source == "explicit" and confidence < 0.8)
        or (source != "explicit" and confidence < 0.7)
        or not isinstance(candidate.get("attribute"), str)
        or not candidate["attribute"].strip()
        or not isinstance(candidate.get("value"), dict)
    ):
        return None

    subject_id = str(candidate.get("subject_id") or resolved_subject_id)
    if not subject_id or subject_id != resolved_subject_id:
        return None

    section = candidate.get("section")
    if section not in ALL_SECTIONS:
        section = _derive_section(fact_type)
    if section is None:
        return None  # transient / non-profile fact, drop silently

    valid_until = _optional_float(candidate.get("valid_until"))
    if valid_until is None and subject_id != "self":
        valid_until = timestamp + _config.PROFILE_THIRD_PARTY_TTL_SECONDS

    verification_status = str(candidate.get("verification_status") or "unconfirmed")
    if verification_status not in VERIFICATION_STATUSES:
        verification_status = "unconfirmed"
    source_kind = str(candidate.get("source_kind") or "chat_explicit")
    if source_kind not in SOURCE_KINDS:
        source_kind = "chat_explicit"
    reporter_role = candidate.get("reporter_role")
    if reporter_role not in REPORTER_ROLES:
        reporter_role = "patient" if subject_id == "self" else "related_person"

    return ProfileFact(
        profile_fact_id=str(candidate.get("profile_fact_id") or uuid.uuid4().hex),
        owner_id=owner_id,
        subject_id=subject_id,
        section=section,
        fact_type=str(fact_type),
        entity_type=str(entity_type) if entity_type else None,
        entity_id=str(candidate["entity_id"]) if candidate.get("entity_id") else None,
        attribute=candidate["attribute"].strip(),
        value=candidate["value"],
        temporal_status=str(temporal_status),
        confidence=confidence,
        verification_status=verification_status,
        source_kind=source_kind,
        reporter_role=reporter_role,
        valid_from=_optional_float(candidate.get("valid_from")),
        valid_until=valid_until,
        superseded_by=None,
        created_at=timestamp,
        updated_at=timestamp,
        coding_system=str(candidate.get("coding_system")) if candidate.get("coding_system") else None,
        coding_code=str(candidate.get("coding_code")) if candidate.get("coding_code") else None,
        coding_display=str(candidate.get("coding_display")) if candidate.get("coding_display") else None,
        source_turn_id=source_turn_id,
    )


def _derive_section(fact_type: str) -> str | None:
    fact_type = str(fact_type or "")
    for section, fact_types in SECTION_FACT_TYPES.items():
        if fact_type in fact_types:
            return section
    return None


def _supersede_matching(
    conn: sqlite3.Connection,
    fact: ProfileFact,
    *,
    timestamp: float,
) -> None:
    conn.execute(
        "UPDATE medical_profile_fact "
        "SET superseded_by = ?, updated_at = ? "
        "WHERE owner_id = ? AND subject_id = ? AND fact_type = ? "
        "AND COALESCE(entity_type, '') = COALESCE(?, '') "
        "AND COALESCE(entity_id, '') = COALESCE(?, '') "
        "AND attribute = ? AND superseded_by IS NULL "
        "AND profile_fact_id != ?",
        (
            fact.profile_fact_id,
            timestamp,
            fact.owner_id,
            fact.subject_id,
            fact.fact_type,
            fact.entity_type,
            fact.entity_id,
            fact.attribute,
            fact.profile_fact_id,
        ),
    )


def _find_duplicate(
    conn: sqlite3.Connection, fact: ProfileFact
) -> ProfileFact | None:
    row = conn.execute(
        f"SELECT {_FACT_COLUMNS} FROM medical_profile_fact "
        "WHERE owner_id = ? AND subject_id = ? AND fact_type = ? "
        "AND COALESCE(entity_type, '') = COALESCE(?, '') "
        "AND COALESCE(entity_id, '') = COALESCE(?, '') "
        "AND attribute = ? AND value_json = ? AND temporal_status = ? "
        "AND superseded_by IS NULL LIMIT 1",
        (
            fact.owner_id,
            fact.subject_id,
            fact.fact_type,
            fact.entity_type,
            fact.entity_id,
            fact.attribute,
            json.dumps(fact.value, ensure_ascii=False, sort_keys=True),
            fact.temporal_status,
        ),
    ).fetchone()
    return _row_to_fact(row) if row else None


def write_profile_fact(
    *,
    owner_id: str,
    subject_id: str,
    fact_type: str,
    section: str,
    value: dict,
    attribute: str = "value",
    entity_type: str | None = None,
    entity_id: str | None = None,
    coding_system: str | None = None,
    coding_code: str | None = None,
    coding_display: str | None = None,
    temporal_status: str = "current",
    now: float | None = None,
) -> ProfileFact:
    """Create a confirmed profile fact (manual UI entry)."""
    _require_owner_key(owner_id)
    if section not in ALL_SECTIONS:
        raise ValueError(f"Invalid profile section: {section}")
    timestamp = time.time() if now is None else now
    fact = ProfileFact(
        profile_fact_id=uuid.uuid4().hex,
        owner_id=owner_id,
        subject_id=subject_id,
        section=section,
        fact_type=fact_type,
        entity_type=entity_type,
        entity_id=entity_id,
        attribute=attribute,
        value=value,
        temporal_status=temporal_status,
        confidence=1.0,
        verification_status="confirmed",
        source_kind="profile_manual",
        reporter_role="patient" if subject_id == "self" else "related_person",
        valid_from=None,
        valid_until=None,
        superseded_by=None,
        created_at=timestamp,
        updated_at=timestamp,
        confirmed_at=timestamp,
        coding_system=coding_system,
        coding_code=coding_code,
        coding_display=coding_display,
        source_turn_id=f"profile_{uuid.uuid4().hex}",
    )
    with _SQLITE_LOCK:
        conn = get_sqlite()
        try:
            conn.execute("BEGIN IMMEDIATE")
            _ensure_subject_in_transaction(conn, subject_id, owner_id, timestamp)
            conn.execute(
                "INSERT INTO medical_profile_fact "
                f"({_FACT_COLUMNS}) VALUES ({', '.join('?' for _ in range(25))})",
                _fact_values(fact),
            )
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise
    recompute_section_status(owner_id, subject_id, section, now=timestamp)
    return fact


def confirm_fact(
    owner_id: str,
    profile_fact_id: str,
    *,
    now: float | None = None,
) -> ProfileFact | None:
    _require_owner_key(owner_id)
    timestamp = time.time() if now is None else now
    with _SQLITE_LOCK:
        conn = get_sqlite()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT profile_fact_id FROM medical_profile_fact "
                "WHERE owner_id = ? AND profile_fact_id = ?",
                (owner_id, profile_fact_id),
            ).fetchone()
            if not row:
                conn.rollback()
                return None
            conn.execute(
                "UPDATE medical_profile_fact "
                "SET verification_status = 'confirmed', "
                "source_kind = 'profile_edit', "
                "valid_until = NULL, "
                "confirmed_at = ?, updated_at = ?, inactive = 0 "
                "WHERE owner_id = ? AND profile_fact_id = ?",
                (timestamp, timestamp, owner_id, profile_fact_id),
            )
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise
    return get_fact(owner_id, profile_fact_id)


def set_fact_verification(
    owner_id: str,
    profile_fact_id: str,
    verification_status: str,
    *,
    now: float | None = None,
) -> ProfileFact | None:
    _require_owner_key(owner_id)
    if verification_status not in VERIFICATION_STATUSES:
        raise ValueError(f"Invalid verification_status: {verification_status}")
    timestamp = time.time() if now is None else now
    with _SQLITE_LOCK:
        conn = get_sqlite()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT profile_fact_id FROM medical_profile_fact "
                "WHERE owner_id = ? AND profile_fact_id = ?",
                (owner_id, profile_fact_id),
            ).fetchone()
            if not row:
                conn.rollback()
                return None
            inactive = 1 if verification_status in {"refuted", "entered_in_error"} else 0
            confirmed_at = None if inactive else timestamp
            conn.execute(
                "UPDATE medical_profile_fact "
                "SET verification_status = ?, updated_at = ?, inactive = ?, "
                "confirmed_at = ? "
                "WHERE owner_id = ? AND profile_fact_id = ?",
                (verification_status, timestamp, inactive, confirmed_at, owner_id, profile_fact_id),
            )
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise
    return get_fact(owner_id, profile_fact_id)


def set_fact_inactive(
    owner_id: str,
    profile_fact_id: str,
    *,
    inactive: bool = True,
    now: float | None = None,
) -> ProfileFact | None:
    _require_owner_key(owner_id)
    timestamp = time.time() if now is None else now
    with _SQLITE_LOCK:
        conn = get_sqlite()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT profile_fact_id FROM medical_profile_fact "
                "WHERE owner_id = ? AND profile_fact_id = ?",
                (owner_id, profile_fact_id),
            ).fetchone()
            if not row:
                conn.rollback()
                return None
            conn.execute(
                "UPDATE medical_profile_fact "
                "SET inactive = ?, updated_at = ? "
                "WHERE owner_id = ? AND profile_fact_id = ?",
                (1 if inactive else 0, timestamp, owner_id, profile_fact_id),
            )
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise
    return get_fact(owner_id, profile_fact_id)


def replace_fact(
    owner_id: str,
    old_profile_fact_id: str,
    *,
    new_fact: ProfileFact,
    expected_updated_at: float | None,
    now: float | None = None,
) -> tuple[str, ProfileFact | None]:
    _require_owner_key(owner_id)
    if new_fact.owner_id != owner_id:
        raise ValueError("replace_fact: new_fact.owner_id does not match owner_id")
    timestamp = time.time() if now is None else now
    with _SQLITE_LOCK:
        conn = get_sqlite()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT updated_at, superseded_by FROM medical_profile_fact "
                "WHERE owner_id = ? AND profile_fact_id = ?",
                (owner_id, old_profile_fact_id),
            ).fetchone()
            if not row:
                conn.rollback()
                return ("missing", None)
            current_updated_at = float(row[0])
            current_superseded_by = row[1]
            if (
                expected_updated_at is not None
                and abs(current_updated_at - float(expected_updated_at)) > 1e-6
            ) or current_superseded_by is not None:
                conn.rollback()
                return ("stale", None)
            conn.execute(
                "INSERT INTO medical_profile_fact "
                f"({_FACT_COLUMNS}) VALUES ({', '.join('?' for _ in range(25))})",
                _fact_values(new_fact),
            )
            conn.execute(
                "UPDATE medical_profile_fact SET superseded_by = ?, updated_at = ? "
                "WHERE owner_id = ? AND profile_fact_id = ?",
                (new_fact.profile_fact_id, timestamp, owner_id, old_profile_fact_id),
            )
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise
    return ("ok", new_fact)


def delete_fact_with_lineage(owner_id: str, profile_fact_id: str) -> int:
    _require_owner_key(owner_id)
    with _SQLITE_LOCK:
        conn = get_sqlite()
        try:
            conn.execute("BEGIN IMMEDIATE")
            seed = conn.execute(
                "SELECT profile_fact_id FROM medical_profile_fact "
                "WHERE owner_id = ? AND profile_fact_id = ?",
                (owner_id, profile_fact_id),
            ).fetchone()
            if not seed:
                conn.rollback()
                return 0
            lineage: set[str] = set()
            queue: list[str] = [seed[0]]
            while queue:
                current = queue.pop()
                if current in lineage:
                    continue
                lineage.add(current)
                for neighbor in conn.execute(
                    "SELECT profile_fact_id FROM medical_profile_fact "
                    "WHERE owner_id = ? "
                    "AND (superseded_by = ? OR profile_fact_id = "
                    "(SELECT superseded_by FROM medical_profile_fact "
                    "WHERE owner_id = ? AND profile_fact_id = ?))",
                    (owner_id, current, owner_id, current),
                ).fetchall():
                    queue.append(neighbor[0])
            placeholders = ",".join("?" * len(lineage))
            cursor = conn.execute(
                f"DELETE FROM medical_profile_fact "
                f"WHERE owner_id = ? AND profile_fact_id IN ({placeholders})",
                (owner_id, *lineage),
            )
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise
    return cursor.rowcount


def count_fact_lineage(owner_id: str, profile_fact_id: str) -> int:
    _require_owner_key(owner_id)
    with _SQLITE_LOCK:
        conn = get_sqlite()
        seed = conn.execute(
            "SELECT profile_fact_id FROM medical_profile_fact "
            "WHERE owner_id = ? AND profile_fact_id = ?",
            (owner_id, profile_fact_id),
        ).fetchone()
        if not seed:
            return 0
        lineage: set[str] = set()
        queue: list[str] = [seed[0]]
        while queue:
            current = queue.pop()
            if current in lineage:
                continue
            lineage.add(current)
            for neighbor in conn.execute(
                "SELECT profile_fact_id FROM medical_profile_fact "
                "WHERE owner_id = ? "
                "AND (superseded_by = ? OR profile_fact_id = "
                "(SELECT superseded_by FROM medical_profile_fact "
                "WHERE owner_id = ? AND profile_fact_id = ?))",
                (owner_id, current, owner_id, current),
            ).fetchall():
                queue.append(neighbor[0])
    return len(lineage)


def delete_subject_profile(owner_id: str, subject_id: str) -> int:
    _require_owner_key(owner_id)
    with _SQLITE_LOCK:
        conn = get_sqlite()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "DELETE FROM medical_profile_fact "
                "WHERE owner_id = ? AND subject_id = ?",
                (owner_id, subject_id),
            )
            conn.execute(
                "DELETE FROM medical_profile_section_state "
                "WHERE owner_id = ? AND subject_id = ?",
                (owner_id, subject_id),
            )
            conn.execute(
                "DELETE FROM medical_profile_subject "
                "WHERE owner_id = ? AND subject_id = ?",
                (owner_id, subject_id),
            )
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise
    return cursor.rowcount


def delete_medical_profile(owner_id: str) -> int:
    _require_owner_key(owner_id)
    with _SQLITE_LOCK:
        conn = get_sqlite()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "DELETE FROM medical_profile_fact WHERE owner_id = ?",
                (owner_id,),
            )
            conn.execute(
                "DELETE FROM medical_profile_section_state WHERE owner_id = ?",
                (owner_id,),
            )
            conn.execute(
                "DELETE FROM medical_profile_subject WHERE owner_id = ?",
                (owner_id,),
            )
            conn.execute(
                "DELETE FROM medical_profile_preference WHERE owner_id = ?",
                (owner_id,),
            )
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise
    return cursor.rowcount


def migrate_owner_key(current_owner_id: str, previous_owner_ids: tuple[str, ...]) -> bool:
    """Move records from previous HMAC versions without exposing external IDs."""
    _require_owner_key(current_owner_id)
    previous = tuple(
        owner_id
        for owner_id in previous_owner_ids
        if owner_id != current_owner_id and is_owner_key(owner_id)
    )
    if not previous:
        return False
    with _SQLITE_LOCK:
        conn = get_sqlite()
        present = conn.execute(
            "SELECT 1 FROM medical_profile_subject WHERE owner_id = ? "
            "UNION SELECT 1 FROM medical_profile_fact WHERE owner_id = ? "
            "UNION SELECT 1 FROM medical_profile_preference WHERE owner_id = ? LIMIT 1",
            (current_owner_id, current_owner_id, current_owner_id),
        ).fetchone()
        if present:
            return False
        try:
            conn.execute("BEGIN IMMEDIATE")
            migrated = False
            for old_owner_id in previous:
                found = conn.execute(
                    "SELECT 1 FROM medical_profile_subject WHERE owner_id = ? "
                    "UNION SELECT 1 FROM medical_profile_fact WHERE owner_id = ? "
                    "UNION SELECT 1 FROM medical_profile_preference WHERE owner_id = ? LIMIT 1",
                    (old_owner_id, old_owner_id, old_owner_id),
                ).fetchone()
                if not found:
                    continue
                conn.execute(
                    "UPDATE medical_profile_subject SET owner_id = ? WHERE owner_id = ?",
                    (current_owner_id, old_owner_id),
                )
                conn.execute(
                    "UPDATE medical_profile_fact SET owner_id = ? WHERE owner_id = ?",
                    (current_owner_id, old_owner_id),
                )
                conn.execute(
                    "UPDATE medical_profile_section_state SET owner_id = ? WHERE owner_id = ?",
                    (current_owner_id, old_owner_id),
                )
                conn.execute(
                    "UPDATE medical_profile_preference SET owner_id = ? WHERE owner_id = ?",
                    (current_owner_id, old_owner_id),
                )
                migrated = True
                break
            conn.commit()
            return migrated
        except sqlite3.Error:
            conn.rollback()
            raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_ALLOWED_FACT_TYPES = {
    "age",
    "sex",
    "allergy",
    "chronic_disease",
    "medication_use",
    "pregnancy_status",
    "diagnosis",
    "symptom_state",
    "symptom_history",
}
_ALLOWED_ENTITY_TYPES = {None, "symptom", "drug", "disease", "procedure", "person"}
_ALLOWED_TEMPORAL_STATUS = {"current", "historical", "resolved", "unknown"}
_PERSISTABLE_SOURCES = {"explicit", "pronoun", "history"}


def _confidence(value: object) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
