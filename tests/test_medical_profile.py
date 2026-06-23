"""Tests for the IPS-inspired medical profile projection and storage helpers.

Covers:
- Fresh-schema and upgrade-path migrations
- Profile fact lifecycle (write, confirm, refute, inactive, section state)
- Independent storage / personalization consent
- Cautious prompt formatting for unconfirmed entries
- Ranking: confirmed entries outrank unconfirmed ones
- Selector excludes inactive / refuted / entered_in_error facts
- Self / family isolation in the profile projection
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile

import pytest

from src.chat.clients import get_sqlite
from src.chat.profile.domain import ProfileFact
from src.chat.profile.selector import build_context_bundle
from src.chat.profile.projection import (
    build_medical_profile,
    format_profile_for_prompt,
    is_projection_fresh,
)
from src.chat.security.identity import derive_owner_key
from src.chat.profile.repository import (
    confirm_fact,
    count_fact_lineage,
    delete_fact_with_lineage,
    delete_subject_profile,
    ensure_subject,
    get_all_section_states,
    get_fact,
    get_section_state,
    get_user_preference,
    list_facts_paginated,
    list_profile_facts,
    set_fact_inactive,
    set_fact_verification,
    set_section_status,
    set_user_preference,
    update_subject_demographics,
    write_profile_candidates,
    write_profile_fact,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_sqlite(tmp_path, monkeypatch):
    """Reset the global SQLite singleton for each test."""
    from src.chat import clients

    monkeypatch.setattr(clients, "SQLITE_PATH", str(tmp_path / "profile-test.db"))
    clients.get_sqlite.cache_clear()
    yield
    clients.get_sqlite.cache_clear()


@pytest.fixture
def identity_key(monkeypatch):
    import base64
    from src import config
    encoded = base64.b64encode(b"medical-profile-test-key-32-bytes!").decode()
    monkeypatch.setattr(config, "PROFILE_IDENTITY_ACTIVE_VERSION", "v1")
    monkeypatch.setattr(config, "PROFILE_IDENTITY_HMAC_KEY", encoded)
    monkeypatch.delenv("PROFILE_IDENTITY_KEY_V1", raising=False)
    return encoded


@pytest.fixture
def owner_key(identity_key):
    return derive_owner_key("test", "user-1")


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def test_migration_creates_new_columns_on_fresh_db(isolated_sqlite):
    conn = get_sqlite()
    # Fresh DB has only the medical-profile tables, no legacy memory tables.
    legacy_tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    } & {"memory_fact", "memory_subject", "memory_user_preference", "profile_section_state"}
    assert not legacy_tables, f"legacy tables still present: {legacy_tables}"
    cols = {row[1] for row in conn.execute("PRAGMA table_info(medical_profile_fact)")}
    for col in (
        "verification_status", "source_kind", "reporter_role", "confirmed_at",
        "inactive", "section", "coding_system", "coding_code", "coding_display",
    ):
        assert col in cols, f"missing column: {col}"
    subject_cols = {row[1] for row in conn.execute("PRAGMA table_info(medical_profile_subject)")}
    for col in ("birth_date", "gender"):
        assert col in subject_cols, f"missing subject column: {col}"
    # Section state and user preference tables exist.
    conn.execute("SELECT 1 FROM medical_profile_section_state LIMIT 0")
    conn.execute("SELECT 1 FROM medical_profile_preference LIMIT 0")


def test_migration_purges_legacy_tables_on_first_run(tmp_path, monkeypatch):
    """The one-shot migration drops legacy memory tables; nothing is backfilled."""
    import sqlite3
    from src.chat import clients

    legacy_db = tmp_path / "legacy.db"
    lc = sqlite3.connect(str(legacy_db))
    lc.executescript(
        """
        CREATE TABLE memory_fact (memory_id TEXT PRIMARY KEY);
        CREATE TABLE memory_subject (subject_id TEXT PRIMARY KEY);
        CREATE TABLE memory_user_preference (owner_id TEXT PRIMARY KEY);
        CREATE TABLE profile_section_state (owner_id TEXT PRIMARY KEY);
        CREATE TABLE patient_profile (owner_id TEXT PRIMARY KEY);
        """
    )
    lc.commit()
    lc.close()

    monkeypatch.setattr(clients, "SQLITE_PATH", str(legacy_db))
    clients.get_sqlite.cache_clear()
    conn = get_sqlite()
    tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    for legacy in (
        "memory_fact", "memory_subject", "memory_user_preference",
        "profile_section_state", "patient_profile",
    ):
        assert legacy not in tables, f"{legacy} should have been purged"
    # Fresh profile tables exist.
    assert "medical_profile_fact" in tables
    assert "medical_profile_subject" in tables
    assert "medical_profile_section_state" in tables
    assert "medical_profile_preference" in tables


# ---------------------------------------------------------------------------
# Profile fact lifecycle
# ---------------------------------------------------------------------------


def test_chat_extracted_fact_is_unconfirmed_by_default(isolated_sqlite, owner_key):
    write_profile_candidates(
        owner_id=owner_key,
        resolved_subject_id="self",
        source_turn_id="t1",
        profile_candidates=[{
            "fact_type": "allergy", "entity_type": "drug", "entity_id": "penicillin",
            "attribute": "current_use", "value": {"value": True},
            "temporal_status": "current", "confidence": 0.95, "source": "explicit",
        }],
    )
    facts = list_profile_facts(owner_key, "self")
    assert len(facts) == 1
    fact = facts[0]
    assert fact.verification_status == "unconfirmed"
    assert fact.source_kind == "chat_explicit"
    assert fact.section == "allergies"
    assert fact.confirmed_at is None


def test_profile_manual_fact_is_confirmed(isolated_sqlite, owner_key):
    fact = write_profile_fact(
        owner_id=owner_key, subject_id="self",
        fact_type="allergy", section="allergies",
        value={"type": "allergy", "criticality": "high"},
        entity_type="drug", entity_id="penicillin",
        coding_system="http://snomed.info/sct", coding_code="764146007",
        coding_display="Penicillin",
    )
    assert fact.verification_status == "confirmed"
    assert fact.source_kind == "profile_manual"
    assert fact.confirmed_at is not None
    assert fact.coding_code == "764146007"
    state = get_section_state(owner_key, "self", "allergies")
    assert state == "has_entries"


def test_confirm_refute_inactive_lifecycle(isolated_sqlite, owner_key):
    fact = write_profile_fact(
        owner_id=owner_key, subject_id="self",
        fact_type="allergy", section="allergies",
        value={"type": "allergy"},
        entity_type="drug", entity_id="sulfa",
    )
    confirmed = confirm_fact(owner_key, fact.profile_fact_id)
    assert confirmed.verification_status == "confirmed"
    assert confirmed.source_kind == "profile_edit"
    assert confirmed.confirmed_at is not None

    # Refute
    refuted = set_fact_verification(owner_key, fact.profile_fact_id, "refuted")
    assert refuted.verification_status == "refuted"
    assert refuted.inactive is True

    # Mark as entered in error
    err = set_fact_verification(owner_key, fact.profile_fact_id, "entered_in_error")
    assert err.verification_status == "entered_in_error"
    assert err.inactive is True

    # Manual inactive
    fact2 = write_profile_fact(
        owner_id=owner_key, subject_id="self",
        fact_type="medication_use", section="medications",
        value={"status": "stopped"},
        entity_type="drug", entity_id="ibuprofen",
    )
    inactive = set_fact_inactive(owner_key, fact2.profile_fact_id, inactive=True)
    assert inactive.inactive is True


def test_section_state_explicit_none_known(isolated_sqlite, owner_key):
    set_section_status(owner_key, "self", "allergies", "none_known")
    assert get_section_state(owner_key, "self", "allergies") == "none_known"
    states = get_all_section_states(owner_key, "self")
    assert states == {"allergies": "none_known"}


def test_section_state_recomputes_when_entry_added(isolated_sqlite, owner_key):
    set_section_status(owner_key, "self", "allergies", "none_known")
    # Adding a new profile entry should flip the section back to has_entries.
    write_profile_fact(
        owner_id=owner_key, subject_id="self",
        fact_type="allergy", section="allergies",
        value={"type": "allergy"},
        entity_type="drug", entity_id="latex",
    )
    state = get_section_state(owner_key, "self", "allergies")
    assert state == "has_entries"


def test_split_user_preferences_independent(isolated_sqlite, owner_key):
    set_user_preference(owner_key, "storage", True)
    set_user_preference(owner_key, "personalization", False)
    assert get_user_preference(owner_key, "storage") is True
    assert get_user_preference(owner_key, "personalization") is False
    # Toggling one must not affect the other.
    set_user_preference(owner_key, "storage", False)
    assert get_user_preference(owner_key, "personalization") is False


def test_legacy_consent_gates_storage_and_personalization(isolated_sqlite, owner_key, monkeypatch):
    # With no user preference saved, both follow the consent-required default.
    from src import config
    from src.chat.profile.runtime import prepare_context_runtime
    from src.chat.storage.domain import PatientSession

    analysis = {
        "analysis_succeeded": True,
        "guardrail": {"verdict": "allow"},
        "context": {
            "subject": {"id": "self", "source": "explicit", "confidence": 1.0},
            "references": [], "relation": "continue", "needs_medical_profile": True,
            "ambiguous": False, "clarification": "",
        },
    }
    session = PatientSession("session-test")
    # Treat the test environment as a deployed system: the operator kill-switch
    # is on, so profile reads/writes are allowed when the per-user toggles let it.
    monkeypatch.setattr(config, "PROFILE_READ_ENABLED", True)
    monkeypatch.setattr(config, "PROFILE_WRITE_ENABLED", True)
    monkeypatch.setattr(config, "PROFILE_REQUIRE_CONSENT", True)
    runtime = prepare_context_runtime(
        "session-test", session, analysis,
        owner_key=owner_key, profile_persistence_allowed=True,
    )
    # The user-facing personalization flag must be denied by default.
    assert runtime.profile_personalization_allowed is False

    # Both preferences ON => personalization ON.
    set_user_preference(owner_key, "storage", True)
    set_user_preference(owner_key, "personalization", True)
    runtime2 = prepare_context_runtime(
        "session-test", session, analysis,
        owner_key=owner_key, profile_persistence_allowed=True,
    )
    assert runtime2.profile_personalization_allowed is True

    # Storage OFF => personalization OFF.
    set_user_preference(owner_key, "storage", False)
    set_user_preference(owner_key, "personalization", True)
    runtime3 = prepare_context_runtime(
        "session-test", session, analysis,
        owner_key=owner_key, profile_persistence_allowed=True,
    )
    assert runtime3.profile_personalization_allowed is False

    # Storage ON + personalization OFF => reads on, profile text empty.
    set_user_preference(owner_key, "storage", True)
    set_user_preference(owner_key, "personalization", False)
    runtime4 = prepare_context_runtime(
        "session-test", session, analysis,
        owner_key=owner_key, profile_persistence_allowed=True,
    )
    assert runtime4.profile_personalization_allowed is False
    assert runtime4.profile_text == ""


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def test_projection_groups_sections_and_orders_confirmed_first(isolated_sqlite, owner_key):
    # Two allergies, one unconfirmed, one confirmed.
    write_profile_candidates(
        owner_id=owner_key, resolved_subject_id="self", source_turn_id="t1",
        profile_candidates=[{
            "fact_type": "allergy", "entity_type": "drug", "entity_id": "aspirin",
            "attribute": "current_use", "value": {"value": True},
            "temporal_status": "current", "confidence": 0.9, "source": "explicit",
        }],
    )
    profile_fact = write_profile_fact(
        owner_id=owner_key, subject_id="self",
        fact_type="allergy", section="allergies",
        value={"type": "allergy", "criticality": "high"},
        entity_type="drug", entity_id="penicillin",
    )
    profile = build_medical_profile(owner_key, "self")
    assert [a.agent for a in profile.allergies] == ["penicillin", "aspirin"]
    assert profile.allergies[0].confirmed is True
    assert profile.allergies[1].confirmed is False
    assert profile.section_states["allergies"].status == "has_entries"


def test_projection_excludes_transient_symptoms(isolated_sqlite, owner_key):
    write_profile_candidates(
        owner_id=owner_key, resolved_subject_id="self", source_turn_id="t1",
        profile_candidates=[
            {
                "fact_type": "symptom_state", "entity_type": "symptom",
                "entity_id": "headache",
                "attribute": "current", "value": {"value": True},
                "temporal_status": "current", "confidence": 0.9, "source": "explicit",
            },
            {
                "fact_type": "chronic_disease", "entity_type": "disease",
                "entity_id": "asthma", "attribute": "current",
                "value": {"severity": "mild"},
                "temporal_status": "current", "confidence": 0.9, "source": "explicit",
            },
        ],
    )
    profile = build_medical_profile(owner_key, "self")
    assert profile.problems and profile.problems[0].condition == "asthma"
    assert not any(getattr(a, "entity_id", "") == "headache" for a in profile.allergies)


def test_projection_isolation_by_subject(isolated_sqlite, owner_key):
    write_profile_fact(
        owner_id=owner_key, subject_id="self",
        fact_type="allergy", section="allergies",
        value={"type": "allergy"},
        entity_type="drug", entity_id="penicillin",
    )
    write_profile_fact(
        owner_id=owner_key, subject_id="father",
        fact_type="allergy", section="allergies",
        value={"type": "allergy"},
        entity_type="drug", entity_id="shrimp",
    )
    self_profile = build_medical_profile(owner_key, "self")
    father_profile = build_medical_profile(owner_key, "father")
    assert [a.agent for a in self_profile.allergies] == ["penicillin"]
    assert [a.agent for a in father_profile.allergies] == ["shrimp"]


def test_projection_format_omits_none_known(isolated_sqlite, owner_key):
    set_section_status(owner_key, "self", "allergies", "none_known")
    profile = build_medical_profile(owner_key, "self")
    text = format_profile_for_prompt(profile)
    assert "đã xác nhận không có dị ứng" in text
    # The other sections are unreviewed and must be omitted entirely.
    assert "Vấn đề sức khỏe" not in text
    assert "Thuốc đang dùng" not in text


def test_projection_format_uses_cautious_language_for_unconfirmed(isolated_sqlite, owner_key):
    write_profile_candidates(
        owner_id=owner_key, resolved_subject_id="self", source_turn_id="t1",
        profile_candidates=[{
            "fact_type": "allergy", "entity_type": "drug", "entity_id": "aspirin",
            "attribute": "current_use", "value": {"value": True},
            "temporal_status": "current", "confidence": 0.9, "source": "explicit",
        }],
    )
    profile = build_medical_profile(owner_key, "self")
    text = format_profile_for_prompt(profile)
    assert "ghi nhận trước đó" in text
    assert "xác nhận" in text  # caveat for the LLM


def test_projection_is_fresh(isolated_sqlite, owner_key):
    profile = build_medical_profile(owner_key, "self", now=100.0)
    assert is_projection_fresh(profile, now=110.0) is True
    assert is_projection_fresh(profile, now=200.0) is False


def test_fact_lineage_count_matches_delete(isolated_sqlite, owner_key):
    fact = write_profile_fact(
        owner_id=owner_key, subject_id="self",
        fact_type="allergy", section="allergies",
        value={"type": "allergy"},
        entity_type="drug", entity_id="penicillin",
    )
    # Replace once -> lineage of 2 rows.
    new_fact = ProfileFact(
        profile_fact_id="replacement", owner_id=owner_key, subject_id="self",
        section="allergies",
        fact_type="allergy", entity_type="drug", entity_id="penicillin",
        attribute="current_use", value={"value": True}, temporal_status="current",
        confidence=1.0, source_turn_id="manual", valid_from=None, valid_until=None,
        superseded_by=None, created_at=2.0, updated_at=2.0,
        verification_status="confirmed", source_kind="profile_edit",
    )
    from src.chat.profile.repository import replace_fact
    replace_fact(owner_key, fact.profile_fact_id, new_fact=new_fact, expected_updated_at=fact.updated_at)
    assert count_fact_lineage(owner_key, fact.profile_fact_id) == 2
    removed = delete_fact_with_lineage(owner_key, fact.profile_fact_id)
    assert removed == 2
    assert get_fact(owner_key, fact.profile_fact_id) is None


# ---------------------------------------------------------------------------
# Selector ranking + exclusion
# ---------------------------------------------------------------------------


def test_selector_excludes_inactive_refuted_and_entered_in_error(isolated_sqlite, owner_key):
    # Active + inactive + refuted + entered_in_error rows
    write_profile_fact(
        owner_id=owner_key, subject_id="self",
        fact_type="allergy", section="allergies",
        value={"type": "allergy"},
        entity_type="drug", entity_id="penicillin",
    )
    f2 = write_profile_fact(
        owner_id=owner_key, subject_id="self",
        fact_type="allergy", section="allergies",
        value={"type": "allergy"},
        entity_type="drug", entity_id="sulfa",
    )
    set_fact_inactive(owner_key, f2.profile_fact_id, inactive=True)
    f3 = write_profile_fact(
        owner_id=owner_key, subject_id="self",
        fact_type="allergy", section="allergies",
        value={"type": "allergy"},
        entity_type="drug", entity_id="latex",
    )
    set_fact_verification(owner_key, f3.profile_fact_id, "refuted")
    f4 = write_profile_fact(
        owner_id=owner_key, subject_id="self",
        fact_type="allergy", section="allergies",
        value={"type": "allergy"},
        entity_type="drug", entity_id="iodine",
    )
    set_fact_verification(owner_key, f4.profile_fact_id, "entered_in_error")

    facts = list_profile_facts(owner_key, "self")
    from src.chat.context.domain import SessionState
    state = SessionState(session_id="s", owner_id=owner_key)
    analysis = {
        "analysis_succeeded": True,
        "guardrail": {"verdict": "allow"},
        "context": {
            "subject": {"id": "self", "source": "explicit", "confidence": 1.0},
            "references": [], "relation": "continue", "needs_medical_profile": True,
            "ambiguous": False, "clarification": "",
        },
    }
    from src.chat.context.resolver import resolve_subject
    from src.chat.profile.selector import build_context_bundle
    resolution = resolve_subject(analysis["context"], state)
    bundle = build_context_bundle(
        analysis=analysis, state=state, resolution=resolution,
        facts=facts, active_case=None,
    )
    memory_ids = {f.profile_fact_id for f in bundle.safety_profile + bundle.relevant_facts}
    assert memory_ids == {get_fact(owner_key, _id).profile_fact_id for _id in memory_ids}
    # Only the still-active row should survive.
    assert len(memory_ids) == 1


def test_selector_prefers_confirmed_over_unconfirmed(isolated_sqlite, owner_key):
    # Unconfirmed first, confirmed second.
    # (no longer needed; selector is the same for any fact type)
    write_profile_candidates(
        owner_id=owner_key, resolved_subject_id="self", source_turn_id="t1",
        profile_candidates=[{
            "fact_type": "allergy", "entity_type": "drug", "entity_id": "aspirin",
            "attribute": "current_use", "value": {"value": True},
            "temporal_status": "current", "confidence": 0.9, "source": "explicit",
        }],
    )
    confirmed_fact = write_profile_fact(
        owner_id=owner_key, subject_id="self",
        fact_type="allergy", section="allergies",
        value={"type": "allergy"},
        entity_type="drug", entity_id="penicillin",
    )
    facts = list_profile_facts(owner_key, "self")
    from src.chat.context.domain import SessionState
    state = SessionState(session_id="s", owner_id=owner_key)
    analysis = {
        "analysis_succeeded": True,
        "guardrail": {"verdict": "allow"},
        "context": {
            "subject": {"id": "self", "source": "explicit", "confidence": 1.0},
            "references": [], "relation": "continue", "needs_medical_profile": True,
            "ambiguous": False, "clarification": "",
        },
    }
    from src.chat.context.resolver import resolve_subject
    resolution = resolve_subject(analysis["context"], state)
    bundle = build_context_bundle(
        analysis=analysis, state=state, resolution=resolution,
        facts=facts, active_case=None, fact_limit=2,
    )
    ids = [f.profile_fact_id for f in bundle.safety_profile + bundle.relevant_facts]
    # Confirmed allergy (penicillin) must come first.
    assert ids[0] == confirmed_fact.profile_fact_id


# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------


def test_update_subject_demographics_enforces_concurrency(isolated_sqlite, owner_key):
    from src.chat.profile.repository import ensure_subject
    ensure_subject(owner_key, "self", relationship="self", display_name="Lan")
    updated = update_subject_demographics(
        owner_key, "self", birth_date="1990-04-12", gender="female",
        expected_updated_at=1.0,
    )
    assert updated is None  # wrong updated_at -> stale
    # Re-read and update with the right timestamp.
    from src.chat.profile.repository import get_subject
    current = get_subject(owner_key, "self")
    updated = update_subject_demographics(
        owner_key, "self", birth_date="1990-04-12", gender="female",
        expected_updated_at=current["updated_at"],
    )
    assert updated["birth_date"] == "1990-04-12"
    assert updated["gender"] == "female"
    # Clear gender.
    current = get_subject(owner_key, "self")
    cleared = update_subject_demographics(
        owner_key, "self", expected_updated_at=current["updated_at"],
        clear_gender=True,
    )
    assert cleared["gender"] is None
    assert cleared["birth_date"] == "1990-04-12"


# ---------------------------------------------------------------------------
# Section state file-table guard
# ---------------------------------------------------------------------------


def test_rejected_inferred_fact_is_not_persisted(isolated_sqlite, owner_key):
    """source != "explicit" and confidence < 0.7 must be silently dropped."""
    write_profile_candidates(
        owner_id=owner_key, resolved_subject_id="self", source_turn_id="t1",
        profile_candidates=[{
            "fact_type": "allergy", "entity_type": "drug", "entity_id": "aspirin",
            "attribute": "current_use", "value": {"value": True},
            "temporal_status": "current", "confidence": 0.5, "source": "inferred",
        }],
    )
    assert list_profile_facts(owner_key, "self") == []


# ---------------------------------------------------------------------------
# Manager-defect coverage
# ---------------------------------------------------------------------------


class _NoopRedis:
    """Stand-in when we don't need a real Redis client for pure unit checks."""

    def __getattr__(self, name):
        raise RuntimeError("redis unavailable")


def test_delete_fact_with_lineage_uses_selected_fact_only(isolated_sqlite, owner_key):
    # A first allergy lives in `self`. A second allergy lives in `self` too.
    # We want delete to count only the lineage of the first fact, not the
    # subject's total superseded rows.
    f1 = write_profile_fact(
        owner_id=owner_key, subject_id="self",
        fact_type="allergy", section="allergies",
        value={"type": "allergy"},
        entity_type="drug", entity_id="penicillin",
    )
    f2 = write_profile_fact(
        owner_id=owner_key, subject_id="self",
        fact_type="allergy", section="allergies",
        value={"type": "allergy"},
        entity_type="drug", entity_id="sulfa",
    )
    # Replace f1 once -> lineage of 2.
    from src.chat.profile.repository import replace_fact
    new_fact = ProfileFact(
        profile_fact_id="replacement1", owner_id=owner_key, subject_id="self",
        section="allergies",
        fact_type="allergy", entity_type="drug", entity_id="penicillin",
        attribute="current_use", value={"value": True}, temporal_status="current",
        confidence=1.0, source_turn_id="manual", valid_from=None, valid_until=None,
        superseded_by=None, created_at=2.0, updated_at=2.0,
        verification_status="confirmed", source_kind="profile_edit",
    )
    replace_fact(owner_key, f1.profile_fact_id, new_fact=new_fact, expected_updated_at=f1.updated_at)
    # The pre-existing `count_fact_lineage` should now equal 2 for f1 and 1
    # for f2 — not the total of all superseded facts on the subject.
    assert count_fact_lineage(owner_key, f1.profile_fact_id) == 2
    assert count_fact_lineage(owner_key, f2.profile_fact_id) == 1
    # And the fact detail page must surface the lineage count without
    # showing subject-wide superseded totals.
    assert count_fact_lineage(owner_key, f1.profile_fact_id) < 3


def test_fact_delete_enforces_optimistic_concurrency(isolated_sqlite, owner_key):
    f1 = write_profile_fact(
        owner_id=owner_key, subject_id="self",
        fact_type="allergy", section="allergies",
        value={"type": "allergy"},
        entity_type="drug", entity_id="penicillin",
    )
    # First delete succeeds.
    assert delete_fact_with_lineage(owner_key, f1.profile_fact_id) == 1
    # A second delete on the same id must report zero (already gone).
    assert delete_fact_with_lineage(owner_key, f1.profile_fact_id) == 0


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def test_system_prompt_includes_profile_caveats():
    from src.chat.prompts import GENERATOR_SYSTEM
    assert "Hồ sơ y tế cá nhân" in GENERATOR_SYSTEM
    assert "ghi nhận trước đó" in GENERATOR_SYSTEM
    assert "xác nhận" in GENERATOR_SYSTEM
