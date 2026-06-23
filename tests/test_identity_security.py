"""Identity, owner-scoping, and provenance-leak tests for the profile subsystem.

The migration renames the durable personal-memory store to the IPS-inspired
medical profile. These tests assert the same security guarantees hold
against the new module paths:

- The pseudonymous owner key is deterministic, domain-separated, and tenant-scoped.
- Key rotation migrates previous owners transactionally.
- Wrong HMAC keys cannot access another owner's data.
- The prompt context bundle never leaks owner IDs, fact IDs, or source turns.
- Owner-scoped CRUD prevents cross-tenant deletion.
- Profile persistence fails closed when the identity key is missing.
"""

from __future__ import annotations

import base64
import json

import pytest

from src import config
from src.chat.clients import get_sqlite
from src.chat.context.context_store import (
    clear_conversation_context,
    save_conversation_context,
)
from src.chat.context.domain import (
    ConversationContextBundle as ContextBundle,
    SessionState,
)
from src.chat.profile.domain import ProfileFact
from src.chat.profile.repository import (
    delete_fact_with_lineage,
    list_profile_facts,
    migrate_owner_key,
    set_user_preference,
    write_profile_candidates,
)
from src.chat.security.identity import (
    derive_owner_key,
    derive_request_identity,
    derive_session_key,
    is_owner_key,
    is_session_key,
    validate_identity_configuration,
)
from src.chat.storage.feedback import create_feedback_request
from src.chat.storage.sqlite_profile import log_consultation
from src.chat.storage.traces import save_chat_trace
from src.chat.storage.domain import PatientSession


@pytest.fixture
def identity_key(monkeypatch):
    encoded = base64.b64encode(b"identity-test-key-material-32bytes!!").decode()
    monkeypatch.setattr(config, "PROFILE_IDENTITY_ACTIVE_VERSION", "v1")
    monkeypatch.setattr(config, "PROFILE_IDENTITY_HMAC_KEY", encoded)
    monkeypatch.delenv("PROFILE_IDENTITY_KEY_V1", raising=False)
    return encoded


def test_identity_is_deterministic_domain_separated_and_tenant_scoped(identity_key):
    first = derive_owner_key("telegram", "raw-user-123", tenant="clinic-a")
    second = derive_owner_key("telegram", "raw-user-123", tenant="clinic-a")
    other_tenant = derive_owner_key("telegram", "raw-user-123", tenant="clinic-b")
    other_channel = derive_owner_key("messenger", "raw-user-123", tenant="clinic-a")
    session = derive_session_key("telegram", "raw-user-123", tenant="clinic-a")

    assert first == second
    assert len(first.removeprefix("owner_v1_")) == 64
    assert is_owner_key(first)
    assert is_session_key(session)
    assert len({first, other_tenant, other_channel, session}) == 4
    assert "raw-user-123" not in first


def test_key_version_changes_output(identity_key, monkeypatch):
    old = derive_owner_key("api", "user", tenant="tenant", version="v1")
    monkeypatch.setenv(
        "PROFILE_IDENTITY_KEY_V2",
        base64.b64encode(b"new-identity-key-material-32-bytes!").decode(),
    )
    new = derive_owner_key("api", "user", tenant="tenant", version="v2")

    assert old.startswith("owner_v1_")
    assert new.startswith("owner_v2_")
    assert old != new


def test_missing_key_disables_profile_persistence(monkeypatch):
    monkeypatch.setattr(config, "PROFILE_IDENTITY_HMAC_KEY", "")
    monkeypatch.delenv("PROFILE_IDENTITY_KEY_V1", raising=False)

    identity = derive_request_identity("telegram", "123", "123")

    assert identity.owner_key == ""
    assert identity.profile_persistence_allowed is False
    assert is_session_key(identity.session_key)


def test_startup_validation_fails_closed_when_profile_persistence_enabled(monkeypatch):
    monkeypatch.setattr(config, "PROFILE_IDENTITY_HMAC_KEY", "")
    monkeypatch.setattr(config, "PROFILE_READ_ENABLED", True)
    monkeypatch.setattr(config, "PROFILE_WRITE_ENABLED", False)
    monkeypatch.delenv("PROFILE_IDENTITY_KEY_V1", raising=False)

    with pytest.raises(RuntimeError, match="Profile identity key"):
        validate_identity_configuration()


def test_owner_scoped_delete_cannot_delete_another_users_fact(identity_key):
    owner_a = derive_owner_key("api", "a", tenant="tenant")
    owner_b = derive_owner_key("api", "b", tenant="tenant")
    written = write_profile_candidates(
        owner_id=owner_a,
        resolved_subject_id="self",
        source_turn_id="turn_1",
        profile_candidates=[{
            "fact_type": "allergy",
            "entity_type": "drug",
            "entity_id": "penicillin",
            "attribute": "allergic",
            "value": {"status": True},
            "temporal_status": "current",
            "confidence": 1.0,
            "source": "explicit",
        }],
    )
    fact = written[0]

    assert delete_fact_with_lineage(owner_b, fact.profile_fact_id) == 0
    owner_a_facts = [item.profile_fact_id for item in list_profile_facts(owner_a)]
    assert owner_a_facts == [fact.profile_fact_id]
    assert delete_fact_with_lineage(owner_a, fact.profile_fact_id) == 1


def test_sensitive_tables_reject_raw_identifiers(identity_key):
    raw_id = "telegram:987654321"
    session_key = derive_session_key("telegram", "987654321")

    log_consultation(raw_id, "question", "answer")
    with pytest.raises(ValueError):
        create_feedback_request(raw_id, "telegram", raw_id, "question", "answer")
    with pytest.raises(ValueError):
        save_chat_trace(
            trace_id="trace",
            session_id=raw_id,
            internal_session_id=raw_id,
            mode="auto",
            question="question",
            answer="answer",
            meta={},
        )

    log_consultation(session_key, "question", "answer")
    rows = get_sqlite().execute("SELECT session_id FROM consultations").fetchall()
    assert rows == [(session_key,)]
    assert raw_id not in json.dumps(rows)


def test_context_bundle_never_exposes_owner_or_provenance(identity_key):
    owner_key = derive_owner_key("telegram", "123")
    fact = ProfileFact(
        profile_fact_id="profile-internal",
        owner_id=owner_key,
        subject_id="self",
        section="medications",
        fact_type="medication_use",
        entity_type="drug",
        entity_id="warfarin",
        attribute="current_use",
        value={"status": True},
        temporal_status="current",
        confidence=1.0,
        source_turn_id="turn-internal",
        valid_from=None,
        valid_until=None,
        superseded_by=None,
        created_at=1.0,
        updated_at=1.0,
    )
    prompt = ContextBundle(
        subject={"id": "self"},
        safety_profile=[fact],
        relevant_facts=[],
        active_case=None,
        reference_turns=[],
    ).to_prompt_dict()
    serialized = json.dumps(prompt)

    assert owner_key not in serialized
    assert "owner_id" not in serialized
    assert "profile-internal" not in serialized
    assert "turn-internal" not in serialized


def test_user_preference_stores_only_pseudonymous_owner(identity_key):
    owner_key = derive_owner_key("messenger", "raw-sender")
    set_user_preference(owner_key, "storage", True)

    row = get_sqlite().execute(
        "SELECT owner_id, enabled FROM medical_profile_preference"
    ).fetchone()
    assert row == (owner_key, 1)
    assert "raw-sender" not in row[0]


def test_consent_disabled_prevents_profile_read_and_write(identity_key, monkeypatch):
    owner_key = derive_owner_key("telegram", "123")
    monkeypatch.setattr(config, "PROFILE_READ_ENABLED", True)
    monkeypatch.setattr(config, "PROFILE_WRITE_ENABLED", True)
    monkeypatch.setattr(config, "PROFILE_REQUIRE_CONSENT", True)
    analysis = {
        "analysis_succeeded": True,
        "guardrail": {"verdict": "allow"},
        "context": {
            "subject": {"id": "self", "source": "explicit", "confidence": 1.0},
            "references": [],
            "relation": "continue",
            "needs_medical_profile": True,
            "ambiguous": False,
            "clarification": "",
        },
    }
    session_id = "session_ephemeral_" + "b" * 64

    from src.chat.profile.runtime import prepare_context_runtime

    runtime = prepare_context_runtime(
        session_id,
        PatientSession(session_id),
        analysis,
        owner_key=owner_key,
        profile_persistence_allowed=True,
    )

    assert runtime.profile_personalization_allowed is False


def test_wrong_hmac_key_cannot_access_profile(identity_key, monkeypatch):
    correct_owner = derive_owner_key("api", "user", tenant="tenant")
    write_profile_candidates(
        owner_id=correct_owner,
        resolved_subject_id="self",
        source_turn_id="turn_1",
        profile_candidates=[{
            "fact_type": "chronic_disease",
            "entity_type": "disease",
            "entity_id": "asthma",
            "attribute": "diagnosed",
            "value": {"status": True},
            "temporal_status": "current",
            "confidence": 1.0,
            "source": "explicit",
        }],
    )
    monkeypatch.setattr(
        config,
        "PROFILE_IDENTITY_HMAC_KEY",
        base64.b64encode(b"completely-different-key-32-bytes!").decode(),
    )
    wrong_owner = derive_owner_key("api", "user", tenant="tenant")

    assert wrong_owner != correct_owner
    assert list_profile_facts(wrong_owner) == []


def test_key_rotation_migrates_previous_owner_transactionally(identity_key, monkeypatch):
    previous_owner = derive_owner_key("api", "user", tenant="tenant")
    write_profile_candidates(
        owner_id=previous_owner,
        resolved_subject_id="self",
        source_turn_id="turn_1",
        profile_candidates=[{
            "fact_type": "allergy",
            "entity_type": "drug",
            "entity_id": "aspirin",
            "attribute": "allergic",
            "value": {"status": True},
            "temporal_status": "current",
            "confidence": 1.0,
            "source": "explicit",
        }],
    )
    monkeypatch.setenv(
        "PROFILE_IDENTITY_KEY_V2",
        base64.b64encode(b"rotation-key-material-at-least-32-bytes").decode(),
    )
    current_owner = derive_owner_key("api", "user", tenant="tenant", version="v2")

    assert migrate_owner_key(current_owner, (previous_owner,)) is True
    assert list_profile_facts(previous_owner) == []
    assert [fact.entity_id for fact in list_profile_facts(current_owner)] == ["aspirin"]
