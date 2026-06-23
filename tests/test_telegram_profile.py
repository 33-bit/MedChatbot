"""End-to-end tests for the IPS-inspired profile manager flows.

These tests exercise the dispatcher / renderer layer using the existing
fake Redis fixture so they can run without real network access. They
focus on the new section + preferences + none_known flows and on the
defect fixes (delete-while-off, optimistic concurrency on delete, and
section-state recompute when the user toggles storage/personalization).
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from src import config
from src.chat.clients import get_sqlite
from src.chat.profile.repository import (
    confirm_fact,
    count_fact_lineage,
    delete_fact_with_lineage,
    delete_subject_profile,
    ensure_subject,
    get_all_section_states,
    get_fact,
    get_section_state,
    get_subject,
    get_user_preference,
    list_facts_paginated,
    set_fact_inactive,
    set_fact_verification,
    set_section_status,
    set_user_preference,
    update_subject_demographics,
    write_profile_fact,
)
from src.chat.profile.ui_state import (
    invalidate_profile_sessions_for_owner,
    iter_profile_sessions,
    track_profile_session,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def identity_key(monkeypatch):
    encoded = base64.b64encode(b"telegram-profile-test-key-32-bytes!").decode()
    monkeypatch.setattr(config, "PROFILE_IDENTITY_ACTIVE_VERSION", "v1")
    monkeypatch.setattr(config, "PROFILE_IDENTITY_HMAC_KEY", encoded)
    monkeypatch.delenv("PROFILE_IDENTITY_KEY_V1", raising=False)
    return encoded


@pytest.fixture
def isolated_sqlite(tmp_path, monkeypatch):
    from src.chat import clients
    monkeypatch.setattr(clients, "SQLITE_PATH", str(tmp_path / "tg-profile.db"))
    clients.get_sqlite.cache_clear()
    yield
    clients.get_sqlite.cache_clear()


@pytest.fixture
def owner_key(identity_key):
    from src.chat.security.identity import derive_owner_key
    return derive_owner_key("telegram", "user-1")


# ---------------------------------------------------------------------------
# Deletion / mutation flows
# ---------------------------------------------------------------------------


def test_subject_delete_is_allowed_when_profile_is_off(isolated_sqlite, owner_key):
    """Profile storage OFF must still allow destructive cleanup."""
    set_user_preference(owner_key, "storage", False)
    set_user_preference(owner_key, "personalization", False)
    ensure_subject(owner_key, "self", relationship="self", display_name="Lan")
    write_profile_fact(
        owner_id=owner_key, subject_id="self",
        fact_type="allergy", section="allergies",
        value={"type": "allergy"},
        entity_type="drug", entity_id="penicillin",
    )
    deleted = delete_subject_profile(owner_key, "self")
    assert deleted == 1
    assert get_subject(owner_key, "self") is None


def test_fact_delete_with_lineage_counts_only_selected(isolated_sqlite, owner_key):
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
    from src.chat.profile.repository import replace_fact
    from src.chat.profile.domain import ProfileFact
    replacement = ProfileFact(
        profile_fact_id="replacement", owner_id=owner_key, subject_id="self",
        section="allergies",
        fact_type="allergy", entity_type="drug", entity_id="penicillin",
        attribute="current_use", value={"value": True}, temporal_status="current",
        confidence=1.0, source_turn_id="manual", valid_from=None, valid_until=None,
        superseded_by=None, created_at=2.0, updated_at=2.0,
        verification_status="confirmed", source_kind="profile_edit",
    )
    replace_fact(owner_key, f1.profile_fact_id, new_fact=replacement, expected_updated_at=f1.updated_at)
    assert count_fact_lineage(owner_key, f1.profile_fact_id) == 2
    assert count_fact_lineage(owner_key, f2.profile_fact_id) == 1
    assert delete_fact_with_lineage(owner_key, f1.profile_fact_id) == 2


@pytest.fixture
def fake_redis(monkeypatch):
    class _Fake:
        def __init__(self):
            self.data = {}
            self.sets = {}
        def get(self, k): return self.data.get(k)
        def setex(self, k, t, v): self.data[k] = v
        def sadd(self, k, v): self.sets.setdefault(k, set()).add(v)
        def smembers(self, k): return self.sets.get(k, set())
        def expire(self, *a, **kw): pass
        def delete(self, *keys):
            n = 0
            for k in keys:
                if k in self.data:
                    del self.data[k]; n += 1
                if k in self.sets:
                    del self.sets[k]; n += 1
            return n
        def getdel(self, k):
            v = self.data.pop(k, None); return v
    fake = _Fake()
    # Profile UI state lives under src.chat.profile.ui_state; monkeypatch its
    # redis accessor.
    monkeypatch.setattr(
        "src.chat.profile.ui_state.get_redis", lambda: fake,
    )
    yield fake


def test_invalidate_profile_sessions_for_owner_clears_tracked(fake_redis, identity_key, owner_key):
    """A profile mutation must invalidate every tracked session for the owner."""
    track_profile_session(owner_key, "session_1")
    track_profile_session(owner_key, "session_2")
    assert sorted(iter_profile_sessions(owner_key)) == ["session_1", "session_2"]
    cleared = invalidate_profile_sessions_for_owner(owner_key)
    assert cleared == 2
    assert iter_profile_sessions(owner_key) == []


# ---------------------------------------------------------------------------
# Section state and preferences
# ---------------------------------------------------------------------------


def test_none_known_persists_and_is_surfaced_in_prompt(isolated_sqlite, owner_key):
    set_section_status(owner_key, "self", "allergies", "none_known")
    assert get_section_state(owner_key, "self", "allergies") == "none_known"
    states = get_all_section_states(owner_key, "self")
    assert states == {"allergies": "none_known"}
    from src.chat.profile.projection import (
        build_medical_profile, format_profile_for_prompt,
    )
    profile = build_medical_profile(owner_key, "self")
    text = format_profile_for_prompt(profile)
    assert "đã xác nhận không có dị ứng" in text
    # Unreviewed sections remain absent.
    assert "Vấn đề sức khỏe" not in text


def test_split_preferences_round_trip(isolated_sqlite, owner_key):
    set_user_preference(owner_key, "storage", True)
    set_user_preference(owner_key, "personalization", False)
    assert get_user_preference(owner_key, "storage") is True
    assert get_user_preference(owner_key, "personalization") is False
    set_user_preference(owner_key, "storage", False)
    assert get_user_preference(owner_key, "personalization") is False


def test_subject_demographics_round_trip(isolated_sqlite, owner_key):
    ensure_subject(owner_key, "self", relationship="self", display_name="Lan")
    current = get_subject(owner_key, "self")
    updated = update_subject_demographics(
        owner_key, "self",
        birth_date="1990-04-12", gender="female",
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


def test_fact_lifecycle_keeps_audit_trail(isolated_sqlite, owner_key):
    fact = write_profile_fact(
        owner_id=owner_key, subject_id="self",
        fact_type="allergy", section="allergies",
        value={"type": "allergy"},
        entity_type="drug", entity_id="penicillin",
    )
    confirm_fact(owner_key, fact.profile_fact_id)
    # After confirmation, source_kind and verification_status must reflect it.
    confirmed = get_fact(owner_key, fact.profile_fact_id)
    assert confirmed.verification_status == "confirmed"
    assert confirmed.source_kind == "profile_edit"
    # Inactive rows are skipped by list_facts_paginated.
    set_fact_inactive(owner_key, fact.profile_fact_id, inactive=True)
    assert list_facts_paginated(owner_key, "self") == []
    # Refute another.
    f2 = write_profile_fact(
        owner_id=owner_key, subject_id="self",
        fact_type="allergy", section="allergies",
        value={"type": "allergy"},
        entity_type="drug", entity_id="sulfa",
    )
    set_fact_verification(owner_key, f2.profile_fact_id, "refuted")
    assert list_facts_paginated(owner_key, "self") == []


# ---------------------------------------------------------------------------
# Telegram profile UI regressions
# ---------------------------------------------------------------------------


def test_manual_fact_summaries_are_medically_coherent(isolated_sqlite, owner_key):
    from src.server.channels.telegram_profile import _fact_summary

    medication = write_profile_fact(
        owner_id=owner_key, subject_id="self",
        fact_type="medication_use", section="medications",
        value={"name": "Metformin", "status": True}, entity_id="Metformin",
    )
    allergy = write_profile_fact(
        owner_id=owner_key, subject_id="self",
        fact_type="allergy", section="allergies",
        value={"name": "Penicillin"}, entity_id="Penicillin",
    )
    pregnancy = write_profile_fact(
        owner_id=owner_key, subject_id="self",
        fact_type="pregnancy_status", section="pregnancy",
        value={"value": "not_pregnant"}, entity_id="not_pregnant",
    )

    assert "Đang dùng Metformin" in _fact_summary(medication)
    assert "Dị ứng: Penicillin" in _fact_summary(allergy)
    assert "đã ngưng" not in _fact_summary(allergy)
    assert "Không mang thai" in _fact_summary(pregnancy)


def test_birth_date_parser_accepts_friendly_format_and_rejects_invalid_dates():
    from src.server.channels.telegram_profile import _parse_birth_date

    assert _parse_birth_date("25/08/1955") == "1955-08-25"
    assert _parse_birth_date("1955-08-25") == "1955-08-25"
    assert _parse_birth_date("31/02/1955") is None
    assert _parse_birth_date("01/01/2200") is None


def test_section_callback_is_acknowledged_only_once(
    isolated_sqlite, fake_redis, identity_key, owner_key, monkeypatch,
):
    from src.chat.profile.ui_state import issue_profile_token
    from src.server.channels import telegram_profile

    ensure_subject(owner_key, "self", relationship="self")
    token = issue_profile_token(
        "section_set_state", chat_id=123, owner_key=owner_key,
        payload={
            "subject_id": "self",
            "section": "allergies",
            "status": "unknown",
        },
    )
    answers = []

    async def fake_answer(callback_id, text):
        answers.append((callback_id, text))

    async def fake_send(*args, **kwargs):
        return None

    monkeypatch.setattr(telegram_profile, "_answer_callback_query", fake_answer)
    monkeypatch.setattr(telegram_profile, "_send", fake_send)

    handled = asyncio.run(telegram_profile._dispatch_profile_callback(
        f"prof:section_set_state:{token}", 123, "user-1", "callback-1",
    ))

    assert handled is True
    assert answers == [("callback-1", "")]


def test_profile_root_uses_plain_language_and_separates_settings(
    isolated_sqlite, fake_redis, owner_key, monkeypatch,
):
    from src.server.channels import telegram_profile

    ensure_subject(owner_key, "self", relationship="self", display_name="Lan")
    messages = []

    async def fake_send(chat_id, text, *, inline_keyboard=None):
        messages.append((text, inline_keyboard))

    monkeypatch.setattr(telegram_profile, "_send", fake_send)
    asyncio.run(telegram_profile._render_profile_root(123, owner_key, "self"))

    text, keyboard = messages[-1]
    button_texts = [
        button["text"]
        for row in keyboard["inline_keyboard"]
        for button in row
    ]
    assert "Hồ sơ sức khỏe" in text
    assert "IPS-inspired" not in text
    assert "Chủ thể" not in text
    assert "👤 Tên, ngày sinh và giới tính" in button_texts
    assert "🔒 Cài đặt lưu và sử dụng hồ sơ" in button_texts


def test_new_pregnancy_choice_replaces_old_choice(
    isolated_sqlite, fake_redis, owner_key, monkeypatch,
):
    from src.server.channels import telegram_profile

    ensure_subject(owner_key, "self", relationship="self")
    write_profile_fact(
        owner_id=owner_key, subject_id="self",
        fact_type="pregnancy_status", section="pregnancy",
        value={"value": "pregnant"}, entity_id="pregnant",
    )

    async def fake_send(*args, **kwargs):
        return None

    monkeypatch.setattr(telegram_profile, "_send", fake_send)
    asyncio.run(telegram_profile._apply_pregnancy_status(
        123, owner_key, "self", "not_pregnant",
    ))

    active = [
        fact for fact in list_facts_paginated(owner_key, "self")
        if fact.section == "pregnancy"
    ]
    assert len(active) == 1
    assert active[0].value == {"value": "not_pregnant"}
