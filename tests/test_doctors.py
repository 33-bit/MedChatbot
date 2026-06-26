from __future__ import annotations

import time

from src.chat.clients import get_sqlite
from src.chat.storage import doctors


def test_doctor_tables_exist():
    conn = get_sqlite()
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert "doctor" in tables
    assert "doctor_consultation" in tables


def test_doctor_table_columns_exist():
    conn = get_sqlite()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(doctor)")}
    assert cols >= {
        "id",
        "name",
        "specialty",
        "tier",
        "price",
        "telegram_user_id",
        "active",
        "degree",
        "experience_years",
        "hospital",
        "bio",
    }


def test_doctor_consultation_table_columns_exist():
    conn = get_sqlite()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(doctor_consultation)")}
    assert cols >= {
        "id",
        "patient_chat_id",
        "doctor_id",
        "doctor_chat_id",
        "tier",
        "fee",
        "status",
        "created_at",
        "accepted_at",
        "ended_at",
        "last_activity_at",
    }


def test_create_and_get_doctor():
    doctor_id = doctors.create_doctor(
        name="BS An",
        specialty="Nội tổng quát",
        tier="free",
        price=0,
        telegram_user_id=111,
        degree="Thạc sĩ, Bác sĩ",
        experience_years=8,
        hospital="Bệnh viện Bạch Mai",
        bio="Tư vấn các vấn đề nội khoa thường gặp.",
    )

    row = doctors.get_doctor(doctor_id)

    assert row == {
        "id": doctor_id,
        "name": "BS An",
        "specialty": "Nội tổng quát",
        "tier": "free",
        "price": 0,
        "telegram_user_id": 111,
        "active": 1,
        "available": True,
        "degree": "Thạc sĩ, Bác sĩ",
        "experience_years": 8,
        "hospital": "Bệnh viện Bạch Mai",
        "bio": "Tư vấn các vấn đề nội khoa thường gặp.",
    }


def test_seed_default_doctors_adds_mock_profiles_for_every_doctor():
    from src.chat.storage.seed_doctors import seed_default_doctors

    seed_default_doctors()

    rows = doctors.list_doctors("free") + doctors.list_doctors("paid")

    assert len(rows) == 30
    for row in rows:
        assert row["degree"]
        assert row["experience_years"] > 0
        assert row["hospital"]
        assert row["bio"]


def test_doctor_without_profile_fields_gets_mock_profile():
    doctor_id = doctors.create_doctor("BS Legacy", "Nhi khoa", "free", 0, 204)

    row = doctors.get_doctor(doctor_id)

    assert row["degree"]
    assert row["experience_years"] > 0
    assert row["hospital"]
    assert row["bio"]


def test_list_doctors_filters_tier_and_inactive():
    free_id = doctors.create_doctor("BS Free", "Nhi", "free", 0, 201)
    paid_id = doctors.create_doctor("BS Paid", "Da liễu", "paid", 50_000, 202)
    inactive_id = doctors.create_doctor("BS Off", "Tim mạch", "free", 0, 203)
    doctors.set_doctor_active(inactive_id, False)

    free = doctors.list_doctors("free")
    paid = doctors.list_doctors("paid")
    inactive = doctors.get_doctor(inactive_id)

    assert [d["id"] for d in free] == [free_id]
    assert [d["id"] for d in paid] == [paid_id]
    assert free[0]["available"] is True
    assert paid[0]["price"] == 50_000
    assert inactive["available"] is False


def test_admin_doctor_helpers_update_list_and_soft_delete():
    doctor_id = doctors.create_doctor("BS Old", "Nội", "free", 0, 205)

    changed = doctors.update_doctor(
        doctor_id,
        name="BS New",
        specialty="Tim mạch",
        tier="paid",
        price=75_000,
        telegram_user_id=206,
        degree="Bác sĩ chuyên khoa II",
        experience_years=12,
        hospital="Bệnh viện E",
        bio="Tư vấn tim mạch.",
    )

    row = doctors.get_doctor(doctor_id)
    assert changed is True
    assert row["name"] == "BS New"
    assert row["specialty"] == "Tim mạch"
    assert row["tier"] == "paid"
    assert row["price"] == 75_000
    assert row["telegram_user_id"] == 206
    assert row["degree"] == "Bác sĩ chuyên khoa II"
    assert row["experience_years"] == 12
    assert row["hospital"] == "Bệnh viện E"
    assert row["bio"] == "Tư vấn tim mạch."

    assert [d["id"] for d in doctors.list_all_doctors(include_inactive=False)] == [doctor_id]
    assert doctors.delete_doctor(doctor_id) is True
    assert doctors.get_doctor(doctor_id)["active"] == 0
    assert doctors.list_all_doctors(include_inactive=False) == []
    assert [d["id"] for d in doctors.list_all_doctors()] == [doctor_id]


def test_consultation_create_accept_lookup_and_end():
    doctor_id = doctors.create_doctor("BS Relay", "Nội", "free", 0, 301)
    doctor = doctors.get_doctor(doctor_id)
    consult_id = doctors.create_consultation(patient_chat_id=9001, doctor=doctor, tier="free")

    pending = doctors.get_consultation(consult_id)
    assert pending["status"] == "pending"
    assert pending["patient_chat_id"] == 9001
    assert pending["doctor_chat_id"] == 301
    assert pending["doctor_name"] == "BS Relay"

    assert doctors.accept_consultation(consult_id) is True
    active = doctors.active_consultation_for_chat(9001)
    assert active["id"] == consult_id
    assert doctors.active_consultation_for_chat(301)["id"] == consult_id

    doctors.end_consultation(consult_id)
    assert doctors.active_consultation_for_chat(9001) is None
    assert doctors.get_consultation(consult_id)["status"] == "ended"


def test_decline_consultation():
    doctor_id = doctors.create_doctor("BS Decline", "Nội", "free", 0, 302)
    consult_id = doctors.create_consultation(9002, doctors.get_doctor(doctor_id), "free")

    doctors.decline_consultation(consult_id)

    row = doctors.get_consultation(consult_id)
    assert row["status"] == "declined"
    assert doctors.active_consultation_for_chat(9002) is None


def test_accept_consultation_enforces_one_active_per_doctor():
    doctor_id = doctors.create_doctor("BS Busy", "Nội", "free", 0, 303)
    first = doctors.create_consultation(9101, doctors.get_doctor(doctor_id), "free")
    second = doctors.create_consultation(9102, doctors.get_doctor(doctor_id), "free")

    assert doctors.accept_consultation(first) is True
    assert doctors.accept_consultation(second) is False
    assert doctors.get_consultation(second)["status"] == "pending"
    assert doctors.get_doctor(doctor_id)["available"] is False


def test_patient_cannot_have_pending_or_active_consultation():
    doctor_id = doctors.create_doctor("BS PatientGuard", "Nội", "free", 0, 304)
    doctor = doctors.get_doctor(doctor_id)
    first = doctors.create_consultation(9201, doctor, "free")

    assert doctors.open_consultation_for_patient(9201)["id"] == first


def test_sweep_idle_ends_old_active_consultation():
    doctor_id = doctors.create_doctor("BS Idle", "Nội", "free", 0, 401)
    consult_id = doctors.create_consultation(9301, doctors.get_doctor(doctor_id), "free")
    assert doctors.accept_consultation(consult_id) is True

    conn = get_sqlite()
    conn.execute(
        "UPDATE doctor_consultation SET last_activity_at = ? WHERE id = ?",
        (time.time() - 3600, consult_id),
    )
    conn.commit()

    ended = doctors.sweep_idle(max_idle_seconds=60)

    assert [c["id"] for c in ended] == [consult_id]
    assert doctors.get_consultation(consult_id)["status"] == "ended"


def test_seed_doctors_upserts_by_telegram_user_id():
    from src.chat.storage.seed_doctors import seed_doctors

    rows = [
        {"name": "BS Demo", "specialty": "Nội", "tier": "free", "price": 0, "telegram_user_id": 12345},
        {"name": "BS Demo Updated", "specialty": "Nội", "tier": "free", "price": 0, "telegram_user_id": 12345},
    ]

    seed_doctors(rows)

    listed = doctors.list_doctors("free")
    matching = [d for d in listed if d["telegram_user_id"] == 12345]
    assert len(matching) == 1
    assert matching[0]["name"] == "BS Demo Updated"


def test_default_doctors_cover_all_specialties_both_tiers():
    from src.chat.storage.seed_doctors import DEFAULT_DOCTORS

    specialties = sorted({d["specialty"] for d in DEFAULT_DOCTORS})
    assert len(specialties) == 15

    free = [d for d in DEFAULT_DOCTORS if d["tier"] == "free"]
    paid = [d for d in DEFAULT_DOCTORS if d["tier"] == "paid"]
    assert {d["specialty"] for d in free} == set(specialties)
    assert {d["specialty"] for d in paid} == set(specialties)

    # Every paid doctor bills at the 2000 VND/min base rate; free doctors are free.
    assert all(d["price"] == 0 for d in free)
    assert all(d["price"] == 2000 for d in paid)

    # telegram_user_id must be unique across the seed set (UNIQUE index).
    ids = [d["telegram_user_id"] for d in DEFAULT_DOCTORS]
    assert len(ids) == len(set(ids))


def test_seed_default_doctors_is_idempotent():
    from src.chat.storage.seed_doctors import DEFAULT_DOCTORS, seed_default_doctors

    seed_default_doctors()
    seed_default_doctors()

    free = doctors.list_doctors("free")
    paid = doctors.list_doctors("paid")
    seeded_ids = {d["telegram_user_id"] for d in DEFAULT_DOCTORS}
    free_seeded = [d for d in free if d["telegram_user_id"] in seeded_ids]
    paid_seeded = [d for d in paid if d["telegram_user_id"] in seeded_ids]

    assert len(free_seeded) == 15
    assert len(paid_seeded) == 15


def test_accept_free_consultation_sets_five_minute_expiry():
    doctor_id = doctors.create_doctor("BS FreeCap", "Nội", "free", 0, 1201)
    consult_id = doctors.create_consultation(9501, doctors.get_doctor(doctor_id), "free")

    assert doctors.accept_consultation(consult_id) is True

    row = doctors.get_consultation(consult_id)
    assert row["expires_at"] is not None
    remaining = row["expires_at"] - row["accepted_at"]
    assert abs(remaining - doctors.FREE_SESSION_SECONDS) < 2


def test_free_cooldown_active_after_recent_free_session():
    doctor_id = doctors.create_doctor("BS Cool", "Nội", "free", 0, 1202)
    consult_id = doctors.create_consultation(9502, doctors.get_doctor(doctor_id), "free")
    doctors.accept_consultation(consult_id)

    # No cooldown while no session has ended.
    assert doctors.free_cooldown_remaining(9502) == 0

    doctors.end_consultation(consult_id)

    remaining = doctors.free_cooldown_remaining(9502)
    assert remaining > 0
    assert remaining <= doctors.FREE_COOLDOWN_SECONDS


def test_free_cooldown_expires_after_window():
    doctor_id = doctors.create_doctor("BS CoolGone", "Nội", "free", 0, 1203)
    consult_id = doctors.create_consultation(9503, doctors.get_doctor(doctor_id), "free")
    doctors.accept_consultation(consult_id)
    doctors.end_consultation(consult_id)

    # Backdate the end so the cooldown window has fully elapsed.
    conn = get_sqlite()
    conn.execute(
        "UPDATE doctor_consultation SET ended_at = ? WHERE id = ?",
        (time.time() - doctors.FREE_COOLDOWN_SECONDS - 5, consult_id),
    )
    conn.commit()

    assert doctors.free_cooldown_remaining(9503) == 0


def test_paid_session_does_not_trigger_free_cooldown():
    doctor_id = doctors.create_doctor("BS PaidNoCool", "Nội", "paid", 2000, 1204)
    consult_id = doctors.create_consultation(9504, doctors.get_doctor(doctor_id), "paid")
    doctors.accept_consultation(consult_id)
    doctors.end_consultation(consult_id)

    assert doctors.free_cooldown_remaining(9504) == 0
