from __future__ import annotations

import time

from src.chat.clients import get_sqlite
from src.chat.storage import doctors


def _busy_doctor(doctor_tg: int, tier: str = "free", rate: int = 0) -> int:
    """Create a doctor and occupy them with an active consultation."""
    doctor_id = doctors.create_doctor("BS Busy WL", "Nội", tier, rate, doctor_tg)
    consult_id = doctors.create_consultation(70000 + doctor_tg, doctors.get_doctor(doctor_id), tier)
    doctors.accept_consultation(consult_id)
    return doctor_id


def test_join_waitlist_returns_position():
    doctor_id = _busy_doctor(7301)
    p1 = doctors.join_waitlist(doctor_id, 81001, "free")
    p2 = doctors.join_waitlist(doctor_id, 81002, "free")
    assert p1 == 1
    assert p2 == 2


def test_join_waitlist_is_idempotent_per_patient():
    doctor_id = _busy_doctor(7302)
    doctors.join_waitlist(doctor_id, 81010, "free")
    again = doctors.join_waitlist(doctor_id, 81010, "free")
    # Same patient re-joining keeps their existing place, not a new row.
    assert again == 1
    status = doctors.waitlist_status(doctor_id, 81010)
    assert status["total_waiting"] == 1


def test_waitlist_status_reports_position_total_and_estimate():
    doctor_id = _busy_doctor(7303, tier="paid", rate=2000)
    doctors.join_waitlist(doctor_id, 81020, "paid")
    doctors.join_waitlist(doctor_id, 81021, "paid")

    s1 = doctors.waitlist_status(doctor_id, 81020)
    s2 = doctors.waitlist_status(doctor_id, 81021)
    assert s1["position"] == 1
    assert s2["position"] == 2
    assert s1["total_waiting"] == 2
    # Estimated wait grows with position and is > 0 for those behind the front.
    assert s2["estimated_wait_seconds"] > s1["estimated_wait_seconds"]


def test_leave_waitlist_removes_and_reorders():
    doctor_id = _busy_doctor(7304)
    doctors.join_waitlist(doctor_id, 81030, "free")
    doctors.join_waitlist(doctor_id, 81031, "free")

    doctors.leave_waitlist(doctor_id, 81030)

    assert doctors.waitlist_status(doctor_id, 81030) is None
    s = doctors.waitlist_status(doctor_id, 81031)
    assert s["position"] == 1
    assert s["total_waiting"] == 1


def test_promote_waitlist_offers_front_of_queue():
    doctor_id = _busy_doctor(7305)
    doctors.join_waitlist(doctor_id, 81040, "free")
    doctors.join_waitlist(doctor_id, 81041, "free")

    offered = doctors.promote_waitlist(doctor_id)
    assert offered["patient_chat_id"] == 81040
    assert offered["status"] == "offered"
    assert offered["notified_at"] is not None

    # The other patient is now at the front for the next promotion.
    assert doctors.waitlist_status(doctor_id, 81041)["position"] == 1


def test_promote_waitlist_none_when_empty():
    doctor_id = _busy_doctor(7306)
    assert doctors.promote_waitlist(doctor_id) is None


def test_sweep_waitlist_offers_expire_and_advance():
    doctor_id = _busy_doctor(7307)
    doctors.join_waitlist(doctor_id, 81050, "free")
    doctors.join_waitlist(doctor_id, 81051, "free")
    offered = doctors.promote_waitlist(doctor_id)

    # Backdate the offer beyond the claim window.
    conn = get_sqlite()
    conn.execute(
        "UPDATE doctor_waitlist SET notified_at = ? WHERE id = ?",
        (time.time() - doctors.WAITLIST_CLAIM_SECONDS - 5, offered["id"]),
    )
    conn.commit()

    effects = doctors.sweep_waitlist(now=time.time())
    kinds = {(e["kind"], e["entry"]["patient_chat_id"]) for e in effects}
    # First patient's offer expired; next patient gets a fresh offer.
    assert ("offer_expired", 81050) in kinds
    assert ("offer", 81051) in kinds


def test_sweep_waitlist_promotes_when_doctor_becomes_free():
    # Doctor is free (no active consultation) but has waiters and no offer yet.
    doctor_id = doctors.create_doctor("BS FreedUp", "Nội", "free", 0, 7308)
    doctors.join_waitlist(doctor_id, 81060, "free")
    doctors.join_waitlist(doctor_id, 81061, "free")

    effects = doctors.sweep_waitlist(now=time.time())
    offers = [(e["kind"], e["entry"]["patient_chat_id"]) for e in effects if e["kind"] == "offer"]
    assert offers == [("offer", 81060)]


def test_sweep_waitlist_repeats_offer_reminder_every_30_seconds():
    doctor_id = doctors.create_doctor("BS Reminder", "Nội", "free", 0, 7312)
    doctors.join_waitlist(doctor_id, 81100, "free")
    offered_at = time.time()
    offered = doctors.promote_waitlist(doctor_id, now=offered_at)

    assert doctors.sweep_waitlist(now=offered_at + doctors.WAITLIST_REMINDER_SECONDS - 1) == []

    effects = doctors.sweep_waitlist(now=offered_at + doctors.WAITLIST_REMINDER_SECONDS + 1)
    assert [(e["kind"], e["entry"]["patient_chat_id"]) for e in effects] == [("offer_reminder", 81100)]

    assert doctors.sweep_waitlist(now=offered_at + doctors.WAITLIST_REMINDER_SECONDS + 2) == []

    effects = doctors.sweep_waitlist(now=offered_at + doctors.WAITLIST_REMINDER_SECONDS * 2 + 2)
    assert [(e["kind"], e["entry"]["id"]) for e in effects] == [("offer_reminder", offered["id"])]


def test_sweep_waitlist_no_promote_while_doctor_busy():
    doctor_id = _busy_doctor(7309)
    doctors.join_waitlist(doctor_id, 81070, "free")

    effects = doctors.sweep_waitlist(now=time.time())
    assert effects == []


def test_waitlist_count_reports_waiting_total():
    doctor_id = _busy_doctor(7310)
    assert doctors.waitlist_count(doctor_id) == 0
    doctors.join_waitlist(doctor_id, 81080, "free")
    doctors.join_waitlist(doctor_id, 81081, "free")
    assert doctors.waitlist_count(doctor_id) == 2


def test_waitlist_count_excludes_promoted_offer():
    doctor_id = _busy_doctor(7311)
    doctors.join_waitlist(doctor_id, 81090, "free")
    doctors.join_waitlist(doctor_id, 81091, "free")
    doctors.promote_waitlist(doctor_id)
    # The promoted entry is 'offered', leaving one still 'waiting'.
    assert doctors.waitlist_count(doctor_id) == 1

