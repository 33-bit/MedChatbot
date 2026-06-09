from __future__ import annotations

import time

from src.chat.clients import get_sqlite
from src.chat.storage import doctors


def _make_active(patient_chat_id: int, doctor_tg: int, tier: str = "free") -> dict:
    doctor_id = doctors.create_doctor("BS Sweep", "Nội", tier, 0, doctor_tg)
    consult_id = doctors.create_consultation(patient_chat_id, doctors.get_doctor(doctor_id), tier)
    doctors.accept_consultation(consult_id)
    return doctors.get_consultation(consult_id)


def _make_active_paid(patient_chat_id: int, doctor_tg: int, rate: int = 2000) -> dict:
    doctor_id = doctors.create_doctor("BS Paid", "Nội", "paid", rate, doctor_tg)
    consult_id = doctors.create_consultation(patient_chat_id, doctors.get_doctor(doctor_id), "paid")
    doctors.accept_consultation(consult_id)
    return doctors.get_consultation(consult_id)


def _set(consult_id: int, **cols) -> None:
    conn = get_sqlite()
    assignments = ", ".join(f"{k} = ?" for k in cols)
    conn.execute(
        f"UPDATE doctor_consultation SET {assignments} WHERE id = ?",
        (*cols.values(), consult_id),
    )
    conn.commit()


def test_sweep_sessions_ends_expired_active_consultation():
    row = _make_active(5001, 6001)
    now = time.time()
    _set(row["id"], expires_at=now - 10, last_activity_at=now)

    effects = doctors.sweep_sessions(now=now)

    ended = [e for e in effects if e["kind"] == "ended_timeout" and e["consultation"]["id"] == row["id"]]
    assert len(ended) == 1
    assert doctors.get_consultation(row["id"])["status"] == "ended"
    assert doctors.get_consultation(row["id"])["end_reason"] == "timeout"


def test_sweep_sessions_warns_before_expiry_once():
    row = _make_active(5002, 6002)
    now = time.time()
    # Expires in 30s; warn window is 60s, so it should warn now.
    _set(row["id"], expires_at=now + 30, last_activity_at=now)

    effects = doctors.sweep_sessions(now=now)
    warns = [e for e in effects if e["kind"] == "warn" and e["consultation"]["id"] == row["id"]]
    assert len(warns) == 1
    assert doctors.get_consultation(row["id"])["warned_at"] is not None

    # Second sweep must not warn again.
    effects2 = doctors.sweep_sessions(now=now)
    warns2 = [e for e in effects2 if e["kind"] == "warn" and e["consultation"]["id"] == row["id"]]
    assert warns2 == []


def test_sweep_sessions_skips_unexpired_and_null_expiry():
    row_future = _make_active(5003, 6003)
    row_null = _make_active(5004, 6004)
    now = time.time()
    _set(row_future["id"], expires_at=now + 600, last_activity_at=now)
    # row_null keeps expires_at = NULL (no time cap)
    _set(row_null["id"], expires_at=None, last_activity_at=now)

    effects = doctors.sweep_sessions(now=now)
    touched_ids = {e["consultation"]["id"] for e in effects}
    assert row_future["id"] not in touched_ids
    assert row_null["id"] not in touched_ids


def test_accept_paid_sets_block_rate_and_15min_expiry():
    row = _make_active_paid(5101, 6101, rate=2000)
    assert row["expires_at"] is not None
    assert abs((row["expires_at"] - row["accepted_at"]) - doctors.PAID_BLOCK_SECONDS) < 2
    assert row["block_index"] == 0
    assert row["rate_per_min"] == 2000
    assert row["block_started_at"] is not None
    assert row["minutes_billed_block"] == 0


def test_sweep_bills_whole_minutes_at_block_rate():
    row = _make_active_paid(5102, 6102, rate=2000)
    start = row["block_started_at"]
    # 3 minutes and 30 seconds into the block → 3 whole minutes billable.
    now = start + 210

    effects = doctors.sweep_sessions(now=now)
    bills = [e for e in effects if e["kind"] == "bill" and e["consultation"]["id"] == row["id"]]
    assert len(bills) == 1
    assert bills[0]["amount"] == 3 * 2000
    assert bills[0]["account_id"] == "tg:5102"
    assert doctors.get_consultation(row["id"])["minutes_billed_block"] == 3

    # A later tick only bills the new minute, never re-bills.
    effects2 = doctors.sweep_sessions(now=start + 270)  # 4m30s → minute 4
    bills2 = [e for e in effects2 if e["kind"] == "bill" and e["consultation"]["id"] == row["id"]]
    assert len(bills2) == 1
    assert bills2[0]["amount"] == 1 * 2000
    assert doctors.get_consultation(row["id"])["minutes_billed_block"] == 4


def test_sweep_caps_billing_at_block_length():
    row = _make_active_paid(5103, 6103, rate=2000)
    start = row["block_started_at"]
    # Far past the 15-minute block; never bill more than 15 minutes for it.
    now = start + doctors.PAID_BLOCK_SECONDS + 600
    _set(row["id"], expires_at=now + 600)  # keep active so we isolate billing cap

    effects = doctors.sweep_sessions(now=now)
    bills = [e for e in effects if e["kind"] == "bill" and e["consultation"]["id"] == row["id"]]
    total = sum(b["amount"] for b in bills)
    assert total == 15 * 2000
    assert doctors.get_consultation(row["id"])["minutes_billed_block"] == 15


def test_sweep_paid_near_expiry_emits_extend_offer():
    row = _make_active_paid(5104, 6104, rate=2000)
    now = row["expires_at"] - 30  # within warn window

    effects = doctors.sweep_sessions(now=now)
    offers = [e for e in effects if e["kind"] == "extend_offer" and e["consultation"]["id"] == row["id"]]
    assert len(offers) == 1
    # Next block would bill at +1000/min.
    assert offers[0]["next_rate"] == 3000


def test_request_extension_renews_block_with_higher_rate():
    row = _make_active_paid(5105, 6105, rate=2000)
    # Drive the session to the brink, then extend.
    near = row["expires_at"] - 10
    doctors.sweep_sessions(now=near)  # emits offer, stamps warned_at

    ok = doctors.request_extension(row["id"], now=near)
    assert ok is True

    renewed = doctors.get_consultation(row["id"])
    assert renewed["block_index"] == 1
    assert renewed["rate_per_min"] == 3000
    assert renewed["minutes_billed_block"] == 0
    assert renewed["warned_at"] is None
    assert abs((renewed["expires_at"] - near) - doctors.PAID_BLOCK_SECONDS) < 2


def test_sweep_paid_timeout_without_extension_ends_session():
    row = _make_active_paid(5106, 6106, rate=2000)
    now = row["expires_at"] + 5

    effects = doctors.sweep_sessions(now=now)
    ended = [e for e in effects if e["kind"] == "ended_timeout" and e["consultation"]["id"] == row["id"]]
    assert len(ended) == 1
    assert doctors.get_consultation(row["id"])["status"] == "ended"


def test_paid_pair_cooldown_after_end():
    row = _make_active_paid(5107, 6107, rate=2000)
    doctor_id = row["doctor_id"]
    doctors.end_consultation(row["id"])

    remaining = doctors.paid_pair_cooldown_remaining(5107, doctor_id)
    assert remaining > 0
    assert remaining <= doctors.PAID_PAIR_COOLDOWN_SECONDS

    # A different doctor is not under cooldown.
    assert doctors.paid_pair_cooldown_remaining(5107, doctor_id + 99999) == 0


def test_settle_block_minutes_bills_unbilled_whole_minutes():
    row = _make_active_paid(5108, 6108, rate=2000)
    start = row["block_started_at"]
    # 2m40s elapsed, nothing billed yet → 2 whole minutes to settle.
    settled = doctors.settle_block_minutes(row["id"], now=start + 160)
    assert settled == ("tg:5108", 2 * 2000)
    assert doctors.get_consultation(row["id"])["minutes_billed_block"] == 2

    # Nothing new to settle on an immediate repeat.
    assert doctors.settle_block_minutes(row["id"], now=start + 160) is None


def test_settle_block_minutes_none_for_free_session():
    row = _make_active(5109, 6109)
    assert doctors.settle_block_minutes(row["id"]) is None


