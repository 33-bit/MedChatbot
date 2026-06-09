"""
doctors.py
----------
SQLite-backed doctor registry and doctor consultation lifecycle.

This module contains no Telegram API calls. Channel code handles messaging.
"""

from __future__ import annotations

import time
from threading import RLock

from src.chat.clients import get_sqlite

_SQLITE_LOCK = RLock()
DEFAULT_IDLE_TIMEOUT_SECONDS = 30 * 60
WARN_BEFORE_EXPIRY_SECONDS = 60

FREE_SESSION_SECONDS = 5 * 60
FREE_COOLDOWN_SECONDS = 30 * 60

PAID_BLOCK_SECONDS = 15 * 60
PAID_BLOCK_MINUTES = 15
# Each renewed 15-minute block adds this much to the per-minute rate.
PAID_RATE_STEP_PER_MIN = 1000
PAID_PAIR_COOLDOWN_SECONDS = 30 * 60

# How long a promoted waitlist entry has to be picked up before it expires and
# the slot advances to the next person in line.
WAITLIST_CLAIM_SECONDS = 2 * 60
WAITLIST_REMINDER_SECONDS = 30
# Per-position estimate used to show an approximate wait time.
WAITLIST_ESTIMATE_PER_POSITION = {
    "free": FREE_SESSION_SECONDS,
    "paid": PAID_BLOCK_SECONDS,
}

_MOCK_DEGREES = (
    "Bác sĩ chuyên khoa I",
    "Thạc sĩ, Bác sĩ",
    "Bác sĩ chuyên khoa II",
    "Tiến sĩ, Bác sĩ",
)

_MOCK_HOSPITALS = (
    "Bệnh viện Bạch Mai",
    "Bệnh viện Đại học Y Hà Nội",
    "Bệnh viện Trung ương Quân đội 108",
    "Bệnh viện E",
    "Bệnh viện Hữu nghị Việt Đức",
)


def _doctor_row(row) -> dict | None:
    if row is None:
        return None
    degree = row[8]
    experience_years = row[9]
    hospital = row[10]
    bio = row[11]
    if not degree or experience_years is None or not hospital or not bio:
        seed = f"{row[0]}:{row[1]}:{row[2]}:{row[3]}"
        key = sum(ord(ch) for ch in seed)
        specialty = str(row[2] or "nội khoa").strip() or "nội khoa"
        degree = degree or _MOCK_DEGREES[key % len(_MOCK_DEGREES)]
        experience_years = experience_years if experience_years is not None else 5 + (key % 8) + (2 if row[3] == "paid" else 0)
        hospital = hospital or _MOCK_HOSPITALS[(key // 3) % len(_MOCK_HOSPITALS)]
        bio = bio or (
            f"{row[1]} có kinh nghiệm tư vấn các vấn đề {specialty.lower()} "
            "và hướng dẫn người bệnh đi khám đúng lúc."
        )
    return {
        "id": row[0],
        "name": row[1],
        "specialty": row[2],
        "tier": row[3],
        "price": row[4],
        "telegram_user_id": row[5],
        "active": row[6],
        "available": bool(row[7]),
        "degree": degree,
        "experience_years": experience_years,
        "hospital": hospital,
        "bio": bio,
    }


def create_doctor(
    name: str,
    specialty: str | None,
    tier: str,
    price: int,
    telegram_user_id: int,
    degree: str | None = None,
    experience_years: int | None = None,
    hospital: str | None = None,
    bio: str | None = None,
) -> int:
    with _SQLITE_LOCK:
        conn = get_sqlite()
        cur = conn.execute(
            "INSERT INTO doctor ("
            "name, specialty, tier, price, telegram_user_id, active, "
            "degree, experience_years, hospital, bio"
            ") VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?)",
            (name, specialty, tier, price, telegram_user_id, degree, experience_years, hospital, bio),
        )
        conn.commit()
        return int(cur.lastrowid)


def set_doctor_active(doctor_id: int, active: bool) -> None:
    with _SQLITE_LOCK:
        conn = get_sqlite()
        conn.execute("UPDATE doctor SET active = ? WHERE id = ?", (1 if active else 0, doctor_id))
        conn.commit()


def get_doctor(doctor_id: int) -> dict | None:
    with _SQLITE_LOCK:
        conn = get_sqlite()
        row = conn.execute(
            """
            SELECT d.id, d.name, d.specialty, d.tier, d.price, d.telegram_user_id, d.active,
                   CASE WHEN d.active = 0 THEN 0
                        WHEN EXISTS (
                            SELECT 1 FROM doctor_consultation c
                            WHERE c.doctor_id = d.id AND c.status = 'active'
                        ) THEN 0 ELSE 1 END AS available,
                   d.degree, d.experience_years, d.hospital, d.bio
            FROM doctor d WHERE d.id = ?
            """,
            (doctor_id,),
        ).fetchone()
    return _doctor_row(row)


def list_doctors(tier: str) -> list[dict]:
    with _SQLITE_LOCK:
        conn = get_sqlite()
        rows = conn.execute(
            """
            SELECT d.id, d.name, d.specialty, d.tier, d.price, d.telegram_user_id, d.active,
                   CASE WHEN EXISTS (
                       SELECT 1 FROM doctor_consultation c
                       WHERE c.doctor_id = d.id AND c.status = 'active'
                   ) THEN 0 ELSE 1 END AS available,
                   d.degree, d.experience_years, d.hospital, d.bio
            FROM doctor d
            WHERE d.tier = ? AND d.active = 1
            ORDER BY d.id
            """,
            (tier,),
        ).fetchall()
    return [_doctor_row(row) for row in rows]


def _consultation_row(row) -> dict | None:
    if row is None:
        return None
    return {
        "id": row[0],
        "patient_chat_id": row[1],
        "doctor_id": row[2],
        "doctor_chat_id": row[3],
        "tier": row[4],
        "fee": row[5],
        "status": row[6],
        "created_at": row[7],
        "accepted_at": row[8],
        "ended_at": row[9],
        "last_activity_at": row[10],
        "expires_at": row[11],
        "warned_at": row[12],
        "end_reason": row[13],
        "block_index": row[14],
        "rate_per_min": row[15],
        "block_started_at": row[16],
        "minutes_billed_block": row[17],
        "extend_requested": row[18],
        "doctor_name": row[19] if len(row) > 19 else None,
        "doctor_specialty": row[20] if len(row) > 20 else None,
    }


def _consultation_select(where: str = "") -> str:
    return f"""
        SELECT c.id, c.patient_chat_id, c.doctor_id, c.doctor_chat_id, c.tier,
               c.fee, c.status, c.created_at, c.accepted_at, c.ended_at,
               c.last_activity_at, c.expires_at, c.warned_at, c.end_reason,
               c.block_index, c.rate_per_min, c.block_started_at,
               c.minutes_billed_block, c.extend_requested, d.name, d.specialty
        FROM doctor_consultation c
        JOIN doctor d ON d.id = c.doctor_id
        {where}
    """


def get_consultation(consultation_id: int) -> dict | None:
    with _SQLITE_LOCK:
        conn = get_sqlite()
        row = conn.execute(
            _consultation_select("WHERE c.id = ?"),
            (consultation_id,),
        ).fetchone()
    return _consultation_row(row)


def _open_consultation_for_chat(conn, chat_id: int) -> dict | None:
    row = conn.execute(
        _consultation_select(
            "WHERE c.patient_chat_id = ? AND c.status IN ('pending', 'active') "
            "ORDER BY c.id DESC LIMIT 1"
        ),
        (int(chat_id),),
    ).fetchone()
    return _consultation_row(row)


def _active_consultation_for_chat(conn, chat_id: int) -> dict | None:
    row = conn.execute(
        _consultation_select(
            "WHERE c.status = 'active' AND (c.patient_chat_id = ? OR c.doctor_chat_id = ?) "
            "ORDER BY c.id DESC LIMIT 1"
        ),
        (int(chat_id), int(chat_id)),
    ).fetchone()
    return _consultation_row(row)


def create_consultation(patient_chat_id: int | str, doctor: dict, tier: str) -> int:
    with _SQLITE_LOCK:
        conn = get_sqlite()
        if not isinstance(doctor, dict):
            raise ValueError("doctor not found")
        if not doctor.get("active"):
            raise ValueError("doctor is inactive")
        if not doctor.get("available"):
            raise ValueError("doctor is not available")
        doctor_id = doctor["id"]
        doctor_chat_id = int(doctor["telegram_user_id"])
        patient_chat_id = int(patient_chat_id)
        fee = int(doctor.get("price") or 0)
        if _open_consultation_for_chat(conn, patient_chat_id) is not None:
            raise ValueError("patient chat already has an open consultation")
        now = time.time()
        cur = conn.execute(
            """
            INSERT INTO doctor_consultation (
                patient_chat_id, doctor_id, doctor_chat_id, tier, fee,
                status, created_at, last_activity_at
            ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
            """,
            (patient_chat_id, doctor_id, doctor_chat_id, tier, fee, now, now),
        )
        conn.commit()
        return int(cur.lastrowid)


def open_consultation_for_patient(patient_chat_id: int) -> dict | None:
    with _SQLITE_LOCK:
        conn = get_sqlite()
        row = conn.execute(
            _consultation_select(
                "WHERE c.patient_chat_id = ? AND c.status IN ('pending', 'active') "
                "ORDER BY c.id DESC LIMIT 1"
            ),
            (int(patient_chat_id),),
        ).fetchone()
    return _consultation_row(row)


def accept_consultation(consultation_id: int) -> bool:
    with _SQLITE_LOCK:
        conn = get_sqlite()
        consultation = get_consultation(consultation_id)
        if consultation is None or consultation["status"] != "pending":
            return False
        active = conn.execute(
            """
            SELECT 1 FROM doctor_consultation
            WHERE status = 'active'
              AND id != ?
              AND doctor_id = ?
            LIMIT 1
            """,
            (consultation_id, consultation["doctor_id"]),
        ).fetchone()
        if active is not None:
            return False
        now = time.time()
        tier = consultation["tier"]
        if tier == "free":
            cur = conn.execute(
                """
                UPDATE doctor_consultation
                SET status = 'active', accepted_at = ?, last_activity_at = ?, expires_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (now, now, now + FREE_SESSION_SECONDS, consultation_id),
            )
        else:
            rate = consultation["fee"] or 0
            cur = conn.execute(
                """
                UPDATE doctor_consultation
                SET status = 'active', accepted_at = ?, last_activity_at = ?,
                    expires_at = ?, block_index = 0, rate_per_min = ?,
                    block_started_at = ?, minutes_billed_block = 0
                WHERE id = ? AND status = 'pending'
                """,
                (now, now, now + PAID_BLOCK_SECONDS, rate, now, consultation_id),
            )
        conn.commit()
        return cur.rowcount == 1


def decline_consultation(consultation_id: int) -> None:
    with _SQLITE_LOCK:
        now = time.time()
        conn = get_sqlite()
        conn.execute(
            """
            UPDATE doctor_consultation
            SET status = 'declined', ended_at = ?, last_activity_at = ?
            WHERE id = ? AND status = 'pending'
            """,
            (now, now, consultation_id),
        )
        conn.commit()


def end_consultation(consultation_id: int) -> None:
    with _SQLITE_LOCK:
        now = time.time()
        conn = get_sqlite()
        conn.execute(
            """
            UPDATE doctor_consultation
            SET status = 'ended', ended_at = ?, last_activity_at = ?
            WHERE id = ? AND status IN ('pending', 'active')
            """,
            (now, now, consultation_id),
        )
        conn.commit()


def free_cooldown_remaining(patient_chat_id: int, now: float | None = None) -> float:
    """Seconds until the patient may start another FREE consultation.

    Derived from history (single source of truth): the most recent free
    consultation that was actually accepted (accepted_at set) and has ended.
    Cooldown runs FREE_COOLDOWN_SECONDS from that end. Returns 0 if free.
    """
    if now is None:
        now = time.time()
    with _SQLITE_LOCK:
        conn = get_sqlite()
        row = conn.execute(
            "SELECT ended_at FROM doctor_consultation "
            "WHERE patient_chat_id = ? AND tier = 'free' "
            "AND accepted_at IS NOT NULL AND ended_at IS NOT NULL "
            "ORDER BY ended_at DESC LIMIT 1",
            (int(patient_chat_id),),
        ).fetchone()
    if row is None or row[0] is None:
        return 0
    remaining = (row[0] + FREE_COOLDOWN_SECONDS) - now
    return remaining if remaining > 0 else 0


def active_consultation_for_chat(chat_id: int) -> dict | None:
    with _SQLITE_LOCK:
        conn = get_sqlite()
        row = conn.execute(
            _consultation_select(
                "WHERE c.status = 'active' AND (c.patient_chat_id = ? OR c.doctor_chat_id = ?) "
                "ORDER BY c.id DESC LIMIT 1"
            ),
            (int(chat_id), int(chat_id)),
        ).fetchone()
    return _consultation_row(row)


def touch_activity(consultation_id: int) -> dict | None:
    with _SQLITE_LOCK:
        if get_consultation(consultation_id) is None:
            return None
        conn = get_sqlite()
        conn.execute(
            "UPDATE doctor_consultation SET last_activity_at = ? WHERE id = ?",
            (time.time(), consultation_id),
        )
        conn.commit()
        return get_consultation(consultation_id)


def sweep_idle(max_idle_seconds: int = DEFAULT_IDLE_TIMEOUT_SECONDS) -> list[dict]:
    cutoff = time.time() - max_idle_seconds
    with _SQLITE_LOCK:
        conn = get_sqlite()
        rows = conn.execute(
            _consultation_select(
                "WHERE c.status IN ('pending', 'active') AND c.last_activity_at < ? "
                "ORDER BY c.last_activity_at"
            ),
            (cutoff,),
        ).fetchall()
        ended = [_consultation_row(row) for row in rows]
        now = time.time()
        for consultation in ended:
            conn.execute(
                "UPDATE doctor_consultation SET status = 'ended', ended_at = ?, "
                "last_activity_at = ? WHERE id = ?",
                (now, now, consultation["id"]),
            )
        conn.commit()
    return ended


def sweep_sessions(now: float | None = None) -> list[dict]:
    """Time-driven sweep of active consultations.

    Pure of any Telegram/IO: returns a list of effect dicts describing what the
    channel layer should do. Each effect is {"kind": str, "consultation": dict, ...}.
    State mutations (billing minutes, ending expired sessions, stamping warnings)
    happen here so they are exactly-once across ticks. The channel layer performs
    the wallet debit for "bill" effects.

    Effect kinds emitted here:
      - "bill":          paid session accrued whole minutes; carries account_id + amount.
      - "warn":          free session within WARN_BEFORE_EXPIRY_SECONDS of expiry.
      - "extend_offer":  paid session near expiry; carries next_rate for renewal.
      - "ended_timeout": session passed expires_at and was ended (end_reason set).
    """
    if now is None:
        now = time.time()
    effects: list[dict] = []
    with _SQLITE_LOCK:
        conn = get_sqlite()
        rows = conn.execute(
            _consultation_select(
                "WHERE c.status = 'active' AND c.expires_at IS NOT NULL "
                "ORDER BY c.expires_at"
            )
        ).fetchall()
        sessions = [_consultation_row(row) for row in rows]

        for s in sessions:
            expires_at = s["expires_at"]
            if expires_at is None:
                continue

            # Paid per-minute billing (whole minutes elapsed in the current
            # block, capped at the block length). Runs before expiry handling so
            # the final minutes of a block are billed when it times out.
            if s["tier"] == "paid" and s["rate_per_min"] and s["block_started_at"]:
                elapsed_min = int((now - s["block_started_at"]) // 60)
                elapsed_min = min(elapsed_min, PAID_BLOCK_MINUTES)
                billable = elapsed_min - s["minutes_billed_block"]
                if billable > 0:
                    amount = billable * s["rate_per_min"]
                    conn.execute(
                        "UPDATE doctor_consultation SET minutes_billed_block = ? WHERE id = ?",
                        (elapsed_min, s["id"]),
                    )
                    effects.append({
                        "kind": "bill",
                        "consultation": get_consultation(s["id"]),
                        "account_id": f"tg:{s['patient_chat_id']}",
                        "amount": amount,
                    })

            if now >= expires_at:
                conn.execute(
                    "UPDATE doctor_consultation SET status = 'ended', ended_at = ?, "
                    "last_activity_at = ?, end_reason = COALESCE(end_reason, 'timeout') "
                    "WHERE id = ? AND status = 'active'",
                    (now, now, s["id"]),
                )
                ended = get_consultation(s["id"])
                effects.append({"kind": "ended_timeout", "consultation": ended})
                continue

            if expires_at - now <= WARN_BEFORE_EXPIRY_SECONDS and not s["warned_at"]:
                conn.execute(
                    "UPDATE doctor_consultation SET warned_at = ? WHERE id = ?",
                    (now, s["id"]),
                )
                refreshed = get_consultation(s["id"])
                if s["tier"] == "paid":
                    effects.append({
                        "kind": "extend_offer",
                        "consultation": refreshed,
                        "next_rate": s["rate_per_min"] + PAID_RATE_STEP_PER_MIN,
                    })
                else:
                    effects.append({"kind": "warn", "consultation": refreshed})
        conn.commit()
    return effects


def request_extension(consultation_id: int, now: float | None = None) -> bool:
    """Renew a paid session for another 15-minute block at a higher rate.

    Each renewal bumps block_index, raises rate_per_min by PAID_RATE_STEP_PER_MIN,
    resets the per-block billing counter and warning, and pushes expiry out by one
    block. Returns False if the consultation is not an active paid session.
    """
    if now is None:
        now = time.time()
    with _SQLITE_LOCK:
        conn = get_sqlite()
        row = get_consultation(consultation_id)
        if row is None or row["status"] != "active" or row["tier"] != "paid":
            return False
        new_rate = (row["rate_per_min"] or 0) + PAID_RATE_STEP_PER_MIN
        cur = conn.execute(
            """
            UPDATE doctor_consultation
            SET block_index = block_index + 1,
                rate_per_min = ?,
                block_started_at = ?,
                minutes_billed_block = 0,
                warned_at = NULL,
                expires_at = ?,
                extend_requested = 0,
                last_activity_at = ?
            WHERE id = ? AND status = 'active'
            """,
            (new_rate, now, now + PAID_BLOCK_SECONDS, now, consultation_id),
        )
        conn.commit()
        return cur.rowcount == 1


def paid_pair_cooldown_remaining(
    patient_chat_id: int, doctor_id: int, now: float | None = None
) -> float:
    """Seconds until the patient may reconnect to THIS doctor on the paid tier.

    Prevents fee-farming by reconnecting immediately after a short paid session.
    Derived from the most recent ended paid consultation for this exact
    (patient, doctor) pair. Returns 0 if none / elapsed.
    """
    if now is None:
        now = time.time()
    with _SQLITE_LOCK:
        conn = get_sqlite()
        row = conn.execute(
            "SELECT ended_at FROM doctor_consultation "
            "WHERE patient_chat_id = ? AND doctor_id = ? AND tier = 'paid' "
            "AND ended_at IS NOT NULL "
            "ORDER BY ended_at DESC LIMIT 1",
            (int(patient_chat_id), int(doctor_id)),
        ).fetchone()
    if row is None or row[0] is None:
        return 0
    remaining = (row[0] + PAID_PAIR_COOLDOWN_SECONDS) - now
    return remaining if remaining > 0 else 0


def settle_block_minutes(consultation_id: int, now: float | None = None) -> tuple[str, int] | None:
    """Bill any not-yet-billed whole minutes of a paid session's current block.

    Used when a paid session ends between ticks (e.g. patient /end) so the final
    whole minutes are still charged. Mirrors the sweep's per-minute math, capped
    at the block length. Returns (account_id, amount) if there is something to
    bill, else None. The caller performs the wallet debit.
    """
    if now is None:
        now = time.time()
    with _SQLITE_LOCK:
        conn = get_sqlite()
        s = get_consultation(consultation_id)
        if s is None or s["tier"] != "paid" or not s["rate_per_min"] or not s["block_started_at"]:
            return None
        elapsed_min = int((now - s["block_started_at"]) // 60)
        elapsed_min = min(elapsed_min, PAID_BLOCK_MINUTES)
        billable = elapsed_min - s["minutes_billed_block"]
        if billable <= 0:
            return None
        conn.execute(
            "UPDATE doctor_consultation SET minutes_billed_block = ? WHERE id = ?",
            (elapsed_min, consultation_id),
        )
        conn.commit()
    return (f"tg:{s['patient_chat_id']}", billable * s["rate_per_min"])


def _waitlist_row(row) -> dict | None:
    if row is None:
        return None
    return {
        "id": row[0],
        "doctor_id": row[1],
        "patient_chat_id": row[2],
        "tier": row[3],
        "created_at": row[4],
        "notified_at": row[5],
        "last_reminded_at": row[6],
        "status": row[7],
    }


def _waitlist_select(where: str = "") -> str:
    return (
        "SELECT id, doctor_id, patient_chat_id, tier, created_at, notified_at, last_reminded_at, status "
        f"FROM doctor_waitlist {where}"
    )


def join_waitlist(doctor_id: int, patient_chat_id: int, tier: str) -> int:
    """Add a patient to a doctor's waitlist; return their 1-based position.

    Idempotent per (doctor, patient): re-joining keeps the existing place.
    """
    with _SQLITE_LOCK:
        conn = get_sqlite()
        existing = conn.execute(
            "SELECT id FROM doctor_waitlist "
            "WHERE doctor_id = ? AND patient_chat_id = ? AND status IN ('waiting', 'offered')",
            (int(doctor_id), int(patient_chat_id)),
        ).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO doctor_waitlist "
                "(doctor_id, patient_chat_id, tier, created_at, status) "
                "VALUES (?, ?, ?, ?, 'waiting')",
                (int(doctor_id), int(patient_chat_id), tier, time.time()),
            )
            conn.commit()
    status = waitlist_status(doctor_id, patient_chat_id)
    return status["position"] if status else 1


def _waiting_rows(conn, doctor_id: int) -> list[dict]:
    rows = conn.execute(
        _waitlist_select(
            "WHERE doctor_id = ? AND status = 'waiting' "
            "ORDER BY created_at, id"
        ),
        (int(doctor_id),),
    ).fetchall()
    return [_waitlist_row(r) for r in rows]


def waitlist_status(doctor_id: int, patient_chat_id: int, now: float | None = None) -> dict | None:
    """Return {position, total_waiting, estimated_wait_seconds} or None.

    Position is 1-based among waiting/offered entries ordered by join time.
    Estimated wait scales with the number of people ahead.
    """
    with _SQLITE_LOCK:
        conn = get_sqlite()
        queue = _waiting_rows(conn, doctor_id)
    total = len(queue)
    for idx, entry in enumerate(queue):
        if int(entry["patient_chat_id"]) == int(patient_chat_id):
            position = idx + 1
            per = WAITLIST_ESTIMATE_PER_POSITION.get(entry["tier"], FREE_SESSION_SECONDS)
            return {
                "position": position,
                "total_waiting": total,
                "estimated_wait_seconds": idx * per,
            }
    return None


def waitlist_count(doctor_id: int) -> int:
    """Number of patients currently waiting (status 'waiting') for a doctor.

    Excludes a patient who has already been offered the slot ('offered').
    """
    with _SQLITE_LOCK:
        conn = get_sqlite()
        row = conn.execute(
            "SELECT COUNT(*) FROM doctor_waitlist WHERE doctor_id = ? AND status = 'waiting'",
            (int(doctor_id),),
        ).fetchone()
    return int(row[0]) if row else 0


def leave_waitlist(doctor_id: int, patient_chat_id: int) -> None:
    with _SQLITE_LOCK:
        conn = get_sqlite()
        conn.execute(
            "UPDATE doctor_waitlist SET status = 'left' "
            "WHERE doctor_id = ? AND patient_chat_id = ? AND status IN ('waiting', 'offered')",
            (int(doctor_id), int(patient_chat_id)),
        )
        conn.commit()


def promote_waitlist(doctor_id: int, now: float | None = None) -> dict | None:
    """Offer the slot to the front-of-queue waiter; return that entry or None.

    Marks the front 'waiting' entry as 'offered' and stamps notified_at. Does
    nothing if there is already an outstanding offer for this doctor.
    """
    if now is None:
        now = time.time()
    with _SQLITE_LOCK:
        conn = get_sqlite()
        outstanding = conn.execute(
            "SELECT 1 FROM doctor_waitlist WHERE doctor_id = ? AND status = 'offered' LIMIT 1",
            (int(doctor_id),),
        ).fetchone()
        if outstanding is not None:
            return None
        front = conn.execute(
            _waitlist_select(
                "WHERE doctor_id = ? AND status = 'waiting' ORDER BY created_at, id LIMIT 1"
            ),
            (int(doctor_id),),
        ).fetchone()
        if front is None:
            return None
        entry = _waitlist_row(front)
        conn.execute(
            "UPDATE doctor_waitlist SET status = 'offered', notified_at = ?, last_reminded_at = ? WHERE id = ?",
            (now, now, entry["id"]),
        )
        conn.commit()
    return _waitlist_row((entry["id"], entry["doctor_id"], entry["patient_chat_id"],
                          entry["tier"], entry["created_at"], now, now, "offered"))


def sweep_waitlist(now: float | None = None) -> list[dict]:
    """Expire stale offers and promote the next waiter.

    Returns effect dicts: {"kind": "offer_expired"|"offer", "entry": dict}.
    An offer that is not claimed within WAITLIST_CLAIM_SECONDS is expired and the
    slot advances to the next person in line.
    """
    if now is None:
        now = time.time()
    cutoff = now - WAITLIST_CLAIM_SECONDS
    effects: list[dict] = []
    with _SQLITE_LOCK:
        conn = get_sqlite()
        stale = conn.execute(
            _waitlist_select(
                "WHERE status = 'offered' AND notified_at IS NOT NULL AND notified_at < ?"
            ),
            (cutoff,),
        ).fetchall()
        stale_entries = [_waitlist_row(r) for r in stale]
        for entry in stale_entries:
            conn.execute(
                "UPDATE doctor_waitlist SET status = 'expired' WHERE id = ?",
                (entry["id"],),
            )
            effects.append({"kind": "offer_expired", "entry": entry})

        reminder_cutoff = now - WAITLIST_REMINDER_SECONDS
        reminders = conn.execute(
            _waitlist_select(
                "WHERE status = 'offered' AND notified_at IS NOT NULL AND notified_at >= ? "
                "AND (last_reminded_at IS NULL OR last_reminded_at <= ?)"
            ),
            (cutoff, reminder_cutoff),
        ).fetchall()
        reminder_entries = [_waitlist_row(r) for r in reminders]
        for entry in reminder_entries:
            conn.execute(
                "UPDATE doctor_waitlist SET last_reminded_at = ? WHERE id = ?",
                (now, entry["id"]),
            )
            entry["last_reminded_at"] = now
            effects.append({"kind": "offer_reminder", "entry": entry})
        conn.commit()

    # For each doctor whose offer just expired, advance the slot to the next
    # waiter. Promotion does not gate on availability here: an expired offer
    # means the slot was never taken. The real availability check happens when
    # the next patient clicks Accept (via the pick flow).
    promoted_doctors: set[int] = set()
    for entry in stale_entries:
        doctor_id = entry["doctor_id"]
        if doctor_id in promoted_doctors:
            continue
        promoted_doctors.add(doctor_id)
        nxt = promote_waitlist(doctor_id, now=now)
        if nxt is not None:
            effects.append({"kind": "offer", "entry": nxt})

    # Proactively promote the front waiter for any free doctor that has waiters
    # but no outstanding offer. This is how a queue drains when a doctor's
    # session ends (timeout, /end, decline) — the ticker is the single driver,
    # so every end path is covered without scattering promotion calls.
    with _SQLITE_LOCK:
        conn = get_sqlite()
        doctor_ids = [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT doctor_id FROM doctor_waitlist WHERE status = 'waiting'"
            ).fetchall()
        ]
    for doctor_id in doctor_ids:
        if doctor_id in promoted_doctors:
            continue
        doctor = get_doctor(doctor_id)
        if doctor is None or not doctor["available"]:
            continue
        promoted_doctors.add(doctor_id)
        nxt = promote_waitlist(doctor_id, now=now)
        if nxt is not None:
            effects.append({"kind": "offer", "entry": nxt})
    return effects
