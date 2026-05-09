from __future__ import annotations

import time
from threading import RLock

from src.chat.clients import get_sqlite
from src.chat.storage.domain import PatientSession

_SQLITE_LOCK = RLock()


def log_consultation(session_id: str, question: str, answer: str) -> None:
    with _SQLITE_LOCK:
        conn = get_sqlite()
        conn.execute(
            "INSERT INTO consultations "
            "(session_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
            (session_id, question, answer, time.time()),
        )
        conn.commit()


def get_past_consultations(session_id: str, limit: int = 5) -> list[dict]:
    with _SQLITE_LOCK:
        conn = get_sqlite()
        rows = conn.execute(
            "SELECT question, answer, created_at FROM consultations "
            "WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [{"question": q, "answer": a, "created_at": t} for q, a, t in rows]


def save_profile(session: PatientSession) -> None:
    with _SQLITE_LOCK:
        conn = get_sqlite()
        conn.execute(
            "INSERT INTO patient_profile (session_id, profile_json, updated_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET profile_json=excluded.profile_json, "
            "updated_at=excluded.updated_at",
            (session.session_id, session.to_json(), time.time()),
        )
        conn.commit()
