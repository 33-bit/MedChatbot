from __future__ import annotations

import time
from threading import RLock

from src.chat.clients import get_sqlite
from src.chat.storage.domain import PatientSession
from src.chat.security.identity import is_session_key

_SQLITE_LOCK = RLock()


def log_consultation(session_id: str, question: str, answer: str) -> None:
    if not is_session_key(session_id):
        return
    with _SQLITE_LOCK:
        conn = get_sqlite()
        conn.execute(
            "INSERT INTO consultations "
            "(session_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
            (session_id, question, answer, time.time()),
        )
        conn.commit()


def get_past_consultations(session_id: str, limit: int = 5) -> list[dict]:
    if not is_session_key(session_id):
        return []
    with _SQLITE_LOCK:
        conn = get_sqlite()
        rows = conn.execute(
            "SELECT question, answer, created_at FROM consultations "
            "WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [{"question": q, "answer": a, "created_at": t} for q, a, t in rows]


def save_profile(session: PatientSession) -> None:
    """Legacy whole-session profile writes are permanently disabled.

    Existing rows remain readable only for controlled retention/deletion. They
    lack fact-level subject provenance and must never be migrated automatically.
    """
    del session
