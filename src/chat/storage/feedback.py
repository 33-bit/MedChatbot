from __future__ import annotations

import secrets
import time
from threading import RLock

from src.chat.clients import get_sqlite
from src.chat.security.identity import is_session_key

_SQLITE_LOCK = RLock()


def create_feedback_request(
    session_id: str,
    channel: str,
    recipient_id: str,
    question: str,
    answer: str,
) -> str:
    if not is_session_key(session_id) or recipient_id != session_id:
        raise ValueError("Feedback persistence requires a pseudonymous session key")
    token = secrets.token_urlsafe(16)
    with _SQLITE_LOCK:
        conn = get_sqlite()
        conn.execute(
            "INSERT INTO response_feedback "
            "(token, session_id, channel, recipient_id, question, answer, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (token, session_id, channel, recipient_id, question, answer, time.time()),
        )
        conn.commit()
    return token


def record_feedback_rating(token: str, rating: int) -> bool:
    if rating < 1 or rating > 5:
        return False
    with _SQLITE_LOCK:
        conn = get_sqlite()
        cursor = conn.execute(
            "UPDATE response_feedback SET rating = ?, rated_at = ? "
            "WHERE token = ? AND rating IS NULL",
            (rating, time.time(), token),
        )
        conn.commit()
    return cursor.rowcount > 0


def get_feedback(token: str) -> dict | None:
    with _SQLITE_LOCK:
        conn = get_sqlite()
        row = conn.execute(
            "SELECT token, session_id, channel, recipient_id, question, answer, rating "
            "FROM response_feedback WHERE token = ?",
            (token,),
        ).fetchone()
    if row is None:
        return None
    token, session_id, channel, recipient_id, question, answer, rating = row
    return {
        "token": token,
        "session_id": session_id,
        "channel": channel,
        "recipient_id": recipient_id,
        "question": question,
        "answer": answer,
        "rating": rating,
    }
