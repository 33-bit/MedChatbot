"""
session.py
----------
Compatibility facade for patient session state and persistence helpers.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from threading import RLock

import redis

from src.chat.clients import get_redis, get_sqlite
from src.chat.storage.domain import PatientSession
from src.config import RATE_LIMIT_PER_MINUTE, SESSION_TTL_SECONDS

log = logging.getLogger(__name__)
WEBHOOK_UPDATE_TTL_SECONDS = 7 * 24 * 60 * 60
_SQLITE_LOCK = RLock()


def _session_key(session_id: str) -> str:
    return f"session:{session_id}"


def load_session(session_id: str) -> PatientSession:
    try:
        raw = get_redis().get(_session_key(session_id))
    except (RuntimeError, redis.RedisError) as e:
        log.warning("Redis load failed for %s: %s", session_id, e)
        return PatientSession(session_id=session_id)
    if raw:
        try:
            return PatientSession.from_json(raw)
        except (json.JSONDecodeError, TypeError) as e:
            log.warning("Session JSON parse failed for %s: %s", session_id, e)
    return PatientSession(session_id=session_id)


def save_session(session: PatientSession) -> None:
    try:
        get_redis().setex(
            _session_key(session.session_id),
            SESSION_TTL_SECONDS,
            session.to_json(),
        )
    except (RuntimeError, redis.RedisError) as e:
        log.warning("Redis save failed for %s: %s", session.session_id, e)


def clear_session(session_id: str) -> None:
    try:
        get_redis().delete(_session_key(session_id))
    except (RuntimeError, redis.RedisError) as e:
        log.warning("Redis delete failed for %s: %s", session_id, e)


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


def reserve_webhook_update(channel: str, update_id: str | int) -> bool:
    """Return True only for the first delivery of a webhook update."""
    now = time.time()
    with _SQLITE_LOCK:
        conn = get_sqlite()
        try:
            conn.execute(
                "DELETE FROM webhook_update WHERE created_at < ?",
                (now - WEBHOOK_UPDATE_TTL_SECONDS,),
            )
            cur = conn.execute(
                "INSERT OR IGNORE INTO webhook_update (channel, update_id, created_at) "
                "VALUES (?, ?, ?)",
                (channel, str(update_id), now),
            )
            conn.commit()
            return cur.rowcount == 1
        except sqlite3.Error as e:
            conn.rollback()
            log.warning(
                "Webhook update reservation failed for %s:%s: %s",
                channel,
                update_id,
                e,
            )
            return True


def check_rate_limit(session_id: str) -> bool:
    """Return True if allowed, False if over limit. Sliding window 60s."""
    now = time.time()
    window_start = now - 60.0
    with _SQLITE_LOCK:
        conn = get_sqlite()
        conn.execute("DELETE FROM rate_limit WHERE ts < ?", (window_start,))
        count = conn.execute(
            "SELECT COUNT(*) FROM rate_limit WHERE session_id = ? AND ts >= ?",
            (session_id, window_start),
        ).fetchone()[0]
        if count >= RATE_LIMIT_PER_MINUTE:
            conn.commit()
            return False
        conn.execute(
            "INSERT INTO rate_limit (session_id, ts) VALUES (?, ?)",
            (session_id, now),
        )
        conn.commit()
    return True


__all__ = [
    "PatientSession",
    "RATE_LIMIT_PER_MINUTE",
    "check_rate_limit",
    "clear_session",
    "get_past_consultations",
    "get_redis",
    "get_sqlite",
    "load_session",
    "log_consultation",
    "reserve_webhook_update",
    "save_profile",
    "save_session",
]
