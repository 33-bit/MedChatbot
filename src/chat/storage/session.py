"""
session.py
----------
Patient session state.

PatientSession lives in Redis (TTL 24h).
patient_profile, consultations, and rate_limit rows live in SQLite.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field

import redis

from src.chat.clients import get_redis, get_sqlite
from src.config import RATE_LIMIT_PER_MINUTE, SESSION_TTL_SECONDS

log = logging.getLogger(__name__)


@dataclass
class PatientSession:
    session_id: str
    symptoms: list[dict] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    conversation: list[dict] = field(default_factory=list)
    candidate_diseases: list[dict] = field(default_factory=list)
    answered_questions: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> "PatientSession":
        return cls(**json.loads(data))

    def add_message(self, role: str, content: str, max_history: int = 20) -> None:
        self.conversation.append({"role": role, "content": content})
        if len(self.conversation) > max_history:
            self.conversation = self.conversation[-max_history:]

    def upsert_symptom(self, entry: dict) -> None:
        sid = entry.get("symptom_id")
        if not sid:
            return
        for i, s in enumerate(self.symptoms):
            if s.get("symptom_id") == sid:
                self.symptoms[i] = {**s, **{k: v for k, v in entry.items() if v}}
                return
        self.symptoms.append(entry)

    def add_medication(self, drug_id: str) -> None:
        if drug_id and drug_id not in self.medications:
            self.medications.append(drug_id)


def _session_key(session_id: str) -> str:
    return f"session:{session_id}"


def load_session(session_id: str) -> PatientSession:
    try:
        raw = get_redis().get(_session_key(session_id))
    except redis.RedisError as e:
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
    except redis.RedisError as e:
        log.warning("Redis save failed for %s: %s", session.session_id, e)


def clear_session(session_id: str) -> None:
    try:
        get_redis().delete(_session_key(session_id))
    except redis.RedisError as e:
        log.warning("Redis delete failed for %s: %s", session_id, e)


def log_consultation(session_id: str, question: str, answer: str) -> None:
    conn = get_sqlite()
    conn.execute(
        "INSERT INTO consultations (session_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
        (session_id, question, answer, time.time()),
    )
    conn.commit()


def get_past_consultations(session_id: str, limit: int = 5) -> list[dict]:
    conn = get_sqlite()
    rows = conn.execute(
        "SELECT question, answer, created_at FROM consultations "
        "WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    return [{"question": q, "answer": a, "created_at": t} for q, a, t in rows]


def save_profile(session: PatientSession) -> None:
    conn = get_sqlite()
    conn.execute(
        "INSERT INTO patient_profile (session_id, profile_json, updated_at) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT(session_id) DO UPDATE SET profile_json=excluded.profile_json, "
        "updated_at=excluded.updated_at",
        (session.session_id, session.to_json(), time.time()),
    )
    conn.commit()


def check_rate_limit(session_id: str) -> bool:
    """Return True if allowed, False if over limit. Sliding window 60s."""
    conn = get_sqlite()
    now = time.time()
    window_start = now - 60.0
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
