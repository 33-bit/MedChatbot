"""
session.py
----------
Patient session state:
  - Redis: active session memory (TTL 24h)
  - SQLite: persistent patient profile, past consultations, rate limit log

PatientSession fields:
  symptoms       — list of {symptom_id, name, onset, severity, pattern, associated}
  medications    — list of drug_ids
  conversation   — list of {role, content} for query rewriting
  candidate_diseases — current diagnostic shortlist
  answered_questions — clarification keys already asked
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import redis

from src.config import (
    RATE_LIMIT_PER_MINUTE,
    REDIS_URL,
    SESSION_TTL_SECONDS,
    SQLITE_PATH,
)


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


# ----- Redis (session memory) -----

@lru_cache(maxsize=1)
def _redis() -> redis.Redis:
    if not REDIS_URL:
        raise RuntimeError("REDIS_URL not configured")
    return redis.from_url(REDIS_URL, decode_responses=True)


def _session_key(session_id: str) -> str:
    return f"session:{session_id}"


def load_session(session_id: str) -> PatientSession:
    try:
        raw = _redis().get(_session_key(session_id))
    except Exception:
        raw = None
    if raw:
        try:
            return PatientSession.from_json(raw)
        except Exception:
            pass
    return PatientSession(session_id=session_id)


def save_session(session: PatientSession) -> None:
    try:
        _redis().setex(
            _session_key(session.session_id),
            SESSION_TTL_SECONDS,
            session.to_json(),
        )
    except Exception:
        pass


def clear_session(session_id: str) -> None:
    try:
        _redis().delete(_session_key(session_id))
    except Exception:
        pass


# ----- SQLite (persistent profile + rate limit) -----

_SCHEMA = """
CREATE TABLE IF NOT EXISTS consultations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_consult_session ON consultations(session_id);

CREATE TABLE IF NOT EXISTS patient_profile (
    session_id TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS rate_limit (
    session_id TEXT NOT NULL,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rl_session_ts ON rate_limit(session_id, ts);
"""


@lru_cache(maxsize=1)
def _db() -> sqlite3.Connection:
    Path(SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def log_consultation(session_id: str, question: str, answer: str) -> None:
    conn = _db()
    conn.execute(
        "INSERT INTO consultations (session_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
        (session_id, question, answer, time.time()),
    )
    conn.commit()


def get_past_consultations(session_id: str, limit: int = 5) -> list[dict]:
    conn = _db()
    rows = conn.execute(
        "SELECT question, answer, created_at FROM consultations "
        "WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    return [{"question": q, "answer": a, "created_at": t} for q, a, t in rows]


def save_profile(session: PatientSession) -> None:
    conn = _db()
    conn.execute(
        "INSERT INTO patient_profile (session_id, profile_json, updated_at) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT(session_id) DO UPDATE SET profile_json=excluded.profile_json, "
        "updated_at=excluded.updated_at",
        (session.session_id, session.to_json(), time.time()),
    )
    conn.commit()


# ----- Rate limit -----

def check_rate_limit(session_id: str) -> bool:
    """Return True if allowed, False if over limit. Sliding window 60s."""
    conn = _db()
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
