"""
clients.py
----------
Single public factory module for external service clients.
Every module that needs Redis / Neo4j / SQLite / OpenAI imports from here.

All factories are lazy singletons (@lru_cache).
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

import redis
from neo4j import GraphDatabase

from src.config import (
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    REDIS_URL,
    SQLITE_PATH,
    make_openai_client,
)

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS consultations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_consult_session ON consultations(session_id);

CREATE TABLE IF NOT EXISTS chat_trace (
    trace_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    internal_session_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    meta_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_trace_session_created
    ON chat_trace(session_id, created_at);

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

CREATE TABLE IF NOT EXISTS webhook_update (
    channel TEXT NOT NULL,
    update_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (channel, update_id)
);

CREATE TABLE IF NOT EXISTS response_feedback (
    token TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    recipient_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    rating INTEGER,
    created_at REAL NOT NULL,
    rated_at REAL
);
CREATE INDEX IF NOT EXISTS idx_feedback_session ON response_feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_channel_recipient ON response_feedback(channel, recipient_id);

CREATE TABLE IF NOT EXISTS account_balance (
    account_id TEXT PRIMARY KEY,
    balance INTEGER NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS topup_order (
    order_code INTEGER PRIMARY KEY,
    account_id TEXT NOT NULL,
    amount INTEGER NOT NULL,
    status TEXT NOT NULL,
    payment_link_id TEXT,
    created_at REAL NOT NULL,
    paid_at REAL,
    qr_message_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_topup_account ON topup_order(account_id);

CREATE TABLE IF NOT EXISTS admin_credit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_code INTEGER NOT NULL,
    admin_user_id INTEGER NOT NULL,
    account_id TEXT NOT NULL,
    amount INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_admin_credit_order ON admin_credit(order_code);

CREATE TABLE IF NOT EXISTS doctor (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    specialty TEXT,
    tier TEXT NOT NULL,
    price INTEGER NOT NULL DEFAULT 0,
    telegram_user_id INTEGER NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    degree TEXT,
    experience_years INTEGER,
    hospital TEXT,
    bio TEXT
);
CREATE INDEX IF NOT EXISTS idx_doctor_tier_active ON doctor(tier, active);
CREATE UNIQUE INDEX IF NOT EXISTS idx_doctor_telegram_user ON doctor(telegram_user_id);

CREATE TABLE IF NOT EXISTS doctor_consultation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_chat_id INTEGER NOT NULL,
    doctor_id INTEGER NOT NULL,
    doctor_chat_id INTEGER NOT NULL,
    tier TEXT NOT NULL,
    fee INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    accepted_at REAL,
    ended_at REAL,
    last_activity_at REAL
);
CREATE INDEX IF NOT EXISTS idx_dconsult_patient ON doctor_consultation(patient_chat_id, status);
CREATE INDEX IF NOT EXISTS idx_dconsult_doctor ON doctor_consultation(doctor_chat_id, status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_dconsult_one_open_patient
    ON doctor_consultation(patient_chat_id)
    WHERE status IN ('pending', 'active');
CREATE UNIQUE INDEX IF NOT EXISTS idx_dconsult_one_active_doctor
    ON doctor_consultation(doctor_id)
    WHERE status = 'active';
CREATE UNIQUE INDEX IF NOT EXISTS idx_dconsult_one_active_doctor_chat
    ON doctor_consultation(doctor_chat_id)
    WHERE status = 'active';

CREATE TABLE IF NOT EXISTS doctor_waitlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doctor_id INTEGER NOT NULL,
    patient_chat_id INTEGER NOT NULL,
    tier TEXT NOT NULL,
    created_at REAL NOT NULL,
    notified_at REAL,
    last_reminded_at REAL,
    status TEXT NOT NULL DEFAULT 'waiting'
);
CREATE INDEX IF NOT EXISTS idx_waitlist_doctor ON doctor_waitlist(doctor_id, status, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_waitlist_one_per_patient_doctor
    ON doctor_waitlist(doctor_id, patient_chat_id)
    WHERE status IN ('waiting', 'offered');
"""


@lru_cache(maxsize=1)
def get_redis() -> redis.Redis:
    if not REDIS_URL:
        raise RuntimeError("REDIS_URL not configured")
    return redis.from_url(REDIS_URL, decode_responses=True)


@lru_cache(maxsize=1)
def get_neo4j():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


@lru_cache(maxsize=1)
def get_sqlite() -> sqlite3.Connection:
    Path(SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    conn.executescript(_SQLITE_SCHEMA)
    _migrate_sqlite(conn)
    conn.commit()
    return conn


def _migrate_sqlite(conn: sqlite3.Connection) -> None:
    """Idempotent column additions for tables that predate a column.

    CREATE TABLE IF NOT EXISTS won't add columns to an existing table, so add
    any missing ones here. Safe to run on every connection.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(topup_order)")}
    if "qr_message_id" not in cols:
        conn.execute("ALTER TABLE topup_order ADD COLUMN qr_message_id INTEGER")

    dconsult_cols = {row[1] for row in conn.execute("PRAGMA table_info(doctor_consultation)")}
    for col, ddl in (
        ("expires_at", "ALTER TABLE doctor_consultation ADD COLUMN expires_at REAL"),
        ("warned_at", "ALTER TABLE doctor_consultation ADD COLUMN warned_at REAL"),
        ("end_reason", "ALTER TABLE doctor_consultation ADD COLUMN end_reason TEXT"),
        ("block_index", "ALTER TABLE doctor_consultation ADD COLUMN block_index INTEGER NOT NULL DEFAULT 0"),
        ("rate_per_min", "ALTER TABLE doctor_consultation ADD COLUMN rate_per_min INTEGER NOT NULL DEFAULT 0"),
        ("block_started_at", "ALTER TABLE doctor_consultation ADD COLUMN block_started_at REAL"),
        ("minutes_billed_block", "ALTER TABLE doctor_consultation ADD COLUMN minutes_billed_block INTEGER NOT NULL DEFAULT 0"),
        ("extend_requested", "ALTER TABLE doctor_consultation ADD COLUMN extend_requested INTEGER NOT NULL DEFAULT 0"),
    ):
        if col not in dconsult_cols:
            conn.execute(ddl)

    balance_cols = {row[1] for row in conn.execute("PRAGMA table_info(account_balance)")}
    if "debt_since" not in balance_cols:
        conn.execute("ALTER TABLE account_balance ADD COLUMN debt_since REAL")

    waitlist_cols = {row[1] for row in conn.execute("PRAGMA table_info(doctor_waitlist)")}
    if "last_reminded_at" not in waitlist_cols:
        conn.execute("ALTER TABLE doctor_waitlist ADD COLUMN last_reminded_at REAL")

    doctor_cols = {row[1] for row in conn.execute("PRAGMA table_info(doctor)")}
    for col, ddl in (
        ("degree", "ALTER TABLE doctor ADD COLUMN degree TEXT"),
        ("experience_years", "ALTER TABLE doctor ADD COLUMN experience_years INTEGER"),
        ("hospital", "ALTER TABLE doctor ADD COLUMN hospital TEXT"),
        ("bio", "ALTER TABLE doctor ADD COLUMN bio TEXT"),
    ):
        if col not in doctor_cols:
            conn.execute(ddl)


@lru_cache(maxsize=1)
def get_openai():
    return make_openai_client()
