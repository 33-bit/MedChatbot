"""
clients.py
----------
Single public factory module for external service clients.
Every module that needs Redis / Neo4j / SQLite / OpenAI imports from here.

All factories are lazy singletons (@lru_cache).
"""

from __future__ import annotations

import sqlite3
import os
import time
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

CREATE TABLE IF NOT EXISTS medical_profile_subject (
    subject_id TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    relationship TEXT NOT NULL,
    display_name TEXT,
    birth_date TEXT,
    gender TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (owner_id, subject_id)
);
CREATE INDEX IF NOT EXISTS idx_medical_profile_subject_updated
    ON medical_profile_subject(owner_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS medical_profile_fact (
    profile_fact_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    section TEXT NOT NULL,
    fact_type TEXT NOT NULL,
    entity_type TEXT,
    entity_id TEXT,
    attribute TEXT NOT NULL,
    value_json TEXT NOT NULL,
    temporal_status TEXT NOT NULL,
    confidence REAL NOT NULL,
    verification_status TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    reporter_role TEXT,
    valid_from REAL,
    valid_until REAL,
    superseded_by TEXT,
    inactive INTEGER NOT NULL DEFAULT 0,
    coding_system TEXT,
    coding_code TEXT,
    coding_display TEXT,
    source_turn_id TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    confirmed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_medical_profile_subject
    ON medical_profile_fact(owner_id, subject_id);
CREATE INDEX IF NOT EXISTS idx_medical_profile_section
    ON medical_profile_fact(owner_id, subject_id, section);
CREATE INDEX IF NOT EXISTS idx_medical_profile_active
    ON medical_profile_fact(owner_id, subject_id, superseded_by, inactive);
CREATE INDEX IF NOT EXISTS idx_medical_profile_expiry
    ON medical_profile_fact(valid_until);

CREATE TABLE IF NOT EXISTS medical_profile_section_state (
    owner_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    section TEXT NOT NULL,
    status TEXT NOT NULL,
    reviewed_at REAL,
    PRIMARY KEY (owner_id, subject_id, section)
);

CREATE TABLE IF NOT EXISTS medical_profile_preference (
    owner_id TEXT NOT NULL,
    preference TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 0,
    consented_at REAL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (owner_id, preference)
);

CREATE TABLE IF NOT EXISTS app_schema_migration (
    id TEXT PRIMARY KEY,
    applied_at REAL NOT NULL
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


def _migrate_medical_profile_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(medical_profile_fact)")}
    if "confirmed_at" not in existing:
        conn.execute("ALTER TABLE medical_profile_fact ADD COLUMN confirmed_at REAL")


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
    try:
        os.chmod(SQLITE_PATH, 0o600)
    except OSError:
        pass
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

    # Medical-profile fact column additions for databases created before
    # the IPS-inspired schema was finalised.
    _migrate_medical_profile_columns(conn)

    # --- One-shot cutover: drop legacy medical / patient tables ---
    # The IPS-inspired medical profile subsystem replaces the old medical
    # memory store. Legacy data is intentionally not migrated.
    if not _migration_applied(conn, "medical_profile_cutover_v1"):
        _purge_legacy_medical_tables(conn)
        conn.execute(
            "INSERT INTO app_schema_migration (id, applied_at) "
            "VALUES (?, ?)",
            ("medical_profile_cutover_v1", time.time()),
        )


def _migration_applied(conn: sqlite3.Connection, migration_id: str) -> bool:
    try:
        row = conn.execute(
            "SELECT 1 FROM app_schema_migration WHERE id = ?", (migration_id,),
        ).fetchone()
        return bool(row)
    except sqlite3.OperationalError:
        # app_schema_migration does not exist yet on truly fresh DBs; the
        # CREATE TABLE above will run before the migration is recorded.
        return False


def _purge_legacy_medical_tables(conn: sqlite3.Connection) -> None:
    """Drop the legacy medical / patient tables and their indexes.

    Idempotent: missing tables / indexes are silently skipped.
    """
    legacy_tables = (
        "memory_fact", "memory_subject", "memory_preference",
        "memory_user_preference", "profile_section_state", "patient_profile",
    )
    for table in legacy_tables:
        try:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        except sqlite3.Error:
            pass


@lru_cache(maxsize=1)
def get_openai():
    return make_openai_client()
