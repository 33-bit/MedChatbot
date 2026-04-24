"""
clients.py
----------
Single public factory module for external service clients.
Every module that needs Redis / Neo4j / SQLite / xAI imports from here.

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
    make_xai_client,
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
    conn.commit()
    return conn


@lru_cache(maxsize=1)
def get_xai():
    return make_xai_client()
