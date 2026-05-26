"""
session.py
----------
Compatibility facade for patient session state and persistence helpers.
"""

from __future__ import annotations

from src.chat.clients import get_redis, get_sqlite
from src.chat.storage.domain import PatientSession
from src.chat.storage.rate_limit import check_rate_limit
from src.chat.storage.redis_session import clear_session, load_session, save_session
from src.chat.storage.sqlite_profile import (
    get_past_consultations,
    log_consultation,
    save_profile,
)
from src.chat.storage.webhook_dedupe import reserve_webhook_update
from src.config import RATE_LIMIT_PER_MINUTE

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
