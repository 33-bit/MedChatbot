"""
quota.py
--------
LLM usage quotas tracked in Redis:
  - Global: max calls per minute (hard cap protecting the wallet)
  - Per-session: max calls per day (abuse per account)

Call `check_and_consume_*` BEFORE making an LLM call.
On over-quota, return False and let caller short-circuit.
"""

from __future__ import annotations

import logging
import time
from datetime import date

import redis

from src.chat.clients import get_redis
from src.config import GLOBAL_LLM_QUOTA_PER_MINUTE, SESSION_LLM_QUOTA_PER_DAY

log = logging.getLogger(__name__)


def _check_and_incr(key: str, limit: int, ttl: int) -> bool:
    """Atomic increment with TTL on first write. Returns True if allowed.
    Fails open (allow) when Redis is unreachable so we don't block users."""
    try:
        r = get_redis()
        count = r.incr(key)
        if count == 1:
            r.expire(key, ttl)
        return count <= limit
    except redis.RedisError as e:
        log.warning("Quota check skipped (Redis error): %s", e)
        return True
    except RuntimeError as e:
        log.warning("Quota check skipped (Redis not configured): %s", e)
        return True


def check_global_quota() -> bool:
    minute = int(time.time() // 60)
    return _check_and_incr(
        f"llm_global:{minute}", GLOBAL_LLM_QUOTA_PER_MINUTE, ttl=120,
    )


def check_session_quota(session_id: str) -> bool:
    day = date.today().isoformat()
    return _check_and_incr(
        f"llm_session:{session_id}:{day}", SESSION_LLM_QUOTA_PER_DAY, ttl=86400,
    )


def check_both(session_id: str) -> tuple[bool, str]:
    """Returns (allowed, reason). reason is non-empty only when blocked."""
    if not check_session_quota(session_id):
        return False, "Bạn đã đạt giới hạn số câu hỏi trong ngày. Quay lại vào ngày mai nhé."
    if not check_global_quota():
        return False, "Hệ thống đang quá tải. Vui lòng thử lại sau 1 phút."
    return True, ""
