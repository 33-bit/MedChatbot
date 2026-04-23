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

import time
from datetime import date

from src.chat.session import _redis
from src.config import GLOBAL_LLM_QUOTA_PER_MINUTE, SESSION_LLM_QUOTA_PER_DAY


def _check_and_incr(key: str, limit: int, ttl: int) -> bool:
    """Atomic increment with TTL on first write. Returns True if allowed."""
    try:
        r = _redis()
    except Exception:
        return True  # Redis down → fail open (don't block users)
    try:
        count = r.incr(key)
        if count == 1:
            r.expire(key, ttl)
        return count <= limit
    except Exception:
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
