from __future__ import annotations

import time
from threading import RLock

from src.chat.clients import get_sqlite
from src.config import RATE_LIMIT_PER_MINUTE

_SQLITE_LOCK = RLock()


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
