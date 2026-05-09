from __future__ import annotations

import logging
import sqlite3
import time
from threading import RLock

from src.chat.clients import get_sqlite

log = logging.getLogger(__name__)
WEBHOOK_UPDATE_TTL_SECONDS = 7 * 24 * 60 * 60
_SQLITE_LOCK = RLock()


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
