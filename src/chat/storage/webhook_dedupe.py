from __future__ import annotations

import logging
import sqlite3
import time
from collections import OrderedDict
from threading import RLock

from src.chat.clients import get_sqlite

log = logging.getLogger(__name__)
WEBHOOK_UPDATE_TTL_SECONDS = 7 * 24 * 60 * 60
WEBHOOK_UPDATE_MEMORY_MAX = 4096
_SQLITE_LOCK = RLock()
_MEMORY_UPDATES: OrderedDict[tuple[str, str], float] = OrderedDict()


def _reserve_memory(channel: str, update_id: str | int, now: float) -> bool:
    cutoff = now - WEBHOOK_UPDATE_TTL_SECONDS
    for key, created_at in list(_MEMORY_UPDATES.items()):
        if created_at < cutoff:
            _MEMORY_UPDATES.pop(key, None)

    key = (channel, str(update_id))
    if key in _MEMORY_UPDATES:
        _MEMORY_UPDATES.move_to_end(key)
        return False

    _MEMORY_UPDATES[key] = now
    while len(_MEMORY_UPDATES) > WEBHOOK_UPDATE_MEMORY_MAX:
        _MEMORY_UPDATES.popitem(last=False)
    return True


def reserve_webhook_update(channel: str, update_id: str | int) -> bool:
    """Return True only for the first delivery of a webhook update."""
    now = time.time()
    with _SQLITE_LOCK:
        if not _reserve_memory(channel, update_id, now):
            return False

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
