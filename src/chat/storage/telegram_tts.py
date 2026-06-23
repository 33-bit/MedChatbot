from __future__ import annotations

import time

from src.chat.clients import get_sqlite


def _ensure_table() -> None:
    conn = get_sqlite()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS telegram_tts_preference (
            chat_id TEXT PRIMARY KEY,
            enabled INTEGER NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.commit()


def is_tts_enabled(chat_id: str | int) -> bool:
    _ensure_table()
    row = get_sqlite().execute(
        "SELECT enabled FROM telegram_tts_preference WHERE chat_id = ?",
        (str(chat_id),),
    ).fetchone()
    return bool(row and row[0])


def set_tts_enabled(chat_id: str | int, enabled: bool) -> None:
    _ensure_table()
    conn = get_sqlite()
    conn.execute(
        """
        INSERT INTO telegram_tts_preference (chat_id, enabled, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(chat_id) DO UPDATE SET
            enabled = excluded.enabled,
            updated_at = excluded.updated_at
        """,
        (str(chat_id), int(enabled), time.time()),
    )
    conn.commit()
