import json
import time
from threading import RLock
from src.chat.clients import get_sqlite

_REMINDERS_LOCK = RLock()

def init_reminders_db() -> None:
    with _REMINDERS_LOCK:
        conn = get_sqlite()
        conn.execute("""
        CREATE TABLE IF NOT EXISTS telegram_reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            medical_type TEXT NOT NULL,
            reminder_text TEXT NOT NULL,
            schedule_json TEXT NOT NULL,
            next_fire_at INTEGER NOT NULL,
            end_date TEXT,
            source TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            leased_until INTEGER NOT NULL DEFAULT 0
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reminders_next_fire_at ON telegram_reminders(next_fire_at) WHERE status = 'active';")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reminders_chat_user ON telegram_reminders(chat_id, user_id);")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS telegram_reminder_drafts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            medical_type TEXT NOT NULL,
            reminder_text TEXT NOT NULL,
            schedule_json TEXT NOT NULL,
            next_fire_at INTEGER NOT NULL,
            end_date TEXT,
            source TEXT NOT NULL,
            expires_at INTEGER NOT NULL,
            created_at INTEGER NOT NULL
        );
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS telegram_reminder_conversations (
            chat_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            original_request TEXT NOT NULL,
            partial_fields_json TEXT NOT NULL,
            turns_json TEXT NOT NULL,
            missing_fields_json TEXT NOT NULL,
            expires_at INTEGER NOT NULL,
            PRIMARY KEY (chat_id, user_id)
        );
        """)
        conn.commit()

def cleanup_expired_drafts() -> None:
    now = int(time.time())
    with _REMINDERS_LOCK:
        conn = get_sqlite()
        conn.execute("DELETE FROM telegram_reminder_drafts WHERE expires_at < ?", (now,))
        conn.execute("DELETE FROM telegram_reminder_conversations WHERE expires_at < ?", (now,))
        conn.commit()


def create_reminder_draft(
    chat_id: int,
    user_id: int,
    medical_type: str,
    reminder_text: str,
    schedule: dict,
    next_fire_at: int,
    end_date: str | None,
    source: str
) -> int:
    cleanup_expired_drafts()
    now = int(time.time())
    expires_at = now + 900  # 15 minutes
    schedule_json = json.dumps(schedule)
    with _REMINDERS_LOCK:
        conn = get_sqlite()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO telegram_reminder_drafts "
            "(chat_id, user_id, medical_type, reminder_text, schedule_json, next_fire_at, end_date, source, expires_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (chat_id, user_id, medical_type, reminder_text, schedule_json, next_fire_at, end_date, source, expires_at, now)
        )
        conn.commit()
        return cursor.lastrowid

def get_reminder_draft(draft_id: int) -> dict | None:
    cleanup_expired_drafts()
    with _REMINDERS_LOCK:
        conn = get_sqlite()
        row = conn.execute(
            "SELECT id, chat_id, user_id, medical_type, reminder_text, schedule_json, next_fire_at, end_date, source "
            "FROM telegram_reminder_drafts WHERE id = ?",
            (draft_id,)
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "chat_id": row[1],
            "user_id": row[2],
            "medical_type": row[3],
            "reminder_text": row[4],
            "schedule": json.loads(row[5]),
            "next_fire_at": row[6],
            "end_date": row[7],
            "source": row[8],
        }

def delete_reminder_draft(draft_id: int) -> None:
    with _REMINDERS_LOCK:
        conn = get_sqlite()
        conn.execute("DELETE FROM telegram_reminder_drafts WHERE id = ?", (draft_id,))
        conn.commit()

def count_active_reminders(chat_id: int, user_id: int) -> int:
    with _REMINDERS_LOCK:
        conn = get_sqlite()
        res = conn.execute(
            "SELECT COUNT(*) FROM telegram_reminders WHERE chat_id = ? AND user_id = ? AND status = 'active'",
            (chat_id, user_id)
        ).fetchone()
        return res[0] if res else 0

def check_duplicate_active_or_pending(
    chat_id: int,
    user_id: int,
    medical_type: str,
    reminder_text: str,
    schedule_json: str,
    exclude_draft_id: int | None = None
) -> bool:
    cleanup_expired_drafts()
    with _REMINDERS_LOCK:
        conn = get_sqlite()
        active_dup = conn.execute(
            "SELECT 1 FROM telegram_reminders "
            "WHERE chat_id = ? AND user_id = ? AND medical_type = ? AND reminder_text = ? AND schedule_json = ? AND status = 'active'",
            (chat_id, user_id, medical_type, reminder_text, schedule_json)
        ).fetchone()
        if active_dup:
            return True
        
        query = (
            "SELECT 1 FROM telegram_reminder_drafts "
            "WHERE chat_id = ? AND user_id = ? AND medical_type = ? AND reminder_text = ? AND schedule_json = ?"
        )
        params = [chat_id, user_id, medical_type, reminder_text, schedule_json]
        if exclude_draft_id is not None:
            query += " AND id != ?"
            params.append(exclude_draft_id)
            
        pending_dup = conn.execute(query, params).fetchone()
        return pending_dup is not None

def confirm_reminder_draft(chat_id: int, user_id: int, draft_id: int) -> dict | None:
    with _REMINDERS_LOCK:
        draft = get_reminder_draft(draft_id)
        if not draft:
            return None
        
        if count_active_reminders(chat_id, user_id) >= 20:
            return None
            
        schedule_str = json.dumps(draft["schedule"])
        if check_duplicate_active_or_pending(chat_id, user_id, draft["medical_type"], draft["reminder_text"], schedule_str, exclude_draft_id=draft_id):
            delete_reminder_draft(draft_id)
            conn = get_sqlite()
            row = conn.execute(
                "SELECT id, chat_id, user_id, medical_type, reminder_text, schedule_json, next_fire_at, end_date, source, status "
                "FROM telegram_reminders "
                "WHERE chat_id = ? AND user_id = ? AND medical_type = ? AND reminder_text = ? AND schedule_json = ? AND status = 'active'",
                (chat_id, user_id, draft["medical_type"], draft["reminder_text"], schedule_str)
            ).fetchone()
            if row:
                return {
                    "id": row[0],
                    "chat_id": row[1],
                    "user_id": row[2],
                    "medical_type": row[3],
                    "reminder_text": row[4],
                    "schedule": json.loads(row[5]),
                    "next_fire_at": row[6],
                    "end_date": row[7],
                    "source": row[8],
                    "status": row[9]
                }
            return None

        conn = get_sqlite()
        cursor = conn.cursor()
        now = int(time.time())
        cursor.execute(
            "INSERT INTO telegram_reminders "
            "(chat_id, user_id, medical_type, reminder_text, schedule_json, next_fire_at, end_date, source, status, created_at, leased_until) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, 0)",
            (chat_id, user_id, draft["medical_type"], draft["reminder_text"], schedule_str, draft["next_fire_at"], draft["end_date"], draft["source"], now)
        )
        reminder_id = cursor.lastrowid
        delete_reminder_draft(draft_id)
        conn.commit()
        return {
            "id": reminder_id,
            "chat_id": chat_id,
            "user_id": user_id,
            "medical_type": draft["medical_type"],
            "reminder_text": draft["reminder_text"],
            "schedule": draft["schedule"],
            "next_fire_at": draft["next_fire_at"],
            "end_date": draft["end_date"],
            "source": draft["source"],
            "status": "active"
        }

def list_active_reminders(chat_id: int, user_id: int) -> list[dict]:
    with _REMINDERS_LOCK:
        conn = get_sqlite()
        rows = conn.execute(
            "SELECT id, chat_id, user_id, medical_type, reminder_text, schedule_json, next_fire_at, end_date, source, status "
            "FROM telegram_reminders "
            "WHERE chat_id = ? AND user_id = ? AND status = 'active' "
            "ORDER BY created_at ASC",
            (chat_id, user_id)
        ).fetchall()
        return [{
            "id": r[0],
            "chat_id": r[1],
            "user_id": r[2],
            "medical_type": r[3],
            "reminder_text": r[4],
            "schedule": json.loads(r[5]),
            "next_fire_at": r[6],
            "end_date": r[7],
            "source": r[8],
            "status": r[9],
        } for r in rows]

def delete_reminder(chat_id: int, user_id: int, reminder_id: int) -> bool:
    with _REMINDERS_LOCK:
        conn = get_sqlite()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM telegram_reminders WHERE id = ? AND chat_id = ? AND user_id = ?",
            (reminder_id, chat_id, user_id)
        )
        conn.commit()
        return cursor.rowcount > 0

def claim_due_reminders(now: int, lease_duration: int = 300) -> list[dict]:
    with _REMINDERS_LOCK:
        conn = get_sqlite()
        rows = conn.execute(
            "SELECT id, chat_id, user_id, medical_type, reminder_text, schedule_json, next_fire_at, end_date, source, status "
            "FROM telegram_reminders "
            "WHERE status = 'active' AND next_fire_at <= ? AND leased_until < ?",
            (now, now)
        ).fetchall()
        
        if not rows:
            return []
            
        ids = [r[0] for r in rows]
        conn.execute(
            f"UPDATE telegram_reminders SET leased_until = ? WHERE id IN ({','.join('?' for _ in ids)})",
            [now + lease_duration] + ids
        )
        conn.commit()
        
        return [{
            "id": r[0],
            "chat_id": r[1],
            "user_id": r[2],
            "medical_type": r[3],
            "reminder_text": r[4],
            "schedule_json": r[5],
            "next_fire_at": r[6],
            "end_date": r[7],
            "source": r[8],
            "status": r[9],
        } for r in rows]

def complete_delivery(reminder_id: int, next_fire_at: int | None) -> None:
    with _REMINDERS_LOCK:
        conn = get_sqlite()
        if next_fire_at is None:
            conn.execute("DELETE FROM telegram_reminders WHERE id = ?", (reminder_id,))
        else:
            conn.execute(
                "UPDATE telegram_reminders SET next_fire_at = ?, leased_until = 0 WHERE id = ?",
                (next_fire_at, reminder_id)
            )
        conn.commit()

def release_claim(reminder_id: int) -> None:
    with _REMINDERS_LOCK:
        conn = get_sqlite()
        conn.execute("UPDATE telegram_reminders SET leased_until = 0 WHERE id = ?", (reminder_id,))
        conn.commit()

def reschedule_transient_failure(reminder_id: int, new_fire_at: int) -> None:
    with _REMINDERS_LOCK:
        conn = get_sqlite()
        conn.execute(
            "UPDATE telegram_reminders SET next_fire_at = ?, leased_until = 0 WHERE id = ?",
            (new_fire_at, reminder_id)
        )
        conn.commit()

def delete_reminder_permanently(reminder_id: int) -> None:
    with _REMINDERS_LOCK:
        conn = get_sqlite()
        conn.execute("DELETE FROM telegram_reminders WHERE id = ?", (reminder_id,))
        conn.commit()

def disable_chat_reminders(chat_id: int) -> None:
    with _REMINDERS_LOCK:
        conn = get_sqlite()
        conn.execute("DELETE FROM telegram_reminders WHERE chat_id = ?", (chat_id,))
        conn.commit()


def get_pending_conversation(chat_id: int, user_id: int) -> dict | None:
    now = int(time.time())
    with _REMINDERS_LOCK:
        conn = get_sqlite()
        row = conn.execute(
            "SELECT chat_id, user_id, original_request, partial_fields_json, turns_json, missing_fields_json, expires_at "
            "FROM telegram_reminder_conversations WHERE chat_id = ? AND user_id = ?",
            (chat_id, user_id)
        ).fetchone()
        if not row:
            return None
        expires_at = row[6]
        if expires_at < now:
            conn.execute(
                "DELETE FROM telegram_reminder_conversations WHERE chat_id = ? AND user_id = ?",
                (chat_id, user_id)
            )
            conn.commit()
            return None
        return {
            "chat_id": row[0],
            "user_id": row[1],
            "original_request": row[2],
            "partial_fields": json.loads(row[3]),
            "turns": json.loads(row[4]),
            "missing_fields": json.loads(row[5]),
            "expires_at": expires_at
        }


def upsert_pending_conversation(
    chat_id: int,
    user_id: int,
    original_request: str,
    partial_fields: dict,
    turns: list,
    missing_fields: list,
    expires_at: int | None = None
) -> None:
    now = int(time.time())
    if expires_at is None:
        expires_at = now + 900  # 15 minutes
    partial_fields_json = json.dumps(partial_fields)
    turns_json = json.dumps(turns)
    missing_fields_json = json.dumps(missing_fields)
    
    with _REMINDERS_LOCK:
        conn = get_sqlite()
        conn.execute(
            "INSERT OR REPLACE INTO telegram_reminder_conversations "
            "(chat_id, user_id, original_request, partial_fields_json, turns_json, missing_fields_json, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (chat_id, user_id, original_request, partial_fields_json, turns_json, missing_fields_json, expires_at)
        )
        conn.commit()


def delete_pending_conversation(chat_id: int, user_id: int) -> None:
    with _REMINDERS_LOCK:
        conn = get_sqlite()
        conn.execute(
            "DELETE FROM telegram_reminder_conversations WHERE chat_id = ? AND user_id = ?",
            (chat_id, user_id)
        )
        conn.commit()

