from __future__ import annotations

import json
import time
from threading import RLock
from typing import Any

from src.chat.clients import get_sqlite
from src.chat.security.identity import is_session_key

_SQLITE_LOCK = RLock()


def _decode_trace(row: tuple) -> dict[str, Any]:
    (
        trace_id,
        session_id,
        internal_session_id,
        mode,
        question,
        answer,
        meta_json,
        created_at,
    ) = row
    return {
        "trace_id": trace_id,
        "session_id": session_id,
        "internal_session_id": internal_session_id,
        "mode": mode,
        "question": question,
        "answer": answer,
        "created_at": created_at,
        "meta": json.loads(meta_json),
    }


def _summary(trace: dict[str, Any]) -> dict[str, Any]:
    meta = trace.get("meta") or {}
    answer = str(trace.get("answer") or "")
    return {
        "trace_id": trace["trace_id"],
        "session_id": trace["session_id"],
        "mode": trace["mode"],
        "question": trace["question"],
        "answer_preview": answer[:180],
        "created_at": trace["created_at"],
        "latency_ms_total": meta.get("latency_ms_total"),
        "route": meta.get("route_label") or meta.get("outcome"),
    }


def save_chat_trace(
    *,
    trace_id: str,
    session_id: str,
    internal_session_id: str,
    mode: str,
    question: str,
    answer: str,
    meta: dict[str, Any],
    created_at: float | None = None,
) -> dict[str, Any]:
    if (
        not is_session_key(session_id)
        or not is_session_key(internal_session_id)
        or session_id != internal_session_id
    ):
        raise ValueError("Trace persistence requires a pseudonymous session key")
    created = time.time() if created_at is None else created_at
    safe_meta = _remove_owner_identifiers(meta)
    meta_json = json.dumps(safe_meta, ensure_ascii=False)
    with _SQLITE_LOCK:
        conn = get_sqlite()
        conn.execute(
            "INSERT INTO chat_trace "
            "(trace_id, session_id, internal_session_id, mode, question, answer, meta_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trace_id,
                session_id,
                internal_session_id,
                mode,
                question,
                answer,
                meta_json,
                created,
            ),
        )
        conn.commit()
    return {
        "trace_id": trace_id,
        "session_id": session_id,
        "internal_session_id": internal_session_id,
        "mode": mode,
        "question": question,
        "answer": answer,
        "created_at": created,
        "meta": safe_meta,
    }


def _remove_owner_identifiers(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _remove_owner_identifiers(item)
            for key, item in value.items()
            if key not in {"owner_id", "owner_key"}
        }
    if isinstance(value, list):
        return [_remove_owner_identifiers(item) for item in value]
    if isinstance(value, tuple):
        return [_remove_owner_identifiers(item) for item in value]
    return value


def get_chat_trace(trace_id: str, *, internal_session_id: str) -> dict[str, Any] | None:
    with _SQLITE_LOCK:
        conn = get_sqlite()
        row = conn.execute(
            "SELECT trace_id, session_id, internal_session_id, mode, question, answer, meta_json, created_at "
            "FROM chat_trace WHERE trace_id = ? AND internal_session_id = ?",
            (trace_id, internal_session_id),
        ).fetchone()
    return _decode_trace(row) if row else None


def list_chat_traces(
    *,
    internal_session_id: str,
    trace_id: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    capped_limit = max(1, min(int(limit), 100))
    clauses = ["internal_session_id = ?"]
    params: list[Any] = [internal_session_id]
    if trace_id:
        clauses.append("trace_id = ?")
        params.append(trace_id)
    where = f" WHERE {' AND '.join(clauses)}"
    params.append(capped_limit)
    with _SQLITE_LOCK:
        conn = get_sqlite()
        rows = conn.execute(
            "SELECT trace_id, session_id, internal_session_id, mode, question, answer, meta_json, created_at "
            f"FROM chat_trace{where} ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
    return [_summary(_decode_trace(row)) for row in rows]
