from __future__ import annotations

import json
import logging

import redis

from src.chat.clients import get_redis
from src.chat.storage.domain import PatientSession
from src.config import SESSION_TTL_SECONDS

log = logging.getLogger(__name__)


def _session_key(session_id: str) -> str:
    return f"session:{session_id}"


def load_session(session_id: str) -> PatientSession:
    try:
        raw = get_redis().get(_session_key(session_id))
    except (RuntimeError, redis.RedisError) as e:
        log.warning("Redis load failed: %s", e)
        return PatientSession(session_id=session_id)
    if raw:
        try:
            return PatientSession.from_json(raw)
        except (json.JSONDecodeError, TypeError) as e:
            log.warning("Session JSON parse failed: %s", e)
    return PatientSession(session_id=session_id)


def save_session(session: PatientSession) -> None:
    try:
        get_redis().setex(
            _session_key(session.session_id),
            SESSION_TTL_SECONDS,
            session.to_json(),
        )
    except (RuntimeError, redis.RedisError) as e:
        log.warning("Redis save failed: %s", e)


def clear_session(session_id: str) -> None:
    try:
        client = get_redis()
        client.delete(_session_key(session_id))
        # Best-effort cleanup of legacy conversation-context keys; they
        # are unreachable from the new namespace and would otherwise just
        # expire under their existing TTLs.
        for legacy_suffix in (session_id, f"{session_id}:cases", f"{session_id}:state"):
            client.delete(f"conversation_context:v1:{legacy_suffix}")
    except (RuntimeError, redis.RedisError) as e:
        log.warning("Redis delete failed: %s", e)
