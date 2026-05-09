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
        log.warning("Redis load failed for %s: %s", session_id, e)
        return PatientSession(session_id=session_id)
    if raw:
        try:
            return PatientSession.from_json(raw)
        except (json.JSONDecodeError, TypeError) as e:
            log.warning("Session JSON parse failed for %s: %s", session_id, e)
    return PatientSession(session_id=session_id)


def save_session(session: PatientSession) -> None:
    try:
        get_redis().setex(
            _session_key(session.session_id),
            SESSION_TTL_SECONDS,
            session.to_json(),
        )
    except (RuntimeError, redis.RedisError) as e:
        log.warning("Redis save failed for %s: %s", session.session_id, e)


def clear_session(session_id: str) -> None:
    try:
        get_redis().delete(_session_key(session_id))
    except (RuntimeError, redis.RedisError) as e:
        log.warning("Redis delete failed for %s: %s", session_id, e)
