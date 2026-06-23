"""Redis-only storage for the conversation context.

The context holds the active subject, recent turns, the in-flight diagnostic
case, and any pending clarification. It expires with the session. It does
NOT hold any medical profile data; that lives in SQLite.
"""

from __future__ import annotations

import json
import logging
import time

import redis

from src.chat.clients import get_redis
from src.chat.context.domain import ClinicalCase, SessionState
from src import config

log = logging.getLogger(__name__)

NAMESPACE = "conversation_context:v1:"


def _context_key(session_id: str) -> str:
    return f"{NAMESPACE}{session_id}"


def load_conversation_context(
    session_id: str,
    owner_id: str | None,
) -> tuple[SessionState, dict[str, ClinicalCase], bool]:
    state = SessionState(session_id=session_id, owner_id=owner_id or "")
    try:
        raw = get_redis().get(_context_key(session_id))
    except (RuntimeError, redis.RedisError) as exc:
        log.warning("Conversation context load failed: %s", exc)
        return state, {}, False
    if not raw:
        return state, {}, True
    try:
        payload = json.loads(raw)
        loaded_state = SessionState.from_dict(payload["state"])
        if (
            loaded_state.session_id != session_id
            or loaded_state.owner_id != (owner_id or "")
        ):
            raise ValueError("conversation context identity mismatch")
        cases = {
            case_id: ClinicalCase(**case)
            for case_id, case in payload.get("cases", {}).items()
        }
        return loaded_state, cases, True
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        log.warning("Conversation context JSON parse failed: %s", exc)
        return state, {}, False


def save_conversation_context(
    state: SessionState,
    cases: dict[str, ClinicalCase],
    *,
    expected_revision: int,
) -> bool:
    """Optimistic save under a short per-session Redis lock."""
    try:
        client = get_redis()
        from contextlib import nullcontext
        lock_factory = getattr(client, "lock", None)
        lock = (
            lock_factory(
                f"lock:{_context_key(state.session_id)}",
                timeout=5,
                blocking_timeout=1,
            )
            if lock_factory
            else nullcontext()
        )
        with lock:
            raw = client.get(_context_key(state.session_id))
            current_revision = 0
            if raw:
                current_revision = int(json.loads(raw).get("state", {}).get("revision", 0))
            if current_revision != expected_revision:
                return False
            state.revision = expected_revision + 1
            client.setex(
                _context_key(state.session_id),
                config.SESSION_TTL_SECONDS,
                json.dumps(
                    {
                        "state": state.to_dict(),
                        "cases": {
                            case_id: case.__dict__ for case_id, case in cases.items()
                        },
                    },
                    ensure_ascii=False,
                ),
            )
        return True
    except (RuntimeError, redis.RedisError, ValueError, json.JSONDecodeError) as exc:
        log.warning("Conversation context save failed: %s", exc)
        return False


def clear_conversation_context(session_id: str) -> None:
    try:
        get_redis().delete(_context_key(session_id))
    except (RuntimeError, redis.RedisError) as exc:
        log.warning("Conversation context delete failed: %s", exc)
