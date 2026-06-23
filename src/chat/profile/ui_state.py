"""Redis-backed Telegram callback tokens, pending edits, and session index."""

from __future__ import annotations

import json
import logging
import secrets
import time

import redis

from src.chat.clients import get_redis
from src import config

log = logging.getLogger(__name__)


_PROFILE_TOKEN_TTL_SECONDS = 600
_PROFILE_EDIT_TTL_SECONDS = 600

_ALLOWED_PROFILE_TOKEN_SCOPES = frozenset({
    "profile_root",
    "profile_settings",
    "profile_cancel_pending",
    "subjects_page",
    "subject_view",
    "subject_add_menu",
    "subject_add_relationship",
    "subject_rename_confirm",
    "subject_delete_confirm",
    "subject_delete",
    "section_view",
    "section_set_state",
    "section_add_entry",
    "pregnancy_set",
    "subject_demographics_confirm",
    "subject_demographics_field",
    "subject_gender_set",
    "profile_set_preference",
    "facts_page",
    "fact_view",
    "fact_edit_field_confirm",
    "fact_edit_apply",
    "fact_delete_confirm",
    "fact_delete",
})


def _token_key(token: str) -> str:
    return f"profile_cb:v1:{token}"


def _edit_key(chat_id: int | str, owner_key: str) -> str:
    return f"profile_edit:v1:{chat_id}:{owner_key}"


def _owner_sessions_key(owner_key: str) -> str:
    return f"profile_sessions:v1:{owner_key}"


def issue_profile_token(
    scope: str,
    *,
    chat_id: int | str,
    owner_key: str,
    payload: dict | None = None,
) -> str:
    if scope not in _ALLOWED_PROFILE_TOKEN_SCOPES:
        raise ValueError(f"Unknown profile token scope: {scope}")
    token = secrets.token_urlsafe(12)
    body = json.dumps(
        {
            "scope": scope,
            "chat_id": str(chat_id),
            "owner_key": owner_key,
            "payload": payload or {},
            "issued_at": time.time(),
        },
        ensure_ascii=False,
    )
    try:
        get_redis().setex(
            _token_key(token),
            _PROFILE_TOKEN_TTL_SECONDS,
            body,
        )
    except (RuntimeError, redis.RedisError) as exc:
        log.warning("Profile token mint failed (non-fatal): %s", exc)
    return token


def consume_profile_token(
    token: str,
    *,
    expected_chat_id: int | str,
    expected_owner_key: str,
) -> dict | None:
    """One-shot read+delete; returns None on miss or scope/owner mismatch."""
    try:
        client = get_redis()
        getdel = getattr(client, "getdel", None)
        if callable(getdel):
            raw = getdel(_token_key(token))
        else:
            raw = client.get(_token_key(token))
            if raw is not None:
                client.delete(_token_key(token))
    except (RuntimeError, redis.RedisError) as exc:
        log.warning("Profile token consume failed: %s", exc)
        return None
    if not raw:
        return None
    try:
        body = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        str(body.get("chat_id") or "") != str(expected_chat_id)
        or body.get("owner_key") != expected_owner_key
        or body.get("scope") not in _ALLOWED_PROFILE_TOKEN_SCOPES
    ):
        return None
    return body


def set_pending_profile_edit(
    chat_id: int | str,
    owner_key: str,
    *,
    payload: dict,
) -> bool:
    body = json.dumps(
        {
            "chat_id": str(chat_id),
            "owner_key": owner_key,
            "payload": payload,
            "issued_at": time.time(),
        },
        ensure_ascii=False,
    )
    try:
        get_redis().setex(
            _edit_key(chat_id, owner_key),
            _PROFILE_EDIT_TTL_SECONDS,
            body,
        )
        return True
    except (RuntimeError, redis.RedisError) as exc:
        log.warning("Profile edit stage failed: %s", exc)
        return False


def pop_pending_profile_edit(
    chat_id: int | str,
    owner_key: str,
) -> dict | None:
    key = _edit_key(chat_id, owner_key)
    try:
        client = get_redis()
        getdel = getattr(client, "getdel", None)
        if callable(getdel):
            raw = getdel(key)
        else:
            pipe = client.pipeline()
            pipe.get(key)
            pipe.delete(key)
            raw = pipe.execute()[0]
    except (RuntimeError, redis.RedisError) as exc:
        log.warning("Profile edit pop failed: %s", exc)
        return None
    if not raw:
        return None
    try:
        body = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        str(body.get("chat_id") or "") != str(chat_id)
        or body.get("owner_key") != owner_key
    ):
        return None
    payload = body.get("payload")
    return payload if isinstance(payload, dict) else None


def clear_pending_profile_edit(chat_id: int | str, owner_key: str) -> None:
    """Best-effort cancellation for an unfinished profile input step."""
    try:
        get_redis().delete(_edit_key(chat_id, owner_key))
    except (RuntimeError, redis.RedisError) as exc:
        log.debug("Profile edit cancellation failed (non-fatal): %s", exc)


def track_profile_session(owner_key: str, session_key: str) -> None:
    """Best-effort: remember this session under its owner for invalidation."""
    if not owner_key or not session_key:
        return
    try:
        client = get_redis()
        client.sadd(_owner_sessions_key(owner_key), session_key)
        client.expire(_owner_sessions_key(owner_key), config.SESSION_TTL_SECONDS)
    except (RuntimeError, redis.RedisError) as exc:
        log.debug("Profile session track failed (non-fatal): %s", exc)


def iter_profile_sessions(owner_key: str) -> list[str]:
    if not owner_key:
        return []
    try:
        members = get_redis().smembers(_owner_sessions_key(owner_key))
    except (RuntimeError, redis.RedisError) as exc:
        log.warning("Profile session iterate failed: %s", exc)
        return []
    return [str(member) for member in (members or [])]


def invalidate_profile_sessions_for_owner(owner_key: str) -> int:
    """Clear the cached conversation context for every tracked session."""
    if not owner_key:
        return 0
    from src.chat.context.context_store import (
        NAMESPACE as _CONVERSATION_CONTEXT_NAMESPACE,
    )
    sessions = iter_profile_sessions(owner_key)
    cleared = 0
    try:
        client = get_redis()
        if sessions:
            client.delete(*[
                f"{_CONVERSATION_CONTEXT_NAMESPACE}{s}" for s in sessions
            ])
            cleared = len(sessions)
        client.delete(_owner_sessions_key(owner_key))
    except (RuntimeError, redis.RedisError) as exc:
        log.warning("Profile session invalidate failed: %s", exc)
    return cleared

