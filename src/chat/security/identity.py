from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import unicodedata
from dataclasses import dataclass

from src import config

_VERSION_RE = re.compile(r"^v[1-9][0-9]*$")
_OWNER_KEY_RE = re.compile(r"^owner_v[1-9][0-9]*_[0-9a-f]{64}$")
_SESSION_KEY_RE = re.compile(
    r"^(?:session_v[1-9][0-9]*_[0-9a-f]{64}|session_ephemeral_[0-9a-f]{64})$"
)
_EPHEMERAL_SESSION_SECRET = secrets.token_bytes(32)


@dataclass(frozen=True)
class RequestIdentity:
    owner_key: str
    session_key: str
    profile_persistence_allowed: bool


def derive_owner_key(
    channel: str,
    external_user_id: str | int,
    *,
    tenant: str | None = None,
    version: str | None = None,
) -> str:
    resolved_version = version or config.PROFILE_IDENTITY_ACTIVE_VERSION
    secret = _identity_secret(resolved_version)
    payload = _canonical_payload(
        "owner",
        resolved_version,
        tenant or config.PROFILE_DEFAULT_TENANT_ID,
        channel,
        external_user_id,
    )
    return f"owner_{resolved_version}_{hmac.new(secret, payload, hashlib.sha256).hexdigest()}"


def derive_session_key(
    channel: str,
    external_session_id: str | int,
    *,
    tenant: str | None = None,
    version: str | None = None,
) -> str:
    resolved_version = version or config.PROFILE_IDENTITY_ACTIVE_VERSION
    secret = _identity_secret(resolved_version)
    payload = _canonical_payload(
        "session",
        resolved_version,
        tenant or config.PROFILE_DEFAULT_TENANT_ID,
        channel,
        external_session_id,
    )
    return f"session_{resolved_version}_{hmac.new(secret, payload, hashlib.sha256).hexdigest()}"


def derive_request_identity(
    channel: str,
    external_user_id: str | int | None,
    external_session_id: str | int,
    *,
    tenant: str | None = None,
) -> RequestIdentity:
    try:
        session_key = derive_session_key(
            channel,
            external_session_id,
            tenant=tenant,
        )
    except (RuntimeError, ValueError):
        session_key = _derive_ephemeral_session_key(
            channel,
            external_session_id,
            tenant or config.PROFILE_DEFAULT_TENANT_ID,
        )
    if external_user_id is None or str(external_user_id).strip() == "":
        return RequestIdentity("", session_key, False)
    try:
        owner_key = derive_owner_key(channel, external_user_id, tenant=tenant)
    except (RuntimeError, ValueError):
        return RequestIdentity("", session_key, False)
    return RequestIdentity(owner_key, session_key, True)


def derive_previous_owner_keys(
    channel: str,
    external_user_id: str | int,
    *,
    tenant: str | None = None,
) -> tuple[str, ...]:
    keys = []
    for version in config.PROFILE_IDENTITY_PREVIOUS_VERSIONS:
        try:
            keys.append(
                derive_owner_key(
                    channel,
                    external_user_id,
                    tenant=tenant,
                    version=version,
                )
            )
        except (RuntimeError, ValueError):
            continue
    return tuple(keys)


def is_owner_key(value: object) -> bool:
    return isinstance(value, str) and bool(_OWNER_KEY_RE.fullmatch(value))


def is_session_key(value: object) -> bool:
    return isinstance(value, str) and bool(_SESSION_KEY_RE.fullmatch(value))


def validate_identity_configuration() -> None:
    profile_enabled = (
        getattr(config, "PROFILE_READ_ENABLED", False)
        or getattr(config, "PROFILE_WRITE_ENABLED", False)
    )
    try:
        _identity_secret(config.PROFILE_IDENTITY_ACTIVE_VERSION)
        for version in config.PROFILE_IDENTITY_PREVIOUS_VERSIONS:
            _identity_secret(version)
    except (RuntimeError, ValueError):
        if profile_enabled:
            raise RuntimeError(
                "Profile identity key is missing or invalid. Set a base64-encoded "
                "PROFILE_IDENTITY_HMAC_KEY (at least 32 bytes) or disable "
                "PROFILE_READ_ENABLED / PROFILE_WRITE_ENABLED."
            ) from None


def _identity_secret(version: str) -> bytes:
    if not _VERSION_RE.fullmatch(version):
        raise ValueError("Invalid profile identity key version")
    raw = os.getenv(f"PROFILE_IDENTITY_KEY_{version.upper()}", "").strip()
    if not raw and version == config.PROFILE_IDENTITY_ACTIVE_VERSION:
        raw = config.PROFILE_IDENTITY_HMAC_KEY
    if not raw:
        raise RuntimeError("Profile identity key is not configured")
    try:
        secret = base64.b64decode(raw, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("Profile identity key is not valid base64") from exc
    if len(secret) < 32:
        raise ValueError("Profile identity key must contain at least 32 bytes")
    return secret


def _canonical_payload(domain: str, version: str, *components: object) -> bytes:
    values = [domain, version, *components]
    canonical = []
    for value in values:
        normalized = unicodedata.normalize("NFKC", str(value)).strip()
        if not normalized or "\0" in normalized:
            raise ValueError("Identity components must be non-empty and contain no NUL")
        canonical.append(normalized)
    return "\0".join(canonical).encode("utf-8", errors="strict")


def _derive_ephemeral_session_key(
    channel: str,
    external_session_id: str | int,
    tenant: str,
) -> str:
    payload = _canonical_payload(
        "session-ephemeral",
        "v1",
        tenant,
        channel,
        external_session_id,
    )
    digest = hmac.new(_EPHEMERAL_SESSION_SECRET, payload, hashlib.sha256).hexdigest()
    return f"session_ephemeral_{digest}"
