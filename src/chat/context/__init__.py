"""Conversation context: short-lived, session-scoped, Redis-only state.

Symptom snapshots, differential candidates, the in-flight clinical case,
the active subject, and recent turns all live here. They expire with the
session. The medical profile lives separately under src/chat/profile/.
"""
from src.chat.context.context_store import (
    clear_conversation_context,
    load_conversation_context,
    save_conversation_context,
)
from src.chat.context.domain import (
    ClinicalCase,
    ClarificationState,
    ContextReference,
    SessionState,
    Turn,
)
from src.chat.context.resolver import (
    SubjectResolution,
    context_references,
    format_subject_address,
    resolve_subject,
)

__all__ = [
    "ClinicalCase",
    "ClarificationState",
    "ContextReference",
    "SessionState",
    "SubjectResolution",
    "Turn",
    "clear_conversation_context",
    "context_references",
    "format_subject_address",
    "load_conversation_context",
    "resolve_subject",
    "save_conversation_context",
]
