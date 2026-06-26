"""Load profile preferences and assemble the safe medical-profile context."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from src import config
from src.chat.profile import repository as repo
from src.chat.context.context_store import (
    clear_conversation_context,
    load_conversation_context,
    save_conversation_context,
)
from src.chat.context.domain import (
    ClinicalCase,
    ClarificationState,
    ConversationContextBundle,
    SessionState,
)
from src.chat.context.resolver import (
    SubjectResolution,
    context_references,
    resolve_subject,
)
from src.chat.profile.domain import ProfileFact
from src.chat.profile.selector import build_context_bundle
from src.chat.profile.ui_state import (
    invalidate_profile_sessions_for_owner,
    track_profile_session,
)
from src.chat.security.identity import is_owner_key

log = logging.getLogger(__name__)


@dataclass
class ConversationContextRuntime:
    state: SessionState
    cases: dict[str, ClinicalCase]
    resolution: SubjectResolution
    bundle: ConversationContextBundle
    source_turn_id: str
    expected_revision: int
    redis_available: bool
    profile_personalization_allowed: bool = False
    profile: object | None = None
    profile_text: str = ""


def prepare_context_runtime(
    session_key: str,
    legacy_session,  # PatientSession (kept for compatibility; not read here)
    analysis: dict,
    *,
    owner_key: str | None = None,
    profile_persistence_allowed: bool = False,
    previous_owner_keys: tuple[str, ...] = (),
    preloaded: tuple | None = None,
) -> ConversationContextRuntime:
    del legacy_session
    state, cases, redis_available = (
        preloaded if preloaded is not None
        else load_conversation_context(session_key, owner_key)
    )
    expected_revision = state.revision
    context = analysis.get("context") if isinstance(analysis.get("context"), dict) else {}
    resolution = resolve_subject(context, state)
    analysis_ok = bool(analysis.get("analysis_succeeded", True))
    relation = context.get("relation")

    if analysis_ok and relation != "off_topic":
        state.active_entity_refs = context_references(context)
        if resolution.subject_id:
            _activate_subject_case(state, cases, resolution.subject_id)

    facts: list[ProfileFact] = []
    consent_enabled = not config.PROFILE_REQUIRE_CONSENT
    storage_enabled = consent_enabled
    personalization_enabled = consent_enabled
    identity_allowed = profile_persistence_allowed and bool(owner_key)
    profile_available = True
    if identity_allowed:
        try:
            repo.migrate_owner_key(owner_key, previous_owner_keys)
            storage_pref = repo.get_user_preference(owner_key, "storage")
            if storage_pref is not None:
                storage_enabled = storage_pref
            personalization_pref = repo.get_user_preference(
                owner_key, "personalization"
            )
            if personalization_pref is not None:
                personalization_enabled = personalization_pref
        except Exception:
            profile_available = False
            log.exception("Profile preference preparation failed")

    profile_read_allowed = bool(
        identity_allowed
        and storage_enabled
        and config.PROFILE_READ_ENABLED
        and profile_available
    )
    profile_write_allowed = bool(
        identity_allowed
        and storage_enabled
        and config.PROFILE_WRITE_ENABLED
        and profile_available
    )
    if (
        analysis_ok
        and profile_read_allowed
        and resolution.subject_id
        and context.get("needs_medical_profile")
        and not resolution.ambiguous
    ):
        try:
            facts = repo.list_profile_facts(owner_key, resolution.subject_id)
        except Exception:
            profile_available = False
            profile_read_allowed = False
            profile_write_allowed = False
            log.exception("Profile fact load failed")

    active_case = cases.get(state.active_case_id or "")
    bundle = build_context_bundle(
        analysis=analysis,
        state=state,
        resolution=resolution,
        facts=facts if profile_available and redis_available else [],
        active_case=active_case,
    )
    profile_personalization_allowed = bool(
        profile_read_allowed and personalization_enabled
    )
    profile_obj = None
    profile_text = ""
    if (
        profile_personalization_allowed
        and profile_available
        and resolution.subject_id
    ):
        try:
            from src.chat.profile.projection import (
                build_medical_profile,
                format_profile_for_prompt,
            )
            profile_obj = build_medical_profile(
                owner_key, resolution.subject_id, facts=facts,
            )
            profile_text = format_profile_for_prompt(profile_obj)
        except Exception:
            log.exception("Medical profile projection failed")
            profile_obj = None
            profile_text = ""

    subject_context = context.get("subject")
    if bundle.subject and isinstance(subject_context, dict):
        for key in ("relationship", "display_name"):
            value = subject_context.get(key)
            if isinstance(value, str) and value.strip():
                bundle.subject[key] = value.strip()

    if resolution.ambiguous:
        state.pending_clarification = ClarificationState(
            question=resolution.clarification,
            created_at=time.time(),
        )
    elif bundle.excluded_reason == "conflicting_facts":
        state.pending_clarification = ClarificationState(
            question="Tôi đang có các thông tin y tế mâu thuẫn. Bạn xác nhận thông tin nào hiện đúng?",
            subject_id=resolution.subject_id,
            created_at=time.time(),
        )
    elif resolution.subject_id:
        state.pending_clarification = None

    return ConversationContextRuntime(
        state=state,
        cases=cases,
        resolution=resolution,
        bundle=bundle,
        source_turn_id=f"turn_{__import__('uuid').uuid4().hex}",
        expected_revision=expected_revision,
        redis_available=redis_available,
        profile_personalization_allowed=profile_personalization_allowed,
        profile=profile_obj,
        profile_text=profile_text,
    )


def persist_context_runtime(
    runtime: ConversationContextRuntime,
    legacy_session,
    *,
    question: str,
    reply: str,
    analysis: dict,
) -> None:
    del legacy_session
    from src.chat.context.domain import Turn
    timestamp = time.time()
    _sync_active_case(runtime.state, runtime.cases, timestamp)
    runtime.state.recent_turns.extend([
        Turn(
            runtime.source_turn_id, "user", question, timestamp, runtime.resolution.subject_id,
        ),
        Turn(
            f"turn_{__import__('uuid').uuid4().hex}",
            "assistant", reply, timestamp, runtime.resolution.subject_id,
        ),
    ])
    runtime.state.recent_turns = runtime.state.recent_turns[-20:]

    if (
        analysis.get("analysis_succeeded", True)
        and runtime.profile_personalization_allowed  # write requires both
        and runtime.redis_available
        and runtime.resolution.subject_id
        and not runtime.resolution.ambiguous
    ):
        candidates = analysis.get("profile_candidates")
        if isinstance(candidates, list) and candidates:
            try:
                repo.write_profile_candidates(
                    owner_id=runtime.state.owner_id,
                    resolved_subject_id=runtime.resolution.subject_id,
                    source_turn_id=runtime.source_turn_id,
                    profile_candidates=candidates,
                )
            except Exception:
                log.exception("Profile write failed")

    if runtime.redis_available and not save_conversation_context(
        runtime.state,
        runtime.cases,
        expected_revision=runtime.expected_revision,
    ):
        log.warning(
            "Skipped stale conversation context save revision=%s",
            runtime.expected_revision,
        )


def _activate_subject_case(
    state: SessionState,
    cases: dict[str, ClinicalCase],
    subject_id: str,
) -> None:
    now = time.time()
    if state.active_subject_id and state.active_subject_id != subject_id:
        _sync_active_case(state, cases, now)
    if state.active_subject_id != subject_id:
        selected = next(
            (
                case for case in cases.values()
                if case.subject_id == subject_id and case.status == "active"
            ),
            None,
        )
        state.active_subject_id = subject_id
        state.active_case_id = selected.case_id if selected else None


def _sync_active_case(
    state: SessionState,
    cases: dict[str, ClinicalCase],
    now: float,
) -> None:
    case = cases.get(state.active_case_id or "")
    if case is None:
        return
    case.updated_at = now
