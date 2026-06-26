"""
pipeline.py
-----------
Orchestration per user turn.

Phases (short-circuit on the first failure):
  A. Input validation      — rate limit, guardrail, LLM quota
  B. State load            — load session from Redis, classify turn, extract entities
  C. Route                 — greeting / emergency / clarification / diagnostic / informational
  D. Persist               — save session, profile, log consultation
"""

from __future__ import annotations

import logging
import threading
import time
import unicodedata
import uuid
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import asdict
from typing import Any, Callable

from src.chat.diagnosis.clarification_options import (
    detail_options_from_catalog,
    detail_selection_mode_from_catalog,
    fallback_detail_options,
    fallback_selection_mode,
    presence_options_from_catalog,
)
from src.chat.diagnosis.differential import (
    build_clarification,
    discriminative_symptoms,
    parse_clarification_answer,
    rank_candidates,
    should_ask_clarification,
    symptom_catalog,
)
from src.chat.diagnosis.entities import normalize_entities
from src.chat.diagnosis.flow import build_general_triage_prompt, direct_diagnostic_prompt
from src.chat.evidence_plan import (
    fallback_domain_for_intent,
    normalize_evidence_plan,
    plan_answer_domain,
    plan_required_facts,
    plan_targets,
    should_run_evidence_planner,
)
from src.chat.guards.drug_policy import OTC_ONLY_REPLY, evaluate_drug_policy
from src.chat.guards.guardrail import VERDICT_REPLIES, regex_check
from src.chat.health_insurance import is_health_insurance_query
from src.chat.guards.quota import check_both as check_llm_quota
from src.chat.llm.analyzer import analyze_turn
from src.chat.llm.evidence_planner import plan_evidence
from src.chat.llm.generator import AnswerDomain, generate
from src.chat.context.domain import ClinicalCase, ConversationContextBundle
from src.chat.context.resolver import format_subject_address
from src.chat.profile.runtime import (
    ConversationContextRuntime,
    persist_context_runtime,
    prepare_context_runtime,
)
from src.chat.security.identity import is_owner_key
from src.chat.mode_policy import (
    ModeDecision,
    apply_mode_policy,
    normalize_intent,
    normalize_mode,
)
from src.chat.preflight import RATE_LIMIT_MSG, preflight
from src.chat.replies import ChatReply, TECHNICAL_ERROR_REPLY, emergency_reply
from src.chat.retrieval import (
    Hit,
    RetrievalScope,
    format_kg_context,
    hybrid_search,
    hybrid_search_with_debug,
    kg_search,
)
from src.chat.storage.session import (
    PatientSession,
    check_rate_limit,
    load_session,
    log_consultation,
    save_profile,
    save_session,
)
from src.chat.context.context_store import load_conversation_context
from src.chat.timing import elapsed_ms, log_trace_timing
from src.config import CONVERSATION_CONTEXT_ENABLED, EMERGENCY_RAG_ENABLED, PIPELINE_EMERGENCY_RAG, RERANK_TOP_K

GREETING_REPLY = ("Xin chào! Tôi là trợ lý y tế. Bạn có thể mô tả triệu chứng "
                  "hoặc hỏi về bệnh/thuốc cụ thể.")
GENERAL_TRIAGE_MARKER = "general:triage"
_START_CHOICES = {"bat dau", "duoc", "ok", "okay", "tiep", "tiep tuc"}
_YES_CHOICES = {"co"}
_NO_CHOICES = {"khong"}
_UNKNOWN_CHOICES = {"khong ro", "khong biet", "chua ro", "chua biet"}
_ANSWER_NOW_CHOICES = {"tra loi luon", "cu tra loi", "tra loi di"}
_DISEASE_INFO_TERMS = (
    "la gi",
    "nguyen nhan",
    "trieu chung",
    "dieu tri",
    "giai doan",
    "phan loai",
    "chan doan",
    "xet nghiem",
    "bien chung",
    "co lay",
)
_MEDICATION_INFO_TERMS = (
    "lieu",
    "lieu dung",
    "cach dung",
    "dung thuoc",
    "tac dung",
    "cong dung",
    "chi dinh",
    "chi đinh",
    "chong chi dinh",
    "chong chi đinh",
    "tuong tac",
    "tac dung phu",
    "qua lieu",
    "bao nhieu",
    "may vien",
    "uong may",
    "uong bao",
    "ngay uong",
)
_TOXICOLOGY_TERMS = (
    "ma tuy",
    "bong cuoi",
    "khi cuoi",
    "ngo doc",
    "lam dung chat",
    "chat gay nghien",
)
_TOXICOLOGY_HEALTH_TERMS = (
    "suc khoe",
    "tac hai",
    "anh huong",
    "co hai",
    "trieu chung",
    "ngo doc",
    "qua lieu",
    "nghien",
    "lam dung",
    "cai",
    "dieu tri",
    "nguy hiem",
    "hau qua",
    "gay hai",
    "gay benh",
)
_EMERGENCY_ACTION_TERMS = (
    "toi phai lam gi",
    "phai lam gi",
    "nen lam gi",
    "can lam gi",
    "lam sao",
    "xu tri the nao",
    "xu tri sao",
    "xu ly the nao",
    "so cuu",
    "cap cuu khong",
    "can cap cuu",
    "co can cap cuu",
    "goi 115",
    "goi cap cuu",
    "di cap cuu",
    "dua di cap cuu",
    "dua den vien",
    "dua vao vien",
)
_EMERGENCY_FACTUAL_TERMS = (
    "la gi",
    "trieu chung",
    "dau hieu",
    "bieu hien",
    "nguyen nhan",
    "chan doan",
    "xet nghiem",
    "bien chung",
    "phan loai",
    "co che",
    "dinh nghia",
    "mo ta",
    "thong tin",
)
_EMERGENCY_PROTOCOL_INFO_TERMS = (
    "truyen dich",
    "bu dich",
    "hoi suc dich",
    "phac do",
    "cap cuu ban dau",
)
_CURRENT_CASE_TERMS = (
    "toi bi",
    "toi dang",
    "minh bi",
    "minh dang",
    "em bi",
    "em dang",
    "bo toi",
    "ba toi",
    "me toi",
    "chong toi",
    "vo toi",
    "con toi",
    "nguoi nha",
    "ban toi",
)
_DETAIL_PREFIX = "detail:"
_DETAIL_SLOT_ORDER = ("onset", "severity", "pattern", "associated")
log = logging.getLogger(__name__)

_META_LOCAL = threading.local()


def _meta() -> dict | None:
    """Return the active per-turn meta dict, or None when not collecting."""
    return getattr(_META_LOCAL, "current", None)


def _event_sink():
    """Return the active per-thread event sink, or None when not streaming."""
    return getattr(_META_LOCAL, "event_sink", None)


def _install_event_sink(sink) -> None:
    """Install (or clear with None) the per-thread streaming event sink."""
    _META_LOCAL.event_sink = sink


def _emit_node_event(stage: str, status: str, ms: float | None) -> None:
    """Push a compact node-progress event to the active sink, if one is installed.

    No-op when no sink is present (every normal answer() call and every test),
    so the public pipeline behavior is unchanged.
    """
    sink = _event_sink()
    if sink is None:
        return
    sink.put({
        "type": "node",
        "id": stage,
        "status": status,
        "ms": round(float(ms), 2) if ms is not None else None,
    })


def _record_hits(hits: list[Hit]) -> None:
    meta = _meta()
    if meta is None:
        return
    include_text = bool(meta.get("_collect_graph"))
    meta.setdefault("retrieved", []).extend(
        {
            "source_type": h.source_type,
            "source_slug": h.source_slug,
            "source_name": h.source_name,
            "heading_path": h.heading_path,
            "id": h.id,
            "chunk_id": h.chunk_id,
            "score": h.score,
            **({"text": h.text} if include_text else {}),
        }
        for h in hits
    )


def _record_usage(stage: str, usage: dict | None, model: str | None = None) -> None:
    meta = _meta()
    if meta is None or not usage:
        return
    entry = {"stage": stage, **usage}
    if model:
        entry["model"] = model
    meta.setdefault("usage", []).append(entry)


def _record_latency(stage: str, ms: float) -> None:
    meta = _meta()
    if meta is None:
        return
    meta.setdefault("latency_ms", {})[stage] = round(ms, 2)


def _record_timing(stage: str, ms: float, fields: dict[str, Any]) -> None:
    meta = _meta()
    if meta is None:
        return
    entry_fields = {
        key: value
        for key, value in fields.items()
        if value is not None
    }
    meta.setdefault("timings", []).append(
        {"stage": stage, "ms": round(ms, 2), "fields": entry_fields}
    )
    if stage == "route" and fields.get("label") is not None:
        meta["route_label"] = fields["label"]
    if stage == "total" and fields.get("outcome") is not None:
        meta["outcome"] = fields["outcome"]


def _record_mode_decision(
    mode: str,
    intent: str,
    decision: ModeDecision,
    question: str,
) -> None:
    meta = _meta()
    if meta is None:
        return
    meta["mode"] = mode
    meta["intent"] = intent
    if decision.suggest_mode:
        meta["suggest_mode"] = decision.suggest_mode
        meta["retry_question"] = question


def _record_diagnostic_snapshot(session, *, extra=None) -> None:
    """Snapshot current diagnostic session state into meta for the graph inspector.

    No-op unless a graph is being collected, so normal answer() calls and channel
    turns are unaffected. Only reads session state — never mutates it.
    """
    meta = _meta()
    if meta is None or not meta.get("_collect_graph"):
        return
    snap = meta.setdefault("_graph_diagnostic", {})
    snap["answered_questions"] = list(session.answered_questions)
    snap["clarification_queue"] = list(session.clarification_queue)
    snap["symptoms"] = [dict(s) for s in session.symptoms]
    snap["candidate_diseases"] = [dict(c) for c in session.candidate_diseases[:8]]
    snap["clarification_plan_started"] = session.clarification_plan_started
    snap["clarification_parse_failures"] = session.clarification_parse_failures
    if extra:
        snap.setdefault("events", {}).update(extra)


def _log_timing(trace_id: str, stage: str, start: float, **fields) -> None:
    ms = elapsed_ms(start)
    _record_timing(stage, ms, fields)
    log_trace_timing(log, "pipeline", trace_id, stage, start, **fields)
    status = "error" if fields.get("failed") else "ok"
    _emit_node_event(stage, status, ms)


def _mark_graph_error_stage(stage: str) -> None:
    meta = _meta()
    if meta is not None and meta.get("_collect_graph"):
        stages = meta.setdefault("_graph_error_stages", [])
        if stage not in stages:
            stages.append(stage)
        meta.setdefault("_graph_error_stage", stage)


def _record_failed_elapsed_timing(
    stage: str,
    ms: float,
    **fields,
) -> None:
    _mark_graph_error_stage(stage)
    _record_timing(stage, ms, {"failed": True, **fields})


def _log_failed_timing(
    trace_id: str,
    stage: str,
    start: float,
    **fields,
) -> None:
    _mark_graph_error_stage(stage)
    _log_timing(trace_id, stage, start, failed=True, **fields)


# ---------- Phase A: input validation ----------

def _guardrail_reply(session_id: str, question: str, verdict: str) -> str:
    if verdict == "abuse":
        return ""
    reply = VERDICT_REPLIES.get(verdict, "")
    session = load_session(session_id)
    session.add_message("user", question)
    session.add_message("assistant", reply)
    save_session(session)
    return reply


# ---------- Phase B helpers ----------

def _last_bot_message(session: PatientSession) -> str:
    for turn in reversed(session.conversation):
        if turn.get("role") == "assistant":
            return turn.get("content", "")
    return ""


def _choice_key(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.strip().casefold().replace("đ", "d"))
    no_marks = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    for char in ".!?":
        no_marks = no_marks.replace(char, " ")
    return " ".join(no_marks.split())


def _is_yes_no_choice(text: str) -> bool:
    return _choice_present(text) is not None


def _choice_present(text: str) -> str | None:
    key = _choice_key(text)
    if key in _UNKNOWN_CHOICES:
        return "unknown"
    if key in _YES_CHOICES or key.startswith("co "):
        return "yes"
    if key in _NO_CHOICES or key.startswith("khong "):
        return "no"
    return None


def _is_answer_now_choice(text: str) -> bool:
    return _choice_key(text) in _ANSWER_NOW_CHOICES


def _looks_like_medication_info_question(text: str) -> bool:
    key = _choice_key(text)
    return any(term in key for term in _MEDICATION_INFO_TERMS)


def _looks_like_disease_info_question(text: str) -> bool:
    key = _choice_key(text)
    return any(term in key for term in _DISEASE_INFO_TERMS)


def _is_medical_toxicology_query(text: str) -> bool:
    key = _choice_key(text)
    has_toxicology_term = any(term in key for term in _TOXICOLOGY_TERMS)
    if not has_toxicology_term:
        return False
    if "ngo doc" in key:
        return True
    return any(term in key for term in _TOXICOLOGY_HEALTH_TERMS)


def _asks_emergency_action(text: str) -> bool:
    key = _choice_key(text)
    return any(term in key for term in _EMERGENCY_ACTION_TERMS)


def _looks_like_current_emergency_case(text: str) -> bool:
    key = _choice_key(text)
    return any(term in key for term in _CURRENT_CASE_TERMS)


def _looks_like_factual_emergency_topic(text: str) -> bool:
    key = _choice_key(text)
    if _asks_emergency_action(key):
        return False
    if any(term in key for term in _EMERGENCY_FACTUAL_TERMS):
        return True
    if _looks_like_current_emergency_case(key):
        return False
    return any(term in key for term in _EMERGENCY_PROTOCOL_INFO_TERMS)


def _allow_rag_drug_info(question: str, label: str, intent: str) -> bool:
    if label == "greeting_other" or intent == "off_scope":
        return False
    return (
        intent in {"contextual_drug_info", "pure_info"}
        or _looks_like_medication_info_question(question)
    )


def _infer_answer_domain(
    question: str,
    hits: list[Hit],
    turn_intent: str = "",
) -> AnswerDomain:
    if turn_intent == "health_insurance_info":
        return "health_insurance_info"
    if turn_intent == "contextual_drug_info":
        return "drug_info"

    top_hits = hits[:5]
    drug_count = sum(1 for hit in top_hits if hit.source_type == "drug")
    disease_count = sum(1 for hit in top_hits if hit.source_type == "disease")

    if drug_count and drug_count >= disease_count:
        return "drug_info"
    if (
        disease_count
        and disease_count >= drug_count
        and (
            _looks_like_disease_info_question(question)
            or turn_intent in {"pure_info", "condition_management_info"}
        )
    ):
        return "disease_info"
    return "symptom_or_care"


def _prepare_evidence_plan(
    question: str,
    analysis: dict[str, Any],
    label: str,
    intent: str,
    trace_id: str,
) -> dict[str, Any]:
    fallback_domain = fallback_domain_for_intent(intent, label)
    raw_plan = analysis.get("evidence_plan")
    analysis_succeeded = analysis.get("analysis_succeeded")
    plan_is_active = isinstance(raw_plan, dict) and analysis_succeeded is not False
    if analysis_succeeded is False:
        raw_plan = None
    plan = normalize_evidence_plan(raw_plan, fallback_domain=fallback_domain)
    plan_lacks_scope = not plan_targets(plan) and not plan_required_facts(plan)
    planner_needed = should_run_evidence_planner(plan) or (
        fallback_domain in {"disease_info", "drug_info"}
        and (
            analysis_succeeded is False
            or (plan_is_active and plan_lacks_scope)
        )
    )
    planner_used = False
    if planner_needed:
        stage_start = time.perf_counter()
        try:
            plan = plan_evidence(
                question,
                analysis=analysis,
                fallback_domain=fallback_domain,
            )
            planner_used = True
        except Exception:
            log.exception("Evidence planner fallback failed trace=%s", trace_id)
            plan = normalize_evidence_plan(
                raw_plan,
                fallback_domain=fallback_domain,
            )
        _log_timing(
            trace_id,
            "evidence_plan",
            stage_start,
            fallback=True,
            domain=plan.get("domain"),
            source_type=plan.get("source_type"),
            answer_slot=plan.get("answer_slot"),
        )
    if planner_used and (plan_targets(plan) or plan_required_facts(plan)):
        plan_is_active = True
    analysis["evidence_plan"] = plan
    meta = _meta()
    if meta is not None:
        meta["evidence_plan"] = plan
    return plan if plan_is_active else None


def _question_variants(value: Any) -> list[str]:
    if not value:
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _detail_question_id(symptom_id: str, slot: str, index: int | None = None) -> str:
    if index is None:
        return f"{_DETAIL_PREFIX}{slot}:{symptom_id}"
    return f"{_DETAIL_PREFIX}{slot}:{index}:{symptom_id}"


def _parse_detail_question_id(question_id: str) -> tuple[str, str, int | None] | None:
    if not question_id.startswith(_DETAIL_PREFIX):
        return None
    remainder = question_id[len(_DETAIL_PREFIX):]
    if ":" not in remainder:
        return None
    slot, tail = remainder.split(":", 1)
    if not slot or not tail:
        return None
    index = None
    symptom_id = tail
    if ":" in tail:
        maybe_index, rest = tail.split(":", 1)
        if maybe_index.isdigit() and rest:
            index = int(maybe_index)
            symptom_id = rest
    return symptom_id, slot, index


def _is_detail_question_id(question_id: str) -> bool:
    return _parse_detail_question_id(question_id) is not None


def _detail_question_text(question_id: str, catalog: dict[str, dict]) -> str:
    parsed = _parse_detail_question_id(question_id)
    if not parsed:
        return ""
    symptom_id, slot, index = parsed
    questions = catalog.get(symptom_id, {}).get("clarification_questions", {}) or {}
    variants = _question_variants(questions.get(slot) if isinstance(questions, dict) else None)
    if index is not None:
        if 0 <= index < len(variants):
            return variants[index]
        return ""
    return variants[0] if variants else ""


def _symptom_has_slot(session: PatientSession, symptom_id: str, slot: str) -> bool:
    return any(
        symptom.get("symptom_id") == symptom_id and bool(symptom.get(slot))
        for symptom in session.symptoms
    )


def _detail_question_ids(
    session: PatientSession,
    symptom_id: str,
    catalog: dict[str, dict],
) -> list[str]:
    questions = catalog.get(symptom_id, {}).get("clarification_questions", {}) or {}
    answered = set(session.answered_questions)
    queued = set(session.clarification_queue)
    detail_ids: list[str] = []
    for slot in _DETAIL_SLOT_ORDER:
        raw_questions = questions.get(slot)
        variants = _question_variants(raw_questions)
        if not variants or _symptom_has_slot(session, symptom_id, slot):
            continue
        use_index = isinstance(raw_questions, list)
        for index, _question in enumerate(variants):
            question_id = _detail_question_id(
                symptom_id,
                slot,
                index if use_index else None,
            )
            if question_id not in answered and question_id not in queued:
                detail_ids.append(question_id)
    return detail_ids


def _prepend_detail_questions(
    session: PatientSession,
    symptom_id: str,
    catalog: dict[str, dict],
) -> None:
    detail_ids = _detail_question_ids(session, symptom_id, catalog)
    if detail_ids:
        session.clarification_queue = detail_ids + session.clarification_queue


def _prepend_known_symptom_detail_questions(
    session: PatientSession,
    catalog: dict[str, dict],
) -> None:
    detail_ids: list[str] = []
    queued = set(session.clarification_queue)
    for symptom in session.symptoms:
        symptom_id = symptom.get("symptom_id")
        if not symptom_id or symptom_id.startswith("raw:"):
            continue
        for question_id in _detail_question_ids(session, symptom_id, catalog):
            if question_id in queued:
                continue
            queued.add(question_id)
            detail_ids.append(question_id)
    if detail_ids:
        session.clarification_queue = detail_ids + session.clarification_queue


def _append_detail_answer(existing: Any, answer: str) -> Any:
    if not existing:
        return answer
    if isinstance(existing, list):
        return existing if answer in existing else [*existing, answer]
    if str(existing).strip() == answer:
        return existing
    return [existing, answer]


def _reset_diagnostic_workflow(session: PatientSession) -> None:
    """Fail closed when a subject switch cannot use subject-aware state."""
    session.symptoms = []
    session.medications = []
    session.candidate_diseases = []
    session.answered_questions = []
    session.clarification_queue = []
    session.clarification_plan_started = False
    session.clarification_parse_failures = 0


def _is_pending_clarification_choice(session: PatientSession, text: str) -> bool:
    if not session.answered_questions:
        return False
    key = _choice_key(text)
    last_question = session.answered_questions[-1]
    if last_question == GENERAL_TRIAGE_MARKER:
        return key in _START_CHOICES
    if _is_detail_question_id(last_question):
        return not _looks_like_medication_info_question(text)
    return _is_yes_no_choice(text) or _is_answer_now_choice(text)


def _clarification_choice_analysis(text: str = "") -> dict:
    answer_now = _is_answer_now_choice(text)
    return {
        "guardrail": {"verdict": "allow", "reason": "clarification choice"},
        "turn": {
            "label": "clarification_answer",
            "intent": "clarification_answer",
            "direct_answer_requested": answer_now,
        },
        "rewrite": {"rewritten": "", "confident": True, "clarification": ""},
        "entities": {"symptoms": [], "medications": []},
    }


def _symptom_group_key(symptom_id: str, catalog: dict[str, dict]) -> str:
    entry = catalog.get(symptom_id, {})
    name = entry.get("name_vi", symptom_id.replace("symptom:S_", "").replace("_", " "))
    key = _choice_key(name)
    if key in {"non", "buon non"} or key.startswith("non ") or "buon non" in key:
        return "nausea_vomiting"
    return key or symptom_id


def _queue_clarification_symptoms(
    session: PatientSession,
    symptom_ids: list[str],
) -> None:
    catalog = symptom_catalog()
    already_asked = {
        sid for sid in session.answered_questions
        if sid != GENERAL_TRIAGE_MARKER and not _is_detail_question_id(sid)
    }
    seen_groups = {
        _symptom_group_key(sid, catalog)
        for sid in already_asked
    }
    queued: list[str] = []
    for sid in symptom_ids:
        if not sid or sid == GENERAL_TRIAGE_MARKER or sid in already_asked:
            continue
        group_key = _symptom_group_key(sid, catalog)
        if group_key in seen_groups:
            continue
        seen_groups.add(group_key)
        queued.append(sid)
    session.clarification_queue = queued
    session.clarification_plan_started = True


def _ask_next_queued_clarification(
    session: PatientSession,
    trace_id: str,
    stage_start: float,
) -> str | None:
    while session.clarification_queue:
        next_symptom = session.clarification_queue.pop(0)
        if _is_detail_question_id(next_symptom):
            detail_question = _detail_question_text(next_symptom, symptom_catalog())
            if not detail_question:
                continue
            session.answered_questions.append(next_symptom)
            _log_timing(
                trace_id,
                "diagnostic_clarification",
                stage_start,
                asked=1,
                queued=len(session.clarification_queue),
                kind="detail",
            )
            _record_diagnostic_snapshot(
                session,
                extra={"asked": next_symptom},
            )
            return detail_question
        if next_symptom in session.answered_questions:
            continue
        session.answered_questions.append(next_symptom)
        _log_timing(
            trace_id,
            "diagnostic_clarification",
            stage_start,
            asked=1,
            queued=len(session.clarification_queue),
            kind="presence",
        )
        _record_diagnostic_snapshot(
            session,
            extra={"asked": next_symptom},
        )
        return build_clarification([next_symptom])
    return None


def _ingest_entities(raw_entities: dict, session: PatientSession) -> None:
    ents = normalize_entities(raw_entities)
    for s in ents["symptoms"]:
        session.upsert_symptom(s)
    for d in ents["medications"]:
        session.add_medication(d)


def _load_kg_context(question: str, trace_id: str) -> str:
    stage_start = time.perf_counter()
    kg_result = kg_search(question)
    kg_text = format_kg_context(kg_result)
    _log_timing(trace_id, "kg_search", stage_start,
                matched=len(kg_result.matched_entities),
                empty=kg_result.is_empty)
    return kg_text


def _load_hybrid_hits(
    question: str,
    trace_id: str,
    scope: RetrievalScope = "medical",
    evidence_plan: dict[str, Any] | None = None,
) -> list[Hit]:
    stage_start = time.perf_counter()
    if scope == "medical":
        if evidence_plan is None:
            hits = hybrid_search(question, top_k=RERANK_TOP_K)
        else:
            hits = hybrid_search(
                question,
                top_k=RERANK_TOP_K,
                evidence_plan=evidence_plan,
            )
    else:
        if evidence_plan is None:
            hits = hybrid_search(question, top_k=RERANK_TOP_K, scope=scope)
        else:
            hits = hybrid_search(
                question,
                top_k=RERANK_TOP_K,
                scope=scope,
                evidence_plan=evidence_plan,
            )
    _log_timing(trace_id, "hybrid_search", stage_start, hits=len(hits))
    return hits


def _load_kg_context_with_debug(
    question: str,
    trace_id: str,
) -> tuple[str, dict[str, Any]]:
    stage_start = time.perf_counter()
    kg_result = kg_search(question)
    kg_context = asdict(kg_result)
    kg_text = format_kg_context(kg_result)
    _log_timing(trace_id, "kg_search", stage_start,
                matched=len(kg_result.matched_entities),
                empty=kg_result.is_empty)
    return kg_text, kg_context


def _load_hybrid_hits_with_debug(
    question: str,
    trace_id: str,
    scope: RetrievalScope = "medical",
    evidence_plan: dict[str, Any] | None = None,
) -> tuple[list[Hit], dict[str, Any]]:
    stage_start = time.perf_counter()

    def _on_stage(stage, status, ms):
        _emit_node_event(stage, status, ms)

    if scope == "medical":
        if evidence_plan is None:
            hits, retrieval_debug = hybrid_search_with_debug(
                question,
                top_k=RERANK_TOP_K,
                on_stage=_on_stage,
            )
        else:
            hits, retrieval_debug = hybrid_search_with_debug(
                question,
                top_k=RERANK_TOP_K,
                on_stage=_on_stage,
                evidence_plan=evidence_plan,
            )
    else:
        if evidence_plan is None:
            hits, retrieval_debug = hybrid_search_with_debug(
                question,
                top_k=RERANK_TOP_K,
                scope=scope,
                on_stage=_on_stage,
            )
        else:
            hits, retrieval_debug = hybrid_search_with_debug(
                question,
                top_k=RERANK_TOP_K,
                scope=scope,
                on_stage=_on_stage,
                evidence_plan=evidence_plan,
            )
    _log_timing(trace_id, "hybrid_search", stage_start, hits=len(hits))
    return hits, retrieval_debug


def _call_with_elapsed(fn: Any, *args: Any) -> tuple[Any, float]:
    stage_start = time.perf_counter()
    try:
        result = fn(*args)
    except Exception as exc:
        exc.pipeline_elapsed_ms = elapsed_ms(stage_start)
        raise
    return result, elapsed_ms(stage_start)


def _retrieval_error_stage(exc: BaseException) -> str:
    debug = getattr(exc, "debug", None)
    if isinstance(debug, dict) and isinstance(debug.get("error_stage"), str):
        return debug["error_stage"]
    error_text = str(exc).casefold()
    return next(
        (
            stage
            for keyword, stage in (
                ("sparse", "sparse_search"),
                ("fusion", "fusion"),
                ("rrf", "fusion"),
                ("rerank", "rerank"),
                ("dense", "dense_search"),
                ("qdrant", "dense_search"),
                ("embedding", "dense_search"),
            )
            if keyword in error_text
        ),
        "dense_search",
    )


# ---------- Phase C: route handlers ----------

def _handle_diagnostic(
    session: PatientSession,
    question: str,
    trace_id: str,
    force_answer: bool = False,
) -> str:
    """New symptoms reported. Rank diseases, maybe ask one clarification question."""
    if not force_answer and not session.answered_questions:
        session.answered_questions.append(GENERAL_TRIAGE_MARKER)
        _log_timing(trace_id, "diagnostic_general_triage", time.perf_counter())
        bundle = getattr(_META_LOCAL, "context_bundle", None)
        subject = bundle.subject if isinstance(bundle, ConversationContextBundle) else None
        return build_general_triage_prompt(session, subject)

    stage_start = time.perf_counter()
    if not force_answer:
        queued_reply = _ask_next_queued_clarification(session, trace_id, stage_start)
        if queued_reply:
            return queued_reply
        if not session.clarification_plan_started:
            _prepend_known_symptom_detail_questions(session, symptom_catalog())
            queued_reply = _ask_next_queued_clarification(session, trace_id, stage_start)
            if queued_reply:
                return queued_reply
    else:
        session.clarification_queue.clear()
        session.clarification_plan_started = True

    symptom_ids = [s["symptom_id"] for s in session.symptoms if s.get("symptom_id")]
    stage_start = time.perf_counter()
    candidates = rank_candidates(symptom_ids)
    session.candidate_diseases = candidates
    _log_timing(trace_id, "diagnostic_rank", stage_start,
                symptoms=len(symptom_ids), candidates=len(candidates))
    _record_diagnostic_snapshot(session)

    stage_start = time.perf_counter()
    if (
        not force_answer
        and not session.clarification_plan_started
        and should_ask_clarification(candidates)
    ):
        cand_ids = [c["disease_id"] for c in candidates]
        known = [s for s in symptom_ids if not s.startswith("raw:")]
        next_symptoms = discriminative_symptoms(cand_ids, known + session.answered_questions)
        if next_symptoms:
            _queue_clarification_symptoms(session, next_symptoms)
            queued_reply = _ask_next_queued_clarification(session, trace_id, stage_start)
            if queued_reply:
                return queued_reply
    _log_timing(trace_id, "diagnostic_clarification", stage_start,
                asked=0, force_answer=force_answer)
    _record_diagnostic_snapshot(session)

    if force_answer or session.clarification_plan_started:
        prompt, retrieval_query = direct_diagnostic_prompt(session, question)
        return _handle_informational(
            session,
            prompt,
            trace_id,
            use_patient_context=True,
            retrieval_question=retrieval_query,
            answer_domain="symptom_or_care",
        )
    # Enough narrowing — answer with RAG + KG and patient context
    return _handle_informational(
        session,
        question,
        trace_id,
        use_patient_context=True,
        answer_domain="symptom_or_care",
    )


def _handle_clarification_answer(
    session: PatientSession,
    user_answer: str,
    trace_id: str,
    force_answer: bool = False,
) -> str:
    """Parse one clarification reply, update slots, then re-narrow."""
    if session.answered_questions and session.answered_questions[-1] == GENERAL_TRIAGE_MARKER:
        session.clarification_parse_failures = 0
        return _handle_diagnostic(session, user_answer, trace_id, force_answer=force_answer)
    if force_answer and _is_answer_now_choice(user_answer):
        session.clarification_parse_failures = 0
        return _handle_diagnostic(session, user_answer, trace_id, force_answer=True)
    last_question = session.answered_questions[-1] if session.answered_questions else ""
    detail_question = _parse_detail_question_id(last_question)
    if detail_question:
        symptom_id, slot, _index = detail_question
        catalog = symptom_catalog()
        existing_value = next(
            (
                symptom.get(slot)
                for symptom in session.symptoms
                if symptom.get("symptom_id") == symptom_id
            ),
            None,
        )
        entry = {
            "symptom_id": symptom_id,
            "name": catalog.get(symptom_id, {}).get("name_vi", symptom_id),
            slot: _append_detail_answer(existing_value, user_answer.strip()),
        }
        session.upsert_symptom(entry)
        session.clarification_parse_failures = 0
        return _handle_diagnostic(session, user_answer, trace_id, force_answer=force_answer)

    asked_ids = [last_question] if last_question else []
    catalog = symptom_catalog()
    asked = [
        {"symptom_id": sid, "name": catalog.get(sid, {}).get("name_vi", sid)}
        for sid in asked_ids
    ]
    stage_start = time.perf_counter()
    present = _choice_present(user_answer)
    if present is not None and len(asked_ids) == 1:
        parsed_answers = [{"symptom_id": asked_ids[0], "present": present}]
    else:
        parsed_answers = parse_clarification_answer(asked, user_answer)
    _log_timing(trace_id, "clarification_parse", stage_start,
                asked=len(asked), parsed=len(parsed_answers))
    if parsed_answers:
        session.clarification_parse_failures = 0
    else:
        session.clarification_parse_failures += 1
        if session.clarification_parse_failures >= 2:
            force_answer = True

    for r in parsed_answers:
        sid = r.get("symptom_id")
        if not sid or r.get("present") != "yes":
            continue
        entry = {"symptom_id": sid,
                 "name": catalog.get(sid, {}).get("name_vi", sid)}
        for k in ("onset", "severity", "pattern", "associated"):
            if r.get(k):
                entry[k] = r[k]
        session.upsert_symptom(entry)
        _prepend_detail_questions(session, sid, catalog)

    _record_diagnostic_snapshot(
        session,
        extra={"parsed_answers": parsed_answers, "last_asked": last_question},
    )
    return _handle_diagnostic(session, user_answer, trace_id, force_answer=force_answer)


def _handle_informational(
    session: PatientSession,
    question: str,
    trace_id: str,
    use_patient_context: bool = False,
    retrieval_question: str | None = None,
    turn_intent: str = "",
    answer_domain: AnswerDomain | None = None,
    evidence_plan: dict[str, Any] | None = None,
) -> str:
    stage_start = time.perf_counter()
    search_question = retrieval_question or question
    active_meta = _meta()
    collect_debug = bool(
        active_meta is not None and active_meta.get("_collect_graph")
    )
    kg_loader = _load_kg_context_with_debug if collect_debug else _load_kg_context
    hits_loader = (
        _load_hybrid_hits_with_debug if collect_debug else _load_hybrid_hits
    )
    parent_sink = _event_sink()

    def _with_sink(fn, *args):
        if parent_sink is not None:
            _install_event_sink(parent_sink)
        try:
            return _call_with_elapsed(fn, *args)
        finally:
            if parent_sink is not None:
                _install_event_sink(None)

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="retrieval") as executor:
        kg_future = executor.submit(_with_sink, kg_loader, search_question, trace_id)
        if evidence_plan is None:
            hits_future = executor.submit(
                _with_sink,
                hits_loader,
                search_question,
                trace_id,
            )
        else:
            hits_future = executor.submit(
                _with_sink,
                hits_loader,
                search_question,
                trace_id,
                "medical",
                evidence_plan,
            )
        done, pending = wait((kg_future, hits_future), return_when=FIRST_EXCEPTION)
        failures: dict[Any, BaseException] = {}
        for future in done:
            exc = future.exception()
            if exc is not None:
                failures[future] = exc
        if failures:
            for pending_future in pending:
                pending_future.cancel()
        else:
            kg_result, kg_elapsed = kg_future.result()
            hits_result, hits_elapsed = hits_future.result()
    if failures:
        if collect_debug:
            meta = _meta()
            if meta is not None:
                for future in (kg_future, hits_future):
                    if future.done() and not future.cancelled():
                        exc = future.exception()
                        if exc is not None:
                            failures.setdefault(future, exc)
                kg_exc = failures.get(kg_future)
                if kg_exc is not None:
                    _record_failed_elapsed_timing(
                        "kg_search",
                        float(getattr(
                            kg_exc,
                            "pipeline_elapsed_ms",
                            elapsed_ms(stage_start),
                        )),
                    )
                hits_exc = failures.get(hits_future)
                if hits_exc is not None:
                    debug = getattr(hits_exc, "debug", None)
                    if isinstance(debug, dict):
                        meta["retrieval_debug"] = debug
                    _mark_graph_error_stage(_retrieval_error_stage(hits_exc))
                if (
                    kg_future.done()
                    and not kg_future.cancelled()
                    and kg_future.exception() is None
                ):
                    kg_result, kg_elapsed = kg_future.result()
                    _kg_text, kg_context = kg_result
                    meta["kg_context"] = kg_context
                    _record_timing(
                        "kg_search",
                        kg_elapsed,
                        {"kg_chars": len(_kg_text)},
                    )
                if (
                    hits_future.done()
                    and not hits_future.cancelled()
                    and hits_future.exception() is None
                ):
                    hits_result, hits_elapsed = hits_future.result()
                    _hits, retrieval_debug = hits_result
                    meta["retrieval_debug"] = retrieval_debug
                    _record_timing(
                        "hybrid_search",
                        hits_elapsed,
                        {"hits": len(_hits)},
                    )
        raise (
            failures.get(kg_future)
            or failures.get(hits_future)
            or next(iter(failures.values()))
        )
    if collect_debug:
        kg_text, kg_context = kg_result
        hits, retrieval_debug = hits_result
        meta = _meta()
        if meta is not None:
            meta["kg_context"] = kg_context
            meta["retrieval_debug"] = retrieval_debug
    else:
        kg_text = kg_result
        hits = hits_result
    _record_timing("kg_search", kg_elapsed, {"kg_chars": len(kg_text)})
    _record_timing("hybrid_search", hits_elapsed, {"hits": len(hits)})
    _log_timing(trace_id, "parallel_retrieval", stage_start,
                hits=len(hits), kg_chars=len(kg_text))
    _record_hits(hits)
    _record_latency("retrieval", elapsed_ms(stage_start))
    planned_domain = plan_answer_domain(evidence_plan, "") if evidence_plan else ""
    answer_domain = answer_domain or planned_domain or _infer_answer_domain(
        search_question,
        hits,
        turn_intent,
    )
    active_meta = _meta()
    if active_meta is not None:
        active_meta["answer_domain"] = answer_domain

    patient_dict = None
    profile_text = ""
    if use_patient_context:
        if CONVERSATION_CONTEXT_ENABLED:
            bundle = getattr(_META_LOCAL, "context_bundle", None)
            if isinstance(bundle, ConversationContextBundle) and bundle.excluded_reason is None:
                if session.symptoms and bundle.subject:
                    existing_case = bundle.active_case
                    bundle.active_case = ClinicalCase(
                        case_id=(existing_case.case_id if existing_case else "current_turn"),
                        subject_id=str(bundle.subject["id"]),
                        symptoms=list(session.symptoms),
                        candidate_diseases=list(session.candidate_diseases),
                        answered_questions=list(session.answered_questions),
                        clarification_queue=list(session.clarification_queue),
                        created_at=(existing_case.created_at if existing_case else time.time()),
                        updated_at=time.time(),
                    )
                patient_dict = bundle.to_prompt_dict()
                runtime = getattr(_META_LOCAL, "context_runtime", None)
                if runtime is not None and getattr(runtime, "profile_text", ""):
                    profile_text = runtime.profile_text
                    if isinstance(patient_dict, dict):
                        patient_dict["medical_profile_text"] = profile_text
        else:
            patient_dict = asdict(session)
    stage_start = time.perf_counter()
    try:
        if _meta() is not None:
            reply, gen_meta = generate(
                question,
                hits,
                kg_text=kg_text,
                patient=patient_dict,
                answer_domain=answer_domain,
                evidence_plan=evidence_plan,
                return_meta=True,
            )
            _record_usage("generator", gen_meta.get("usage"), model=gen_meta.get("model"))
            _meta()["doctor_offer"] = bool(gen_meta.get("doctor_handoff_recommended"))
            _meta()["doctor_specialty"] = gen_meta.get("doctor_specialty")
        else:
            reply = generate(
                question,
                hits,
                kg_text=kg_text,
                patient=patient_dict,
                answer_domain=answer_domain,
                evidence_plan=evidence_plan,
            )
    except Exception:
        _log_failed_timing(trace_id, "generate", stage_start)
        raise
    _log_timing(trace_id, "generate", stage_start, chars=len(reply))
    _record_latency("generator", elapsed_ms(stage_start))
    return reply


def _handle_health_insurance(
    question: str,
    trace_id: str,
    evidence_plan: dict[str, Any] | None = None,
) -> str:
    stage_start = time.perf_counter()
    active_meta = _meta()
    collect_debug = bool(
        active_meta is not None and active_meta.get("_collect_graph")
    )
    if collect_debug:
        if evidence_plan is None:
            hits_result, hits_elapsed = _call_with_elapsed(
                _load_hybrid_hits_with_debug,
                question,
                trace_id,
                "health_insurance",
            )
        else:
            hits_result, hits_elapsed = _call_with_elapsed(
                _load_hybrid_hits_with_debug,
                question,
                trace_id,
                "health_insurance",
                evidence_plan,
            )
        hits, retrieval_debug = hits_result
        if active_meta is not None:
            active_meta["retrieval_debug"] = retrieval_debug
    else:
        if evidence_plan is None:
            hits, hits_elapsed = _call_with_elapsed(
                _load_hybrid_hits,
                question,
                trace_id,
                "health_insurance",
            )
        else:
            hits, hits_elapsed = _call_with_elapsed(
                _load_hybrid_hits,
                question,
                trace_id,
                "health_insurance",
                evidence_plan,
            )

    _record_timing("hybrid_search", hits_elapsed, {"hits": len(hits)})
    _record_hits(hits)
    _record_latency("retrieval", elapsed_ms(stage_start))
    if active_meta is not None:
        active_meta["answer_domain"] = "health_insurance_info"

    stage_start = time.perf_counter()
    try:
        if active_meta is not None:
            reply, gen_meta = generate(
                question,
                hits,
                answer_domain="health_insurance_info",
                evidence_plan=evidence_plan,
                return_meta=True,
            )
            _record_usage(
                "generator",
                gen_meta.get("usage"),
                model=gen_meta.get("model"),
            )
            active_meta["doctor_offer"] = False
            active_meta["doctor_specialty"] = None
        else:
            reply = generate(
                question,
                hits,
                answer_domain="health_insurance_info",
                evidence_plan=evidence_plan,
            )
    except Exception:
        _log_failed_timing(trace_id, "generate", stage_start)
        raise
    _log_timing(trace_id, "generate", stage_start, chars=len(reply))
    _record_latency("generator", elapsed_ms(stage_start))
    return reply


def _route(
    label: str,
    session: PatientSession,
    question: str,
    rewritten: str,
    force_answer: bool,
    trace_id: str,
    use_patient_context_override: bool | None = None,
    emergency_subject: str = "bạn",
    emergency_red_flags: list[str] | None = None,
    emergency_context: dict | None = None,
    on_preliminary_reply: Callable[[str], None] | None = None,
    use_emergency_rag: bool | None = None,
    turn_intent: str = "",
    evidence_plan: dict[str, Any] | None = None,
) -> str:
    if label == "emergency":
        emergency_question = question if rewritten == question else f"{question}\n{rewritten}"
        if use_emergency_rag is None:
            use_emergency_rag = PIPELINE_EMERGENCY_RAG
        # Always emit the deterministic fast reply first (so the user sees
        # 115 + first cues without waiting for retrieval). For channels
        # that want a real two-message UX, ``on_preliminary_reply`` is
        # invoked synchronously with the fast reply.
        from src.chat.replies import emergency_fast_reply
        fast = emergency_fast_reply(
            emergency_subject,
            red_flags=emergency_red_flags,
            question=emergency_question,
            context=emergency_context,
        )
        if on_preliminary_reply is not None:
            try:
                on_preliminary_reply(fast)
            except Exception:
                log.exception("preliminary emergency reply callback failed trace=%s", trace_id)
        if not use_emergency_rag:
            return fast
        try:
            if on_preliminary_reply is not None:
                from src.chat.replies import emergency_first_aid_reply
                return emergency_first_aid_reply(
                    emergency_question,
                    red_flags=emergency_red_flags,
                    subject_address=emergency_subject,
                )
            from src.chat.replies import emergency_reply as _emergency_reply
            return _emergency_reply(
                emergency_subject,
                red_flags=emergency_red_flags,
                question=emergency_question,
                context=emergency_context,
                use_rag=True,
            )
        except Exception:
            log.exception("Emergency RAG failed trace=%s", trace_id)
            return fast
    if label == "health_insurance":
        return _handle_health_insurance(
            rewritten,
            trace_id,
            evidence_plan=evidence_plan,
        )
    if label == "clarification_answer":
        return _handle_clarification_answer(session, question, trace_id, force_answer)
    if label == "diagnostic":
        return _handle_diagnostic(session, rewritten, trace_id, force_answer=force_answer)
    # informational: always answer; personalize if we have symptoms
    use_patient_context = (
        use_patient_context_override
        if use_patient_context_override is not None
        else (
            bool(session.symptoms)
            and not _looks_like_medication_info_question(rewritten)
        )
    )
    return _handle_informational(
        session,
        rewritten,
        trace_id,
        use_patient_context=use_patient_context,
        turn_intent=turn_intent,
        evidence_plan=evidence_plan,
    )


def _persist_final_turn(
    session: PatientSession,
    session_id: str,
    question: str,
    reply: str,
    trace_id: str,
    context_runtime: ConversationContextRuntime | None = None,
    analysis: dict | None = None,
) -> None:
    stage_start = time.perf_counter()
    try:
        save_session(session)
        log_consultation(session_id, question, reply)
    except Exception:
        log.exception("Pipeline persist failed trace=%s", trace_id)
        _log_timing(trace_id, "persist", stage_start, failed=True)
        return
    if context_runtime is not None:
        try:
            persist_context_runtime(
                context_runtime,
                session,
                question=question,
                reply=reply,
                analysis=analysis or {"analysis_succeeded": False},
            )
        except Exception:
            log.exception(
                "Conversation context persistence failed trace=%s",
                trace_id,
            )
    _log_timing(trace_id, "persist", stage_start)


def suggested_choices(reply: str) -> tuple[str, ...]:
    """Return short tap-friendly choices for clarification replies."""
    compact_reply = reply.strip()
    if "Để tôi định hướng tốt hơn, bạn cho tôi hỏi thêm một vài câu hỏi nhé." in reply:
        return ("Bắt đầu",)
    if (
        compact_reply.startswith("Bạn có bị ")
        and compact_reply.endswith(" không?")
        and "\n" not in compact_reply
    ):
        return presence_options_from_catalog(compact_reply, symptom_catalog())
    if compact_reply.endswith("?") and "\n" not in compact_reply:
        return (
            detail_options_from_catalog(compact_reply, symptom_catalog())
            or fallback_detail_options(compact_reply)
        )
    return ()


def suggested_selection_mode(reply: str) -> str:
    """Return whether a clarification reply should render single or multi select."""
    compact_reply = reply.strip()
    if not compact_reply.endswith("?") or "\n" in compact_reply:
        return "single"
    if compact_reply.startswith("Bạn có bị ") and compact_reply.endswith(" không?"):
        return "single"
    return (
        detail_selection_mode_from_catalog(compact_reply, symptom_catalog())
        or fallback_selection_mode(compact_reply)
    )


# ---------- Entry point ----------

def _answer_inner(
    question: str,
    session_id: str,
    trace_id: str,
    request_start: float,
    owner_id: str | None = None,
    previous_owner_ids: tuple[str, ...] = (),
    mode: str = "auto",
    on_preliminary_reply: Callable[[str], None] | None = None,
    use_emergency_rag: bool | None = None,
) -> str:
    question = (question or "").strip()
    if not question:
        return "Bạn hãy đặt câu hỏi cụ thể nhé."
    mode = normalize_mode(mode)
    _META_LOCAL.context_bundle = None
    _emit_node_event("input", "ok", 0.0)

    stage_start = time.perf_counter()
    try:
        session = load_session(session_id)
    except Exception:
        _log_failed_timing(trace_id, "load_session", stage_start)
        raise
    _log_timing(trace_id, "load_session", stage_start)

    preloaded_context = None
    if CONVERSATION_CONTEXT_ENABLED:
        preloaded_context = load_conversation_context(session_id, owner_id)

    is_clarification_choice = _is_pending_clarification_choice(session, question)
    regex_check_for_turn = (lambda text: None) if is_clarification_choice else regex_check

    def guardrail_reply_with_timing(
        guard_session_id: str,
        guard_question: str,
        verdict: str,
    ) -> str:
        persist_start = time.perf_counter()
        try:
            reply = _guardrail_reply(guard_session_id, guard_question, verdict)
        except Exception:
            _log_failed_timing(trace_id, "persist", persist_start)
            raise
        if verdict != "abuse":
            _log_timing(trace_id, "persist", persist_start)
        return reply

    stage_start = time.perf_counter()
    try:
        short_circuit = preflight(
            question,
            session_id,
            guardrail_reply_with_timing,
            check_rate_limit,
            regex_check_for_turn,
            check_llm_quota,
        )
    except Exception:
        meta = _meta()
        error_stages = (
            meta.get("_graph_error_stages", [])
            if isinstance(meta, dict)
            else []
        )
        if "persist" in error_stages:
            _log_timing(trace_id, "preflight", stage_start)
        else:
            _log_failed_timing(trace_id, "preflight", stage_start)
        raise
    _log_timing(trace_id, "preflight", stage_start)
    if short_circuit is not None:
        _log_timing(trace_id, "total", request_start, outcome="short_circuit")
        return short_circuit

    stage_start = time.perf_counter()
    try:
        if is_clarification_choice and not CONVERSATION_CONTEXT_ENABLED:
            analysis = _clarification_choice_analysis(question)
        else:
            analysis = analyze_turn(
                question,
                last_bot_message=_last_bot_message(session),
                history=session.conversation,
                session_context=(
                    {
                        "active_subject_id": preloaded_context[0].active_subject_id,
                        "active_entity_refs": [
                            reference.__dict__
                            for reference in preloaded_context[0].active_entity_refs
                        ],
                        "active_case_id": preloaded_context[0].active_case_id,
                        "pending_clarification": (
                            preloaded_context[0].pending_clarification.__dict__
                            if preloaded_context[0].pending_clarification
                            else None
                        ),
                    }
                    if preloaded_context is not None
                    else None
                ),
            )
    except Exception:
        _log_failed_timing(trace_id, "turn_analysis", stage_start)
        raise
    meta = _meta()
    if meta is not None and meta.get("_collect_graph"):
        meta["_graph_turn_analysis"] = analysis
        rewrite_data = analysis.get("rewrite")
        if isinstance(rewrite_data, dict):
            meta["rewrite_query"] = {
                "original": question,
                "rewritten": rewrite_data.get("rewritten"),
                "confident": rewrite_data.get("confident"),
            }
    guard = analysis["guardrail"]
    turn = analysis["turn"]
    label = turn["label"]
    intent = normalize_intent(turn.get("intent"), label)
    rewrite = analysis.get("rewrite") if isinstance(analysis.get("rewrite"), dict) else {}
    health_insurance_detected = (
        is_health_insurance_query(question)
        or is_health_insurance_query(str(rewrite.get("rewritten") or ""))
    )
    if health_insurance_detected and guard.get("verdict") in {"allow", "off_topic"}:
        guard["verdict"] = "allow"
        guard["reason"] = "deterministic_health_insurance_route"
        label = "informational"
        intent = "health_insurance_info"
        turn["label"] = label
        turn["intent"] = intent
        analysis.setdefault("entities", {}).setdefault("symptoms", [])
        analysis.setdefault("entities", {}).setdefault("medications", [])
        rewrite = analysis.setdefault("rewrite", {})
        rewrite.setdefault("rewritten", question)
        rewrite.setdefault("confident", True)
        rewrite.setdefault("clarification", "")
        analysis["evidence_plan"] = normalize_evidence_plan(
            {
                "domain": "health_insurance_info",
                "source_type": "health_insurance",
                "answer_slot": "insurance_rule",
                "safety_mode": "factual_info",
                "answer_style": "short_explanation",
                "confidence": 1.0,
                "needs_fallback": False,
            },
            fallback_domain="health_insurance_info",
        )
        if meta is not None:
            meta["health_insurance_detector"] = "forced"
    toxicology_detected = (
        _is_medical_toxicology_query(question)
        or _is_medical_toxicology_query(str(rewrite.get("rewritten") or ""))
    )
    if (
        toxicology_detected
        and not health_insurance_detected
        and guard.get("verdict") in {"allow", "off_topic"}
    ):
        guard["verdict"] = "allow"
        guard["reason"] = "deterministic_medical_toxicology_scope"
        label = "informational"
        intent = "pure_info"
        turn["label"] = label
        turn["intent"] = intent
        analysis.setdefault("entities", {}).setdefault("symptoms", [])
        analysis.setdefault("entities", {}).setdefault("medications", [])
        rewrite = analysis.setdefault("rewrite", {})
        rewrite.setdefault("rewritten", question)
        rewrite.setdefault("confident", True)
        rewrite.setdefault("clarification", "")
        if meta is not None:
            meta["medical_toxicology_detector"] = "forced"
    context = analysis.get("context") if isinstance(analysis.get("context"), dict) else {}
    triage = analysis.get("triage") if isinstance(analysis.get("triage"), dict) else {}
    red_flags = triage.get("red_flags")
    emergency_red_flags = red_flags if isinstance(red_flags, list) else None
    try:
        from src.chat.emergency import classify_emergency_intent
        emergency_intent = classify_emergency_intent(question, emergency_red_flags)
    except Exception:  # pragma: no cover - route override must never break chat
        emergency_intent = None
    emergency_topic_info = bool(
        emergency_intent
        and _looks_like_factual_emergency_topic(question)
    )
    emergency_action_needed = bool(
        emergency_intent
        and not emergency_topic_info
        and (
            triage.get("urgency") == "emergency"
            or bool(emergency_red_flags)
            or _asks_emergency_action(question)
            or _looks_like_current_emergency_case(question)
        )
    )
    if emergency_topic_info:
        if label == "emergency" or intent == "emergency":
            label = "informational"
            intent = "pure_info"
            turn["label"] = label
            turn["intent"] = intent
        analysis["evidence_plan"] = normalize_evidence_plan(
            {
                "domain": "disease_info",
                "source_type": "disease",
                "answer_slot": "first_aid",
                "safety_mode": "factual_info",
                "answer_style": "short_explanation",
                "confidence": 0.8,
                "needs_fallback": False,
            },
            fallback_domain="disease_info",
        )
        if meta is not None:
            meta["emergency_intent_topic_info"] = emergency_intent
    elif emergency_action_needed:
        label = "emergency"
        intent = "emergency"
        triage["urgency"] = "emergency"
        analysis["evidence_plan"] = normalize_evidence_plan(
            {
                "domain": "symptom_or_care",
                "source_type": "medical",
                "answer_slot": "first_aid",
                "safety_mode": "emergency_action",
                "answer_style": "stepwise",
                "confidence": 1.0,
                "needs_fallback": False,
            },
            fallback_domain="symptom_or_care",
        )
        meta = _meta()
        if meta is not None:
            meta["emergency_intent_override"] = emergency_intent
    if not CONVERSATION_CONTEXT_ENABLED and context.get("relation") == "switch_subject":
        _reset_diagnostic_workflow(session)
    evidence_plan = _prepare_evidence_plan(
        question,
        analysis,
        label,
        intent,
        trace_id,
    )
    direct_answer_for_log = (
        turn["direct_answer_requested"] if guard["verdict"] == "allow" else False
    )
    _log_timing(trace_id, "turn_analysis", stage_start,
                verdict=guard["verdict"],
                label=label,
                intent=intent,
                mode=mode,
                direct_answer=direct_answer_for_log)
    _emit_node_event("rewrite", "ok", None)
    if guard["verdict"] != "allow":
        stage_start = time.perf_counter()
        try:
            reply = _guardrail_reply(session_id, question, guard["verdict"])
        except Exception:
            _log_failed_timing(trace_id, "persist", stage_start)
            raise
        if guard["verdict"] != "abuse":
            _log_timing(trace_id, "persist", stage_start)
        _log_timing(trace_id, "total", request_start, outcome="guardrail")
        return reply

    stage_start = time.perf_counter()
    drug_policy = evaluate_drug_policy(question, analysis)
    _log_timing(
        trace_id,
        "drug_policy",
        stage_start,
        is_drug_question=drug_policy.is_drug_question,
        allowed=drug_policy.allowed,
        reason=drug_policy.reason,
    )
    if meta is not None:
        meta["drug_policy"] = {
            "is_drug_question": drug_policy.is_drug_question,
            "allowed": drug_policy.allowed,
            "reason": drug_policy.reason,
            "matched_otc_names": list(drug_policy.matched_otc_names),
        }
    allow_rag_drug_info = (
        drug_policy.is_drug_question
        and not drug_policy.allowed
        and drug_policy.reason not in {"unresolved_drug", "unsafe_self_prescribing"}
        and _allow_rag_drug_info(question, label, intent)
    )
    if meta is not None and allow_rag_drug_info:
        meta["drug_policy"]["rag_drug_info_override"] = True
    if drug_policy.is_drug_question and not drug_policy.allowed and not allow_rag_drug_info:
        session.add_message("user", question)
        session.add_message("assistant", OTC_ONLY_REPLY)
        _persist_final_turn(
            session,
            session_id,
            question,
            OTC_ONLY_REPLY,
            trace_id,
            analysis=analysis,
        )
        _log_timing(trace_id, "total", request_start, outcome="drug_policy")
        return OTC_ONLY_REPLY

    context_runtime = None
    if CONVERSATION_CONTEXT_ENABLED and intent != "emergency":
        try:
            context_runtime = prepare_context_runtime(
                session_id,
                session,
                analysis,
                owner_key=owner_id,
                profile_persistence_allowed=is_owner_key(owner_id),
                previous_owner_keys=previous_owner_ids,
                preloaded=preloaded_context,
            )
            _META_LOCAL.context_bundle = context_runtime.bundle
            _META_LOCAL.context_runtime = context_runtime
            if meta is not None:
                meta["context_bundle"] = context_runtime.bundle.to_prompt_dict()
                meta["profile_selection_reasons"] = context_runtime.bundle.selection_reasons
        except Exception:
            log.exception(
                "Conversation context preparation failed trace=%s",
                trace_id,
            )
            _META_LOCAL.context_bundle = ConversationContextBundle(
                subject=None,
                safety_profile=[],
                relevant_facts=[],
                active_case=None,
                reference_turns=[],
                excluded_reason="storage_failure",
            )

    direct_answer_requested = turn["direct_answer_requested"]
    session.add_message("user", question)

    if context_runtime is not None and context_runtime.resolution.ambiguous:
        clarification = context_runtime.resolution.clarification
        session.add_message("assistant", clarification)
        _persist_final_turn(
            session,
            session_id,
            question,
            clarification,
            trace_id,
            context_runtime=context_runtime,
            analysis=analysis,
        )
        _log_timing(trace_id, "total", request_start, outcome="subject_clarification")
        return clarification

    if (
        context_runtime is not None
        and context_runtime.bundle.excluded_reason == "conflicting_facts"
    ):
        if analysis.get("profile_candidates"):
            context_runtime.state.pending_clarification = None
        else:
            clarification = context_runtime.state.pending_clarification.question
            session.add_message("assistant", clarification)
            _persist_final_turn(
                session,
                session_id,
                question,
                clarification,
                trace_id,
                context_runtime=context_runtime,
                analysis=analysis,
            )
            _log_timing(trace_id, "total", request_start, outcome="profile_clarification")
            return clarification

    if label == "greeting_other" and intent != "emergency":
        session.add_message("assistant", GREETING_REPLY)
        stage_start = time.perf_counter()
        try:
            save_session(session)
        except Exception:
            _log_failed_timing(trace_id, "persist", stage_start)
            raise
        _log_timing(trace_id, "persist", stage_start)
        _log_timing(trace_id, "total", request_start, outcome="greeting")
        return GREETING_REPLY

    if label != "clarification_answer" and intent != "emergency":
        rewrite = analysis["rewrite"]
        rewritten = rewrite["rewritten"]
        clarification = rewrite["clarification"] if not rewrite["confident"] else ""
        if clarification:
            session.add_message("assistant", clarification)
            stage_start = time.perf_counter()
            try:
                save_session(session)
            except Exception:
                _log_failed_timing(trace_id, "persist", stage_start)
                raise
            _log_timing(trace_id, "persist", stage_start)
            _log_timing(trace_id, "total", request_start, outcome="clarify_question")
            return clarification
    elif label == "clarification_answer":
        rewritten = question
    else:
        rewritten = analysis["rewrite"]["rewritten"]

    active_flow = bool(session.answered_questions or session.clarification_queue)
    route_stage_start = time.perf_counter()
    route_label = label
    try:
        decision = apply_mode_policy(mode, intent, active_flow=active_flow)
    except Exception:
        _log_failed_timing(trace_id, "route", route_stage_start, label=route_label)
        raise
    _record_mode_decision(mode, intent, decision, question)
    route_label = decision.route_label or label
    if route_label == "informational" and health_insurance_detected:
        route_label = "health_insurance"
        if meta is not None:
            meta["health_insurance_detector"] = "route_fallback"
    meta = _meta()
    if meta is not None and meta.get("_collect_graph"):
        meta["_graph_route"] = {
            "input": {
                "label": label,
                "intent": intent,
                "mode": mode,
                "active_flow": active_flow,
            },
            "decision": asdict(decision),
        }
    if not decision.allow:
        reply = decision.reply or ""
        session.add_message("assistant", reply)
        _log_timing(trace_id, "route", route_stage_start, label=route_label)
        _persist_final_turn(
            session,
            session_id,
            question,
            reply,
            trace_id,
            context_runtime=context_runtime,
            analysis=analysis,
        )
        _log_timing(
            trace_id,
            "total",
            request_start,
            outcome=f"mode_suggest_{decision.suggest_mode or 'blocked'}",
        )
        return reply

    _emit_node_event("route", "ok", None)
    if (
        route_label != "emergency"
        and label in ("diagnostic", "informational", "clarification_answer")
        and not (
            label == "informational"
            and intent == "pure_info"
            and drug_policy.reason == "source_grounded_drug_info"
        )
    ):
        stage_start = time.perf_counter()
        try:
            _ingest_entities(analysis["entities"], session)
        except Exception:
            _log_failed_timing(
                trace_id, "entity_ingest", stage_start,
                symptoms=len(analysis["entities"]["symptoms"]),
                medications=len(analysis["entities"]["medications"]),
            )
            raise
        _log_timing(
            trace_id, "entity_ingest", stage_start,
            symptoms=len(analysis["entities"]["symptoms"]),
            medications=len(analysis["entities"]["medications"]),
        )

    force_answer = (
        decision.force_answer
        if decision.force_answer is not None
        else direct_answer_requested
    )
    use_patient_context_override = decision.use_patient_context
    if CONVERSATION_CONTEXT_ENABLED and context_runtime is not None:
        use_patient_context_override = context_runtime.bundle.excluded_reason is None
    try:
        reply = _route(
            route_label,
            session,
            question,
            rewritten,
            force_answer,
            trace_id,
            use_patient_context_override=use_patient_context_override,
            emergency_subject=format_subject_address(
                (analysis.get("context") or {}).get("subject")
            ),
            emergency_red_flags=emergency_red_flags,
            emergency_context=context,
            on_preliminary_reply=on_preliminary_reply,
            use_emergency_rag=use_emergency_rag,
            turn_intent=intent,
            evidence_plan=evidence_plan,
        )
    except Exception:
        log.exception("Pipeline route failed trace=%s", trace_id)
        meta = _meta()
        if meta is not None and meta.get("_graph_error_stage"):
            _log_timing(trace_id, "route", route_stage_start, label=route_label)
        else:
            _log_failed_timing(trace_id, "route", route_stage_start, label=route_label)
        if meta is not None:
            meta["error"] = "technical_error"
        session.add_message("assistant", TECHNICAL_ERROR_REPLY)
        _persist_final_turn(
            session,
            session_id,
            question,
            TECHNICAL_ERROR_REPLY,
            trace_id,
            context_runtime=context_runtime,
        )
        _log_timing(trace_id, "total", request_start, outcome="technical_error")
        return TECHNICAL_ERROR_REPLY
    _log_timing(trace_id, "route", route_stage_start, label=route_label)

    session.add_message("assistant", reply)
    _persist_final_turn(
        session,
        session_id,
        question,
        reply,
        trace_id,
        context_runtime=context_runtime,
        analysis=analysis,
    )
    _log_timing(trace_id, "total", request_start, outcome=route_label)
    return reply


def answer(
    question: str,
    session_id: str = "default",
    owner_id: str | None = None,
    mode: str = "auto",
    on_preliminary_reply: Callable[[str], None] | None = None,
    use_emergency_rag: bool | None = None,
) -> str:
    trace_id = uuid.uuid4().hex[:8]
    request_start = time.perf_counter()
    try:
        return _answer_inner(
            question,
            session_id,
            trace_id,
            request_start,
            owner_id=owner_id,
            mode=mode,
            on_preliminary_reply=on_preliminary_reply,
            use_emergency_rag=use_emergency_rag,
        )
    except Exception:
        log.exception("Pipeline failed trace=%s", trace_id)
        _log_timing(trace_id, "total", request_start, outcome="technical_error")
        return TECHNICAL_ERROR_REPLY


def answer_with_choices(
    question: str,
    session_id: str = "default",
    owner_id: str | None = None,
    mode: str = "auto",
) -> ChatReply:
    meta: dict[str, Any] = {}
    previous_meta = _meta()
    _META_LOCAL.current = meta
    try:
        answer_kwargs = {"session_id": session_id}
        if owner_id:
            answer_kwargs["owner_id"] = owner_id
        if normalize_mode(mode) == "auto":
            reply = answer(question, **answer_kwargs)
        else:
            reply = answer(question, mode=mode, **answer_kwargs)
    finally:
        _META_LOCAL.current = previous_meta
    choices = suggested_choices(reply)
    selection_mode = suggested_selection_mode(reply) if choices else "single"
    doctor_offer = (
        bool(meta.get("doctor_offer"))
        if not choices and not meta.get("suggest_mode")
        else False
    )
    return ChatReply(
        reply,
        choices,
        selection_mode,
        meta.get("suggest_mode"),
        meta.get("retry_question"),
        doctor_offer,
        meta.get("doctor_specialty") if doctor_offer else None,
    )


def _build_graph_nodes(
    meta: dict[str, Any],
    question: str,
    session_id: str,
    mode: str,
    reply: str,
) -> list[dict[str, Any]]:
    timings = {
        entry["stage"]: entry
        for entry in meta.get("timings", [])
        if isinstance(entry, dict) and entry.get("stage")
    }
    analysis = meta.pop("_graph_turn_analysis", None)
    route_data = meta.pop("_graph_route", None)
    diagnostic_snap = meta.pop("_graph_diagnostic", None)
    error_stage = meta.pop("_graph_error_stage", None)
    error_stages = set()
    raw_error_stages = meta.pop("_graph_error_stages", [])
    if isinstance(raw_error_stages, list):
        error_stages.update(
            stage for stage in raw_error_stages if isinstance(stage, str)
        )
    if isinstance(error_stage, str):
        error_stages.add(error_stage)
    meta.pop("_collect_graph", None)
    rewrite_query = meta.get("rewrite_query")
    kg_context = meta.get("kg_context")
    retrieval_debug = meta.get("retrieval_debug") or {}
    retrieval_timings = (
        retrieval_debug.get("timings_ms")
        if (
            isinstance(retrieval_debug, dict)
            and isinstance(retrieval_debug.get("timings_ms"), dict)
        )
        else {}
    )
    if error_stage is None and isinstance(retrieval_debug, dict):
        debug_error_stage = retrieval_debug.get("error_stage")
        if isinstance(debug_error_stage, str):
            error_stage = debug_error_stage
    if isinstance(retrieval_debug, dict):
        debug_error_stage = retrieval_debug.get("error_stage")
        if isinstance(debug_error_stage, str):
            error_stages.add(debug_error_stage)
    retrieval_query = (
        retrieval_debug.get("query")
        if isinstance(retrieval_debug, dict) else None
    ) or (
        rewrite_query.get("rewritten")
        if isinstance(rewrite_query, dict)
        else question
    )
    nodes: list[dict[str, Any]] = []

    def add_node(
        node_id: str,
        label: str,
        *,
        present: bool,
        timing_stage: str | None = None,
        input_data: Any = None,
        output_data: Any = None,
        raw_data: Any = None,
        ms: float | None = None,
    ) -> None:
        timing = timings.get(timing_stage or "")
        failed = (
            node_id in error_stages
            or bool(timing and timing.get("fields", {}).get("failed"))
        )
        nodes.append({
            "id": node_id,
            "label": label,
            "status": "error" if failed else ("success" if present else "skipped"),
            "ms": timing.get("ms") if ms is None and timing else ms,
            "input": input_data,
            "output": output_data,
            "raw": raw_data,
        })

    request_data = {
        "question": question,
        "session_id": session_id,
        "mode": meta.get("mode", normalize_mode(mode)),
    }
    turn_output = analysis.get("turn") if isinstance(analysis, dict) else None
    load_timing = timings.get("load_session")
    persist_timing = timings.get("persist")
    add_node("input", "Input", present=True, ms=0.0,
             input_data=request_data, raw_data=request_data)
    add_node(
        "load_session", "Load session", present=load_timing is not None,
        timing_stage="load_session", input_data={"session_id": session_id},
        output_data={
            "loaded": bool(load_timing)
            and not bool(load_timing.get("fields", {}).get("failed")),
        },
        raw_data=load_timing,
    )
    add_node(
        "preflight", "Preflight", present="preflight" in timings,
        timing_stage="preflight",
        input_data={"question": question, "session_id": session_id},
        output_data={"short_circuit": meta.get("outcome") == "short_circuit"},
        raw_data=timings.get("preflight"),
    )
    add_node(
        "turn_analysis", "Turn analysis", present=analysis is not None,
        timing_stage="turn_analysis", input_data={"question": question},
        output_data=turn_output, raw_data=analysis,
    )
    add_node(
        "rewrite", "Query rewrite", present=rewrite_query is not None,
        input_data={"question": question},
        output_data=rewrite_query, raw_data=rewrite_query,
    )
    add_node(
        "route", "Route",
        present=route_data is not None or "route_label" in meta,
        timing_stage="route",
        input_data=route_data.get("input") if route_data else None,
        output_data={
            "route_label": meta.get("route_label"),
            "suggest_mode": meta.get("suggest_mode"),
        },
        raw_data=route_data,
    )
    add_node(
        "diagnostic_general_triage", "Triage",
        present="diagnostic_general_triage" in timings,
        timing_stage="diagnostic_general_triage",
        input_data={"question": question},
        output_data=(diagnostic_snap or {}).get("events"),
        raw_data=timings.get("diagnostic_general_triage"),
    )
    add_node(
        "diagnostic_rank", "Diagnostic rank",
        present="diagnostic_rank" in timings,
        timing_stage="diagnostic_rank",
        input_data={"symptoms": (diagnostic_snap or {}).get("symptoms")},
        output_data={"candidate_diseases": (diagnostic_snap or {}).get("candidate_diseases")},
        raw_data=timings.get("diagnostic_rank"),
    )
    add_node(
        "diagnostic_clarification", "Clarification",
        present="diagnostic_clarification" in timings,
        timing_stage="diagnostic_clarification",
        input_data={"clarification_queue": (diagnostic_snap or {}).get("clarification_queue")},
        output_data={
            "answered_questions": (diagnostic_snap or {}).get("answered_questions"),
            "asked": (diagnostic_snap or {}).get("events", {}).get("asked"),
            "clarification_plan_started": (diagnostic_snap or {}).get("clarification_plan_started"),
            "clarification_parse_failures": (diagnostic_snap or {}).get("clarification_parse_failures"),
        },
        raw_data=timings.get("diagnostic_clarification"),
    )
    add_node(
        "clarification_parse", "Parse answer",
        present="clarification_parse" in timings,
        timing_stage="clarification_parse",
        input_data={"last_asked": (diagnostic_snap or {}).get("events", {}).get("last_asked")},
        output_data={
            "parsed_answers": (diagnostic_snap or {}).get("events", {}).get("parsed_answers"),
            "symptoms": (diagnostic_snap or {}).get("symptoms"),
        },
        raw_data=timings.get("clarification_parse"),
    )
    add_node(
        "entity_ingest", "Entity ingest",
        present="entity_ingest" in timings,
        timing_stage="entity_ingest",
        input_data=(
            analysis.get("entities")
            if isinstance(analysis, dict)
            else None
        ),
        output_data=(
            timings.get("entity_ingest", {}).get("fields")
            if isinstance(timings.get("entity_ingest"), dict)
            else None
        ),
        raw_data=timings.get("entity_ingest"),
    )
    add_node(
        "kg_search", "KG search", present="kg_context" in meta,
        timing_stage="kg_search", input_data={"query": retrieval_query},
        output_data=kg_context, raw_data=kg_context,
    )
    for node_id, label, key in (
        ("dense_search", "Dense search", "dense_hits"),
        ("sparse_search", "Sparse search", "sparse_hits"),
        ("fusion", "RRF fusion", "fused_hits"),
        ("rerank", "Rerank", "reranked_hits"),
    ):
        hits = retrieval_debug.get(key) if isinstance(retrieval_debug, dict) else None
        stage_ms = retrieval_timings.get(node_id)
        add_node(
            node_id, label,
            present=isinstance(retrieval_debug, dict) and key in retrieval_debug,
            input_data={"query": retrieval_query},
            output_data=hits, raw_data=hits,
            ms=float(stage_ms) if isinstance(stage_ms, (int, float)) else None,
        )
    generate_timing = timings.get("generate")
    generate_failed = (
        "generate" in error_stages
        or bool(
            generate_timing
            and generate_timing.get("fields", {}).get("failed")
        )
    )
    add_node(
        "generate", "Generate", present="generate" in timings,
        timing_stage="generate",
        input_data={
            "question": (
                rewrite_query.get("rewritten")
                if isinstance(rewrite_query, dict) else question
            ),
            "retrieved_count": len(meta.get("retrieved", [])),
            "has_kg_context": kg_context is not None,
        },
        output_data=None if generate_failed else {"reply": reply},
        raw_data={"usage": meta.get("usage", [])},
    )
    add_node(
        "persist", "Persist", present=persist_timing is not None,
        timing_stage="persist", input_data={"session_id": session_id},
        output_data={
            "saved": bool(persist_timing)
            and not bool(persist_timing.get("fields", {}).get("failed")),
        },
        raw_data=persist_timing,
    )
    add_node(
        "total", "Total",
        present="total" in timings or "latency_ms_total" in meta,
        timing_stage="total",
        output_data={
            "outcome": meta.get("outcome"),
            "latency_ms_total": meta.get("latency_ms_total"),
        },
        raw_data={"outcome": meta.get("outcome"), "error": meta.get("error")},
    )
    return nodes


def answer_with_meta(
    question: str,
    session_id: str = "default",
    owner_id: str | None = None,
    mode: str = "auto",
    on_preliminary_reply: Callable[[str], None] | None = None,
    use_emergency_rag: bool | None = None,
) -> tuple[str, dict[str, Any]]:
    """Like `answer`, but also returns a meta dict capturing usage, retrieved
    hits, and per-stage latency for the turn. Eval-only entry point — channels
    should keep using `answer` since the meta payload changes per turn.
    """
    trace_id = uuid.uuid4().hex[:8]
    request_start = time.perf_counter()
    meta: dict[str, Any] = {"trace_id": trace_id, "_collect_graph": True}
    _META_LOCAL.current = meta
    try:
        try:
            reply = _answer_inner(
                question,
                session_id,
                trace_id,
                request_start,
                owner_id=owner_id,
                mode=mode,
                on_preliminary_reply=on_preliminary_reply,
                use_emergency_rag=use_emergency_rag,
            )
        except Exception:
            log.exception("Pipeline failed trace=%s", trace_id)
            _log_timing(trace_id, "total", request_start, outcome="technical_error")
            reply = TECHNICAL_ERROR_REPLY
            meta["error"] = "technical_error"
        meta["latency_ms_total"] = round(elapsed_ms(request_start), 2)
        meta["graph_nodes"] = _build_graph_nodes(
            meta,
            question,
            session_id,
            mode,
            reply,
        )
        return reply, meta
    finally:
        _META_LOCAL.current = None
