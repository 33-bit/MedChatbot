"""
pipeline.py
-----------
Orchestration per user turn.

Phases (short-circuit on the first failure):
  A. Input validation      — rate limit, guardrail, LLM quota
  B. State load            — load session from Redis, classify turn, extract entities
  C. Route                 — greeting / clarification_answer / diagnostic / informational
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
from typing import Any

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
from src.chat.guards.guardrail import VERDICT_REPLIES, regex_check
from src.chat.guards.quota import check_both as check_llm_quota
from src.chat.llm.analyzer import analyze_turn
from src.chat.llm.generator import generate
from src.chat.mode_policy import (
    ModeDecision,
    apply_mode_policy,
    normalize_intent,
    normalize_mode,
)
from src.chat.preflight import RATE_LIMIT_MSG, preflight
from src.chat.replies import ChatReply, TECHNICAL_ERROR_REPLY
from src.chat.retrieval import (
    Hit,
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
from src.chat.timing import elapsed_ms, log_trace_timing
from src.config import RERANK_TOP_K

GREETING_REPLY = ("Xin chào! Tôi là trợ lý y tế. Bạn có thể mô tả triệu chứng "
                  "hoặc hỏi về bệnh/thuốc cụ thể.")
GENERAL_TRIAGE_MARKER = "general:triage"
_START_CHOICES = {"bat dau", "duoc", "ok", "okay", "tiep", "tiep tuc"}
_YES_CHOICES = {"co"}
_NO_CHOICES = {"khong"}
_UNKNOWN_CHOICES = {"khong ro", "khong biet", "chua ro", "chua biet"}
_ANSWER_NOW_CHOICES = {"tra loi luon", "cu tra loi", "tra loi di"}
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
    normalized = unicodedata.normalize("NFD", text.strip().casefold())
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


def _load_hybrid_hits(question: str, trace_id: str) -> list[Hit]:
    stage_start = time.perf_counter()
    hits = hybrid_search(question, top_k=RERANK_TOP_K)
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
) -> tuple[list[Hit], dict[str, Any]]:
    stage_start = time.perf_counter()

    def _on_stage(stage, status, ms):
        _emit_node_event(stage, status, ms)

    hits, retrieval_debug = hybrid_search_with_debug(
        question,
        top_k=RERANK_TOP_K,
        on_stage=_on_stage,
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
        return build_general_triage_prompt(session)

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
        )
    # Enough narrowing — answer with RAG + KG and patient context
    return _handle_informational(session, question, trace_id, use_patient_context=True)


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
        hits_future = executor.submit(_with_sink, hits_loader, search_question, trace_id)
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

    patient_dict = asdict(session) if use_patient_context else None
    stage_start = time.perf_counter()
    try:
        if _meta() is not None:
            reply, gen_meta = generate(
                question, hits, kg_text=kg_text, patient=patient_dict, return_meta=True,
            )
            _record_usage("generator", gen_meta.get("usage"), model=gen_meta.get("model"))
            _meta()["doctor_offer"] = bool(gen_meta.get("doctor_handoff_recommended"))
            _meta()["doctor_specialty"] = gen_meta.get("doctor_specialty")
        else:
            reply = generate(question, hits, kg_text=kg_text, patient=patient_dict)
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
) -> str:
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
    )


def _persist_final_turn(
    session: PatientSession,
    session_id: str,
    question: str,
    reply: str,
    trace_id: str,
) -> None:
    stage_start = time.perf_counter()
    try:
        save_session(session)
        save_profile(session)
        log_consultation(session_id, question, reply)
    except Exception:
        log.exception("Pipeline persist failed trace=%s session=%s", trace_id, session_id)
        _log_timing(trace_id, "persist", stage_start, failed=True)
        return
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
    mode: str = "auto",
) -> str:
    question = (question or "").strip()
    if not question:
        return "Bạn hãy đặt câu hỏi cụ thể nhé."
    mode = normalize_mode(mode)
    _emit_node_event("input", "ok", 0.0)

    stage_start = time.perf_counter()
    try:
        session = load_session(session_id)
    except Exception:
        _log_failed_timing(trace_id, "load_session", stage_start)
        raise
    _log_timing(trace_id, "load_session", stage_start)

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
        if is_clarification_choice:
            analysis = _clarification_choice_analysis(question)
        else:
            analysis = analyze_turn(
                question,
                last_bot_message=_last_bot_message(session),
                history=session.conversation,
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

    direct_answer_requested = turn["direct_answer_requested"]
    session.add_message("user", question)

    if label == "greeting_other":
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

    if label != "clarification_answer":
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
    else:
        rewritten = question

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
        _persist_final_turn(session, session_id, question, reply, trace_id)
        _log_timing(
            trace_id,
            "total",
            request_start,
            outcome=f"mode_suggest_{decision.suggest_mode or 'blocked'}",
        )
        return reply

    _emit_node_event("route", "ok", None)
    if label in ("diagnostic", "informational", "clarification_answer"):
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
    try:
        reply = _route(
            route_label,
            session,
            question,
            rewritten,
            force_answer,
            trace_id,
            use_patient_context_override=decision.use_patient_context,
        )
    except Exception:
        log.exception("Pipeline route failed trace=%s session=%s", trace_id, session_id)
        meta = _meta()
        if meta is not None and meta.get("_graph_error_stage"):
            _log_timing(trace_id, "route", route_stage_start, label=route_label)
        else:
            _log_failed_timing(trace_id, "route", route_stage_start, label=route_label)
        if meta is not None:
            meta["error"] = "technical_error"
        session.add_message("assistant", TECHNICAL_ERROR_REPLY)
        _persist_final_turn(session, session_id, question, TECHNICAL_ERROR_REPLY, trace_id)
        _log_timing(trace_id, "total", request_start, outcome="technical_error")
        return TECHNICAL_ERROR_REPLY
    _log_timing(trace_id, "route", route_stage_start, label=route_label)

    session.add_message("assistant", reply)
    _persist_final_turn(session, session_id, question, reply, trace_id)
    _log_timing(trace_id, "total", request_start, outcome=route_label)
    return reply


def answer(question: str, session_id: str = "default", mode: str = "auto") -> str:
    trace_id = uuid.uuid4().hex[:8]
    request_start = time.perf_counter()
    try:
        return _answer_inner(question, session_id, trace_id, request_start, mode=mode)
    except Exception:
        log.exception("Pipeline failed trace=%s session=%s", trace_id, session_id)
        _log_timing(trace_id, "total", request_start, outcome="technical_error")
        return TECHNICAL_ERROR_REPLY


def answer_with_choices(
    question: str,
    session_id: str = "default",
    mode: str = "auto",
) -> ChatReply:
    meta: dict[str, Any] = {}
    previous_meta = _meta()
    _META_LOCAL.current = meta
    try:
        if normalize_mode(mode) == "auto":
            reply = answer(question, session_id=session_id)
        else:
            reply = answer(question, session_id=session_id, mode=mode)
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
    mode: str = "auto",
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
            reply = _answer_inner(question, session_id, trace_id, request_start, mode=mode)
        except Exception:
            log.exception("Pipeline failed trace=%s session=%s", trace_id, session_id)
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
