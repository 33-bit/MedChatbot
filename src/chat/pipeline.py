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
import time
import uuid
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import asdict

from src.chat.diagnosis.differential import (
    build_clarification,
    discriminative_symptoms,
    parse_clarification_answer,
    rank_candidates,
    should_ask_clarification,
    symptom_catalog,
)
from src.chat.diagnosis.entities import normalize_entities
from src.chat.diagnosis.flow import direct_diagnostic_prompt
from src.chat.guards.guardrail import VERDICT_REPLIES, regex_check
from src.chat.guards.quota import check_both as check_llm_quota
from src.chat.llm.analyzer import analyze_turn
from src.chat.llm.generator import generate
from src.chat.preflight import RATE_LIMIT_MSG, preflight
from src.chat.replies import TECHNICAL_ERROR_REPLY
from src.chat.retrieval import (
    Hit,
    format_kg_context,
    hybrid_search,
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
from src.chat.timing import log_trace_timing
from src.config import CLARIFICATION_BATCH_SIZE, RERANK_TOP_K

GREETING_REPLY = ("Xin chào! Tôi là trợ lý y tế. Bạn có thể mô tả triệu chứng "
                  "hoặc hỏi về bệnh/thuốc cụ thể.")
log = logging.getLogger(__name__)
_RETRIEVAL_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="retrieval")


def _log_timing(trace_id: str, stage: str, start: float, **fields) -> None:
    log_trace_timing(log, "pipeline", trace_id, stage, start, **fields)


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


# ---------- Phase C: route handlers ----------

def _handle_diagnostic(
    session: PatientSession,
    question: str,
    trace_id: str,
    force_answer: bool = False,
) -> str:
    """New symptoms reported. Rank diseases, maybe ask batch clarification."""
    symptom_ids = [s["symptom_id"] for s in session.symptoms if s.get("symptom_id")]
    stage_start = time.perf_counter()
    candidates = rank_candidates(symptom_ids)
    session.candidate_diseases = candidates
    _log_timing(trace_id, "diagnostic_rank", stage_start,
                symptoms=len(symptom_ids), candidates=len(candidates))

    stage_start = time.perf_counter()
    if not force_answer and should_ask_clarification(candidates):
        cand_ids = [c["disease_id"] for c in candidates]
        known = [s for s in symptom_ids if not s.startswith("raw:")]
        next_symptoms = discriminative_symptoms(cand_ids, known + session.answered_questions)
        if next_symptoms:
            session.answered_questions.extend(next_symptoms)
            names = ", ".join(c["name"] for c in candidates[:5] if c.get("name"))
            header = f"Dựa trên triệu chứng bạn nêu, tôi đang cân nhắc các bệnh: {names}."
            _log_timing(trace_id, "diagnostic_clarification", stage_start,
                        asked=len(next_symptoms))
            return header + "\n\n" + build_clarification(next_symptoms)
    _log_timing(trace_id, "diagnostic_clarification", stage_start,
                asked=0, force_answer=force_answer)

    if force_answer:
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
    """Parse batched clarification reply, update slots, then re-narrow."""
    asked_ids = session.answered_questions[-CLARIFICATION_BATCH_SIZE:]
    catalog = symptom_catalog()
    asked = [
        {"symptom_id": sid, "name": catalog.get(sid, {}).get("name_vi", sid)}
        for sid in asked_ids
    ]
    stage_start = time.perf_counter()
    parsed_answers = parse_clarification_answer(asked, user_answer)
    _log_timing(trace_id, "clarification_parse", stage_start,
                asked=len(asked), parsed=len(parsed_answers))
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
    kg_future = _RETRIEVAL_EXECUTOR.submit(_load_kg_context, search_question, trace_id)
    hits_future = _RETRIEVAL_EXECUTOR.submit(_load_hybrid_hits, search_question, trace_id)
    done, pending = wait((kg_future, hits_future), return_when=FIRST_EXCEPTION)
    for future in done:
        if future.exception() is not None:
            for pending_future in pending:
                pending_future.cancel()
            raise future.exception()
    kg_text = kg_future.result()
    hits = hits_future.result()
    _log_timing(trace_id, "parallel_retrieval", stage_start,
                hits=len(hits), kg_chars=len(kg_text))

    patient_dict = asdict(session) if use_patient_context else None
    stage_start = time.perf_counter()
    reply = generate(question, hits, kg_text=kg_text, patient=patient_dict)
    _log_timing(trace_id, "generate", stage_start, chars=len(reply))
    return reply


def _route(label: str, session: PatientSession, question: str,
           rewritten: str, force_answer: bool, trace_id: str) -> str:
    if label == "clarification_answer":
        return _handle_clarification_answer(session, question, trace_id, force_answer)
    if label == "diagnostic":
        return _handle_diagnostic(session, rewritten, trace_id)
    # informational: always answer; personalize if we have symptoms
    return _handle_informational(
        session,
        rewritten,
        trace_id,
        use_patient_context=bool(session.symptoms),
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


# ---------- Entry point ----------

def _answer_inner(
    question: str,
    session_id: str,
    trace_id: str,
    request_start: float,
) -> str:
    question = (question or "").strip()
    if not question:
        return "Bạn hãy đặt câu hỏi cụ thể nhé."

    stage_start = time.perf_counter()
    short_circuit = preflight(
        question,
        session_id,
        _guardrail_reply,
        check_rate_limit,
        regex_check,
        check_llm_quota,
    )
    _log_timing(trace_id, "preflight", stage_start)
    if short_circuit is not None:
        _log_timing(trace_id, "total", request_start, outcome="short_circuit")
        return short_circuit

    stage_start = time.perf_counter()
    session = load_session(session_id)
    _log_timing(trace_id, "load_session", stage_start)

    stage_start = time.perf_counter()
    analysis = analyze_turn(
        question,
        last_bot_message=_last_bot_message(session),
        history=session.conversation,
    )
    _log_timing(trace_id, "turn_analysis", stage_start,
                verdict=analysis["guardrail"]["verdict"],
                label=analysis["turn"]["label"],
                direct_answer=analysis["turn"]["direct_answer_requested"])
    guard = analysis["guardrail"]
    if guard["verdict"] != "allow":
        _log_timing(trace_id, "total", request_start, outcome="guardrail")
        return _guardrail_reply(session_id, question, guard["verdict"])

    turn = analysis["turn"]
    label = turn["label"]
    direct_answer_requested = turn["direct_answer_requested"]
    if direct_answer_requested:
        label = "clarification_answer"
    session.add_message("user", question)

    if label == "greeting_other":
        session.add_message("assistant", GREETING_REPLY)
        stage_start = time.perf_counter()
        save_session(session)
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
            save_session(session)
            _log_timing(trace_id, "persist", stage_start)
            _log_timing(trace_id, "total", request_start, outcome="clarify_question")
            return clarification
    else:
        rewritten = question

    if label in ("diagnostic", "informational"):
        stage_start = time.perf_counter()
        _ingest_entities(analysis["entities"], session)
        _log_timing(trace_id, "entity_ingest", stage_start,
                    symptoms=len(analysis["entities"]["symptoms"]),
                    medications=len(analysis["entities"]["medications"]))

    stage_start = time.perf_counter()
    reply = _route(
        label,
        session,
        question,
        rewritten,
        direct_answer_requested,
        trace_id,
    )
    _log_timing(trace_id, "route", stage_start, label=label)

    session.add_message("assistant", reply)
    _persist_final_turn(session, session_id, question, reply, trace_id)
    _log_timing(trace_id, "total", request_start, outcome=label)
    return reply


def answer(question: str, session_id: str = "default") -> str:
    trace_id = uuid.uuid4().hex[:8]
    request_start = time.perf_counter()
    try:
        return _answer_inner(question, session_id, trace_id, request_start)
    except Exception:
        log.exception("Pipeline failed trace=%s session=%s", trace_id, session_id)
        _log_timing(trace_id, "total", request_start, outcome="technical_error")
        return TECHNICAL_ERROR_REPLY
