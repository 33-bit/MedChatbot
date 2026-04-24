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

from dataclasses import asdict

from src.chat.diagnosis.differential import (
    build_clarification,
    discriminative_symptoms,
    parse_clarification_answer,
    rank_candidates,
    should_ask_clarification,
    symptom_catalog,
)
from src.chat.diagnosis.entities import extract_entities
from src.chat.guards.classifier import classify
from src.chat.guards.guardrail import check as guardrail_check
from src.chat.guards.quota import check_both as check_llm_quota
from src.chat.guards.rewriter import rewrite
from src.chat.llm.generator import generate
from src.chat.retrieval import (
    format_kg_context,
    hybrid_search,
    kg_search,
)
from src.chat.storage.cache import cache_get, cache_put
from src.chat.storage.session import (
    PatientSession,
    check_rate_limit,
    load_session,
    log_consultation,
    save_profile,
    save_session,
)
from src.config import RERANK_TOP_K

RATE_LIMIT_MSG = "Bạn đang hỏi quá nhanh. Vui lòng chờ 1 phút rồi thử lại."
GREETING_REPLY = ("Xin chào! Tôi là trợ lý y tế. Bạn có thể mô tả triệu chứng "
                  "hoặc hỏi về bệnh/thuốc cụ thể.")


# ---------- Phase A: input validation ----------

def _preflight(question: str, session_id: str) -> str | None:
    """Run cheap checks. Return a short-circuit reply, or None to continue."""
    if not check_rate_limit(session_id):
        return RATE_LIMIT_MSG

    guard = guardrail_check(question)
    if guard["verdict"] != "allow":
        if guard["verdict"] == "abuse":
            return ""  # silent drop
        # Persist guardrail bounce as a conversation turn so follow-ups have context.
        session = load_session(session_id)
        session.add_message("user", question)
        session.add_message("assistant", guard["reply"])
        save_session(session)
        return guard["reply"]

    allowed, quota_msg = check_llm_quota(session_id)
    if not allowed:
        return quota_msg

    return None


# ---------- Phase B helpers ----------

def _last_bot_message(session: PatientSession) -> str:
    for turn in reversed(session.conversation):
        if turn.get("role") == "assistant":
            return turn.get("content", "")
    return ""


def _maybe_rewrite(question: str, session: PatientSession) -> tuple[str, str | None]:
    """Rewrite the query using history. Returns (rewritten, clarification_or_None)."""
    rw = rewrite(question, session.conversation[:-1])
    if not rw["confident"] and rw["clarification"]:
        return question, rw["clarification"]
    return rw["rewritten"], None


def _ingest_entities(question: str, session: PatientSession) -> None:
    ents = extract_entities(question)
    for s in ents["symptoms"]:
        session.upsert_symptom(s)
    for d in ents["medications"]:
        session.add_medication(d)


# ---------- Phase C: route handlers ----------

def _handle_diagnostic(session: PatientSession, question: str) -> str:
    """New symptoms reported. Rank diseases, maybe ask batch clarification."""
    symptom_ids = [s["symptom_id"] for s in session.symptoms if s.get("symptom_id")]
    candidates = rank_candidates(symptom_ids)
    session.candidate_diseases = candidates

    if should_ask_clarification(candidates):
        cand_ids = [c["disease_id"] for c in candidates]
        known = [s for s in symptom_ids if not s.startswith("raw:")]
        next_symptoms = discriminative_symptoms(cand_ids, known + session.answered_questions)
        if next_symptoms:
            session.answered_questions.extend(next_symptoms)
            names = ", ".join(c["name"] for c in candidates[:5] if c.get("name"))
            header = f"Dựa trên triệu chứng bạn nêu, tôi đang cân nhắc các bệnh: {names}."
            return header + "\n\n" + build_clarification(next_symptoms)

    # Enough narrowing — answer with RAG + KG and patient context
    return _handle_informational(session, question, use_patient_context=True)


def _handle_clarification_answer(session: PatientSession, user_answer: str) -> str:
    """Parse batched clarification reply, update slots, then re-narrow."""
    asked_ids = session.answered_questions[-4:]
    catalog = symptom_catalog()
    asked = [
        {"symptom_id": sid, "name": catalog.get(sid, {}).get("name_vi", sid)}
        for sid in asked_ids
    ]
    for r in parse_clarification_answer(asked, user_answer):
        sid = r.get("symptom_id")
        if not sid or r.get("present") != "yes":
            continue
        entry = {"symptom_id": sid,
                 "name": catalog.get(sid, {}).get("name_vi", sid)}
        for k in ("onset", "severity", "pattern", "associated"):
            if r.get(k):
                entry[k] = r[k]
        session.upsert_symptom(entry)

    return _handle_diagnostic(session, user_answer)


def _handle_informational(
    session: PatientSession,
    question: str,
    use_patient_context: bool = False,
    cacheable: bool = False,
) -> str:
    if cacheable:
        cached = cache_get(question)
        if cached:
            return cached

    kg_text = format_kg_context(kg_search(question))
    hits = hybrid_search(question, top_k=RERANK_TOP_K)
    patient_dict = asdict(session) if use_patient_context else None
    reply = generate(question, hits, kg_text=kg_text, patient=patient_dict)

    if cacheable:
        cache_put(question, reply)
    return reply


def _route(label: str, session: PatientSession, question: str,
           rewritten: str, cacheable: bool) -> str:
    if label == "clarification_answer":
        return _handle_clarification_answer(session, question)
    if label == "diagnostic":
        return _handle_diagnostic(session, rewritten)
    # informational: always answer; personalize if we have symptoms
    return _handle_informational(
        session,
        rewritten,
        use_patient_context=bool(session.symptoms),
        cacheable=cacheable and not session.symptoms,
    )


# ---------- Entry point ----------

def answer(question: str, session_id: str = "default") -> str:
    question = (question or "").strip()
    if not question:
        return "Bạn hãy đặt câu hỏi cụ thể nhé."

    short_circuit = _preflight(question, session_id)
    if short_circuit is not None:
        return short_circuit

    session = load_session(session_id)
    turn = classify(question, last_bot_message=_last_bot_message(session))
    label = turn["label"]
    session.add_message("user", question)

    if label == "greeting_other":
        session.add_message("assistant", GREETING_REPLY)
        save_session(session)
        return GREETING_REPLY

    if label != "clarification_answer":
        rewritten, clarification = _maybe_rewrite(question, session)
        if clarification:
            session.add_message("assistant", clarification)
            save_session(session)
            return clarification
    else:
        rewritten = question

    if label in ("diagnostic", "informational"):
        _ingest_entities(question, session)

    reply = _route(label, session, question, rewritten, turn["cacheable"])

    session.add_message("assistant", reply)
    save_session(session)
    save_profile(session)
    log_consultation(session_id, question, reply)
    return reply
