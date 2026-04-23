"""
pipeline.py
-----------
Orchestration per user turn:

  0. Rate limit check
  1. Load patient session from Redis
  2. Classify turn (diagnostic / informational / clarification_answer / greeting_other)
  3. Query rewrite (use fast LLM, ask clarification if not confident)
  4. Extract entities → update session
  5. Route:
       - greeting_other       → canned reply
       - clarification_answer → parse slots, rerank candidates, maybe ask more
       - diagnostic           → narrow diseases, ask batch clarification OR answer
       - informational        → (cache lookup) hybrid RAG + KG → generate
  6. Save session + log consultation
"""

from __future__ import annotations

from dataclasses import asdict

from src.chat.cache import cache_get, cache_put
from src.chat.classifier import classify
from src.chat.differential import (
    build_clarification,
    discriminative_symptoms,
    parse_clarification_answer,
    rank_candidates,
    should_ask_clarification,
    symptom_catalog,
)
from src.chat.entity_extractor import extract_entities
from src.chat.generator import generate
from src.chat.guardrail import check as guardrail_check
from src.chat.kg_retriever import format_kg_context, kg_search
from src.chat.query_rewriter import rewrite
from src.chat.quota import check_both as check_llm_quota
from src.chat.retriever import hybrid_search
from src.chat.session import (
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


def _last_bot_message(session: PatientSession) -> str:
    for turn in reversed(session.conversation):
        if turn.get("role") == "assistant":
            return turn.get("content", "")
    return ""


def _handle_diagnostic(session: PatientSession, question: str) -> str:
    """New symptoms reported. Rank diseases, maybe ask batch clarification."""
    symptom_ids = [s["symptom_id"] for s in session.symptoms if s.get("symptom_id")]
    candidates = rank_candidates(symptom_ids)
    session.candidate_diseases = candidates

    if should_ask_clarification(candidates):
        cand_ids = [c["disease_id"] for c in candidates]
        known = [s for s in symptom_ids if not s.startswith("raw:")]
        next_symptoms = discriminative_symptoms(
            cand_ids, known + session.answered_questions,
        )
        if next_symptoms:
            session.answered_questions.extend(next_symptoms)
            header = "Dựa trên triệu chứng bạn nêu, tôi đang cân nhắc các bệnh: "
            header += ", ".join(c["name"] for c in candidates[:5] if c.get("name")) + "."
            return header + "\n\n" + build_clarification(next_symptoms)

    # Enough narrowing — answer with RAG + KG
    return _handle_informational(session, question, use_patient_context=True)


def _handle_clarification_answer(session: PatientSession, user_answer: str) -> str:
    """Parse batched clarification reply, update slots, then continue."""
    asked_ids = session.answered_questions[-4:]  # last batch
    catalog = symptom_catalog()
    asked = [
        {"symptom_id": sid, "name": catalog.get(sid, {}).get("name_vi", sid)}
        for sid in asked_ids
    ]

    results = parse_clarification_answer(asked, user_answer)
    for r in results:
        sid = r.get("symptom_id")
        if not sid:
            continue
        if r.get("present") == "yes":
            entry = {"symptom_id": sid,
                     "name": catalog.get(sid, {}).get("name_vi", sid)}
            for k in ("onset", "severity", "pattern", "associated"):
                if r.get(k):
                    entry[k] = r[k]
            session.upsert_symptom(entry)

    # Re-run narrowing with updated symptoms
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

    kg_ctx = kg_search(question)
    kg_text = format_kg_context(kg_ctx)
    hits = hybrid_search(question, top_k=RERANK_TOP_K)

    patient_dict = asdict(session) if use_patient_context else None
    answer = generate(question, hits, kg_text=kg_text, patient=patient_dict)

    if cacheable:
        cache_put(question, answer)
    return answer


def answer(question: str, session_id: str = "default") -> str:
    question = (question or "").strip()
    if not question:
        return "Bạn hãy đặt câu hỏi cụ thể nhé."

    # Tier A: cheap per-session rate limit (SQLite sliding window)
    if not check_rate_limit(session_id):
        return RATE_LIMIT_MSG

    # Tier B: guardrail (regex + cheap LLM) — blocks spam/off-topic/injection
    guard = guardrail_check(question)
    if guard["verdict"] != "allow":
        if guard["verdict"] == "abuse":
            return ""  # silent drop
        session = load_session(session_id)
        session.add_message("user", question)
        session.add_message("assistant", guard["reply"])
        save_session(session)
        return guard["reply"]

    # Tier C: LLM quota (global + per-session) — hard cost cap
    allowed, quota_msg = check_llm_quota(session_id)
    if not allowed:
        return quota_msg

    session = load_session(session_id)

    # Classify turn
    turn = classify(question, last_bot_message=_last_bot_message(session))
    label = turn["label"]

    session.add_message("user", question)

    if label == "greeting_other":
        reply = GREETING_REPLY
        session.add_message("assistant", reply)
        save_session(session)
        return reply

    # Query rewriting (only for non-clarification turns)
    if label != "clarification_answer":
        rw = rewrite(question, session.conversation[:-1])
        if not rw["confident"] and rw["clarification"]:
            session.add_message("assistant", rw["clarification"])
            save_session(session)
            return rw["clarification"]
        rewritten = rw["rewritten"]
    else:
        rewritten = question

    # Extract entities (symptoms/drugs) from the user message
    if label in ("diagnostic", "informational"):
        ents = extract_entities(question)
        for s in ents["symptoms"]:
            session.upsert_symptom(s)
        for d in ents["medications"]:
            session.add_medication(d)

    # Route. Informational questions always answer even if we have symptoms;
    # patient context is passed in so the answer is personalized.
    if label == "clarification_answer":
        reply = _handle_clarification_answer(session, question)
    elif label == "diagnostic":
        reply = _handle_diagnostic(session, rewritten)
    else:
        reply = _handle_informational(
            session,
            rewritten,
            use_patient_context=bool(session.symptoms),
            cacheable=turn["cacheable"] and not session.symptoms,
        )

    session.add_message("assistant", reply)
    save_session(session)
    save_profile(session)
    log_consultation(session_id, question, reply)
    return reply
