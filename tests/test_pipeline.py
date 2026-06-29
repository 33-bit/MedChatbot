from __future__ import annotations

import logging
import threading

import pytest

import src.chat.emergency as emergency_module
from src.chat import pipeline
from src.chat.errors import Neo4jUnavailable, QdrantUnavailable
from src.chat.replies import TECHNICAL_ERROR_REPLY
from src.chat.retrieval.kg import KGContext
from src.chat.retrieval.types import Hit
from src.chat.storage.session import PatientSession


def _analysis(
    label: str = "informational",
    intent: str | None = None,
    direct_answer_requested: bool = False,
    rewritten: str = "Tôi bị ho",
    entities: dict | None = None,
    red_flags: list[str] | None = None,
    context: dict | None = None,
    verdict: str = "allow",
) -> dict:
    if intent is None:
        intent = {
            "diagnostic": "symptom_triage",
            "informational": "pure_info",
            "clarification_answer": "clarification_answer",
        }.get(label, "pure_info")
    return {
        "guardrail": {"verdict": verdict, "reason": ""},
        "turn": {
            "label": label,
            "intent": intent,
            "direct_answer_requested": direct_answer_requested,
        },
        "rewrite": {
            "rewritten": rewritten,
            "confident": True,
            "clarification": "",
        },
        "triage": {
            "urgency": "emergency" if intent == "emergency" else "routine",
            "red_flags": red_flags or [],
            "reason": "",
        },
        "entities": entities or {"symptoms": [], "medications": []},
        "context": context or {},
    }


def _patch_preflight_ok(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "check_rate_limit", lambda session_id: True)
    monkeypatch.setattr(pipeline, "regex_check", lambda question: None)
    monkeypatch.setattr(pipeline, "check_llm_quota", lambda session_id: (True, ""))


def _patch_persistence_noop(monkeypatch, session: PatientSession | None = None) -> PatientSession:
    current = session or PatientSession(session_id="test")
    monkeypatch.setattr(pipeline, "load_session", lambda session_id: current)
    monkeypatch.setattr(pipeline, "save_session", lambda saved_session: None)
    monkeypatch.setattr(pipeline, "save_profile", lambda saved_session: None)
    monkeypatch.setattr(pipeline, "log_consultation", lambda *args: None)
    return current


def test_prepare_evidence_plan_falls_back_for_unscoped_disease_info(monkeypatch):
    calls: list[dict] = []

    def fake_plan_evidence(question, *, analysis, fallback_domain):
        calls.append({"question": question, "fallback_domain": fallback_domain})
        return {
            "domain": "disease_info",
            "source_type": "disease",
            "entity": "Basedow",
            "answer_slot": "definition",
            "safety_mode": "factual_info",
            "target_heading_paths": ["Đại cương"],
            "required_facts": ["bản chất bệnh"],
            "answer_style": "short_explanation",
            "confidence": 0.9,
            "needs_fallback": False,
        }

    monkeypatch.setattr(pipeline, "plan_evidence", fake_plan_evidence)
    analysis = _analysis(intent="pure_info", rewritten="Basedow là gì?")
    analysis["analysis_succeeded"] = False

    plan = pipeline._prepare_evidence_plan(
        "Basedow là gì?",
        analysis,
        "informational",
        "pure_info",
        "trace",
    )

    assert calls == [{"question": "Basedow là gì?", "fallback_domain": "disease_info"}]
    assert plan is not None
    assert plan["target_heading_paths"] == ["Đại cương"]
    assert analysis["evidence_plan"]["entity"] == "Basedow"


def test_prepare_evidence_plan_does_not_fallback_for_health_insurance(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        pipeline,
        "plan_evidence",
        lambda *args, **kwargs: calls.append(kwargs) or {},
    )
    analysis = _analysis(intent="health_insurance_info")

    plan = pipeline._prepare_evidence_plan(
        "Thẻ bảo hiểm y tế được cấp thế nào?",
        analysis,
        "informational",
        "health_insurance_info",
        "trace",
    )

    assert plan is None
    assert calls == []


def test_answer_empty_question_short_circuits():
    assert pipeline.answer("", session_id="s") == "Bạn hãy đặt câu hỏi cụ thể nhé."


def test_rate_limit_short_circuits_before_llm(monkeypatch):
    monkeypatch.setattr(pipeline, "check_rate_limit", lambda session_id: False)

    assert pipeline.answer("Tôi bị ho", session_id="s") == pipeline.RATE_LIMIT_MSG


def test_llm_quota_short_circuits(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    monkeypatch.setattr(pipeline, "check_llm_quota", lambda session_id: (False, "quota hit"))

    assert pipeline.answer("Tôi bị ho", session_id="s") == "quota hit"


def test_greeting_regex_short_circuits_before_analyzer(monkeypatch):
    monkeypatch.setattr(pipeline, "check_rate_limit", lambda session_id: True)
    _patch_persistence_noop(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("analyzer should not run")),
    )

    reply = pipeline.answer("Xin chào", session_id="s")

    assert "trợ lý y tế" in reply


def test_analyzer_guardrail_replies_without_retrieval(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(verdict="off_topic"),
    )
    monkeypatch.setattr(
        pipeline,
        "_handle_informational",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrieval should not run")),
    )

    assert pipeline.answer("Viết code Python", session_id="s") == (
        "Tôi chỉ hỗ trợ các câu hỏi về sức khỏe, bệnh lý và thuốc."
    )


def test_blocked_guardrail_turn_logs_direct_answer_false(monkeypatch, caplog):
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch)
    caplog.set_level(logging.INFO, logger=pipeline.log.name)
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(
            verdict="off_topic",
            direct_answer_requested=True,
        ),
    )

    pipeline.answer("Viết code Python", session_id="s")

    assert "stage=turn_analysis" in caplog.text
    assert "direct_answer=False" in caplog.text
    assert "direct_answer=True" not in caplog.text


def test_qdrant_failure_stops_pipeline_with_technical_reply(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch)
    monkeypatch.setattr(pipeline, "analyze_turn", lambda *args, **kwargs: _analysis())
    monkeypatch.setattr(pipeline, "_load_kg_context", lambda question, trace_id: "")

    def broken_hybrid(question: str, trace_id: str) -> list[Hit]:
        raise QdrantUnavailable("qdrant down")

    monkeypatch.setattr(pipeline, "_load_hybrid_hits", broken_hybrid)

    assert pipeline.answer("Tôi bị ho", session_id="s") == TECHNICAL_ERROR_REPLY


def test_neo4j_failure_stops_pipeline_with_technical_reply(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch)
    monkeypatch.setattr(pipeline, "analyze_turn", lambda *args, **kwargs: _analysis())
    monkeypatch.setattr(pipeline, "_load_hybrid_hits", lambda question, trace_id: [])

    def broken_kg(question: str, trace_id: str) -> str:
        raise Neo4jUnavailable("neo4j down")

    monkeypatch.setattr(pipeline, "_load_kg_context", broken_kg)

    assert pipeline.answer("Tôi bị ho", session_id="s") == TECHNICAL_ERROR_REPLY


def test_direct_answer_request_skips_more_clarification(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    session = _patch_persistence_noop(
        monkeypatch,
        PatientSession(
            session_id="s",
            symptoms=[{"symptom_id": "symptom:COUGH", "name": "Ho"}],
            answered_questions=["symptom:FEVER"],
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(
            label="clarification_answer",
            direct_answer_requested=True,
        ),
    )
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {"symptom:FEVER": {"name_vi": "Sốt"}})
    monkeypatch.setattr(pipeline, "parse_clarification_answer", lambda asked, answer: [])
    monkeypatch.setattr(
        pipeline,
        "rank_candidates",
        lambda symptom_ids: [
            {"disease_id": "flu", "name": "Bệnh cúm", "overlap": 1},
            {"disease_id": "covid", "name": "COVID-19", "overlap": 1},
            {"disease_id": "cold", "name": "Cảm lạnh", "overlap": 1},
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "discriminative_symptoms",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not ask more")),
    )
    monkeypatch.setattr(
        pipeline,
        "build_clarification",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not build clarification")),
    )
    captured: dict[str, object] = {}

    def fake_informational(
        saved_session: PatientSession,
        question: str,
        trace_id: str,
        use_patient_context: bool = False,
        retrieval_question: str | None = None,
        **kwargs,
    ) -> str:
        captured["question"] = question
        captured["use_patient_context"] = use_patient_context
        captured["retrieval_question"] = retrieval_question
        return "direct diagnostic answer"

    monkeypatch.setattr(pipeline, "_handle_informational", fake_informational)

    reply = pipeline.answer("tôi không biết, hãy trả lời luôn", session_id="s")

    assert reply == "direct diagnostic answer"
    assert captured["use_patient_context"] is True
    assert "chưa đủ để chẩn đoán chính xác" in str(captured["question"])
    assert "Bệnh cúm" in str(captured["retrieval_question"])
    assert session.candidate_diseases[0]["name"] == "Bệnh cúm"


def test_direct_diagnostic_answer_request_keeps_diagnostic_route(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    session = _patch_persistence_noop(monkeypatch, PatientSession(session_id="s"))
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(
            label="diagnostic",
            direct_answer_requested=True,
            rewritten="Tôi bị ho, trả lời luôn",
            entities={"symptoms": [{"name": "Ho"}], "medications": []},
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "normalize_entities",
        lambda raw: {
            "symptoms": [{"symptom_id": "symptom:COUGH", "name": "Ho"}],
            "medications": [],
        },
    )
    monkeypatch.setattr(
        pipeline,
        "parse_clarification_answer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not parse clarification")),
    )
    monkeypatch.setattr(
        pipeline,
        "rank_candidates",
        lambda symptom_ids: [
            {"disease_id": "flu", "name": "Bệnh cúm", "overlap": 1},
            {"disease_id": "covid", "name": "COVID-19", "overlap": 1},
            {"disease_id": "cold", "name": "Cảm lạnh", "overlap": 1},
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "discriminative_symptoms",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not ask more")),
    )
    captured: dict[str, object] = {}

    def fake_informational(
        saved_session: PatientSession,
        question: str,
        trace_id: str,
        use_patient_context: bool = False,
        retrieval_question: str | None = None,
        **kwargs,
    ) -> str:
        captured["question"] = question
        captured["use_patient_context"] = use_patient_context
        captured["retrieval_question"] = retrieval_question
        return "direct diagnostic answer"

    monkeypatch.setattr(pipeline, "_handle_informational", fake_informational)

    reply = pipeline.answer("Tôi bị ho, trả lời luôn", session_id="s")

    assert reply == "direct diagnostic answer"
    assert session.symptoms == [{"symptom_id": "symptom:COUGH", "name": "Ho"}]
    assert captured["use_patient_context"] is True
    assert "chưa đủ để chẩn đoán chính xác" in str(captured["question"])
    assert "Bệnh cúm" in str(captured["retrieval_question"])


def test_medicine_question_during_pending_detail_uses_analyzer_not_clarification(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    session = _patch_persistence_noop(
        monkeypatch,
        PatientSession(
            session_id="s",
            symptoms=[{"symptom_id": "symptom:ABDOMINAL_PAIN", "name": "Đau bụng"}],
            answered_questions=["detail:pattern:1:symptom:ABDOMINAL_PAIN"],
            clarification_queue=["detail:associated:0:symptom:ABDOMINAL_PAIN"],
        ),
    )
    monkeypatch.setattr(pipeline, "analyze_turn", lambda *args, **kwargs: _analysis(
        label="informational",
        rewritten="liều dùng vitamin A",
        entities={"symptoms": [], "medications": ["vitamin A"]},
    ))
    monkeypatch.setattr(
        pipeline,
        "normalize_entities",
        lambda raw: {"symptoms": [], "medications": ["vitamin A"]},
    )
    monkeypatch.setattr(
        pipeline,
        "_handle_clarification_answer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("medicine question should not be treated as clarification")
        ),
    )
    captured: dict[str, object] = {}

    def fake_informational(
        saved_session: PatientSession,
        question: str,
        trace_id: str,
        use_patient_context: bool = False,
        retrieval_question: str | None = None,
        **kwargs,
    ) -> str:
        captured["question"] = question
        captured["use_patient_context"] = use_patient_context
        captured["retrieval_question"] = retrieval_question
        return "vitamin A dosage answer"

    monkeypatch.setattr(pipeline, "_handle_informational", fake_informational)

    reply = pipeline.answer("liều dùng vitamin A", session_id="s")

    assert reply == "vitamin A dosage answer"
    assert captured == {
        "question": "liều dùng vitamin A",
        "use_patient_context": False,
        "retrieval_question": None,
    }
    assert "vitamin A" in session.medications


def test_pending_detail_detection_keeps_plain_symptom_answers_as_clarification():
    session = PatientSession(
        session_id="s",
        answered_questions=["detail:onset:0:symptom:ABDOMINAL_PAIN"],
    )

    assert pipeline._is_pending_clarification_choice(session, "2 ngày trước") is True
    assert pipeline._is_pending_clarification_choice(session, "liều dùng vitamin A") is False


def test_medication_only_informational_turn_ignores_existing_symptom_context(monkeypatch):
    session = PatientSession(
        session_id="s",
        symptoms=[{"symptom_id": "symptom:ABDOMINAL_PAIN", "name": "Đau bụng"}],
        medications=["vitamin A"],
    )
    captured: dict[str, object] = {}

    def fake_informational(
        saved_session: PatientSession,
        question: str,
        trace_id: str,
        use_patient_context: bool = False,
        retrieval_question: str | None = None,
        **kwargs,
    ) -> str:
        captured["use_patient_context"] = use_patient_context
        return "vitamin A dosage answer"

    monkeypatch.setattr(pipeline, "_handle_informational", fake_informational)

    reply = pipeline._route(
        "informational",
        session,
        "liều dùng vitamin A",
        "liều dùng vitamin A",
        False,
        "trace",
    )

    assert reply == "vitamin A dosage answer"
    assert captured["use_patient_context"] is False


def test_contextual_drug_info_uses_informational_route_not_diagnostic(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    monkeypatch.setattr(pipeline, "CONVERSATION_CONTEXT_ENABLED", False)
    session = _patch_persistence_noop(monkeypatch, PatientSession(session_id="s"))
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(
            label="diagnostic",
            intent="contextual_drug_info",
            rewritten="Tôi bị đau đầu, dùng Paracetamol được không?",
            entities={
                "symptoms": [{"name": "đau đầu"}],
                "medications": ["Paracetamol"],
            },
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "normalize_entities",
        lambda raw: {
            "symptoms": [{"symptom_id": "symptom:HEADACHE", "name": "đau đầu"}],
            "medications": ["Paracetamol"],
        },
    )
    monkeypatch.setattr(
        pipeline,
        "_handle_diagnostic",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("contextual drug info should not start diagnostic narrowing")
        ),
    )
    captured: dict[str, object] = {}

    def fake_informational(
        saved_session: PatientSession,
        question: str,
        trace_id: str,
        use_patient_context: bool = False,
        retrieval_question: str | None = None,
        **kwargs,
    ) -> str:
        captured["question"] = question
        captured["use_patient_context"] = use_patient_context
        captured["retrieval_question"] = retrieval_question
        return "paracetamol safety answer"

    monkeypatch.setattr(pipeline, "_handle_informational", fake_informational)

    reply = pipeline.answer(
        "Tôi bị đau đầu, dùng Paracetamol được không?",
        session_id="s",
        mode="auto",
    )

    assert reply == "paracetamol safety answer"
    assert captured == {
        "question": "Tôi bị đau đầu, dùng Paracetamol được không?",
        "use_patient_context": True,
        "retrieval_question": None,
    }
    assert session.symptoms == [{"symptom_id": "symptom:HEADACHE", "name": "đau đầu"}]
    assert session.medications == ["Paracetamol"]


def test_non_otc_drug_question_stops_before_retrieval(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    session = _patch_persistence_noop(monkeypatch, PatientSession(session_id="s"))
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(
            label="informational",
            intent="pure_info",
            rewritten="Cho tôi thông tin về thuốc kháng histamin H2",
            entities={"symptoms": [], "medications": []},
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_handle_informational",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("restricted drug question must not reach retrieval")
        ),
    )

    reply = pipeline.answer(
        "Cho tôi thông tin về thuốc kháng histamin H2",
        session_id="s",
    )

    assert reply == pipeline.OTC_ONLY_REPLY
    assert session.conversation == [
        {"role": "user", "content": "Cho tôi thông tin về thuốc kháng histamin H2"},
        {"role": "assistant", "content": pipeline.OTC_ONLY_REPLY},
    ]


def test_factual_non_otc_drug_policy_reaches_rag_and_is_exposed_in_meta(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch, PatientSession(session_id="s"))
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(
            label="informational",
            intent="pure_info",
            rewritten="Amoxicillin là gì?",
            entities={"symptoms": [], "medications": ["Amoxicillin"]},
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_handle_informational",
        lambda *args, **kwargs: "amoxicillin monograph [1]",
    )

    reply, meta = pipeline.answer_with_meta("Amoxicillin là gì?", session_id="s")

    assert reply == "amoxicillin monograph [1]"
    assert meta["drug_policy"] == {
        "is_drug_question": True,
        "allowed": True,
        "reason": "source_grounded_drug_info",
        "matched_otc_names": [],
    }


def test_condition_management_info_uses_informational_route_not_diagnostic(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    session = _patch_persistence_noop(monkeypatch, PatientSession(session_id="s"))
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(
            label="diagnostic",
            intent="condition_management_info",
            rewritten="Tôi bị đau thần kinh tọa, điều trị không dùng thuốc được không?",
            entities={"symptoms": [], "medications": []},
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_handle_diagnostic",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("condition management info should not start diagnostic narrowing")
        ),
    )
    captured: dict[str, object] = {}

    def fake_informational(
        saved_session: PatientSession,
        question: str,
        trace_id: str,
        use_patient_context: bool = False,
        retrieval_question: str | None = None,
        **kwargs,
    ) -> str:
        captured["question"] = question
        captured["use_patient_context"] = use_patient_context
        captured["retrieval_question"] = retrieval_question
        return "sciatica management answer"

    monkeypatch.setattr(pipeline, "_handle_informational", fake_informational)

    reply = pipeline.answer(
        "Tôi bị đau thần kinh tọa, điều trị không dùng thuốc được không?",
        session_id="s",
        mode="auto",
    )

    assert reply == "sciatica management answer"
    assert captured == {
        "question": "Tôi bị đau thần kinh tọa, điều trị không dùng thuốc được không?",
        "use_patient_context": False,
        "retrieval_question": None,
    }


def test_pure_info_in_diagnostic_mode_suggests_information_mode(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    session = _patch_persistence_noop(monkeypatch, PatientSession(session_id="s"))
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(
            label="informational",
            intent="pure_info",
            rewritten="Acid Pantothenic là gì?",
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_handle_informational",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("blocked pure info should not answer in diagnostic mode")
        ),
    )

    reply = pipeline.answer("Acid Pantothenic là gì?", session_id="s", mode="diagnostic")

    assert reply == (
        "Câu hỏi này phù hợp với chế độ Thông tin hơn. "
        "Bạn muốn trả lời ở chế độ Thông tin không?"
    )
    assert session.conversation == [
        {"role": "user", "content": "Acid Pantothenic là gì?"},
        {"role": "assistant", "content": reply},
    ]


@pytest.mark.parametrize("mode", ("auto", "information", "diagnostic"))
def test_emergency_intent_uses_dedicated_route_without_retrieval(monkeypatch, mode):
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch, PatientSession(session_id="s"))
    monkeypatch.setattr(pipeline, "CONVERSATION_CONTEXT_ENABLED", False)
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(
            label="diagnostic",
            intent="emergency",
            rewritten="Tôi đang bị đau ngực dữ dội",
            red_flags=["đau ngực dữ dội"],
            entities={
                "symptoms": [{"name": "đau ngực", "severity": "dữ dội"}],
                "medications": [],
            },
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_handle_diagnostic",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("emergency route must not run diagnostic narrowing")
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_handle_informational",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("emergency route must not run retrieval")
        ),
    )

    reply, meta = pipeline.answer_with_meta(
        "Tôi đang bị đau ngực dữ dội",
        session_id="s",
        mode=mode,
    )

    route_node = next(node for node in meta["graph_nodes"] if node["id"] == "route")
    assert reply.startswith("Đây có thể là tình trạng cấp cứu")
    assert "gọi 115 ngay" in reply
    assert "bật loa ngoài" not in reply
    assert "điều phối viên" not in reply
    assert "khoa Cấp cứu" not in reply
    assert "hoặc đưa" not in reply
    assert "ép tim" not in reply
    assert "người đang có triệu chứng" not in reply
    assert "người bệnh" not in reply
    assert meta["route_label"] == "emergency"
    assert meta["outcome"] == "emergency"
    assert route_node["raw"]["decision"]["route_label"] == "emergency"
    assert route_node["raw"]["decision"]["force_answer"] is None
    assert meta.get("retrieved", []) == []


def test_dengue_warning_routes_to_emergency(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch, PatientSession(session_id="s"))
    monkeypatch.setattr(pipeline, "CONVERSATION_CONTEXT_ENABLED", False)
    monkeypatch.setattr(pipeline, "PIPELINE_EMERGENCY_RAG", True)
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(
            label="informational",
            intent="pure_info",
            rewritten="Chồng tôi sốt xuất huyết ngày thứ 4, hết sốt nhưng lừ đừ, đau bụng và tay chân lạnh.",
            red_flags=["lừ đừ", "đau bụng", "tay chân lạnh"],
            entities={"symptoms": [], "medications": []},
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_handle_informational",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dengue warning must not use informational route")
        ),
    )

    reply, meta = pipeline.answer_with_meta(
        "Chồng tôi sốt xuất huyết ngày thứ 4, hết sốt nhưng lừ đừ, đau bụng và tay chân lạnh.",
        session_id="s",
    )

    assert meta["route_label"] == "emergency"
    assert meta["emergency_intent_override"] == "dengue_warning"
    assert "Hướng dẫn sơ cứu ban đầu" in reply
    assert "sốt xuất huyết Dengue" in reply


def test_informational_analyzer_emergency_classifier_forces_route(monkeypatch):
    question = "Bố tôi đau thắt ngực dữ dội kéo dài, lan lên hàm và vã mồ hôi."
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch, PatientSession(session_id="s"))
    monkeypatch.setattr(pipeline, "CONVERSATION_CONTEXT_ENABLED", False)
    monkeypatch.setattr(pipeline, "PIPELINE_EMERGENCY_RAG", True)
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(
            label="informational",
            intent="pure_info",
            rewritten=question,
            red_flags=[],
            entities={"symptoms": [], "medications": []},
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_handle_informational",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("deterministic emergency override must route first")
        ),
    )

    reply, meta = pipeline.answer_with_meta(question, session_id="s")

    assert meta["route_label"] == "emergency"
    assert meta["emergency_intent_override"] == "chest_pain_acs"
    assert "Hướng dẫn sơ cứu ban đầu" in reply
    assert "hội chứng vành cấp" in reply


def test_factual_emergency_topic_uses_informational_route(monkeypatch):
    question = "Triệu chứng của thủng tạng rỗng là gì?"
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch, PatientSession(session_id="s"))
    monkeypatch.setattr(pipeline, "CONVERSATION_CONTEXT_ENABLED", False)
    monkeypatch.setattr(
        emergency_module,
        "classify_emergency_intent",
        lambda actual_question, red_flags=None: "acute_abdomen",
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(
            label="informational",
            intent="pure_info",
            rewritten=question,
            red_flags=["đau bụng cấp"],
            entities={"symptoms": [], "medications": []},
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_handle_informational",
        lambda *args, **kwargs: "triệu chứng thủng tạng rỗng [1]",
    )

    reply, meta = pipeline.answer_with_meta(question, session_id="s")

    assert reply == "triệu chứng thủng tạng rỗng [1]"
    assert meta["route_label"] == "informational"
    assert meta["emergency_intent_topic_info"] == "acute_abdomen"


def test_emergency_protocol_question_uses_informational_route(monkeypatch):
    question = "Bệnh nhân bị sốc nhiễm khuẩn cần được truyền dịch như thế nào trong giai đoạn cấp cứu ban đầu?"
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch, PatientSession(session_id="s"))
    monkeypatch.setattr(pipeline, "CONVERSATION_CONTEXT_ENABLED", False)
    monkeypatch.setattr(
        emergency_module,
        "classify_emergency_intent",
        lambda actual_question, red_flags=None: "septic_shock",
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(
            label="emergency",
            intent="emergency",
            rewritten=question,
            red_flags=["sốc nhiễm khuẩn"],
            entities={"symptoms": [], "medications": []},
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_handle_informational",
        lambda *args, **kwargs: "truyền nhanh 1000 mL dịch tinh thể [1]",
    )

    reply, meta = pipeline.answer_with_meta(question, session_id="s")

    assert reply == "truyền nhanh 1000 mL dịch tinh thể [1]"
    assert meta["route_label"] == "informational"
    assert meta["emergency_intent_topic_info"] == "septic_shock"


def test_current_emergency_action_still_uses_emergency_route(monkeypatch):
    question = "Tôi đau bụng dữ dội, phải làm gì?"
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch, PatientSession(session_id="s"))
    monkeypatch.setattr(pipeline, "CONVERSATION_CONTEXT_ENABLED", False)
    monkeypatch.setattr(
        emergency_module,
        "classify_emergency_intent",
        lambda actual_question, red_flags=None: "acute_abdomen",
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(
            label="informational",
            intent="pure_info",
            rewritten=question,
            red_flags=["đau bụng dữ dội"],
            entities={"symptoms": [{"name": "đau bụng"}], "medications": []},
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_handle_informational",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("actionable emergency must not use informational route")
        ),
    )

    reply, meta = pipeline.answer_with_meta(
        question,
        session_id="s",
        use_emergency_rag=False,
    )

    assert meta["route_label"] == "emergency"
    assert meta["emergency_intent_override"] == "acute_abdomen"
    assert reply.startswith("Đây có thể là tình trạng cấp cứu")


def test_emergency_rag_records_retrieval_and_generation_latency(monkeypatch):
    import src.chat.replies as replies

    question = "Tôi đau bụng dữ dội, phải làm gì?"
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch, PatientSession(session_id="s"))
    monkeypatch.setattr(pipeline, "CONVERSATION_CONTEXT_ENABLED", False)
    monkeypatch.setattr(
        emergency_module,
        "classify_emergency_intent",
        lambda actual_question, red_flags=None: "acute_abdomen",
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(
            label="emergency",
            intent="emergency",
            rewritten=question,
            red_flags=["đau bụng dữ dội"],
        ),
    )

    def fake_emergency_reply(*args, timing_callback=None, **kwargs):
        assert timing_callback is not None
        timing_callback("retrieval", 12.5)
        timing_callback("generator", 3.25)
        return "Đây có thể là tình trạng cấp cứu. Hãy gọi 115 ngay."

    monkeypatch.setattr(replies, "emergency_reply", fake_emergency_reply)

    _, meta = pipeline.answer_with_meta(
        question,
        session_id="s",
        use_emergency_rag=True,
    )

    assert meta["latency_ms"]["retrieval"] == 12.5
    assert meta["latency_ms"]["generator"] == 3.25


@pytest.mark.parametrize("subject", ("bố bạn", "mẹ bạn", "cô Lan"))
def test_emergency_reply_treats_user_as_caregiver_for_other_subjects(subject):
    reply = pipeline.emergency_reply(
        subject,
        red_flags=["khó thở dữ dội", "môi tím tái"],
        question=f"{subject} khó thở dữ dội, môi tím tái",
    )

    assert reply.startswith("Đây có thể là tình trạng cấp cứu")
    assert "gọi 115 ngay" in reply
    assert "bật loa ngoài" not in reply
    assert "điều phối viên" not in reply
    assert "khoa Cấp cứu" not in reply
    assert f"đưa {subject}" not in reply
    assert f"không để {subject} tự lái xe" not in reply
    assert "nhờ người khác đưa" not in reply


@pytest.mark.parametrize(
    ("subject", "question", "red_flags", "old_actions", "unexpected"),
    (
        (
            "bố bạn",
            "Bố tôi đột ngột méo miệng, nói ngọng và yếu liệt nửa người.",
            ["méo miệng", "nói ngọng", "yếu liệt nửa người"],
            ("thời điểm khởi phát", "không cho bố bạn ăn uống", "tự dùng thuốc"),
            ("ép tim", "tự lái xe"),
        ),
        (
            "mẹ bạn",
            "Mẹ tôi co giật toàn thân hơn 5 phút chưa dứt.",
            ["co giật toàn thân hơn 5 phút"],
            ("nằm nghiêng an toàn", "không nhét bất cứ thứ gì vào miệng"),
            ("AED", "tự lái xe"),
        ),
        (
            "người nhà bạn",
            "Người nhà tôi hôn mê sau khi đốt than sưởi trong phòng kín.",
            ["ngộ độc khí CO"],
            ("thoáng khí", "hít khí độc"),
            ("ép tim", "tự lái xe"),
        ),
        (
            "bố bạn",
            "Bố tôi ngã xuống, không đáp lại, không thở bình thường.",
            ["không thở bình thường", "không bắt được mạch"],
            ("ép tim ngoài lồng ngực", "AED"),
            ("ăn uống", "tự dùng thuốc"),
        ),
        (
            "con bạn",
            "Con tôi ăn hải sản xong nổi mề đay, sưng môi, khò khè và khó thở.",
            ["nổi mề đay", "sưng môi", "khò khè", "khó thở"],
            ("epinephrine",),
            ("tự lái xe", "ép tim"),
        ),
        (
            "bạn",
            "Tôi đau bụng dữ dội liên tục, bụng cứng như gỗ và chóng mặt.",
            ["đau bụng dữ dội", "bụng cứng"],
            ("Không tự dùng thuốc giảm đau",),
            ("ép tim", "tự lái xe"),
        ),
    ),
)
def test_emergency_reply_fast_path_has_no_immediate_action_block(
    subject,
    question,
    red_flags,
    old_actions,
    unexpected,
):
    reply = pipeline.emergency_reply(subject, red_flags=red_flags, question=question)

    assert "gọi 115 ngay" in reply
    assert "bật loa ngoài" not in reply
    assert "điều phối viên" not in reply
    assert "khoa Cấp cứu" not in reply
    assert "\n-" not in reply
    for phrase in old_actions:
        assert phrase not in reply
    for phrase in unexpected:
        assert phrase not in reply


def test_symptom_triage_in_information_mode_answers_directly(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    session = _patch_persistence_noop(monkeypatch, PatientSession(session_id="s"))
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(
            label="diagnostic",
            intent="symptom_triage",
            rewritten="Tôi đau lưng lan xuống chân là bệnh gì?",
            entities={"symptoms": [{"name": "đau lưng lan xuống chân"}], "medications": []},
        ),
    )
    captured: dict[str, object] = {}

    def fake_diagnostic(saved_session, question, trace_id, force_answer=False):
        captured["force_answer"] = force_answer
        return "Nhận định sơ bộ về đau lưng lan xuống chân [1]"

    monkeypatch.setattr(pipeline, "_handle_diagnostic", fake_diagnostic)
    monkeypatch.setattr(pipeline, "_ingest_entities", lambda *args, **kwargs: None)

    reply = pipeline.answer(
        "Tôi đau lưng lan xuống chân là bệnh gì?",
        session_id="s",
        mode="information",
    )

    assert reply == "Nhận định sơ bộ về đau lưng lan xuống chân [1]"
    assert captured == {"force_answer": True}
    assert session.conversation == [
        {"role": "user", "content": "Tôi đau lưng lan xuống chân là bệnh gì?"},
        {"role": "assistant", "content": reply},
    ]
    assert all("mode" not in turn for turn in session.conversation)


def test_answer_with_choices_no_mode_suggestion_for_triage_in_information_mode(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch, PatientSession(session_id="s"))
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(
            label="diagnostic",
            intent="symptom_triage",
            rewritten="Tôi đau lưng là bệnh gì?",
            entities={"symptoms": [{"name": "đau lưng"}], "medications": []},
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_handle_diagnostic",
        lambda *args, **kwargs: "Nhận định sơ bộ về đau lưng [1]",
    )
    monkeypatch.setattr(pipeline, "_ingest_entities", lambda *args, **kwargs: None)

    reply = pipeline.answer_with_choices("Tôi đau lưng là bệnh gì?", session_id="s", mode="information")

    assert reply.text == "Nhận định sơ bộ về đau lưng [1]"
    assert reply.suggest_mode is None
    assert reply.retry_question is None


def test_first_diagnostic_turn_gives_overview_before_questions(monkeypatch):
    session = PatientSession(
        session_id="s",
        symptoms=[{"symptom_id": "symptom:ABDOMINAL_PAIN", "name": "đau bụng"}],
    )
    monkeypatch.setattr(
        pipeline,
        "rank_candidates",
        lambda symptom_ids: (_ for _ in ()).throw(AssertionError("first turn should not rank diseases")),
    )

    reply = pipeline._handle_diagnostic(session, "tôi đang bị đau bụng", trace_id="trace")

    assert "Tôi hiểu bạn đang bị đau bụng" in reply
    assert "có thể liên quan đến" in reply
    assert "rối loạn tiêu hóa" in reply
    assert "viêm dạ dày-ruột" in reply
    assert "viêm ruột thừa" in reply
    assert "Trong lúc theo dõi" in reply
    assert "uống từng ngụm nước nhỏ" in reply
    assert "Không nên tự uống thuốc giảm đau mạnh" in reply
    assert "hãy đi cấp cứu ngay" in reply
    assert "Để tôi định hướng tốt hơn, bạn cho tôi hỏi thêm một vài câu hỏi nhé." in reply
    assert "Bạn cho tôi biết 3 điểm" not in reply
    assert "1." not in reply
    assert "2." not in reply
    assert "3." not in reply
    assert "Tiêu chảy do Clostridioides Difficile" not in reply
    assert pipeline.suggested_choices(reply) == ("Bắt đầu",)
    assert pipeline.GENERAL_TRIAGE_MARKER in session.answered_questions


def test_first_diagnostic_turn_addresses_resolved_father_subject():
    session = PatientSession(
        session_id="s",
        symptoms=[{"symptom_id": "symptom:HEADACHE", "name": "đau đầu"}],
    )
    previous = getattr(pipeline._META_LOCAL, "context_bundle", None)
    pipeline._META_LOCAL.context_bundle = pipeline.ConversationContextBundle(
        subject={
            "id": "father",
            "relationship": "father",
            "display_name": "bố bạn",
        },
        safety_profile=[],
        relevant_facts=[],
        active_case=None,
        reference_turns=[],
    )
    try:
        reply = pipeline._handle_diagnostic(
            session,
            "Bố tôi đang bị đau đầu",
            trace_id="trace",
        )
    finally:
        pipeline._META_LOCAL.context_bundle = previous

    assert "Tôi hiểu bố bạn đang bị đau đầu" in reply
    assert "Trong lúc theo dõi, bố bạn nên nghỉ ngơi" in reply
    assert "Tôi hiểu bạn đang bị đau đầu" not in reply


def test_general_triage_clarification_answer_skips_catalog_parse(monkeypatch):
    session = PatientSession(
        session_id="s",
        symptoms=[{"symptom_id": "symptom:ABDOMINAL_PAIN", "name": "đau bụng"}],
        answered_questions=[pipeline.GENERAL_TRIAGE_MARKER],
    )
    monkeypatch.setattr(
        pipeline,
        "parse_clarification_answer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("general triage is free text")),
    )
    monkeypatch.setattr(
        pipeline,
        "_handle_diagnostic",
        lambda saved_session, user_answer, trace_id, force_answer=False: "next diagnostic step",
    )

    assert pipeline._handle_clarification_answer(session, "đau 6 điểm, có nôn", "trace") == "next diagnostic step"


def test_suggested_choices_for_general_triage_starts_separate_questions():
    reply = (
        "Tôi hiểu bạn đang bị đau bụng.\n\n"
        "Triệu chứng này có thể liên quan đến rối loạn tiêu hóa.\n\n"
        "Để tôi định hướng tốt hơn, bạn cho tôi hỏi thêm một vài câu hỏi nhé."
    )

    assert pipeline.suggested_choices(reply) == ("Bắt đầu",)


def test_suggested_choices_for_specific_clarification_are_single_question_buttons(monkeypatch):
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {})
    reply = "Bạn có bị Sốt không?"

    assert pipeline.suggested_choices(reply) == (
        "Có",
        "Không",
        "Không rõ",
        "Trả lời luôn",
    )


def test_suggested_choices_prefers_catalog_presence_options(monkeypatch):
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {
        "symptom:S_FEVER": {
            "name_vi": "Sốt",
            "clarification_options": {
                "presence": ["Có sốt", "Không sốt", "Không rõ", "Trả lời luôn"],
            },
        },
    })

    assert pipeline.suggested_choices("Bạn có bị Sốt không?") == (
        "Có sốt",
        "Không sốt",
        "Không rõ",
        "Trả lời luôn",
    )


def test_suggested_choices_for_detail_clarification_only_allows_answer_now():
    assert pipeline.suggested_choices("Câu hỏi chi tiết khác?") == (
        "Không rõ",
        "Trả lời luôn",
    )


def test_suggested_choices_prefers_catalog_detail_options(monkeypatch):
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {
        "symptom:S_DIARRHEA": {
            "name_vi": "Tiêu chảy",
            "clarification_questions": {
                "onset": "Tiêu chảy bắt đầu từ khi nào?",
            },
            "clarification_options": {
                "onset": [
                    "Dưới 6 giờ",
                    "6-24 giờ",
                    "Trên 24 giờ",
                    "Không rõ",
                    "Trả lời luôn",
                ],
            },
        },
    })

    assert pipeline.suggested_choices("Tiêu chảy bắt đầu từ khi nào?") == (
        "Dưới 6 giờ",
        "6-24 giờ",
        "Trên 24 giờ",
        "Không rõ",
        "Trả lời luôn",
    )


def test_suggested_choices_prefers_catalog_detail_options_for_question_list(monkeypatch):
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {
        "symptom:S_FEVER": {
            "name_vi": "Sốt",
            "clarification_questions": {
                "onset": [
                    "Sốt xuất hiện từ khi nào?",
                    "Sốt có xuất hiện sau khi đi vùng dịch không?",
                ],
            },
            "clarification_options": {
                "onset": [
                    [
                        "Hôm nay",
                        "Hôm qua",
                        "Không rõ",
                        "Trả lời luôn",
                    ],
                    [
                        "Có đi vùng dịch",
                        "Không đi vùng dịch",
                        "Không rõ",
                        "Trả lời luôn",
                    ],
                ],
            },
        },
    })

    assert pipeline.suggested_choices("Sốt có xuất hiện sau khi đi vùng dịch không?") == (
        "Có đi vùng dịch",
        "Không đi vùng dịch",
        "Không rõ",
        "Trả lời luôn",
    )


def test_suggested_choices_flattens_nested_catalog_detail_options(monkeypatch):
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {
        "symptom:S_ABDOMINAL_DISTENSION": {
            "name_vi": "Chướng bụng",
            "clarification_questions": {
                "pattern": [
                    "Chướng bụng xảy ra liên tục hay từng đợt? Có liên quan đến bữa ăn hoặc đại tiện không?",
                ],
            },
            "clarification_options": {
                "pattern": [[
                    "['Liên tục', 'Từng đợt', 'Không rõ', 'Trả lời luôn']",
                    ["Sau bữa ăn", "Trước đại tiện", "Không liên quan", "Trả lời luôn"],
                    "Không rõ",
                ]],
            },
        },
    })

    assert pipeline.suggested_choices(
        "Chướng bụng xảy ra liên tục hay từng đợt? Có liên quan đến bữa ăn hoặc đại tiện không?"
    ) == (
        "Liên tục",
        "Từng đợt",
        "Sau bữa ăn",
        "Trước đại tiện",
        "Không liên quan",
        "Không rõ",
        "Trả lời luôn",
    )


def test_suggested_choices_for_detail_onset_question(monkeypatch):
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {})
    assert pipeline.suggested_choices("Tình trạng đái rắt bắt đầu từ khi nào?") == (
        "Hôm nay",
        "Hôm qua",
        "2-3 ngày",
        "Trên 3 ngày",
        "Không rõ",
        "Trả lời luôn",
    )


def test_suggested_choices_for_urinary_frequency_question(monkeypatch):
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {})
    assert pipeline.suggested_choices(
        "Tần suất đi tiểu nhiều đến mức nào trong ngày và đêm?"
    ) == (
        "Vài lần/ngày",
        "Mỗi 1-2 giờ",
        "30 phút/lần",
        "10 phút/lần",
        "Không rõ",
        "Trả lời luôn",
    )


def test_suggested_choices_for_pattern_detail_questions(monkeypatch):
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {})
    assert pipeline.suggested_choices("Có đái rắt liên tục hay theo từng đợt?") == (
        "Liên tục",
        "Từng đợt",
        "Không rõ",
        "Trả lời luôn",
    )
    assert pipeline.suggested_choices("Sốt liên tục hay ngắt quãng?") == (
        "Liên tục",
        "Ngắt quãng",
        "Không rõ",
        "Trả lời luôn",
    )


def test_suggested_choices_for_associated_detail_question(monkeypatch):
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {})
    assert pipeline.suggested_choices(
        "Có kèm theo tiểu buốt, đái máu hoặc đau bụng dưới không?"
    ) == (
        "Có tiểu buốt",
        "Có đái máu",
        "Có đau bụng dưới",
        "Nhiều triệu chứng",
        "Không",
        "Không rõ",
        "Trả lời luôn",
    )


def test_suggested_choices_for_generic_yes_no_detail_question(monkeypatch):
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {})
    assert pipeline.suggested_choices("Có tăng nhu động ruột không?") == (
        "Có",
        "Không",
        "Không rõ",
        "Trả lời luôn",
    )


def test_suggested_choices_for_two_item_associated_detail_question(monkeypatch):
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {})
    assert pipeline.suggested_choices("Có kèm theo nôn hoặc vàng da không?") == (
        "Có nôn",
        "Có vàng da",
        "Cả hai",
        "Không",
        "Không rõ",
        "Trả lời luôn",
    )
    assert pipeline.suggested_choices(
        "Có liên quan đến đau đầu hoặc rối loạn ý thức không?"
    ) == (
        "Có đau đầu",
        "Có rối loạn ý thức",
        "Cả hai",
        "Không",
        "Không rõ",
        "Trả lời luôn",
    )


def test_suggested_selection_mode_prefers_catalog_detail_mode(monkeypatch):
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {
        "symptom:S_DIARRHEA": {
            "name_vi": "Tiêu chảy",
            "clarification_questions": {
                "associated": ["Có kèm theo nôn hoặc vàng da không?"],
            },
            "clarification_options": {
                "associated": [["Có nôn", "Có vàng da", "Cả hai", "Không"]],
            },
            "clarification_selection_modes": {
                "presence": "single",
                "associated": ["multi"],
            },
        },
    })

    assert pipeline.suggested_selection_mode("Có kèm theo nôn hoặc vàng da không?") == "multi"


def test_suggested_selection_mode_overrides_catalog_single_for_combined_dimensions(monkeypatch):
    question = (
        "Đau liên tục hay từng cơn? "
        "Vị trí đau chính ở đâu (thượng vị, quanh rốn, hạ vị...)?"
    )
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {
        "symptom:S_ABDOMINAL_PAIN": {
            "name_vi": "Đau bụng",
            "clarification_questions": {
                "pattern": [question],
            },
            "clarification_options": {
                "pattern": [[
                    "Liên tục",
                    "Từng cơn",
                    "Thượng vị",
                    "Quanh rốn",
                    "Hạ vị",
                    "Không rõ",
                    "Trả lời luôn",
                ]],
            },
            "clarification_selection_modes": {
                "pattern": ["single"],
            },
        },
    })

    assert pipeline.suggested_selection_mode(question) == "multi"


def test_suggested_selection_mode_keeps_single_for_single_clause_pattern_alternatives(monkeypatch):
    question = "Đau có lan ra sau lưng hoặc thay đổi theo tư thế không?"
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {
        "symptom:S_ABDOMINAL_PAIN": {
            "name_vi": "Đau bụng",
            "clarification_questions": {
                "pattern": [question],
            },
            "clarification_options": {
                "pattern": [[
                    "Có lan ra lưng",
                    "Thay đổi theo tư thế",
                    "Không",
                    "Không rõ",
                    "Trả lời luôn",
                ]],
            },
            "clarification_selection_modes": {
                "pattern": ["single"],
            },
        },
    })

    assert pipeline.suggested_selection_mode(question) == "single"


def test_answer_with_choices_includes_selection_mode(monkeypatch):
    monkeypatch.setattr(pipeline, "answer", lambda question, session_id="default": "Có kèm theo nôn hoặc vàng da không?")
    monkeypatch.setattr(pipeline, "suggested_choices", lambda reply: ("Có nôn", "Có vàng da", "Không"))
    monkeypatch.setattr(pipeline, "suggested_selection_mode", lambda reply: "multi")

    reply = pipeline.answer_with_choices("x", session_id="s")

    assert reply == pipeline.ChatReply(
        "Có kèm theo nôn hoặc vàng da không?",
        ("Có nôn", "Có vàng da", "Không"),
        "multi",
    )


def test_answer_with_choices_includes_doctor_specialty(monkeypatch):
    def fake_answer(question, session_id="default"):
        pipeline._meta()["doctor_offer"] = True
        pipeline._meta()["doctor_specialty"] = "Tiêu hóa"
        return "Bạn nên đi khám bác sĩ."

    monkeypatch.setattr(pipeline, "answer", fake_answer)

    reply = pipeline.answer_with_choices("Tôi bị đau bụng", session_id="s")

    assert reply.doctor_offer is True
    assert reply.doctor_specialty == "Tiêu hóa"


def test_answer_with_choices_keeps_standard_retrieval(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch)
    monkeypatch.setattr(pipeline, "analyze_turn", lambda *args, **kwargs: _analysis())
    monkeypatch.setattr(pipeline, "_load_kg_context", lambda question, trace_id: "")
    monkeypatch.setattr(pipeline, "_load_hybrid_hits", lambda question, trace_id: [])
    monkeypatch.setattr(
        pipeline,
        "_load_kg_context_with_debug",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("answer_with_choices should not collect graph metadata")
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_load_hybrid_hits_with_debug",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("answer_with_choices should not collect graph metadata")
        ),
    )

    def fake_generate(question, hits, kg_text="", patient=None, return_meta=False, **kwargs):
        assert return_meta is True
        return "answer", {"usage": {}, "model": "test-model"}

    monkeypatch.setattr(pipeline, "generate", fake_generate)

    reply = pipeline.answer_with_choices("Tôi bị ho", session_id="s")

    assert reply.text == "answer"


def test_suggested_choices_for_fever_severity_question(monkeypatch):
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {})
    assert pipeline.suggested_choices("Sốt cao đến mức nào?") == (
        "< 38.5 độ",
        "38.5-39 độ",
        "> 39 độ",
        "Không đo",
        "Không rõ",
        "Trả lời luôn",
    )


def test_tapped_answer_now_for_pending_clarification_forces_answer(monkeypatch):
    monkeypatch.setattr(pipeline, "check_rate_limit", lambda session_id: True)
    monkeypatch.setattr(pipeline, "check_llm_quota", lambda session_id: (True, ""))
    session = _patch_persistence_noop(
        monkeypatch,
        PatientSession(
            session_id="s",
            symptoms=[{"symptom_id": "symptom:ABDOMINAL_PAIN", "name": "đau bụng"}],
            answered_questions=[pipeline.GENERAL_TRIAGE_MARKER, "symptom:FEVER"],
            clarification_queue=["symptom:DIARRHEA"],
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("tap choice should not need analyzer")
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "parse_clarification_answer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("answer-now tap should not parse symptom slots")
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "rank_candidates",
        lambda symptom_ids: [
            {"disease_id": "a", "name": "A", "overlap": 1},
            {"disease_id": "b", "name": "B", "overlap": 1},
            {"disease_id": "c", "name": "C", "overlap": 1},
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "discriminative_symptoms",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("answer-now tap should not ask more symptoms")
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_handle_informational",
        lambda *args, **kwargs: "answer from current context",
    )

    assert pipeline.answer("Trả lời luôn", session_id="s") == "answer from current context"
    assert session.clarification_queue == []
    assert session.clarification_plan_started is True


def test_tapped_answer_now_for_pending_detail_clarification_forces_answer(monkeypatch):
    monkeypatch.setattr(pipeline, "check_rate_limit", lambda session_id: True)
    monkeypatch.setattr(pipeline, "check_llm_quota", lambda session_id: (True, ""))
    session = _patch_persistence_noop(
        monkeypatch,
        PatientSession(
            session_id="s",
            symptoms=[{"symptom_id": "symptom:ABDOMINAL_PAIN", "name": "đau bụng"}],
            answered_questions=["detail:onset:0:symptom:ABDOMINAL_PAIN"],
            clarification_queue=["detail:pattern:0:symptom:ABDOMINAL_PAIN"],
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("answer-now tap should not need analyzer")
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "parse_clarification_answer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("answer-now tap should not parse detail slots")
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "rank_candidates",
        lambda symptom_ids: [
            {"disease_id": "a", "name": "A", "overlap": 1},
            {"disease_id": "b", "name": "B", "overlap": 1},
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "_handle_informational",
        lambda *args, **kwargs: "answer from current detail context",
    )

    assert pipeline.answer("Trả lời luôn", session_id="s") == "answer from current detail context"
    assert session.clarification_queue == []
    assert session.clarification_plan_started is True
    assert session.symptoms == [
        {"symptom_id": "symptom:ABDOMINAL_PAIN", "name": "đau bụng"}
    ]


def test_answer_now_route_error_persists_cleared_clarification_state(monkeypatch):
    monkeypatch.setattr(pipeline, "check_rate_limit", lambda session_id: True)
    monkeypatch.setattr(pipeline, "check_llm_quota", lambda session_id: (True, ""))
    session = PatientSession(
        session_id="s",
        symptoms=[{"symptom_id": "symptom:ABDOMINAL_PAIN", "name": "đau bụng"}],
        answered_questions=["detail:onset:0:symptom:ABDOMINAL_PAIN"],
        clarification_queue=["detail:pattern:0:symptom:ABDOMINAL_PAIN"],
    )
    saved_sessions: list[PatientSession] = []
    monkeypatch.setattr(pipeline, "load_session", lambda session_id: session)
    monkeypatch.setattr(pipeline, "save_session", lambda saved_session: saved_sessions.append(saved_session))
    monkeypatch.setattr(pipeline, "save_profile", lambda saved_session: None)
    monkeypatch.setattr(pipeline, "log_consultation", lambda *args: None)
    monkeypatch.setattr(
        pipeline,
        "rank_candidates",
        lambda symptom_ids: [{"disease_id": "a", "name": "A", "overlap": 1}],
    )
    monkeypatch.setattr(
        pipeline,
        "_handle_informational",
        lambda *args, **kwargs: (_ for _ in ()).throw(QdrantUnavailable("qdrant down")),
    )

    reply = pipeline.answer("Trả lời luôn", session_id="s")

    assert reply == TECHNICAL_ERROR_REPLY
    assert saved_sessions[-1] is session
    assert session.clarification_queue == []
    assert session.clarification_plan_started is True
    assert session.conversation[-2:] == [
        {"role": "user", "content": "Trả lời luôn"},
        {"role": "assistant", "content": TECHNICAL_ERROR_REPLY},
    ]


def test_tapped_yes_queues_detail_questions_before_next_symptom(monkeypatch):
    monkeypatch.setattr(pipeline, "check_rate_limit", lambda session_id: True)
    monkeypatch.setattr(pipeline, "check_llm_quota", lambda session_id: (True, ""))
    session = _patch_persistence_noop(
        monkeypatch,
        PatientSession(
            session_id="s",
            symptoms=[{"symptom_id": "symptom:ABDOMINAL_PAIN", "name": "đau bụng"}],
            answered_questions=[pipeline.GENERAL_TRIAGE_MARKER, "symptom:FEVER"],
            clarification_queue=["symptom:VOMIT"],
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("tap choice should not need analyzer")
        ),
    )
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {
        "symptom:FEVER": {
            "name_vi": "Sốt",
            "clarification_questions": {
                "onset": "Sốt xuất hiện từ khi nào?",
                "severity": "Sốt cao đến mức nào?",
                "pattern": "Sốt liên tục hay ngắt quãng?",
            },
        },
        "symptom:VOMIT": {"name_vi": "Nôn", "clarification_questions": {}},
    })
    monkeypatch.setattr(
        pipeline,
        "rank_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("details should be asked before reranking")
        ),
    )

    reply = pipeline.answer("Có", session_id="s")

    assert reply == "Sốt xuất hiện từ khi nào?"
    assert session.answered_questions[-1] == "detail:onset:symptom:FEVER"
    assert session.clarification_queue == [
        "detail:severity:symptom:FEVER",
        "detail:pattern:symptom:FEVER",
        "symptom:VOMIT",
    ]
    assert {"symptom_id": "symptom:FEVER", "name": "Sốt"} in session.symptoms


def test_tapped_yes_queues_list_detail_questions_before_next_symptom(monkeypatch):
    monkeypatch.setattr(pipeline, "check_rate_limit", lambda session_id: True)
    monkeypatch.setattr(pipeline, "check_llm_quota", lambda session_id: (True, ""))
    session = _patch_persistence_noop(
        monkeypatch,
        PatientSession(
            session_id="s",
            symptoms=[{"symptom_id": "symptom:ABDOMINAL_PAIN", "name": "đau bụng"}],
            answered_questions=[pipeline.GENERAL_TRIAGE_MARKER, "symptom:FEVER"],
            clarification_queue=["symptom:VOMIT"],
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("tap choice should not need analyzer")
        ),
    )
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {
        "symptom:FEVER": {
            "name_vi": "Sốt",
            "clarification_questions": {
                "onset": [
                    "Sốt xuất hiện từ khi nào?",
                    "Sốt có xuất hiện sau khi đi vùng dịch không?",
                ],
                "severity": ["Sốt cao đến mức nào?"],
            },
        },
        "symptom:VOMIT": {"name_vi": "Nôn", "clarification_questions": {}},
    })
    monkeypatch.setattr(
        pipeline,
        "rank_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("details should be asked before reranking")
        ),
    )

    reply = pipeline.answer("Có", session_id="s")

    assert reply == "Sốt xuất hiện từ khi nào?"
    assert session.answered_questions[-1] == "detail:onset:0:symptom:FEVER"
    assert session.clarification_queue == [
        "detail:onset:1:symptom:FEVER",
        "detail:severity:0:symptom:FEVER",
        "symptom:VOMIT",
    ]


def test_tapped_yes_after_multiple_presence_questions_uses_current_question(monkeypatch):
    monkeypatch.setattr(pipeline, "check_rate_limit", lambda session_id: True)
    monkeypatch.setattr(pipeline, "check_llm_quota", lambda session_id: (True, ""))
    session = _patch_persistence_noop(
        monkeypatch,
        PatientSession(
            session_id="s",
            symptoms=[{"symptom_id": "symptom:ABDOMINAL_PAIN", "name": "đau bụng"}],
            answered_questions=[
                pipeline.GENERAL_TRIAGE_MARKER,
                "symptom:FEVER",
                "symptom:VOMIT",
                "symptom:DIARRHEA",
            ],
            clarification_plan_started=True,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("tap choice should not need analyzer")
        ),
    )
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {
        "symptom:FEVER": {
            "name_vi": "Sốt",
            "clarification_questions": {"onset": "Sốt xuất hiện từ khi nào?"},
        },
        "symptom:VOMIT": {"name_vi": "Nôn", "clarification_questions": {}},
        "symptom:DIARRHEA": {
            "name_vi": "Tiêu chảy",
            "clarification_questions": {
                "onset": "Tiêu chảy bắt đầu từ khi nào?",
            },
        },
    })
    monkeypatch.setattr(
        pipeline,
        "parse_clarification_answer",
        lambda asked, answer: [{"symptom_id": "symptom:FEVER", "present": "yes"}],
    )
    monkeypatch.setattr(
        pipeline,
        "rank_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("current symptom detail should be asked before reranking")
        ),
    )

    reply = pipeline.answer("Có", session_id="s")

    assert reply == "Tiêu chảy bắt đầu từ khi nào?"
    assert session.answered_questions[-1] == "detail:onset:symptom:DIARRHEA"
    assert {"symptom_id": "symptom:DIARRHEA", "name": "Tiêu chảy"} in session.symptoms
    assert not any(
        symptom.get("symptom_id") == "symptom:FEVER"
        for symptom in session.symptoms
    )


def test_detail_answer_updates_symptom_and_asks_next_detail(monkeypatch):
    monkeypatch.setattr(pipeline, "check_rate_limit", lambda session_id: True)
    monkeypatch.setattr(pipeline, "check_llm_quota", lambda session_id: (True, ""))
    session = _patch_persistence_noop(
        monkeypatch,
        PatientSession(
            session_id="s",
            symptoms=[{"symptom_id": "symptom:FEVER", "name": "Sốt"}],
            answered_questions=["detail:onset:symptom:FEVER"],
            clarification_queue=["detail:severity:symptom:FEVER", "symptom:VOMIT"],
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("detail answer should not need analyzer")
        ),
    )
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {
        "symptom:FEVER": {
            "name_vi": "Sốt",
            "clarification_questions": {
                "severity": "Sốt cao đến mức nào?",
            },
        },
        "symptom:VOMIT": {"name_vi": "Nôn", "clarification_questions": {}},
    })
    monkeypatch.setattr(
        pipeline,
        "rank_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("remaining detail should be asked before reranking")
        ),
    )

    reply = pipeline.answer("Từ hôm qua", session_id="s")

    assert reply == "Sốt cao đến mức nào?"
    assert session.symptoms == [
        {"symptom_id": "symptom:FEVER", "name": "Sốt", "onset": "Từ hôm qua"}
    ]
    assert session.answered_questions[-1] == "detail:severity:symptom:FEVER"


def test_completed_clarification_plan_answers_with_patient_context_query(monkeypatch):
    session = PatientSession(
        session_id="s",
        symptoms=[
            {
                "symptom_id": "symptom:FEVER",
                "name": "Sốt",
                "onset": "Từ hôm qua",
                "severity": "39 độ",
            }
        ],
        answered_questions=[pipeline.GENERAL_TRIAGE_MARKER, "symptom:FEVER"],
        clarification_plan_started=True,
    )
    monkeypatch.setattr(
        pipeline,
        "rank_candidates",
        lambda symptom_ids: [
            {"disease_id": "flu", "name": "Bệnh cúm", "overlap": 1},
            {"disease_id": "gastro", "name": "Viêm dạ dày-ruột", "overlap": 1},
        ],
    )
    captured: dict[str, object] = {}

    def fake_informational(
        saved_session: PatientSession,
        question: str,
        trace_id: str,
        use_patient_context: bool = False,
        retrieval_question: str | None = None,
        **kwargs,
    ) -> str:
        captured["question"] = question
        captured["retrieval_question"] = retrieval_question
        captured["use_patient_context"] = use_patient_context
        return "contextual answer"

    monkeypatch.setattr(pipeline, "_handle_informational", fake_informational)

    assert pipeline._handle_diagnostic(session, "39 độ", trace_id="trace") == "contextual answer"
    assert captured["use_patient_context"] is True
    assert "Sốt" in str(captured["retrieval_question"])
    assert "Từ hôm qua" in str(captured["retrieval_question"])
    assert "39 độ" in str(captured["retrieval_question"])
    assert "Không hỏi thêm câu hỏi làm rõ" in str(captured["question"])


def test_unknown_clarification_answer_can_keep_asking(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    session = _patch_persistence_noop(
        monkeypatch,
        PatientSession(
            session_id="s",
            symptoms=[{"symptom_id": "symptom:COUGH", "name": "Ho"}],
            answered_questions=["symptom:FEVER"],
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(
            label="clarification_answer",
            direct_answer_requested=False,
        ),
    )
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {"symptom:FEVER": {"name_vi": "Sốt"}})
    monkeypatch.setattr(pipeline, "parse_clarification_answer", lambda asked, answer: [])
    monkeypatch.setattr(
        pipeline,
        "rank_candidates",
        lambda symptom_ids: [
            {"disease_id": "flu", "name": "Bệnh cúm", "overlap": 1},
            {"disease_id": "covid", "name": "COVID-19", "overlap": 1},
            {"disease_id": "cold", "name": "Cảm lạnh", "overlap": 1},
        ],
    )
    monkeypatch.setattr(pipeline, "discriminative_symptoms", lambda *args, **kwargs: ["symptom:RUNNY_NOSE"])
    monkeypatch.setattr(pipeline, "build_clarification", lambda symptoms: "Để thu hẹp chẩn đoán...")
    monkeypatch.setattr(
        pipeline,
        "_handle_informational",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should keep clarifying")),
    )

    assert pipeline.answer("tôi không biết", session_id="s") == "Để thu hẹp chẩn đoán..."
    assert session.answered_questions == ["symptom:FEVER", "symptom:RUNNY_NOSE"]


def test_tapped_yes_for_pending_clarification_bypasses_trivial_guardrail(monkeypatch):
    monkeypatch.setattr(pipeline, "check_rate_limit", lambda session_id: True)
    monkeypatch.setattr(pipeline, "check_llm_quota", lambda session_id: (True, ""))
    session = _patch_persistence_noop(
        monkeypatch,
        PatientSession(
            session_id="s",
            symptoms=[{"symptom_id": "symptom:ABDOMINAL_PAIN", "name": "đau bụng"}],
            answered_questions=[pipeline.GENERAL_TRIAGE_MARKER, "symptom:FEVER"],
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("tap choice should not need analyzer")),
    )
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {"symptom:FEVER": {"name_vi": "Sốt"}})
    monkeypatch.setattr(
        pipeline,
        "parse_clarification_answer",
        lambda asked, answer: [{"symptom_id": "symptom:FEVER", "present": "yes"}],
    )
    monkeypatch.setattr(
        pipeline,
        "rank_candidates",
        lambda symptom_ids: [
            {"disease_id": "flu", "name": "Bệnh cúm", "overlap": 1},
            {"disease_id": "cold", "name": "Cảm lạnh", "overlap": 1},
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "_handle_informational",
        lambda *args, **kwargs: "answer with fever context",
    )

    assert pipeline.answer("Có", session_id="s") == "answer with fever context"
    assert {"symptom_id": "symptom:FEVER", "name": "Sốt"} in session.symptoms


def test_clarification_uses_fixed_top_k_queue_then_answers(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    session = _patch_persistence_noop(
        monkeypatch,
        PatientSession(
            session_id="s",
            symptoms=[{"symptom_id": "symptom:ABDOMINAL_PAIN", "name": "đau bụng"}],
            answered_questions=[pipeline.GENERAL_TRIAGE_MARKER],
        ),
    )
    monkeypatch.setattr(pipeline, "build_clarification", lambda symptoms: f"ASK {symptoms[0]}")
    monkeypatch.setattr(
        pipeline,
        "rank_candidates",
        lambda symptom_ids: [
            {"disease_id": "a", "name": "A", "overlap": 1},
            {"disease_id": "b", "name": "B", "overlap": 1},
            {"disease_id": "c", "name": "C", "overlap": 1},
        ],
    )
    discriminative_calls: list[object] = []

    def fake_discriminative(*args, **kwargs):
        discriminative_calls.append((args, kwargs))
        return ["symptom:FEVER", "symptom:VOMIT", "symptom:DIARRHEA"]

    monkeypatch.setattr(pipeline, "discriminative_symptoms", fake_discriminative)
    monkeypatch.setattr(
        pipeline,
        "parse_clarification_answer",
        lambda asked, answer: [{"symptom_id": asked[0]["symptom_id"], "present": "no"}],
    )
    monkeypatch.setattr(pipeline, "_handle_informational", lambda *args, **kwargs: "final answer")

    first = pipeline._handle_clarification_answer(session, "Bắt đầu", "trace")
    second = pipeline._handle_clarification_answer(session, "Không", "trace")
    third = pipeline._handle_clarification_answer(session, "Không", "trace")
    final = pipeline._handle_clarification_answer(session, "Không", "trace")

    assert "ASK symptom:FEVER" in first
    assert "ASK symptom:VOMIT" in second
    assert "ASK symptom:DIARRHEA" in third
    assert final == "final answer"
    assert len(discriminative_calls) == 1
    assert session.answered_questions == [
        pipeline.GENERAL_TRIAGE_MARKER,
        "symptom:FEVER",
        "symptom:VOMIT",
        "symptom:DIARRHEA",
    ]


def test_general_triage_start_asks_known_symptom_details_before_other_symptoms(monkeypatch):
    session = PatientSession(
        session_id="s",
        symptoms=[{"symptom_id": "symptom:ABDOMINAL_PAIN", "name": "đau bụng"}],
        answered_questions=[pipeline.GENERAL_TRIAGE_MARKER],
    )
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {
        "symptom:ABDOMINAL_PAIN": {
            "name_vi": "Đau bụng",
            "clarification_questions": {
                "onset": ["Đau bụng bắt đầu từ khi nào?"],
                "severity": ["Đau bụng mức độ nào?"],
            },
        },
    })
    monkeypatch.setattr(
        pipeline,
        "rank_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("known symptom details should be asked before ranking")
        ),
    )

    reply = pipeline._handle_clarification_answer(session, "Bắt đầu", "trace")

    assert reply == "Đau bụng bắt đầu từ khi nào?"
    assert session.answered_questions[-1] == "detail:onset:0:symptom:ABDOMINAL_PAIN"
    assert session.clarification_queue == [
        "detail:severity:0:symptom:ABDOMINAL_PAIN",
    ]


def test_clarification_queue_filters_duplicate_nausea_and_vomiting(monkeypatch):
    session = PatientSession(
        session_id="s",
        symptoms=[{"symptom_id": "symptom:ABDOMINAL_PAIN", "name": "đau bụng"}],
        answered_questions=[pipeline.GENERAL_TRIAGE_MARKER],
    )
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {
        "symptom:FEVER": {"name_vi": "Sốt"},
        "symptom:NAUSEA": {"name_vi": "Buồn nôn"},
        "symptom:VOMIT": {"name_vi": "Nôn"},
        "symptom:DIARRHEA": {"name_vi": "Tiêu chảy"},
    })
    monkeypatch.setattr(
        pipeline,
        "rank_candidates",
        lambda symptom_ids: [
            {"disease_id": "a", "name": "A", "overlap": 1},
            {"disease_id": "b", "name": "B", "overlap": 1},
            {"disease_id": "c", "name": "C", "overlap": 1},
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "discriminative_symptoms",
        lambda *args, **kwargs: [
            "symptom:FEVER",
            "symptom:NAUSEA",
            "symptom:VOMIT",
            "symptom:DIARRHEA",
        ],
    )

    reply = pipeline._handle_diagnostic(session, "Bắt đầu", trace_id="trace")

    assert "symptom:FEVER" in reply
    assert session.clarification_queue == ["symptom:NAUSEA", "symptom:DIARRHEA"]


def test_repeated_clarification_parse_failures_force_answer(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    session = _patch_persistence_noop(
        monkeypatch,
        PatientSession(
            session_id="s",
            symptoms=[{"symptom_id": "symptom:COUGH", "name": "Ho"}],
            answered_questions=["symptom:FEVER"],
        ),
    )
    session.clarification_parse_failures = 1
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(
            label="clarification_answer",
            direct_answer_requested=False,
        ),
    )
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {"symptom:FEVER": {"name_vi": "Sốt"}})
    monkeypatch.setattr(pipeline, "parse_clarification_answer", lambda asked, answer: [])
    monkeypatch.setattr(
        pipeline,
        "rank_candidates",
        lambda symptom_ids: [
            {"disease_id": "flu", "name": "Bệnh cúm", "overlap": 1},
            {"disease_id": "covid", "name": "COVID-19", "overlap": 1},
            {"disease_id": "cold", "name": "Cảm lạnh", "overlap": 1},
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "discriminative_symptoms",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should stop clarifying")),
    )
    monkeypatch.setattr(
        pipeline,
        "_handle_informational",
        lambda *args, **kwargs: "answer from available symptoms",
    )

    assert pipeline.answer("tôi không biết", session_id="s") == "answer from available symptoms"
    assert session.clarification_parse_failures == 2


def test_diagnostic_stops_asking_when_shortlist_is_small(monkeypatch):
    session = PatientSession(
        session_id="s",
        symptoms=[{"symptom_id": "symptom:COUGH", "name": "Ho"}],
        answered_questions=[pipeline.GENERAL_TRIAGE_MARKER],
    )
    monkeypatch.setattr(
        pipeline,
        "rank_candidates",
        lambda symptom_ids: [
            {"disease_id": "flu", "name": "Bệnh cúm", "overlap": 1},
            {"disease_id": "cold", "name": "Cảm lạnh", "overlap": 1},
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "build_clarification",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not ask")),
    )
    monkeypatch.setattr(
        pipeline,
        "_handle_informational",
        lambda saved_session, question, trace_id, use_patient_context=False, retrieval_question=None, **kwargs: "shortlist answer",
    )

    assert pipeline._handle_diagnostic(session, "Tôi bị ho", trace_id="trace") == "shortlist answer"


def test_kg_and_hybrid_retrieval_run_in_parallel(monkeypatch, caplog):
    barrier = threading.Barrier(2, timeout=1.0)
    calls: list[str] = []
    caplog.set_level(logging.INFO, logger=pipeline.log.name)

    def fake_kg(question: str, trace_id: str) -> str:
        calls.append("kg")
        barrier.wait()
        return "KG context"

    def fake_hybrid(question: str, trace_id: str) -> list[Hit]:
        calls.append("hybrid")
        barrier.wait()
        return [
            Hit(
                text="context",
                score=1.0,
                source_type="disease",
                source_name="Nguồn",
                heading_path="",
                source_slug="nguon",
            )
        ]

    monkeypatch.setattr(pipeline, "_load_kg_context", fake_kg)
    monkeypatch.setattr(pipeline, "_load_hybrid_hits", fake_hybrid)
    monkeypatch.setattr(pipeline, "generate", lambda question, hits, kg_text="", patient=None, **kwargs: "answer")

    reply = pipeline._handle_informational(
        PatientSession(session_id="s"),
        "Tôi bị ho",
        trace_id="trace",
    )

    assert reply == "answer"
    assert sorted(calls) == ["hybrid", "kg"]
    assert "stage=parallel_retrieval" in caplog.text


def test_infer_answer_domain_prefers_named_disease_over_unrelated_drug_hits():
    hits = [
        Hit(
            text="Chống chỉ định Diclofenac ở phụ nữ có thai.",
            score=1.0,
            source_type="drug",
            source_name="Diclofenac",
            heading_path="5 Chống chỉ định",
            source_slug="diclofenac",
        ),
        Hit(
            text="Điều trị suy giáp phụ nữ có thai cần điều chỉnh levothyroxine.",
            score=0.9,
            source_type="disease",
            source_name="Suy giáp",
            heading_path="III. Điều trị",
            source_slug="suy_giap",
        ),
    ]

    assert (
        pipeline._infer_answer_domain(
            "Tôi bị suy giáp và đang mang thai. Tôi cần điều chỉnh liều thuốc thế nào?",
            hits,
            "contextual_drug_info",
        )
        == "disease_info"
    )


def test_handle_informational_uses_inferred_domain_when_plan_conflicts(monkeypatch):
    hits = [
        Hit(
            text="Chống chỉ định Diclofenac ở phụ nữ có thai.",
            score=1.0,
            source_type="drug",
            source_name="Diclofenac",
            heading_path="5 Chống chỉ định",
            source_slug="diclofenac",
        ),
        Hit(
            text="Điều trị suy giáp phụ nữ có thai cần điều chỉnh levothyroxine.",
            score=0.9,
            source_type="disease",
            source_name="Suy giáp",
            heading_path="III. Điều trị",
            source_slug="suy_giap",
        ),
    ]
    captured: dict = {}

    monkeypatch.setattr(pipeline, "_load_kg_context", lambda question, trace_id: "")
    monkeypatch.setattr(pipeline, "_load_hybrid_hits", lambda *args: hits)
    monkeypatch.setattr(
        pipeline,
        "generate",
        lambda question, hits, kg_text="", patient=None, **kwargs: (
            captured.update(kwargs) or "answer"
        ),
    )

    reply = pipeline._handle_informational(
        PatientSession(session_id="s"),
        "Tôi bị suy giáp và đang mang thai. Tôi cần điều chỉnh liều thuốc thế nào?",
        trace_id="trace",
        turn_intent="contextual_drug_info",
        evidence_plan={
            "domain": "drug_info",
            "source_type": "drug",
            "entity": "Diclofenac",
            "answer_slot": "dose",
            "confidence": 0.9,
        },
    )

    assert reply == "answer"
    assert captured["answer_domain"] == "disease_info"


def test_informational_raises_when_kg_retrieval_fails(monkeypatch):
    hits = [
        Hit(
            text="context",
            score=1.0,
            source_type="disease",
            source_name="Nguồn",
            heading_path="",
            source_slug="nguon",
        )
    ]

    def fail_kg(question: str, trace_id: str) -> str:
        raise Neo4jUnavailable("Neo4j fulltext search failed for disease_name")

    monkeypatch.setattr(pipeline, "_load_kg_context", fail_kg)
    monkeypatch.setattr(pipeline, "_load_hybrid_hits", lambda question, trace_id: hits)

    with pytest.raises(Neo4jUnavailable):
        pipeline._handle_informational(
            PatientSession(session_id="s"),
            "Tôi bị sốt",
            trace_id="trace",
        )


def test_pipeline_logs_core_timing_stages(monkeypatch, caplog):
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch)
    caplog.set_level(logging.INFO, logger=pipeline.log.name)
    monkeypatch.setattr(pipeline, "analyze_turn", lambda *args, **kwargs: _analysis())
    monkeypatch.setattr(pipeline, "_load_kg_context", lambda question, trace_id: "")
    monkeypatch.setattr(pipeline, "_load_hybrid_hits", lambda question, trace_id: [])
    monkeypatch.setattr(pipeline, "generate", lambda question, hits, kg_text="", patient=None, **kwargs: "answer")

    assert pipeline.answer("Tôi bị ho", session_id="s") == "answer"

    for stage in ("preflight", "load_session", "turn_analysis", "parallel_retrieval", "generate", "total"):
        assert f"stage={stage}" in caplog.text


def test_answer_with_meta_records_timing_timeline(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch)
    monkeypatch.setattr(pipeline, "analyze_turn", lambda *args, **kwargs: _analysis())
    monkeypatch.setattr(
        pipeline,
        "_load_kg_context_with_debug",
        lambda question, trace_id: ("", {}),
    )
    monkeypatch.setattr(
        pipeline,
        "_load_hybrid_hits_with_debug",
        lambda question, trace_id: (
            [],
            {
                "query": question,
                "dense_hits": [],
                "sparse_hits": [],
                "fused_hits": [],
                "reranked_hits": [],
            },
        ),
    )

    def fake_generate(question, hits, kg_text="", patient=None, return_meta=False, **kwargs):
        if return_meta:
            return "answer", {
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                "model": "test-model",
            }
        return "answer"

    monkeypatch.setattr(pipeline, "generate", fake_generate)

    reply, meta = pipeline.answer_with_meta("Tôi bị ho", session_id="s")

    stages = [entry["stage"] for entry in meta["timings"]]
    assert reply == "answer"
    for stage in ("load_session", "preflight", "turn_analysis", "parallel_retrieval", "generate", "route", "persist", "total"):
        assert stage in stages
    assert all(isinstance(entry["ms"], float) for entry in meta["timings"])
    assert meta["route_label"] == "informational"
    assert meta["outcome"] == "informational"
    assert meta["usage"][0]["stage"] == "generator"


def test_answer_with_meta_records_structured_graph_trace(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch)
    analysis = _analysis(rewritten="Ho kéo dài")
    monkeypatch.setattr(pipeline, "analyze_turn", lambda *args, **kwargs: analysis)

    kg_context = KGContext(
        matched_entities=[{"id": "symptom:COUGH", "type": "symptom", "name": "Ho"}],
        related_diseases=[{"id": "disease:FLU", "name": "Cúm"}],
        related_drugs=[{"id": "drug:DEXTROMETHORPHAN", "name": "Dextromethorphan"}],
        related_symptoms=[{"id": "symptom:FEVER", "name": "Sốt"}],
        related_adrs=[{"id": "symptom:DROWSINESS", "name": "Buồn ngủ"}],
        relationships=["Ho là triệu chứng của Cúm"],
    )
    monkeypatch.setattr(pipeline, "kg_search", lambda question: kg_context)

    def fake_format_kg_context(actual_context: KGContext) -> str:
        assert actual_context is kg_context
        return "KG prompt context"

    monkeypatch.setattr(pipeline, "format_kg_context", fake_format_kg_context)
    hits = [
        Hit(
            text="context",
            score=0.9,
            source_type="disease",
            source_name="Cúm",
            heading_path="Triệu chứng",
            source_slug="cum",
            chunk_id="disease:cum:trieu-chung",
        )
    ]
    retrieval_debug = {
        "query": "Ho kéo dài",
        "dense_hits": [{"rank": 1, "stage": "dense", "chunk_id": "dense-1"}],
        "sparse_hits": [{"rank": 1, "stage": "sparse", "chunk_id": "sparse-1"}],
        "fused_hits": [{"rank": 1, "stage": "fused", "chunk_id": "fused-1"}],
        "reranked_hits": [
            {"rank": 1, "stage": "reranked", "chunk_id": "disease:cum:trieu-chung"}
        ],
        "timings_ms": {
            "dense_search": 1.1,
            "sparse_search": 2.2,
            "fusion": 3.3,
            "rerank": 4.4,
            "hybrid_total": 11.0,
        },
    }
    monkeypatch.setattr(
        pipeline,
        "hybrid_search_with_debug",
        lambda question, top_k, on_stage=None: (hits, retrieval_debug),
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "hybrid_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("meta collection should use hybrid_search_with_debug")
        ),
    )

    def fake_generate(question, actual_hits, kg_text="", patient=None, return_meta=False, **kwargs):
        assert question == "Ho kéo dài"
        assert actual_hits is hits
        assert kg_text == "KG prompt context"
        assert return_meta is True
        return "answer", {
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            "model": "test-model",
        }

    monkeypatch.setattr(pipeline, "generate", fake_generate)

    reply, meta = pipeline.answer_with_meta("Tôi bị ho", session_id="s")

    assert reply == "answer"
    assert meta["rewrite_query"] == {
        "original": "Tôi bị ho",
        "rewritten": "Ho kéo dài",
        "confident": True,
    }
    assert meta["kg_context"] == {
        "matched_entities": kg_context.matched_entities,
        "related_diseases": kg_context.related_diseases,
        "related_drugs": kg_context.related_drugs,
        "related_symptoms": kg_context.related_symptoms,
        "related_adrs": kg_context.related_adrs,
        "relationships": kg_context.relationships,
    }
    assert meta["retrieval_debug"] is retrieval_debug
    assert [node["id"] for node in meta["graph_nodes"]] == [
        "input",
        "load_session",
        "preflight",
        "turn_analysis",
        "rewrite",
        "route",
        "diagnostic_general_triage",
        "diagnostic_rank",
        "diagnostic_clarification",
        "clarification_parse",
        "entity_ingest",
        "kg_search",
        "dense_search",
        "sparse_search",
        "fusion",
        "rerank",
        "generate",
        "persist",
        "total",
    ]
    nodes = {node["id"]: node for node in meta["graph_nodes"]}
    assert all(
        set(node) == {"id", "label", "status", "ms", "input", "output", "raw"}
        for node in meta["graph_nodes"]
    )
    # Diagnostic nodes never ran on this informational turn -> skipped (grey).
    diagnostic_ids = {
        "diagnostic_general_triage",
        "diagnostic_rank",
        "diagnostic_clarification",
        "clarification_parse",
    }
    assert all(
        node["status"] == "skipped"
        for node in meta["graph_nodes"]
        if node["id"] in diagnostic_ids
    )
    assert all(
        node["status"] == "success"
        for node in meta["graph_nodes"]
        if node["id"] not in diagnostic_ids
    )
    assert nodes["input"]["input"] == {
        "question": "Tôi bị ho",
        "session_id": "s",
        "mode": "auto",
    }
    assert nodes["turn_analysis"]["raw"] == analysis
    assert nodes["rewrite"]["output"] == meta["rewrite_query"]
    assert nodes["kg_search"]["output"] == meta["kg_context"]
    assert nodes["dense_search"]["output"] == retrieval_debug["dense_hits"]
    assert nodes["sparse_search"]["output"] == retrieval_debug["sparse_hits"]
    assert nodes["fusion"]["output"] == retrieval_debug["fused_hits"]
    assert nodes["rerank"]["output"] == retrieval_debug["reranked_hits"]
    assert nodes["dense_search"]["ms"] == 1.1
    assert nodes["sparse_search"]["ms"] == 2.2
    assert nodes["fusion"]["ms"] == 3.3
    assert nodes["rerank"]["ms"] == 4.4
    assert nodes["generate"]["output"] == {"reply": "answer"}
    assert nodes["total"]["output"] == {
        "outcome": "informational",
        "latency_ms_total": meta["latency_ms_total"],
    }
    assert not any(key.startswith("_") for key in meta)
    for field in (
        "timings",
        "retrieved",
        "usage",
        "route_label",
        "outcome",
        "latency_ms_total",
    ):
        assert field in meta


def test_answer_with_meta_streams_input_rewrite_route_node_events(monkeypatch):
    import queue as _queue

    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch)
    analysis = _analysis(rewritten="Ho kéo dài")
    monkeypatch.setattr(pipeline, "analyze_turn", lambda *args, **kwargs: analysis)
    monkeypatch.setattr(pipeline, "kg_search", lambda question: KGContext())
    monkeypatch.setattr(pipeline, "format_kg_context", lambda kg_result: "KG context")
    hits = [
        Hit(
            text="c", score=1.0, source_type="disease",
            source_name="N", heading_path="", source_slug="n",
        )
    ]
    monkeypatch.setattr(
        pipeline,
        "hybrid_search_with_debug",
        lambda question, top_k, on_stage=None: (hits, {"query": question}),
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "generate",
        lambda question, actual_hits, kg_text="", patient=None, return_meta=False, **kwargs: (
            ("answer", {}) if return_meta else "answer"
        ),
    )

    sink: _queue.Queue = _queue.Queue()
    pipeline._install_event_sink(sink)
    try:
        reply, meta = pipeline.answer_with_meta("Tôi bị ho", session_id="s")
    finally:
        pipeline._install_event_sink(None)

    assert reply == "answer"
    streamed_ids = []
    while not sink.empty():
        streamed_ids.append(sink.get_nowait()["id"])

    # The three nodes that previously never lit up live.
    assert {"input", "rewrite", "route"}.issubset(set(streamed_ids))
    # Live emits must not corrupt the recorded timings / final render.
    timing_stages = [entry["stage"] for entry in meta["timings"]]
    assert "input" not in timing_stages
    assert "rewrite" not in timing_stages


def test_diagnostic_turn_surfaces_diagnostic_nodes_and_snapshot(monkeypatch):
    import queue as _queue

    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {
        "symptom:ABDOMINAL_PAIN": {"name_vi": "Đau bụng"},
        "symptom:FEVER": {"name_vi": "Sốt"},
    })
    monkeypatch.setattr(pipeline, "rank_candidates", lambda symptom_ids: [
        {"disease_id": "flu", "name": "Cúm", "overlap": 1},
        {"disease_id": "covid", "name": "COVID-19", "overlap": 1},
        {"disease_id": "cold", "name": "Cảm lạnh", "overlap": 1},
    ])
    monkeypatch.setattr(pipeline, "should_ask_clarification", lambda candidates: True)
    monkeypatch.setattr(pipeline, "discriminative_symptoms", lambda *a, **k: ["symptom:FEVER"])
    monkeypatch.setattr(pipeline, "build_clarification", lambda symptoms: "Bạn có bị Sốt không?")

    session = PatientSession(
        session_id="s",
        symptoms=[{"symptom_id": "symptom:ABDOMINAL_PAIN", "name": "Đau bụng"}],
        answered_questions=[pipeline.GENERAL_TRIAGE_MARKER],
    )

    meta = {"trace_id": "t", "_collect_graph": True}
    sink: _queue.Queue = _queue.Queue()
    pipeline._META_LOCAL.current = meta
    pipeline._install_event_sink(sink)
    try:
        reply = pipeline._handle_diagnostic(session, "tôi bị đau bụng", trace_id="t")
    finally:
        pipeline._install_event_sink(None)
        pipeline._META_LOCAL.current = None

    assert reply == "Bạn có bị Sốt không?"

    # Snapshot reflects post-rank / post-ask session state.
    snap = meta["_graph_diagnostic"]
    assert snap["candidate_diseases"][0]["disease_id"] == "flu"
    assert "symptom:FEVER" in snap["answered_questions"]

    # Diagnostic stages stream live to the sink.
    streamed = []
    while not sink.empty():
        streamed.append(sink.get_nowait()["id"])
    assert "diagnostic_rank" in streamed
    assert "diagnostic_clarification" in streamed

    nodes = {
        node["id"]: node
        for node in pipeline._build_graph_nodes(
            meta, "tôi bị đau bụng", "s", "auto", reply
        )
    }
    assert {
        "diagnostic_general_triage",
        "diagnostic_rank",
        "diagnostic_clarification",
        "clarification_parse",
    }.issubset(nodes)
    assert nodes["diagnostic_rank"]["status"] == "success"
    assert nodes["diagnostic_clarification"]["status"] == "success"
    # Stages that never ran on this turn stay grey.
    assert nodes["diagnostic_general_triage"]["status"] == "skipped"
    assert nodes["clarification_parse"]["status"] == "skipped"
    assert nodes["diagnostic_rank"]["output"] == {
        "candidate_diseases": snap["candidate_diseases"]
    }
    assert nodes["diagnostic_clarification"]["output"]["asked"] == "symptom:FEVER"
    assert (
        nodes["diagnostic_clarification"]["output"]["clarification_plan_started"]
        == snap["clarification_plan_started"]
    )


def test_clarification_answer_turn_lights_parse_node(monkeypatch):
    import queue as _queue

    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {"symptom:FEVER": {"name_vi": "Sốt"}})
    monkeypatch.setattr(pipeline, "_handle_diagnostic", lambda *a, **k: "next step")

    session = PatientSession(
        session_id="s",
        symptoms=[{"symptom_id": "symptom:COUGH", "name": "Ho"}],
        answered_questions=["symptom:FEVER"],
    )

    meta = {"trace_id": "t", "_collect_graph": True}
    sink: _queue.Queue = _queue.Queue()
    pipeline._META_LOCAL.current = meta
    pipeline._install_event_sink(sink)
    try:
        reply = pipeline._handle_clarification_answer(session, "Có", trace_id="t")
    finally:
        pipeline._install_event_sink(None)
        pipeline._META_LOCAL.current = None

    assert reply == "next step"

    snap = meta["_graph_diagnostic"]
    assert snap["events"]["parsed_answers"] == [
        {"symptom_id": "symptom:FEVER", "present": "yes"}
    ]
    assert snap["events"]["last_asked"] == "symptom:FEVER"

    streamed = []
    while not sink.empty():
        streamed.append(sink.get_nowait()["id"])
    assert "clarification_parse" in streamed

    nodes = {
        node["id"]: node
        for node in pipeline._build_graph_nodes(meta, "Có", "s", "auto", reply)
    }
    assert nodes["clarification_parse"]["status"] == "success"
    assert nodes["clarification_parse"]["output"]["parsed_answers"] == [
        {"symptom_id": "symptom:FEVER", "present": "yes"}
    ]


def test_diagnostic_snapshot_is_noop_without_collect_graph(monkeypatch):
    monkeypatch.setattr(pipeline, "symptom_catalog", lambda: {
        "symptom:ABDOMINAL_PAIN": {"name_vi": "Đau bụng"},
        "symptom:FEVER": {"name_vi": "Sốt"},
    })
    monkeypatch.setattr(pipeline, "rank_candidates", lambda symptom_ids: [
        {"disease_id": "flu", "name": "Cúm", "overlap": 1},
    ])
    monkeypatch.setattr(pipeline, "should_ask_clarification", lambda candidates: True)
    monkeypatch.setattr(pipeline, "discriminative_symptoms", lambda *a, **k: ["symptom:FEVER"])
    monkeypatch.setattr(pipeline, "build_clarification", lambda symptoms: "Bạn có bị Sốt không?")

    session = PatientSession(
        session_id="s",
        symptoms=[{"symptom_id": "symptom:ABDOMINAL_PAIN", "name": "Đau bụng"}],
        answered_questions=[pipeline.GENERAL_TRIAGE_MARKER],
    )

    # No meta installed -> _meta() is None -> snapshot helper must be a no-op
    # and must not raise or alter the reply.
    assert pipeline._meta() is None
    reply = pipeline._handle_diagnostic(session, "tôi bị đau bụng", trace_id="t")
    assert reply == "Bạn có bị Sốt không?"


def test_answer_with_meta_records_regex_guard_persistence(monkeypatch):
    _patch_persistence_noop(monkeypatch)
    monkeypatch.setattr(pipeline, "check_rate_limit", lambda session_id: True)
    monkeypatch.setattr(
        pipeline,
        "regex_check",
        lambda question: {"verdict": "off_topic"},
    )
    monkeypatch.setattr(
        pipeline,
        "check_llm_quota",
        lambda session_id: (_ for _ in ()).throw(
            AssertionError("quota should not run after regex guard")
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("analyzer should not run after regex guard")
        ),
    )

    reply, meta = pipeline.answer_with_meta("Viết code Python", session_id="s")

    nodes = {node["id"]: node for node in meta["graph_nodes"]}
    assert reply == "Tôi chỉ hỗ trợ các câu hỏi về sức khỏe, bệnh lý và thuốc."
    assert nodes["preflight"]["status"] == "success"
    assert nodes["persist"]["status"] == "success"
    assert nodes["persist"]["ms"] is not None


def test_answer_with_meta_marks_regex_guard_persistence_failure(monkeypatch):
    _patch_persistence_noop(monkeypatch)
    monkeypatch.setattr(pipeline, "check_rate_limit", lambda session_id: True)
    monkeypatch.setattr(
        pipeline,
        "regex_check",
        lambda question: {"verdict": "off_topic"},
    )
    monkeypatch.setattr(
        pipeline,
        "save_session",
        lambda session: (_ for _ in ()).throw(RuntimeError("persist failed")),
    )

    reply, meta = pipeline.answer_with_meta("Viết code Python", session_id="s")

    nodes = {node["id"]: node for node in meta["graph_nodes"]}
    assert reply == TECHNICAL_ERROR_REPLY
    assert nodes["preflight"]["status"] != "error"
    assert nodes["persist"]["status"] == "error"
    assert nodes["persist"]["ms"] is not None


def test_answer_with_meta_records_blocked_mode_route_before_persist(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch)
    monkeypatch.setattr(pipeline, "analyze_turn", lambda *args, **kwargs: _analysis())
    monkeypatch.setattr(
        pipeline,
        "_route",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("blocked mode should not route")
        ),
    )

    reply, meta = pipeline.answer_with_meta(
        "Cho tôi thông tin về cúm",
        session_id="s",
        mode="diagnostic",
    )

    stages = [entry["stage"] for entry in meta["timings"]]
    assert "chế độ Thông tin" in reply
    assert stages.index("route") < stages.index("persist")


def test_answer_with_meta_marks_failed_retrieval_node(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch)
    monkeypatch.setattr(pipeline, "analyze_turn", lambda *args, **kwargs: _analysis())
    monkeypatch.setattr(
        pipeline,
        "_load_kg_context_with_debug",
        lambda question, trace_id: ("", {}),
    )
    partial_debug = {
        "query": "Tôi bị ho",
        "dense_hits": [{"rank": 1, "stage": "dense", "chunk_id": "dense-1"}],
        "timings_ms": {
            "dense_search": 1.1,
            "sparse_search": 2.2,
            "hybrid_total": 3.3,
        },
        "error_stage": "sparse_search",
    }

    class PartialRetrievalFailure(QdrantUnavailable):
        def __init__(self) -> None:
            super().__init__("Sparse retrieval failed")
            self.debug = partial_debug

    monkeypatch.setattr(
        pipeline,
        "_load_hybrid_hits_with_debug",
        lambda question, trace_id: (_ for _ in ()).throw(
            PartialRetrievalFailure()
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "generate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("generation should not run after retrieval failure")
        ),
    )

    reply, meta = pipeline.answer_with_meta("Tôi bị ho", session_id="s")

    nodes = {node["id"]: node for node in meta["graph_nodes"]}
    assert reply == TECHNICAL_ERROR_REPLY
    assert meta["outcome"] == "technical_error"
    assert meta["retrieval_debug"] == partial_debug
    assert nodes["kg_search"]["status"] == "success"
    assert nodes["kg_search"]["ms"] is not None
    assert nodes["dense_search"]["status"] == "success"
    assert nodes["dense_search"]["output"] == partial_debug["dense_hits"]
    assert nodes["dense_search"]["ms"] == 1.1
    assert nodes["sparse_search"]["status"] == "error"
    assert nodes["sparse_search"]["ms"] == 2.2
    assert nodes["fusion"]["status"] == "skipped"
    assert nodes["rerank"]["status"] == "skipped"
    assert nodes["generate"]["status"] == "skipped"
    assert nodes["total"]["status"] == "success"
    assert not any(key.startswith("_") for key in meta)


def test_answer_with_meta_preserves_retrieval_sibling_completed_during_shutdown(
    monkeypatch,
):
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch)
    monkeypatch.setattr(pipeline, "analyze_turn", lambda *args, **kwargs: _analysis())
    release_hybrid = threading.Event()
    hybrid_started = threading.Event()
    original_executor = pipeline.ThreadPoolExecutor
    retrieval_debug = {
        "query": "Tôi bị ho",
        "dense_hits": [{"rank": 1, "stage": "dense", "chunk_id": "dense-1"}],
        "sparse_hits": [{"rank": 1, "stage": "sparse", "chunk_id": "sparse-1"}],
        "fused_hits": [{"rank": 1, "stage": "fused", "chunk_id": "fused-1"}],
        "reranked_hits": [{"rank": 1, "stage": "reranked", "chunk_id": "reranked-1"}],
    }

    class ReleasingExecutor(original_executor):
        def __exit__(self, exc_type, exc_value, traceback):
            release_hybrid.set()
            return super().__exit__(exc_type, exc_value, traceback)

    def slow_hybrid(question: str, trace_id: str):
        hybrid_started.set()
        assert release_hybrid.wait(timeout=1.0)
        return [], retrieval_debug

    def fail_kg(question: str, trace_id: str):
        assert hybrid_started.wait(timeout=1.0)
        raise Neo4jUnavailable("Neo4j search failed")

    monkeypatch.setattr(pipeline, "ThreadPoolExecutor", ReleasingExecutor)
    monkeypatch.setattr(pipeline, "_load_kg_context_with_debug", fail_kg)
    monkeypatch.setattr(pipeline, "_load_hybrid_hits_with_debug", slow_hybrid)

    reply, meta = pipeline.answer_with_meta("Tôi bị ho", session_id="s")

    nodes = {node["id"]: node for node in meta["graph_nodes"]}
    assert reply == TECHNICAL_ERROR_REPLY
    assert meta["retrieval_debug"] == retrieval_debug
    assert nodes["kg_search"]["status"] == "error"
    assert nodes["dense_search"]["status"] == "success"
    assert nodes["sparse_search"]["status"] == "success"
    assert nodes["fusion"]["status"] == "success"
    assert nodes["rerank"]["status"] == "success"


def test_answer_with_meta_marks_kg_failure_with_elapsed(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch)
    monkeypatch.setattr(pipeline, "analyze_turn", lambda *args, **kwargs: _analysis())
    retrieval_debug = {
        "query": "Tôi bị ho",
        "dense_hits": [],
        "sparse_hits": [],
        "fused_hits": [],
        "reranked_hits": [],
    }
    monkeypatch.setattr(
        pipeline,
        "_load_kg_context_with_debug",
        lambda question, trace_id: (_ for _ in ()).throw(
            Neo4jUnavailable("Neo4j search failed")
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_load_hybrid_hits_with_debug",
        lambda question, trace_id: ([], retrieval_debug),
    )

    reply, meta = pipeline.answer_with_meta("Tôi bị ho", session_id="s")

    nodes = {node["id"]: node for node in meta["graph_nodes"]}
    assert reply == TECHNICAL_ERROR_REPLY
    assert nodes["kg_search"]["status"] == "error"
    assert nodes["kg_search"]["ms"] is not None


def test_answer_with_meta_marks_both_parallel_retrieval_failures(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch)
    monkeypatch.setattr(pipeline, "analyze_turn", lambda *args, **kwargs: _analysis())
    original_wait = pipeline.wait
    retrieval_debug = {
        "query": "Tôi bị ho",
        "timings_ms": {"dense_search": 1.1, "hybrid_total": 1.1},
        "error_stage": "dense_search",
    }

    class PartialRetrievalFailure(QdrantUnavailable):
        def __init__(self) -> None:
            super().__init__("Dense retrieval failed")
            self.debug = retrieval_debug

    def wait_for_both(futures, return_when=None):
        original_wait(futures)
        return list(futures), set()

    monkeypatch.setattr(pipeline, "wait", wait_for_both)
    monkeypatch.setattr(
        pipeline,
        "_load_kg_context_with_debug",
        lambda question, trace_id: (_ for _ in ()).throw(
            Neo4jUnavailable("Neo4j search failed")
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_load_hybrid_hits_with_debug",
        lambda question, trace_id: (_ for _ in ()).throw(
            PartialRetrievalFailure()
        ),
    )

    reply, meta = pipeline.answer_with_meta("Tôi bị ho", session_id="s")

    nodes = {node["id"]: node for node in meta["graph_nodes"]}
    assert reply == TECHNICAL_ERROR_REPLY
    assert nodes["kg_search"]["status"] == "error"
    assert nodes["kg_search"]["ms"] is not None
    assert meta["retrieval_debug"] == retrieval_debug
    assert nodes["dense_search"]["status"] == "error"
    assert nodes["dense_search"]["ms"] == 1.1


def test_answer_with_meta_marks_generation_failure_not_route(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch)
    monkeypatch.setattr(pipeline, "analyze_turn", lambda *args, **kwargs: _analysis())
    monkeypatch.setattr(
        pipeline,
        "_load_kg_context_with_debug",
        lambda question, trace_id: ("", {}),
    )
    monkeypatch.setattr(
        pipeline,
        "_load_hybrid_hits_with_debug",
        lambda question, trace_id: (
            [],
            {
                "query": question,
                "dense_hits": [],
                "sparse_hits": [],
                "fused_hits": [],
                "reranked_hits": [],
            },
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "generate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("generation failed")
        ),
    )

    reply, meta = pipeline.answer_with_meta("Tôi bị ho", session_id="s")

    nodes = {node["id"]: node for node in meta["graph_nodes"]}
    assert reply == TECHNICAL_ERROR_REPLY
    assert nodes["generate"]["status"] == "error"
    assert nodes["generate"]["ms"] is not None
    assert nodes["generate"]["output"] is None
    assert nodes["route"]["status"] != "error"


@pytest.mark.parametrize("early_return", ("greeting", "clarification"))
def test_answer_with_meta_marks_early_persistence_failure(monkeypatch, early_return):
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch)
    analysis = _analysis(label="greeting_other")
    if early_return == "clarification":
        analysis = _analysis()
        analysis["rewrite"].update({
            "confident": False,
            "clarification": "Bạn có thể mô tả rõ hơn không?",
        })
    monkeypatch.setattr(pipeline, "analyze_turn", lambda *args, **kwargs: analysis)
    monkeypatch.setattr(
        pipeline,
        "save_session",
        lambda session: (_ for _ in ()).throw(RuntimeError("persist failed")),
    )

    reply, meta = pipeline.answer_with_meta("Tôi bị ho", session_id="s")

    nodes = {node["id"]: node for node in meta["graph_nodes"]}
    assert reply == TECHNICAL_ERROR_REPLY
    assert nodes["persist"]["status"] == "error"
    assert nodes["persist"]["ms"] is not None


def test_answer_with_meta_marks_mode_policy_failure_as_route_error(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch)
    monkeypatch.setattr(pipeline, "analyze_turn", lambda *args, **kwargs: _analysis())
    monkeypatch.setattr(
        pipeline,
        "apply_mode_policy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("mode policy failed")
        ),
    )

    reply, meta = pipeline.answer_with_meta("Tôi bị ho", session_id="s")

    nodes = {node["id"]: node for node in meta["graph_nodes"]}
    assert reply == TECHNICAL_ERROR_REPLY
    assert nodes["route"]["status"] == "error"
    assert nodes["route"]["ms"] is not None


def test_answer_with_meta_marks_guardrail_persistence_failure(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(verdict="off_topic"),
    )
    monkeypatch.setattr(
        pipeline,
        "save_session",
        lambda session: (_ for _ in ()).throw(RuntimeError("persist failed")),
    )

    reply, meta = pipeline.answer_with_meta("Viết code Python", session_id="s")

    nodes = {node["id"]: node for node in meta["graph_nodes"]}
    assert reply == TECHNICAL_ERROR_REPLY
    assert nodes["persist"]["status"] == "error"
    assert nodes["persist"]["ms"] is not None


def test_answer_with_meta_marks_entity_ingest_failure(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch)
    monkeypatch.setattr(pipeline, "analyze_turn", lambda *args, **kwargs: _analysis())
    monkeypatch.setattr(
        pipeline,
        "_ingest_entities",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("entity ingest failed")
        ),
    )

    reply, meta = pipeline.answer_with_meta("Tôi bị ho", session_id="s")

    nodes = {node["id"]: node for node in meta["graph_nodes"]}
    assert reply == TECHNICAL_ERROR_REPLY
    assert nodes["entity_ingest"]["status"] == "error"
    assert nodes["entity_ingest"]["ms"] is not None


@pytest.mark.parametrize(
    "failing_stage",
    ("load_session", "preflight", "turn_analysis", "route"),
)
def test_answer_with_meta_marks_non_retrieval_failure_stage(
    monkeypatch,
    failing_stage,
):
    def fail(*args, **kwargs):
        raise RuntimeError(f"{failing_stage} failed")

    if failing_stage == "load_session":
        _patch_preflight_ok(monkeypatch)
        monkeypatch.setattr(pipeline, "load_session", fail)
    elif failing_stage == "preflight":
        _patch_persistence_noop(monkeypatch)
        monkeypatch.setattr(pipeline, "preflight", fail)
    elif failing_stage == "turn_analysis":
        _patch_preflight_ok(monkeypatch)
        _patch_persistence_noop(monkeypatch)
        monkeypatch.setattr(pipeline, "analyze_turn", fail)
    else:
        _patch_preflight_ok(monkeypatch)
        _patch_persistence_noop(monkeypatch)
        monkeypatch.setattr(
            pipeline,
            "analyze_turn",
            lambda *args, **kwargs: _analysis(),
        )
        monkeypatch.setattr(pipeline, "_route", fail)

    reply, meta = pipeline.answer_with_meta("Tôi bị ho", session_id="s")

    nodes = {node["id"]: node for node in meta["graph_nodes"]}
    assert reply == TECHNICAL_ERROR_REPLY
    assert nodes[failing_stage]["status"] == "error"
    assert nodes[failing_stage]["status"] != "skipped"
    assert nodes[failing_stage]["ms"] is not None
    assert meta["outcome"] == "technical_error"
    assert not any(key.startswith("_") for key in meta)


def test_answer_with_meta_records_worker_retrieval_timings(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch)
    monkeypatch.setattr(pipeline, "analyze_turn", lambda *args, **kwargs: _analysis())
    monkeypatch.setattr(
        pipeline,
        "kg_search",
        lambda question: KGContext(matched_entities=[{"name": "Ho"}]),
    )
    monkeypatch.setattr(pipeline, "format_kg_context", lambda result: "KG context")
    monkeypatch.setattr(
        pipeline,
        "hybrid_search_with_debug",
        lambda question, top_k, on_stage=None: (
            [
                Hit(
                    text="context",
                    score=1.0,
                    source_type="disease",
                    source_name="Nguồn",
                    heading_path="",
                    source_slug="nguon",
                )
            ],
            {
                "query": question,
                "dense_hits": [],
                "sparse_hits": [],
                "fused_hits": [],
                "reranked_hits": [],
            },
        ),
    )

    def fake_generate(question, hits, kg_text="", patient=None, return_meta=False, **kwargs):
        if return_meta:
            return "answer", {"usage": {}, "model": "test-model"}
        return "answer"

    monkeypatch.setattr(pipeline, "generate", fake_generate)

    reply, meta = pipeline.answer_with_meta("Tôi bị ho", session_id="s")

    kg_timings = [entry for entry in meta["timings"] if entry["stage"] == "kg_search"]
    hybrid_timings = [entry for entry in meta["timings"] if entry["stage"] == "hybrid_search"]
    assert reply == "answer"
    assert len(kg_timings) == 1
    assert kg_timings[0]["fields"] == {"kg_chars": len("KG context")}
    assert len(hybrid_timings) == 1
    assert hybrid_timings[0]["fields"] == {"hits": 1}


def test_emit_node_event_is_noop_without_sink():
    from src.chat import pipeline

    pipeline._install_event_sink(None)  # ensure no sink on this thread
    pipeline._emit_node_event("route", "ok", 1.5)  # no exception = pass


def test_emit_node_event_pushes_to_installed_sink():
    import queue as _queue
    from src.chat import pipeline

    sink: _queue.Queue = _queue.Queue()
    pipeline._install_event_sink(sink)
    try:
        pipeline._emit_node_event("kg_search", "ok", 12.0)
    finally:
        pipeline._install_event_sink(None)

    event = sink.get_nowait()
    assert event == {"type": "node", "id": "kg_search", "status": "ok", "ms": 12.0}


def test_log_timing_emits_node_event_when_sink_installed():
    import queue as _queue
    import time
    from src.chat import pipeline

    sink: _queue.Queue = _queue.Queue()
    pipeline._install_event_sink(sink)
    try:
        pipeline._log_timing("trace-x", "generate", time.perf_counter(), chars=10)
    finally:
        pipeline._install_event_sink(None)

    event = sink.get_nowait()
    assert event["type"] == "node"
    assert event["id"] == "generate"
    assert event["status"] == "ok"
    assert isinstance(event["ms"], float)


def test_parallel_retrieval_emits_node_events_to_sink_across_executor(monkeypatch):
    import queue as _queue
    from src.chat import pipeline

    # Patch the retrieval primitives (not the loaders) so the REAL
    # _load_kg_context/_load_hybrid_hits run inside the worker threads and
    # emit their kg_search/hybrid_search node events via _log_timing.
    monkeypatch.setattr(pipeline, "kg_search", lambda question: KGContext())
    monkeypatch.setattr(pipeline, "format_kg_context", lambda kg_result: "KG context")
    monkeypatch.setattr(
        pipeline, "hybrid_search",
        lambda question, top_k=None: [
            Hit(text="c", score=1.0, source_type="disease",
                source_name="N", heading_path="", source_slug="n")
        ],
    )
    monkeypatch.setattr(pipeline, "generate",
                        lambda question, hits, kg_text="", patient=None, **kwargs: "answer")

    sink: _queue.Queue = _queue.Queue()
    pipeline._install_event_sink(sink)
    try:
        reply = pipeline._handle_informational(
            PatientSession(session_id="s"), "Tôi bị ho", trace_id="trace",
        )
    finally:
        pipeline._install_event_sink(None)

    assert reply == "answer"
    ids = []
    while not sink.empty():
        ids.append(sink.get_nowait()["id"])
    # kg_search + hybrid_search are emitted from WORKER threads; parallel_retrieval
    # + generate from the request thread. All must reach the sink.
    assert "kg_search" in ids
    assert "hybrid_search" in ids
    assert "parallel_retrieval" in ids
