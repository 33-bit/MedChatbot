from __future__ import annotations

import logging
import threading

import pytest

from src.chat import pipeline
from src.chat.errors import Neo4jUnavailable, QdrantUnavailable
from src.chat.replies import TECHNICAL_ERROR_REPLY
from src.chat.retrieval.types import Hit
from src.chat.storage.session import PatientSession


def _analysis(
    label: str = "informational",
    intent: str | None = None,
    direct_answer_requested: bool = False,
    rewritten: str = "Tôi bị ho",
    entities: dict | None = None,
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
        "entities": entities or {"symptoms": [], "medications": []},
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
    session = _patch_persistence_noop(monkeypatch, PatientSession(session_id="s"))
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(
            label="diagnostic",
            intent="contextual_drug_info",
            rewritten="Tôi bị rụng tóc, dùng Acid Pantothenic được không?",
            entities={
                "symptoms": [{"name": "rụng tóc"}],
                "medications": ["Acid Pantothenic"],
            },
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "normalize_entities",
        lambda raw: {
            "symptoms": [{"symptom_id": "symptom:HAIR_LOSS", "name": "rụng tóc"}],
            "medications": ["Acid Pantothenic"],
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
    ) -> str:
        captured["question"] = question
        captured["use_patient_context"] = use_patient_context
        captured["retrieval_question"] = retrieval_question
        return "acid pantothenic safety answer"

    monkeypatch.setattr(pipeline, "_handle_informational", fake_informational)

    reply = pipeline.answer(
        "Tôi bị rụng tóc, dùng Acid Pantothenic được không?",
        session_id="s",
        mode="auto",
    )

    assert reply == "acid pantothenic safety answer"
    assert captured == {
        "question": "Tôi bị rụng tóc, dùng Acid Pantothenic được không?",
        "use_patient_context": True,
        "retrieval_question": None,
    }
    assert session.symptoms == [{"symptom_id": "symptom:HAIR_LOSS", "name": "rụng tóc"}]
    assert session.medications == ["Acid Pantothenic"]


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


def test_symptom_triage_in_information_mode_suggests_diagnostic_mode(monkeypatch):
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
    monkeypatch.setattr(
        pipeline,
        "_handle_diagnostic",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("blocked symptom triage should not run diagnostic flow")
        ),
    )

    reply = pipeline.answer(
        "Tôi đau lưng lan xuống chân là bệnh gì?",
        session_id="s",
        mode="information",
    )

    assert reply == (
        "Câu hỏi này giống tư vấn triệu chứng hơn. "
        "Bạn muốn trả lời ở chế độ Chẩn đoán không?"
    )
    assert session.conversation == [
        {"role": "user", "content": "Tôi đau lưng lan xuống chân là bệnh gì?"},
        {"role": "assistant", "content": reply},
    ]
    assert all("mode" not in turn for turn in session.conversation)


def test_answer_with_choices_exposes_mode_suggestion_metadata(monkeypatch):
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

    reply = pipeline.answer_with_choices("Tôi đau lưng là bệnh gì?", session_id="s", mode="information")

    assert reply.text == (
        "Câu hỏi này giống tư vấn triệu chứng hơn. "
        "Bạn muốn trả lời ở chế độ Chẩn đoán không?"
    )
    assert reply.suggest_mode == "diagnostic"
    assert reply.retry_question == "Tôi đau lưng là bệnh gì?"


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
        lambda saved_session, question, trace_id, use_patient_context=False, retrieval_question=None: "shortlist answer",
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
    monkeypatch.setattr(pipeline, "generate", lambda question, hits, kg_text="", patient=None: "answer")

    reply = pipeline._handle_informational(
        PatientSession(session_id="s"),
        "Tôi bị ho",
        trace_id="trace",
    )

    assert reply == "answer"
    assert sorted(calls) == ["hybrid", "kg"]
    assert "stage=parallel_retrieval" in caplog.text


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
    monkeypatch.setattr(pipeline, "generate", lambda question, hits, kg_text="", patient=None: "answer")

    assert pipeline.answer("Tôi bị ho", session_id="s") == "answer"

    for stage in ("preflight", "load_session", "turn_analysis", "parallel_retrieval", "generate", "total"):
        assert f"stage={stage}" in caplog.text
