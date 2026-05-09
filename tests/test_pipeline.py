from __future__ import annotations

import logging
import threading

from src.chat import pipeline
from src.chat.errors import Neo4jUnavailable, QdrantUnavailable
from src.chat.replies import TECHNICAL_ERROR_REPLY
from src.chat.retrieval.types import Hit
from src.chat.storage.session import PatientSession


def _analysis(
    label: str = "informational",
    direct_answer_requested: bool = False,
    rewritten: str = "Tôi bị ho",
    entities: dict | None = None,
    verdict: str = "allow",
) -> dict:
    return {
        "guardrail": {"verdict": verdict, "reason": ""},
        "turn": {
            "label": label,
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


def test_unknown_clarification_answer_can_keep_asking(monkeypatch):
    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(
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

    assert pipeline.answer("tôi không biết", session_id="s") == (
        "Dựa trên triệu chứng bạn nêu, tôi đang cân nhắc các bệnh: "
        "Bệnh cúm, COVID-19, Cảm lạnh.\n\nĐể thu hẹp chẩn đoán..."
    )


def test_diagnostic_stops_asking_when_shortlist_is_small(monkeypatch):
    session = PatientSession(
        session_id="s",
        symptoms=[{"symptom_id": "symptom:COUGH", "name": "Ho"}],
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
