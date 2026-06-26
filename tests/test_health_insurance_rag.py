from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from eval import core as eval_core
from src.chat import pipeline
from src.chat.health_insurance import (
    expand_health_insurance_query,
    is_health_insurance_query,
)
from src.chat.llm import generator
from src.chat.mode_policy import apply_mode_policy
from src.chat.retrieval import dense, service
from src.chat.retrieval.types import Hit
from src.processing.health_insurance import parse as health_parser
from src.rag import chunker
from src.server.source_documents import resolve_health_insurance_source_pdf

rerank_module = importlib.import_module("src.chat.retrieval.rerank")
retrieval_health = importlib.import_module("src.chat.retrieval.health_insurance")


def _health_hit(article: str = "22") -> Hit:
    return Hit(
        text="Mức hưởng bảo hiểm y tế theo quy định của Luật.",
        score=1.0,
        source_type="health_insurance",
        source_name="Luật Bảo hiểm y tế (22/VBHN-VPQH)",
        heading_path=f"Điều {article}",
        source_slug="22-vbhn-vpqh",
        chunk_id=f"health:{article}",
        id=f"uuid:{article}",
        metadata={
            "article_number": article,
            "article_title": "Mức hưởng bảo hiểm y tế",
            "page_start": 25,
        },
    )


def test_health_insurance_detector_catches_explicit_and_legal_bhyt_terms():
    assert is_health_insurance_query("Thẻ bảo hiểm y tế bản điện tử có giá trị không?")
    assert is_health_insurance_query("Mức đóng người lao động và người sử dụng lao động là bao nhiêu?")
    assert is_health_insurance_query("Tự đi khám trái tuyến thì được thanh toán thế nào?")
    assert is_health_insurance_query("Chuyển cơ sở khám chữa bệnh khi đang điều trị nội trú cần hồ sơ gì?")
    assert not is_health_insurance_query("Tôi bị đau bụng thì nên ăn gì?")


def test_health_insurance_query_expansion_adds_cross_article_anchors():
    expanded = expand_health_insurance_query(
        "Tôi tham gia bảo hiểm y tế hộ gia đình lần đầu thì thẻ có giá trị ngay không?"
    )

    assert "khoản 5 Điều 12" in expanded
    assert "Điều 16" in expanded


def test_real_health_insurance_pdf_has_expected_structure():
    if not health_parser.DEFAULT_PDF.exists():
        pytest.skip("ignored source PDF is not provisioned")

    document = health_parser.parse_pdf(health_parser.DEFAULT_PDF)

    assert len(document["chapters"]) == health_parser.EXPECTED_CHAPTERS
    assert len(document["articles"]) == health_parser.EXPECTED_ARTICLES
    article_numbers = {article["article_number"] for article in document["articles"]}
    assert {"1", "7a", "22", "48b", "52"}.issubset(article_numbers)


def test_health_insurance_chunks_are_stable_and_article_scoped(
    tmp_path,
    monkeypatch,
):
    document_path = tmp_path / "law.json"
    document_path.write_text(
        json.dumps({
            "document_title": "Luật Bảo hiểm y tế",
            "document_number": "22/VBHN-VPQH",
            "issued_date": "2025-02-26",
            "articles": [{
                "chapter_number": "IV",
                "chapter_title": "PHẠM VI ĐƯỢC HƯỞNG",
                "article_number": "22",
                "article_title": "Mức hưởng bảo hiểm y tế",
                "body": "1. Quy định thứ nhất.\n2. Quy định thứ hai.",
                "page_start": 25,
                "page_end": 27,
            }],
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(chunker, "HEALTH_INSURANCE_DOCUMENT", document_path)

    first = chunker.chunk_health_insurance()
    second = chunker.chunk_health_insurance()

    assert first == second
    assert len(first) == 1
    assert len({item["id"] for item in first}) == 1
    assert all(item["source_type"] == "health_insurance" for item in first)
    assert all(item["metadata"]["article_number"] == "22" for item in first)


def test_health_insurance_scope_reaches_only_scoped_retrievers(monkeypatch):
    calls: list[tuple[str, str]] = []
    hit = _health_hit()

    def fake_dense(query, top_k, scope="medical"):
        calls.append(("dense", scope))
        return [hit]

    def fake_sparse(query, top_k, scope="medical"):
        calls.append(("sparse", scope))
        return [hit]

    monkeypatch.setattr(dense, "dense_search", fake_dense)
    monkeypatch.setattr(service, "bm25_search", fake_sparse)
    monkeypatch.setattr(service, "rrf_merge", lambda d, s, top_k: [hit])
    monkeypatch.setattr(rerank_module, "rerank", lambda q, hits, top_k: hits)

    results = service.hybrid_search(
        "Mức hưởng BHYT",
        top_k=1,
        scope="health_insurance",
    )

    assert results == [hit]
    assert calls == [
        ("dense", "health_insurance"),
        ("sparse", "health_insurance"),
    ]


def test_health_insurance_cross_reference_expansion_adds_referenced_clause(monkeypatch):
    source = Hit(
        text=(
            "3. Thời điểm thẻ có giá trị. Người tham gia theo khoản 4 và khoản 5 Điều 12 "
            "thì thẻ có giá trị sau 30 ngày.\n"
            "39 Khoản này được sửa đổi theo khoản 14 Điều 1 của Luật số 51/2024/QH15."
        ),
        score=1.0,
        source_type="health_insurance",
        source_name="Luật Bảo hiểm y tế (22/VBHN-VPQH)",
        heading_path="Điều 16",
        source_slug="22-vbhn-vpqh",
        chunk_id="article:16",
        id="uuid:16",
        metadata={"article_number": "16", "article_title": "Thẻ bảo hiểm y tế"},
    )
    clause_5 = {
        "id": "uuid:12:5",
        "text": "5. Nhóm tự đóng bảo hiểm y tế bao gồm người thuộc hộ gia đình.",
        "source_type": "health_insurance",
        "source_name": "Luật Bảo hiểm y tế (22/VBHN-VPQH)",
        "heading_path": "Điều 12 > Khoản 5",
        "source_slug": "22-vbhn-vpqh",
        "chunk_id": "article:12:clause:5",
        "metadata": {
            "article_number": "12",
            "article_title": "Đối tượng tham gia bảo hiểm y tế",
            "clause_number": "5",
        },
    }
    footnote_target = {
        "id": "uuid:1:14",
        "text": "Điều 1 footnote target",
        "source_type": "health_insurance",
        "source_name": "Luật Bảo hiểm y tế (22/VBHN-VPQH)",
        "heading_path": "Điều 1 > Khoản 14",
        "source_slug": "22-vbhn-vpqh",
        "chunk_id": "article:1:clause:14",
        "metadata": {"article_number": "1", "clause_number": "14"},
    }
    monkeypatch.setattr(
        retrieval_health,
        "_health_chunks",
        lambda: (clause_5, footnote_target),
    )

    expanded = retrieval_health.expand_health_insurance_hits([source])

    assert [hit.id for hit in expanded] == ["uuid:16", "uuid:12:5"]


def test_mode_policy_routes_health_insurance_without_changing_medical_route():
    insurance = apply_mode_policy("auto", "health_insurance_info")
    medical = apply_mode_policy("auto", "pure_info")

    assert insurance.route_label == "health_insurance"
    assert medical.route_label == "informational"


def test_pipeline_health_insurance_route_uses_dedicated_handler(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "_handle_health_insurance",
        lambda question, trace_id: f"insurance:{question}:{trace_id}",
    )

    result = pipeline._route(
        "health_insurance",
        pipeline.PatientSession(session_id="test"),
        "question",
        "Mức hưởng BHYT",
        False,
        "trace",
    )

    assert result == "insurance:Mức hưởng BHYT:trace"


def test_pipeline_detector_overrides_off_topic_bhyt_guard(monkeypatch):
    session = pipeline.PatientSession(session_id="test")
    monkeypatch.setattr(pipeline, "CONVERSATION_CONTEXT_ENABLED", False)
    monkeypatch.setattr(pipeline, "check_rate_limit", lambda session_id: True)
    monkeypatch.setattr(pipeline, "regex_check", lambda question: None)
    monkeypatch.setattr(pipeline, "check_llm_quota", lambda session_id: (True, ""))
    monkeypatch.setattr(pipeline, "load_session", lambda session_id: session)
    monkeypatch.setattr(pipeline, "save_session", lambda saved_session: None)
    monkeypatch.setattr(pipeline, "log_consultation", lambda *args: None)
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: {
            "guardrail": {"verdict": "off_topic", "reason": "llm_miss"},
            "turn": {
                "label": "greeting_other",
                "intent": "off_scope",
                "direct_answer_requested": False,
            },
            "rewrite": {
                "rewritten": "Bảo hiểm y tế hộ gia đình lần đầu thẻ có giá trị khi nào?",
                "confident": True,
                "clarification": "",
            },
            "triage": {"urgency": "routine", "red_flags": [], "reason": ""},
            "entities": {"symptoms": [], "medications": []},
            "context": {},
        },
    )
    monkeypatch.setattr(
        pipeline,
        "_handle_health_insurance",
        lambda question, trace_id: f"health:{question}",
    )

    reply = pipeline.answer("BHYT hộ gia đình lần đầu thẻ có giá trị khi nào?", session_id="test")

    assert reply == "health:Bảo hiểm y tế hộ gia đình lần đầu thẻ có giá trị khi nào?"


def test_health_insurance_source_label_is_article_specific(monkeypatch):
    monkeypatch.setattr(generator, "PUBLIC_BASE_URL", "https://chat.example.vn")

    rendered = generator._format_sources([_health_hit("22"), _health_hit("23")])

    assert "Điều 22" in rendered
    assert "Điều 23" in rendered
    assert "22-vbhn-vpqh.pdf#page=25" in rendered


def test_health_insurance_article_mentions_get_matching_citations():
    answer, cited = generator._ensure_health_insurance_article_citations(
        "Hộ gia đình thuộc khoản 5 Điều 12. Thẻ có giá trị sau 30 ngày [1].",
        [_health_hit("16"), _health_hit("12")],
        [1],
    )

    assert "Điều 12 [2]" in answer
    assert cited == [1, 2]


def test_health_insurance_article_citation_does_not_match_prefix_article():
    answer, cited = generator._ensure_health_insurance_article_citations(
        "Theo Điều 23, thiết bị y tế thay thế không được hưởng.",
        [_health_hit("2"), _health_hit("23")],
        [],
    )

    assert "Điều 2 [1]3" not in answer
    assert "Điều 23 [2]" in answer
    assert cited == [2]


def test_eval_records_health_insurance_uuid_and_semantic_chunk_ids():
    row = {
        "passed": True,
        "hard_fail": False,
        "checks": [],
    }
    case = {
        "category": "health_insurance_info",
        "source_docs": [{"path": "outputs/health_insurance/22-vbhn-vpqh.json"}],
        "gold_chunks": ["uuid:22"],
    }
    meta = {
        "retrieved": [{
            "id": "uuid:22",
            "chunk_id": "health_insurance:22-vbhn-vpqh:article:22",
            "source_slug": "22-vbhn-vpqh",
            "source_type": "health_insurance",
        }]
    }

    eval_core._attach_meta_metrics(row, case, meta)

    assert row["retrieved_chunks"] == ["uuid:22"]
    assert row["retrieved_semantic_chunks"] == [
        "health_insurance:22-vbhn-vpqh:article:22"
    ]
    assert row["retrieval"]["chunk_recall@5"] == 1.0
    assert row["checks"][-1]["passed"] is True


def test_health_insurance_source_resolver(tmp_path):
    pdf = tmp_path / "law.pdf"
    assert resolve_health_insurance_source_pdf(pdf) is None

    pdf.write_bytes(b"%PDF-1.4\ntest")

    assert resolve_health_insurance_source_pdf(pdf) == pdf


def test_health_insurance_pdf_endpoint(app_client, monkeypatch, tmp_path):
    client, app_module = app_client
    pdf = tmp_path / "law.pdf"
    pdf.write_bytes(b"%PDF-1.4\ntest")
    monkeypatch.setattr(
        app_module,
        "resolve_health_insurance_source_pdf",
        lambda: pdf,
    )

    response = client.get("/sources/health-insurance/22-vbhn-vpqh.pdf")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content == b"%PDF-1.4\ntest"
