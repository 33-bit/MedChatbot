from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from src.chat import pipeline
from src.chat.llm import generator
from src.chat.mode_policy import apply_mode_policy
from src.chat.retrieval import dense, service
from src.chat.retrieval.types import Hit
from src.processing.health_insurance import parse as health_parser
from src.rag import chunker
from src.server.source_documents import resolve_health_insurance_source_pdf

rerank_module = importlib.import_module("src.chat.retrieval.rerank")


def _health_hit(article: str = "22") -> Hit:
    return Hit(
        text="Mức hưởng bảo hiểm y tế theo quy định của Luật.",
        score=1.0,
        source_type="health_insurance",
        source_name="Luật Bảo hiểm y tế (22/VBHN-VPQH)",
        heading_path=f"Điều {article}",
        source_slug="22-vbhn-vpqh",
        chunk_id=f"health:{article}",
        metadata={
            "article_number": article,
            "article_title": "Mức hưởng bảo hiểm y tế",
            "page_start": 25,
        },
    )


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


def test_health_insurance_source_label_is_article_specific(monkeypatch):
    monkeypatch.setattr(generator, "PUBLIC_BASE_URL", "https://chat.example.vn")

    rendered = generator._format_sources([_health_hit("22"), _health_hit("23")])

    assert "Điều 22" in rendered
    assert "Điều 23" in rendered
    assert "22-vbhn-vpqh.pdf#page=25" in rendered


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
