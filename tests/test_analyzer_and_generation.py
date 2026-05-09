from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.chat.llm import analyzer
from src.chat.llm import generator
from src.chat.replies import TECHNICAL_ERROR_REPLY
from src.chat.retrieval.types import Hit


def test_analyzer_falls_back_to_informational_when_mini_llm_fails(monkeypatch):
    monkeypatch.setattr(analyzer, "call_mini", lambda *args, **kwargs: None)

    result = analyzer.analyze_turn("Tôi bị ho")

    assert result["guardrail"]["verdict"] == "allow"
    assert result["turn"] == {
        "label": "informational",
        "direct_answer_requested": False,
    }
    assert result["rewrite"]["rewritten"] == "Tôi bị ho"
    assert result["entities"] == {"symptoms": [], "medications": []}


def test_analyzer_normalizes_direct_answer_requested_from_one_shot_llm(monkeypatch):
    monkeypatch.setattr(
        analyzer,
        "call_mini",
        lambda *args, **kwargs: {
            "guardrail": {"verdict": "allow"},
            "turn": {
                "label": "clarification_answer",
                "direct_answer_requested": "true",
            },
            "rewrite": {"rewritten": "tôi không biết, hãy trả lời luôn", "confident": True},
            "entities": {"symptoms": [], "medications": []},
        },
    )

    result = analyzer.analyze_turn("tôi không biết, hãy trả lời luôn")

    assert result["turn"]["label"] == "clarification_answer"
    assert result["turn"]["direct_answer_requested"] is True


def test_generator_only_lists_sources_cited_in_answer(monkeypatch):
    hits = [
        Hit("Vitamin B9 adverse effects", 1.0, "drug", "Acid Folic", "", "acid-folic"),
        Hit("Niacin context", 0.9, "drug", "Nicotinamide", "", "nicotinamide"),
        Hit("Bismuth context", 0.8, "drug", "Bismuth", "", "bismuth"),
    ]

    fake_response = {
        "choices": [
            {
                "message": {
                    "content": "Tác dụng không mong muốn gồm buồn nôn và nổi ban [1]."
                }
            }
        ]
    }
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: fake_response)
        )
    )
    monkeypatch.setattr(generator, "get_openai", lambda: fake_client)

    answer = generator.generate("Tác dụng không mong muốn của vitamin B9?", hits)

    assert "[1] Dược thư Quốc gia 2022 - [Acid Folic]" in answer
    assert "Nicotinamide" not in answer
    assert "Bismuth" not in answer


def test_generator_returns_no_data_reply_without_retrieval_context():
    assert generator.generate("Bệnh hiếm XYZ điều trị thế nào?", []) == generator.NO_DATA_REPLY


def test_generator_returns_technical_reply_when_llm_fails(monkeypatch):
    hits = [Hit("context", 1.0, "disease", "Bệnh cúm", "", "cum")]
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("llm down"))
            )
        )
    )
    monkeypatch.setattr(generator, "get_openai", lambda: fake_client)

    assert generator.generate("Bệnh cúm là gì?", hits) == TECHNICAL_ERROR_REPLY


def test_no_response_cache_runtime_code_remains_removed():
    runtime_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("src/chat").rglob("*.py")
    )

    assert "GPTCache" not in runtime_text
    assert "gptcache" not in runtime_text.lower()
    assert "response_cache" not in runtime_text
    assert "cache_get" not in runtime_text
    assert "cache_put" not in runtime_text
