from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.chat import prompts
from src.chat.diagnosis import differential
from src.chat.diagnosis.flow import direct_diagnostic_prompt
from src.chat.llm import analyzer
from src.chat.llm import generator
from src.chat.llm import mini
from src.chat.guards.guardrail import VALID_VERDICTS, VERDICT_REPLIES
from src.chat.replies import TECHNICAL_ERROR_REPLY
from src.chat.retrieval.types import Hit
from src.chat.storage.session import PatientSession


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


def test_guardrail_reply_verdicts_are_valid_for_analyzer():
    assert set(VERDICT_REPLIES).issubset(set(VALID_VERDICTS))
    assert "self_medication" not in VALID_VERDICTS
    assert "self_medication" not in VERDICT_REPLIES


def test_analyzer_treats_self_medication_as_regular_medical_question(monkeypatch):
    monkeypatch.setattr(
        analyzer,
        "call_mini",
        lambda *args, **kwargs: {
            "guardrail": {"verdict": "self_medication", "reason": "unsafe self medication"},
            "turn": {"label": "informational"},
            "rewrite": {"rewritten": "x"},
            "entities": {"symptoms": [{"name": "đau"}], "medications": ["amoxicillin"]},
        },
    )

    result = analyzer.analyze_turn("Tôi tự mua amoxicillin uống được không?")

    assert result["guardrail"] == {"verdict": "allow", "reason": "unsafe self medication"}
    assert result["rewrite"]["rewritten"] == "x"
    assert result["entities"] == {
        "symptoms": [{"name": "đau"}],
        "medications": ["amoxicillin"],
    }


def test_analyzer_preserves_trivial_guardrail_verdict(monkeypatch):
    monkeypatch.setattr(
        analyzer,
        "call_mini",
        lambda *args, **kwargs: {
            "guardrail": {"verdict": "trivial", "reason": "too short"},
            "turn": {"label": "informational"},
            "rewrite": {"rewritten": "x"},
            "entities": {"symptoms": [{"name": "ho"}], "medications": ["para"]},
        },
    )

    result = analyzer.analyze_turn("?")

    assert result["guardrail"] == {"verdict": "trivial", "reason": "too short"}
    assert result["rewrite"]["rewritten"] == "?"
    assert result["entities"] == {"symptoms": [], "medications": []}


def test_analyzer_uses_guardrail_model_settings(monkeypatch):
    calls: list[dict] = []

    def fake_call_mini(*args, **kwargs):
        calls.append(kwargs)
        return {
            "guardrail": {"verdict": "allow"},
            "turn": {"label": "informational"},
            "rewrite": {"rewritten": "Tôi bị ho", "confident": True},
            "entities": {"symptoms": [], "medications": []},
        }

    monkeypatch.setattr(analyzer, "GUARDRAIL_MODEL", "guard-model")
    monkeypatch.setattr(analyzer, "GUARDRAIL_MAX_TOKENS", 123)
    monkeypatch.setattr(analyzer, "call_mini", fake_call_mini)

    analyzer.analyze_turn("Tôi bị ho")

    assert calls[0]["model"] == "guard-model"
    assert calls[0]["max_tokens"] == 123


def test_turn_analysis_prompt_recognizes_tapped_clarification_replies():
    text = prompts.TURN_ANALYSIS_SYSTEM

    assert "Để tôi định hướng tốt hơn" in text
    assert "Bắt đầu" in text
    assert "Bạn có bị" in text
    assert "Có / Không / Không rõ" in text


def test_generator_system_prompt_adopts_patient_friendly_template_rules():
    text = prompts.GENERATOR_SYSTEM

    for phrase in (
        "đồng cảm, bình tĩnh",
        "không làm người dùng hoảng sợ",
        "Ghi nhận",
        "Nhận định sơ bộ",
        "tối đa 4 phần",
        "Không dùng dòng phân cách",
        "dấu hiệu nguy hiểm",
        "gọi cấp cứu 115",
    ):
        assert phrase in text


def test_direct_diagnostic_prompt_uses_template_answer_structure():
    session = PatientSession(
        session_id="s",
        symptoms=[{"symptom_id": "symptom:COUGH", "name": "Ho"}],
        candidate_diseases=[
            {"disease_id": "flu", "name": "Bệnh cúm", "overlap": 1},
            {"disease_id": "cold", "name": "Cảm lạnh", "overlap": 1},
        ],
    )

    prompt, retrieval_query = direct_diagnostic_prompt(session, "trả lời luôn")

    assert "Ho" in retrieval_query
    assert "Ghi nhận" in prompt
    assert "không chẩn đoán chắc chắn" in prompt
    assert "2-3 khả năng" in prompt
    assert "dấu hiệu nguy hiểm" in prompt
    assert "Không hỏi thêm câu hỏi làm rõ" in prompt
    assert "tối đa 4 phần" in prompt
    assert "Không dùng dòng phân cách" in prompt


def test_generator_strips_horizontal_rules_from_llm_answer(monkeypatch):
    hits = [Hit("Bù nước khi tiêu chảy.", 1.0, "disease", "Tiêu chảy cấp", "", "tieu-chay-cap")]
    fake_response = {
        "choices": [
            {
                "message": {
                    "content": "Ghi nhận: đau bụng và tiêu chảy [1].\n\n---\n\nBạn nên bù nước."
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

    answer = generator.generate("Tôi đau bụng", hits)

    assert "\n---\n" not in answer
    assert "Ghi nhận: đau bụng và tiêu chảy [1].\n\nBạn nên bù nước." in answer


def test_build_clarification_uses_patient_friendly_intro(monkeypatch):
    monkeypatch.setattr(
        differential,
        "symptom_catalog",
        lambda: {
            "symptom:FEVER": {
                "name_vi": "sốt",
                "clarification_questions": {
                    "onset": "Sốt bắt đầu từ khi nào?",
                    "severity": "Nhiệt độ cao nhất đo được là bao nhiêu?",
                    "associated": "Có kèm rét run hoặc phát ban không?",
                },
            }
        },
    )

    text = differential.build_clarification(["symptom:FEVER"])

    assert text == "Bạn có bị sốt không?"
    assert "Để định hướng tốt hơn" not in text
    assert "Có / Không / Không rõ" not in text


def test_build_clarification_is_short_and_choice_friendly(monkeypatch):
    monkeypatch.setattr(
        differential,
        "symptom_catalog",
        lambda: {
            "symptom:FEVER": {
                "name_vi": "sốt",
                "clarification_questions": {
                    "onset": "Sốt bắt đầu từ khi nào?",
                    "severity": "Nhiệt độ cao nhất đo được là bao nhiêu?",
                },
            },
            "symptom:VOMIT": {
                "name_vi": "nôn",
                "clarification_questions": {
                    "onset": "Nôn bắt đầu từ khi nào?",
                    "pattern": "Nôn vọt hay nôn thường?",
                },
            },
        },
    )

    text = differential.build_clarification(["symptom:FEVER", "symptom:VOMIT"])

    assert text == "Bạn có bị sốt không?"
    assert "Bạn có bị **nôn** không?" not in text
    assert "Sốt bắt đầu từ khi nào?" not in text
    assert "Nôn vọt hay nôn thường?" not in text


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


def test_generator_omits_thinking_for_mistral_base_url(monkeypatch):
    calls: list[dict] = []
    hits = [Hit("context", 1.0, "disease", "Bệnh cúm", "", "cum")]
    fake_response = {
        "choices": [
            {
                "message": {
                    "content": "Cúm có thể gây sốt và ho [1]."
                }
            }
        ]
    }
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: calls.append(kwargs) or fake_response
            )
        )
    )
    monkeypatch.setattr(generator, "BASE_URL", "https://api.mistral.ai/v1", raising=False)
    monkeypatch.setattr(generator, "get_openai", lambda: fake_client)

    generator.generate("Bệnh cúm là gì?", hits)

    assert "extra_body" not in calls[0]


def test_call_mini_omits_thinking_for_mistral_base_url(monkeypatch):
    calls: list[dict] = []
    fake_response = {"choices": [{"message": {"content": '{"ok": true}'}}]}
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: calls.append(kwargs) or fake_response
            )
        )
    )
    monkeypatch.setattr(mini, "BASE_URL", "https://api.mistral.ai/v1", raising=False)
    monkeypatch.setattr(mini, "get_openai", lambda: fake_client)

    assert mini.call_mini("system", "user") == {"ok": True}
    assert "extra_body" not in calls[0]


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
