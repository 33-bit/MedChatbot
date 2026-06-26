from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.chat import prompts
from src.chat.diagnosis import differential
from src.chat.diagnosis.flow import direct_diagnostic_prompt
from src.chat.llm import analyzer
from src.chat.llm import answer_verifier
from src.chat.llm import evidence_brief
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
        "intent": "pure_info",
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
    assert result["turn"]["intent"] == "clarification_answer"
    assert result["turn"]["direct_answer_requested"] is True


def test_analyzer_normalizes_scalar_memory_fact_value(monkeypatch):
    monkeypatch.setattr(
        analyzer,
        "call_mini",
        lambda *args, **kwargs: {
            "guardrail": {"verdict": "allow"},
            "turn": {"label": "diagnostic", "intent": "symptom_triage"},
            "rewrite": {"rewritten": "Tôi bị dị ứng penicillin", "confident": True},
            "entities": {"symptoms": [], "medications": ["penicillin"]},
            "context": {
                "subject": {
                    "id": "self",
                    "source": "explicit",
                    "confidence": 1.0,
                },
                "references": [],
                "relation": "new_entity",
                "needs_medical_profile": True,
            },
            "profile_candidates": [{
                "subject_id": "self",
                "fact_type": "allergy",
                "entity_type": "drug",
                "entity_id": "penicillin",
                "attribute": "reaction_type",
                "value": "allergic",
                "temporal_status": "current",
                "confidence": 1.0,
                "source": "explicit",
            }],
        },
    )

    result = analyzer.analyze_turn("Tôi bị dị ứng penicillin")

    assert result["profile_candidates"][0]["value"] == {"value": "allergic"}


def test_analyzer_preserves_refined_intent_from_one_shot_llm(monkeypatch):
    monkeypatch.setattr(
        analyzer,
        "call_mini",
        lambda *args, **kwargs: {
            "guardrail": {"verdict": "allow"},
            "turn": {
                "label": "diagnostic",
                "intent": "contextual_drug_info",
                "direct_answer_requested": False,
            },
            "rewrite": {
                "rewritten": "Tôi bị rụng tóc, dùng Acid Pantothenic được không?",
                "confident": True,
            },
            "entities": {
                "symptoms": [{"name": "rụng tóc"}],
                "medications": ["Acid Pantothenic"],
            },
        },
    )

    result = analyzer.analyze_turn("Tôi bị rụng tóc, dùng Acid Pantothenic được không?")

    assert result["turn"]["label"] == "diagnostic"
    assert result["turn"]["intent"] == "contextual_drug_info"


def test_analyzer_preserves_semantic_evidence_plan(monkeypatch):
    monkeypatch.setattr(
        analyzer,
        "call_mini",
        lambda *args, **kwargs: {
            "guardrail": {"verdict": "allow"},
            "turn": {
                "label": "informational",
                "intent": "contextual_drug_info",
                "direct_answer_requested": False,
            },
            "rewrite": {
                "rewritten": "Almagate uống thế nào?",
                "confident": True,
            },
            "entities": {"symptoms": [], "medications": ["Almagate"]},
            "evidence_plan": {
                "domain": "drug_info",
                "source_type": "drug",
                "entity": "Almagate",
                "answer_slot": "dose",
                "safety_mode": "factual_info",
                "target_heading_paths": ["Liều dùng và cách dùng"],
                "required_facts": ["liều", "đường dùng"],
                "answer_style": "exact_list",
                "confidence": 0.9,
            },
        },
    )

    result = analyzer.analyze_turn("Almagate uống thế nào?")

    assert result["evidence_plan"]["domain"] == "drug_info"
    assert result["evidence_plan"]["source_type"] == "drug"
    assert result["evidence_plan"]["answer_slot"] == "dose"
    assert result["evidence_plan"]["target_heading_paths"] == ["Liều dùng và cách dùng"]


def test_analyzer_routes_emergency_urgency_independently_of_turn_intent(monkeypatch):
    monkeypatch.setattr(
        analyzer,
        "call_mini",
        lambda *args, **kwargs: {
            "guardrail": {"verdict": "allow"},
            "turn": {
                "label": "diagnostic",
                "intent": "symptom_triage",
                "direct_answer_requested": False,
            },
            "triage": {
                "urgency": "emergency",
                "red_flags": ["đau ngực kèm khó thở"],
                "reason": "Có dấu hiệu đe dọa tim hoặc hô hấp.",
            },
            "rewrite": {
                "rewritten": "Tôi bị đau ngực, khó thở",
                "confident": True,
            },
            "entities": {
                "symptoms": [{"name": "đau ngực"}, {"name": "khó thở"}],
                "medications": [],
            },
        },
    )

    result = analyzer.analyze_turn("Tôi bị đau ngực, khó thở")

    assert result["turn"]["label"] == "diagnostic"
    assert result["turn"]["intent"] == "emergency"
    assert result["triage"] == {
        "urgency": "emergency",
        "red_flags": ["đau ngực kèm khó thở"],
        "reason": "Có dấu hiệu đe dọa tim hoặc hô hấp.",
    }


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
    assert "contextual_drug_info" in text
    assert "condition_management_info" in text
    assert "symptom_triage" in text


def test_turn_analysis_prompt_assesses_urgency_independently_of_intent():
    text = prompts.TURN_ANALYSIS_SYSTEM

    assert '"urgency": "routine" | "urgent" | "emergency"' in text
    assert "ĐỘC LẬP với turn.intent" in text
    assert "tình huống giả định" in text
    assert "TOÀN BỘ cụm triệu chứng và ngữ cảnh" in text


def test_turn_analysis_prompt_includes_evidence_plan_schema():
    text = prompts.TURN_ANALYSIS_SYSTEM

    assert '"evidence_plan"' in text
    assert '"answer_slot"' in text
    assert "không dùng mẹo danh sách từ khóa" in text
    assert "needs_fallback=true" in text


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


def test_context_bundle_formats_exact_subject_address():
    text = generator._format_patient({
        "subject": {
            "id": "father",
            "relationship": "father",
            "display_name": "bố bạn",
        },
        "safety_profile": [],
        "relevant_facts": [],
        "active_case": None,
        "reference_turns": [],
    })

    assert "Chủ thể y tế: bố bạn" in text
    assert "Luôn gọi đúng người này" in text


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


def test_generator_uses_retrieved_drug_usage_when_model_claims_missing(monkeypatch):
    hits = [
        Hit(
            text="Liều dùng: người lớn uống 1-2 gói/lần, ngày 3 lần.",
            score=1.0,
            source_type="drug",
            source_name="Almagate",
            heading_path="Liều dùng và cách dùng",
            source_slug="almagate",
            chunk_id="drug:almagate:lieu-dung-va-cach-dung",
            id="uuid-usage",
        )
    ]
    fake_response = {
        "choices": [{
            "message": {
                "content": "Tài liệu được cung cấp không đủ thông tin để xác nhận liều/cách dùng."
            }
        }]
    }
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: fake_response)
        )
    )
    monkeypatch.setattr(generator, "get_openai", lambda: fake_client)
    monkeypatch.setattr(generator, "build_evidence_brief", lambda **kwargs: None)
    monkeypatch.setattr(
        generator,
        "repair_answer_with_evidence",
        lambda **kwargs: (
            kwargs["answer"]
            + "\n- Nên sử dụng Ketoprofen cùng với thức ăn để giảm nguy cơ tiêu hóa. [1]"
        ),
    )

    answer = generator.generate(
        "Almagate liều dùng thế nào?",
        hits,
        answer_domain="drug_info",
    )

    assert "người lớn uống 1-2 gói/lần, ngày 3 lần" in answer
    assert "không đủ thông tin" not in answer.lower()
    assert "Nguồn:" in answer


def test_generator_drug_usage_fallback_includes_later_cach_dung_chunk(monkeypatch):
    hits = [
        Hit(
            text=(
                "5 Liều dùng - Cách dùng > 5.1 Liều dùng\n"
                "Liều người lớn thông thường cho bệnh thiếu máu do thiếu sắt:\n"
                "Liều khởi đầu: 960 mg/ngày ferrous gluconate (120 mg/ngày sắt nguyên tố) trong 3 tháng\n"
                "Dùng chia làm nhiều lần (1 đến 3 lần mỗi ngày)"
            ),
            score=1.0,
            source_type="drug",
            source_name="Sắt Gluconat",
            heading_path="5 Liều dùng - Cách dùng > 5.1 Liều dùng",
            source_slug="sat-gluconat",
            chunk_id="drug:sat-gluconat:lieu-dung",
            id="uuid-dose",
        ),
        Hit(
            text=(
                "5 Liều dùng - Cách dùng > 5.2 Cách dùng\n"
                "Uống gluconate sắt khi bụng đói, ít nhất 1 giờ trước hoặc 2 giờ sau bữa ăn.\n"
                "Gluconate sắt có thể được dùng cùng với thức ăn nếu nó làm đau dạ dày. "
                "Uống gluconate sắt với một ly nước đầy hoặc nước trái cây."
            ),
            score=0.9,
            source_type="drug",
            source_name="Sắt Gluconat",
            heading_path="5 Liều dùng - Cách dùng > 5.2 Cách dùng",
            source_slug="sat-gluconat",
            chunk_id="drug:sat-gluconat:cach-dung",
            id="uuid-cach-dung",
        ),
    ]
    fake_response = {
        "choices": [{
            "message": {
                "content": "Tài liệu được cung cấp không đủ thông tin để xác nhận liều/cách dùng."
            }
        }]
    }
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: fake_response)
        )
    )
    monkeypatch.setattr(generator, "get_openai", lambda: fake_client)
    monkeypatch.setattr(generator, "build_evidence_brief", lambda **kwargs: None)
    monkeypatch.setattr(
        generator,
        "repair_answer_with_evidence",
        lambda **kwargs: kwargs["answer"],
    )

    answer = generator.generate(
        "Tôi cần uống Sắt Gluconat như thế nào để đạt hiệu quả tốt nhất?",
        hits,
        answer_domain="drug_info",
    )

    assert "960 mg/ngày" in answer
    assert "bụng đói" in answer
    assert "1 giờ trước hoặc 2 giờ sau bữa ăn" in answer
    assert "nước đầy hoặc nước trái cây" in answer
    assert "5 Liều dùng - Cách dùng >" not in answer
    assert "không đủ thông tin" not in answer.lower()


def test_generator_includes_evidence_plan_in_prompt(monkeypatch):
    captured: dict = {}
    hits = [
        Hit(
            text="Liều dùng: người lớn uống 1-2 gói/lần, ngày 3 lần.",
            score=1.0,
            source_type="drug",
            source_name="Almagate",
            heading_path="Liều dùng và cách dùng",
            source_slug="almagate",
            chunk_id="drug:almagate:lieu-dung-va-cach-dung",
            id="uuid-usage",
        )
    ]

    def fake_create(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "Uống theo tài liệu [1]."}}]}

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    monkeypatch.setattr(generator, "get_openai", lambda: fake_client)
    monkeypatch.setattr(generator, "build_evidence_brief", lambda **kwargs: None)
    monkeypatch.setattr(
        generator,
        "repair_answer_with_evidence",
        lambda **kwargs: kwargs["answer"],
    )

    generator.generate(
        "Almagate uống thế nào?",
        hits,
        answer_domain="drug_info",
        evidence_plan={
            "domain": "drug_info",
            "source_type": "drug",
            "entity": "Almagate",
            "answer_slot": "dose",
            "target_heading_paths": ["Liều dùng và cách dùng"],
            "required_facts": ["liều dùng", "tần suất"],
            "confidence": 0.9,
        },
    )

    user_prompt = captured["messages"][1]["content"]
    assert "Kế hoạch bằng chứng cần bám sát" in user_prompt
    assert "Loại thông tin cần trả lời: dose" in user_prompt
    assert "Liều dùng và cách dùng" in user_prompt
    assert "liều dùng; tần suất" in user_prompt


def test_generator_prioritizes_planned_disease_sections_without_dropping_context(monkeypatch):
    captured: dict = {}
    hits = [
        Hit(
            text="Bệnh Basedow là bệnh cường giáp tự miễn, có tính chất gia đình.",
            score=1.0,
            source_type="disease",
            source_name="Basedow",
            heading_path="I. Đại cương",
            source_slug="basedow",
            chunk_id="disease:basedow:intro",
        ),
        Hit(
            text="Triệu chứng gồm bướu giáp, mắt lồi và run tay.",
            score=0.9,
            source_type="disease",
            source_name="Basedow",
            heading_path="II. Triệu chứng lâm sàng",
            source_slug="basedow",
            chunk_id="disease:basedow:symptoms",
        ),
        Hit(
            text="Điều trị có thể dùng thuốc kháng giáp tổng hợp.",
            score=0.8,
            source_type="disease",
            source_name="Basedow",
            heading_path="IV. Điều trị",
            source_slug="basedow",
            chunk_id="disease:basedow:treatment",
        ),
        Hit(
            text="Biến chứng có thể gồm loạn nhịp tim.",
            score=0.7,
            source_type="disease",
            source_name="Basedow",
            heading_path="V. Biến chứng",
            source_slug="basedow",
            chunk_id="disease:basedow:complications",
        ),
    ]

    def fake_create(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "Basedow là bệnh tự miễn [1]."}}]}

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    monkeypatch.setattr(generator, "get_openai", lambda: fake_client)
    monkeypatch.setattr(generator, "build_evidence_brief", lambda **kwargs: None)
    monkeypatch.setattr(
        generator,
        "repair_answer_with_evidence",
        lambda **kwargs: kwargs["answer"],
    )

    generator.generate(
        "Bệnh Basedow là gì và có biểu hiện nào?",
        hits,
        answer_domain="disease_info",
        evidence_plan={
            "domain": "disease_info",
            "source_type": "disease",
            "entity": "Basedow",
            "answer_slot": "overview",
            "target_heading_paths": ["Đại cương", "Triệu chứng lâm sàng"],
            "required_facts": ["bản chất bệnh", "triệu chứng"],
            "confidence": 0.9,
        },
    )

    user_prompt = captured["messages"][1]["content"]
    assert "Ưu tiên tài liệu theo kế hoạch bằng chứng" in user_prompt
    assert "có tính chất gia đình" in user_prompt
    assert "bướu giáp" in user_prompt
    assert "thuốc kháng giáp" in user_prompt
    assert "loạn nhịp tim" in user_prompt
    assert user_prompt.index("có tính chất gia đình") < user_prompt.index("thuốc kháng giáp")


def test_generator_uses_evidence_brief_to_prioritize_context(monkeypatch):
    captured: dict = {}
    hits = [
        Hit(
            text="Basedow là bệnh tự miễn, có tính chất gia đình.",
            score=1.0,
            source_type="disease",
            source_name="Basedow",
            heading_path="I. Đại cương",
            source_slug="basedow",
            chunk_id="disease:basedow:intro",
        ),
        Hit(
            text="Chẩn đoán dựa vào TRAb và siêu âm.",
            score=0.9,
            source_type="disease",
            source_name="Basedow",
            heading_path="II. Chẩn đoán",
            source_slug="basedow",
            chunk_id="disease:basedow:diagnosis",
        ),
        Hit(
            text="Điều trị có thể dùng thuốc kháng giáp tổng hợp.",
            score=0.8,
            source_type="disease",
            source_name="Basedow",
            heading_path="III. Điều trị",
            source_slug="basedow",
            chunk_id="disease:basedow:treatment",
        ),
    ]

    def fake_create(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "Basedow có tính chất gia đình [1]."}}]}

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    monkeypatch.setattr(generator, "get_openai", lambda: fake_client)
    monkeypatch.setattr(
        generator,
        "build_evidence_brief",
        lambda **kwargs: {
            "selected_indexes": [1],
            "must_include_facts": ["có tính chất gia đình"],
            "avoid_topics": ["chẩn đoán", "điều trị"],
            "brief": "Trả lời bằng phần đại cương.",
        },
    )
    monkeypatch.setattr(
        generator,
        "repair_answer_with_evidence",
        lambda **kwargs: kwargs["answer"],
    )

    generator.generate(
        "Basedow là gì?",
        hits,
        answer_domain="disease_info",
    )

    user_prompt = captured["messages"][1]["content"]
    assert "Brief bằng chứng đã chọn" in user_prompt
    assert "có tính chất gia đình" in user_prompt
    assert "TRAb và siêu âm" in user_prompt
    assert "thuốc kháng giáp" in user_prompt
    assert user_prompt.index("có tính chất gia đình") < user_prompt.index("TRAb và siêu âm")


def test_disease_overview_contract_removes_scope_drift_without_specific_fact_injection():
    hits = [
        Hit(
            text=(
                "Basedow là một bệnh cường giáp do hoạt động quá mức không ức chế "
                "được của tuyến giáp. Nó là một bệnh tự miễn, có tính chất gia đình, "
                "bệnh thường gặp ở phụ nữ (3%). Tỷ lệ nữ/nam: 7,5/1."
            ),
            score=1.0,
            source_type="disease",
            source_name="Basedow",
            heading_path="I. ĐẠI CƯƠNG",
            source_slug="basedow",
            chunk_id="disease:basedow:intro",
        ),
        Hit(
            text="Điều trị có thể dùng thuốc kháng giáp tổng hợp.",
            score=0.8,
            source_type="disease",
            source_name="Basedow",
            heading_path="III. Điều trị",
            source_slug="basedow",
            chunk_id="disease:basedow:treatment",
        ),
    ]
    answer = (
        "Basedow: bệnh cường giáp tự miễn [1]. "
        "Cơ chế: hệ miễn dịch kích thích sản xuất hormone tuyến giáp (T3, T4) [1]. "
        "Điều trị: thuốc kháng giáp [1]. "
        "Biến chứng: loạn nhịp tim [1]."
    )

    cleaned = generator.enforce_info_answer_contract(
        answer,
        hits,
        "disease_info",
        "Bệnh Basedow là gì và nó ảnh hưởng đến cơ thể như thế nào?",
    )

    assert "Cơ chế" in cleaned
    assert "gia đình" not in cleaned
    assert "phụ nữ (3%)" not in cleaned
    assert "Điều trị" not in cleaned
    assert "Biến chứng" not in cleaned


def test_generator_keeps_drug_usage_chunk_when_brief_omits_it(monkeypatch):
    captured: dict = {}
    hits = [
        Hit(
            text="Atapulgit dùng điều trị triệu chứng tiêu chảy.",
            score=1.0,
            source_type="drug",
            source_name="Atapulgit",
            heading_path="4 Chỉ định",
            source_slug="atapulgit",
            chunk_id="drug:atapulgit:chi-dinh",
        ),
        Hit(
            text="Liều thường dùng: uống 1,2 g đến 1,5 g mỗi lần đi phân lỏng.",
            score=0.8,
            source_type="drug",
            source_name="Atapulgit",
            heading_path="10 Liều lượng và cách dùng",
            source_slug="atapulgit",
            chunk_id="drug:atapulgit:lieu-dung",
        ),
    ]

    def fake_create(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "Atapulgit dùng cho tiêu chảy [1]."}}]}

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    monkeypatch.setattr(generator, "get_openai", lambda: fake_client)
    monkeypatch.setattr(
        generator,
        "build_evidence_brief",
        lambda **kwargs: {
            "selected_indexes": [1],
            "must_include_facts": ["chỉ định"],
            "avoid_topics": [],
            "brief": "Chỉ định.",
        },
    )
    monkeypatch.setattr(
        generator,
        "repair_answer_with_evidence",
        lambda **kwargs: kwargs["answer"],
    )

    generator.generate(
        "Atapulgit dùng như thế nào?",
        hits,
        answer_domain="drug_info",
        evidence_plan={
            "domain": "drug_info",
            "source_type": "drug",
            "entity": "Atapulgit",
            "answer_slot": "dose",
            "confidence": 0.9,
        },
    )

    user_prompt = captured["messages"][1]["content"]
    assert "Atapulgit dùng điều trị triệu chứng tiêu chảy" in user_prompt
    assert "1,2 g đến 1,5 g" in user_prompt


def test_generator_appends_usage_fallback_when_answer_omits_retrieved_dose(monkeypatch):
    hits = [
        Hit(
            text="Atapulgit dùng điều trị triệu chứng tiêu chảy.",
            score=1.0,
            source_type="drug",
            source_name="Atapulgit",
            heading_path="4 Chỉ định",
            source_slug="atapulgit",
            chunk_id="drug:atapulgit:chi-dinh",
        ),
        Hit(
            text="Liều thường dùng: uống 1,2 g đến 1,5 g mỗi lần đi phân lỏng; không vượt quá 9 g trong 24 giờ.",
            score=0.8,
            source_type="drug",
            source_name="Atapulgit",
            heading_path="10 Liều lượng và cách dùng",
            source_slug="atapulgit",
            chunk_id="drug:atapulgit:lieu-dung",
        ),
    ]
    fake_response = {
        "choices": [{
            "message": {
                "content": "Atapulgit dùng điều trị triệu chứng tiêu chảy [1]."
            }
        }]
    }
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: fake_response)
        )
    )
    monkeypatch.setattr(generator, "get_openai", lambda: fake_client)
    monkeypatch.setattr(generator, "build_evidence_brief", lambda **kwargs: None)
    monkeypatch.setattr(
        generator,
        "repair_answer_with_evidence",
        lambda **kwargs: kwargs["answer"],
    )

    answer = generator.generate(
        "Atapulgit dùng như thế nào?",
        hits,
        answer_domain="drug_info",
        evidence_plan={
            "domain": "drug_info",
            "source_type": "drug",
            "entity": "Atapulgit",
            "answer_slot": "indication",
            "confidence": 0.9,
        },
    )

    assert "Atapulgit dùng điều trị triệu chứng tiêu chảy" in answer
    assert "1,2 g đến 1,5 g" in answer
    assert "không vượt quá 9 g trong 24 giờ" in answer


def test_generator_does_not_replace_valid_drug_usage_answer(monkeypatch):
    hits = [
        Hit(
            text="Liều dùng: uống 1 viên/lần, ngày 2 lần sau ăn.",
            score=1.0,
            source_type="drug",
            source_name="Thuốc A",
            heading_path="Liều dùng và cách dùng",
            source_slug="thuoc-a",
            chunk_id="drug:thuoc-a:lieu-dung",
        )
    ]
    fake_response = {
        "choices": [{
            "message": {"content": "Bạn có thể uống 1 viên/lần, ngày 2 lần sau ăn [1]."}
        }]
    }
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: fake_response)
        )
    )
    monkeypatch.setattr(generator, "get_openai", lambda: fake_client)
    monkeypatch.setattr(generator, "build_evidence_brief", lambda **kwargs: None)
    monkeypatch.setattr(
        generator,
        "repair_answer_with_evidence",
        lambda **kwargs: kwargs["answer"],
    )

    answer = generator.generate(
        "Thuốc A uống như thế nào?",
        hits,
        answer_domain="drug_info",
        evidence_plan={
            "domain": "drug_info",
            "source_type": "drug",
            "entity": "Thuốc A",
            "answer_slot": "dose",
            "confidence": 0.9,
        },
    )

    assert "Bạn có thể uống 1 viên/lần, ngày 2 lần sau ăn" in answer
    assert "Theo chuyên luận thuốc được truy xuất" not in answer


def test_generator_does_not_treat_drug_indication_question_as_usage(monkeypatch):
    hits = [
        Hit(
            text="Thuốc A được sử dụng để hỗ trợ cải thiện đau họng.",
            score=1.0,
            source_type="drug",
            source_name="Thuốc A",
            heading_path="Chỉ định",
            source_slug="thuoc-a",
            chunk_id="drug:thuoc-a:chi-dinh",
        ),
        Hit(
            text="Liều dùng phụ thuộc dạng bào chế và hàm lượng từng chế phẩm.",
            score=0.8,
            source_type="drug",
            source_name="Thuốc A",
            heading_path="Chỉ định - liều dùng",
            source_slug="thuoc-a",
            chunk_id="drug:thuoc-a:lieu-dung",
        ),
    ]
    fake_response = {
        "choices": [{
            "message": {"content": "Có thể dùng trong phạm vi chỉ định hỗ trợ đau họng [1]."}
        }]
    }
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: fake_response)
        )
    )
    monkeypatch.setattr(generator, "get_openai", lambda: fake_client)
    monkeypatch.setattr(generator, "build_evidence_brief", lambda **kwargs: None)
    monkeypatch.setattr(
        generator,
        "repair_answer_with_evidence",
        lambda **kwargs: kwargs["answer"],
    )

    answer = generator.generate(
        "Tôi bị đau họng, có thể dùng thuốc chứa Thuốc A được không?",
        hits,
        answer_domain="drug_info",
        evidence_plan={
            "domain": "drug_info",
            "source_type": "drug",
            "entity": "Thuốc A",
            "answer_slot": "indication",
            "confidence": 0.9,
        },
    )

    assert "Có thể dùng trong phạm vi chỉ định hỗ trợ đau họng" in answer
    assert "Liều dùng phụ thuộc" not in answer.split("Nguồn:", 1)[0]
    assert "Theo chuyên luận thuốc được truy xuất" not in answer


def test_drug_usage_detection_requires_actual_usage_question():
    assert generator._question_asks_drug_usage_detail("Thuốc A uống như thế nào?")
    assert generator._question_asks_drug_usage_detail("Thuốc A liều dùng bao nhiêu?")
    assert not generator._question_asks_drug_usage_detail(
        "Tôi bị loét dạ dày, uống thuốc này có sao không?"
    )
    assert not generator._question_asks_drug_usage_detail(
        "Điều này có đúng không và nó hoạt động như thế nào?"
    )


def test_generator_usage_fallback_prefers_nested_dose_section(monkeypatch):
    hits = [
        Hit(
            text=(
                "Là thuốc giảm đau chống viêm để giảm triệu chứng viêm khớp "
                "và đau bao gồm đau cơ, đau răng, đau đầu do hầu hết các nguyên nhân."
            ),
            score=1.0,
            source_type="drug",
            source_name="Acid Mefenamic",
            heading_path="2 Công dụng và chỉ định",
            source_slug="acid-mefenamic",
            chunk_id="drug:acid-mefenamic:cong-dung-chi-dinh",
        ),
        Hit(
            text=(
                "Acid Mefenamic thường được sử dụng trong thời gian ngắn. "
                "Nên sử dụng Ketoprofen cùng với thức ăn."
            ),
            score=1.0,
            source_type="drug",
            source_name="Acid Mefenamic",
            heading_path="1 Dược lý và cơ chế tác dụng",
            source_slug="acid-mefenamic",
            chunk_id="drug:acid-mefenamic:duoc-ly",
        ),
        Hit(
            text=(
                "Người lớn: 500mg/lần/ngày. Sau đó có thể dùng tiếp 1 liều "
                "250mg nếu cần, thời gian điều trị thường không quá 7 ngày."
            ),
            score=0.8,
            source_type="drug",
            source_name="Acid Mefenamic",
            heading_path=(
                "4 Liều dùng và cách dùng > 4.1 Liều dùng Acid mefenamic > "
                "4.1.1 Đau, đau do viêm xương khớp"
            ),
            source_slug="acid-mefenamic",
            chunk_id="drug:acid-mefenamic:lieu-dung-nguoi-lon",
        ),
    ]
    fake_response = {
        "choices": [{
            "message": {
                "content": "Tài liệu được cung cấp không đủ thông tin để xác nhận liều/cách dùng."
            }
        }]
    }
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: fake_response)
        )
    )
    monkeypatch.setattr(generator, "get_openai", lambda: fake_client)
    monkeypatch.setattr(generator, "build_evidence_brief", lambda **kwargs: None)
    monkeypatch.setattr(
        generator,
        "repair_answer_with_evidence",
        lambda **kwargs: kwargs["answer"],
    )

    answer = generator.generate(
        "Tôi bị đau đầu, tôi có thể dùng Acid Mefenamic được không và dùng như thế nào?",
        hits,
        answer_domain="drug_info",
        evidence_plan={
            "domain": "drug_info",
            "source_type": "drug",
            "entity": "Acid Mefenamic",
            "answer_slot": "dose",
            "confidence": 0.9,
        },
    )

    assert "đau đầu" in answer
    assert "500mg/lần/ngày" in answer
    assert "250mg nếu cần" in answer
    assert "không quá 7 ngày" in answer
    assert "Ketoprofen" not in answer


def test_generator_usage_fallback_ignores_secondary_drug_chunks(monkeypatch):
    hits = [
        Hit(
            text=(
                "Giảm đau/giảm sốt: Uống 300 - 900 mg, lặp lại sau mỗi "
                "4 - 6 giờ nếu cần, tối đa là 4 g/ngày.\n"
                "Chống viêm: có thể dùng liều 4 - 8 g/ngày, chia làm nhiều liều nhỏ."
            ),
            score=1.0,
            source_type="drug",
            source_name="Aspirin",
            heading_path="10 Liều lượng và cách dùng > 10.2 Liều lượng > 10.2.1 Người lớn",
            source_slug="aspirin",
            chunk_id="drug:aspirin:lieu-nguoi-lon",
        ),
        Hit(
            text="Liều uống thông thường: 25 mg/lần, uống 2 - 3 lần mỗi ngày.",
            score=0.7,
            source_type="drug",
            source_name="Indomethacin",
            heading_path="10 Liều lượng và cách dùng > 10.2 Liều dùng",
            source_slug="indomethacin",
            chunk_id="drug:indomethacin:lieu-dung",
        ),
        Hit(
            text="Trẻ em: uống 60 - 130 mg/kg/ngày, chia làm nhiều liều nhỏ.",
            score=0.6,
            source_type="drug",
            source_name="Aspirin",
            heading_path="10 Liều lượng và cách dùng > 10.2 Liều lượng > 10.2.2 Trẻ em",
            source_slug="aspirin",
            chunk_id="drug:aspirin:lieu-tre-em",
        ),
    ]
    fake_response = {
        "choices": [{
            "message": {
                "content": "Tài liệu được cung cấp không đủ thông tin để xác nhận liều/cách dùng."
            }
        }]
    }
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: fake_response)
        )
    )
    monkeypatch.setattr(generator, "get_openai", lambda: fake_client)
    monkeypatch.setattr(generator, "build_evidence_brief", lambda **kwargs: None)
    monkeypatch.setattr(
        generator,
        "repair_answer_with_evidence",
        lambda **kwargs: kwargs["answer"] + "\n- Liều 25 mg/lần. [1]",
    )

    answer = generator.generate(
        "Tôi bị đau khớp, bác sĩ kê Aspirin. Tôi muốn biết liều dùng cho người lớn là bao nhiêu?",
        hits,
        answer_domain="drug_info",
        evidence_plan={
            "domain": "drug_info",
            "source_type": "drug",
            "entity": "Aspirin",
            "answer_slot": "dose",
            "confidence": 0.9,
        },
    )

    assert "300 - 900 mg" in answer
    assert "4 g/ngày" in answer
    assert "4 - 8 g/ngày" in answer
    assert "25 mg/lần" not in answer
    assert "60 - 130 mg/kg/ngày" not in answer
    assert "Indomethacin" not in answer


def test_drug_usage_fallback_keeps_matching_local_regimen_only():
    text = (
        "Giun đũa, giun móc hoặc giun tóc, giun kim:\n"
        "Người lớn và trẻ em trên 2 tuổi: 400 mg uống 1 liều duy nhất.\n"
        "Trẻ em 1 - 2 tuổi: 200 mg uống 1 liều duy nhất.\n"
        "Giun lươn:\n"
        "Người lớn và trẻ em trên 2 tuổi: 400 mg/lần, 2 lần/ngày, trong 3 ngày. "
        "Có thể nhắc lại sau 3 tuần.\n"
        "Điều trị bệnh ấu trùng di chuyển ở da:\n"
        "Người lớn và trẻ em trên 2 tuổi: 400 mg/lần/ngày, uống trong 3 ngày."
    )
    hits = [
        Hit(
            text=text,
            score=1.0,
            source_type="drug",
            source_name="Albendazole",
            heading_path=(
                "7 Liều lượng và cách dùng > 7.2 Liều lượng > "
                "7.2.3 Điều trị nhiễm ký sinh trùng đường ruột như giun đũa"
            ),
            source_slug="albendazole",
            chunk_id="drug:albendazole:lieu-giun-duong-ruot-a",
        ),
        Hit(
            text=text,
            score=0.9,
            source_type="drug",
            source_name="Albendazole",
            heading_path=(
                "7 Liều lượng và cách dùng > 7.2 Liều lượng > "
                "7.2.3 Điều trị nhiễm ký sinh trùng đường ruột như giun đũa"
            ),
            source_slug="albendazole",
            chunk_id="drug:albendazole:lieu-giun-duong-ruot-b",
        ),
    ]

    answer = generator._drug_usage_fallback_answer(
        hits,
        [1, 1],
        question="Tôi bị nhiễm giun đũa, bác sĩ kê Albendazole. Tôi uống thế nào?",
    )

    assert "400 mg uống 1 liều duy nhất" in answer
    assert "400 mg/lần, 2 lần/ngày" not in answer
    assert "ấu trùng di chuyển" not in answer


def test_drug_usage_fallback_filters_age_specific_lines():
    hits = [
        Hit(
            text=(
                "Mày đay, ngứa:\n"
                "Trẻ em 2 - 4 tuổi: 2,5 mg/lần, ngày 3 - 4 lần.\n"
                "Trẻ em 5 - 11 tuổi: 5 mg/lần, ngày 3 - 4 lần.\n"
                "Trẻ em >= 12 tuổi và người lớn: 10 mg/lần, 2 - 3 lần/ngày.\n"
                "Người cao tuổi nên giảm liều: 10 mg/lần, ngày dùng 1 - 2 lần.\n"
                "Tiền mê trước phẫu thuật cho trẻ em:\n"
                "Trẻ em 2 - 7 tuổi: uống liều cao nhất là 2 mg/kg."
            ),
            score=1.0,
            source_type="drug",
            source_name="Alimemazine",
            heading_path="10 Liều lượng và cách dùng > 10.2 Liều lượng",
            source_slug="alimemazine",
            chunk_id="drug:alimemazine:lieu-luong",
        )
    ]

    answer = generator._drug_usage_fallback_answer(
        hits,
        [1],
        question=(
            "Con tôi 3 tuổi bị mày đay, tôi muốn dùng Alimemazine cho con. "
            "Vậy liều dùng như thế nào là phù hợp?"
        ),
    )

    assert "2,5 mg/lần" in answer
    assert "Trẻ em 5 - 11 tuổi" not in answer
    assert "Trẻ em >= 12 tuổi" not in answer
    assert "Người cao tuổi" not in answer
    assert "2 mg/kg" not in answer


def test_drug_contract_removes_unsupported_allergy_minimization():
    hits = [
        Hit(
            text=(
                "Hãy tham khảo ý kiến bác sĩ hoặc dược sĩ trước khi sử dụng.\n"
                "Không sử dụng với người quá mẫn cảm với thành phần của thuốc."
            ),
            score=1.0,
            source_type="drug",
            source_name="Thuốc A",
            heading_path="Thận trọng khi sử dụng",
            source_slug="thuoc-a",
            chunk_id="drug:thuoc-a:than-trong",
        )
    ]

    cleaned = generator.enforce_info_answer_contract(
        (
            "Thuốc A hiếm khi gây dị ứng [1]. "
            "Không ghi nhận dị ứng chéo với thành phần khác [1]. "
            "Hãy tham khảo bác sĩ/dược sĩ trước khi dùng [1]."
        ),
        hits,
        "drug_info",
        "Tôi bị dị ứng với một số thành phần mỹ phẩm, có dùng Thuốc A được không?",
    )

    assert "hiếm khi" not in cleaned
    assert "Không ghi nhận" not in cleaned
    assert "Không sử dụng nếu bạn quá mẫn cảm với thành phần của thuốc" in cleaned


def test_generator_repair_is_limited_to_disease_and_drug(monkeypatch):
    calls: list[dict] = []
    disease_hit = Hit(
        text="Basedow là bệnh cường giáp tự miễn.",
        score=1.0,
        source_type="disease",
        source_name="Basedow",
        heading_path="I. Đại cương",
        source_slug="basedow",
        chunk_id="disease:basedow:intro",
    )
    health_hit = Hit(
        text="Người tham gia bảo hiểm y tế được cấp thẻ bảo hiểm y tế.",
        score=1.0,
        source_type="health_insurance",
        source_name="Luật Bảo hiểm y tế",
        heading_path="Điều 16",
        source_slug="22-vbhn-vpqh",
        chunk_id="health_insurance:22-vbhn-vpqh:article:16",
    )

    fake_response = {
        "choices": [{"message": {"content": "Câu trả lời ban đầu [1]."}}]
    }
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: fake_response)
        )
    )

    def fake_repair(**kwargs):
        calls.append(kwargs)
        return "Câu trả lời đã sửa [1]."

    monkeypatch.setattr(generator, "get_openai", lambda: fake_client)
    monkeypatch.setattr(generator, "repair_answer_with_evidence", fake_repair)

    disease_answer = generator.generate(
        "Basedow là gì?",
        [disease_hit],
        answer_domain="disease_info",
        evidence_plan={
            "domain": "disease_info",
            "source_type": "disease",
            "entity": "Basedow",
            "answer_slot": "overview",
            "target_heading_paths": ["Đại cương"],
            "required_facts": ["bản chất bệnh"],
            "confidence": 0.9,
        },
    )
    assert "Câu trả lời đã sửa" in disease_answer
    assert calls and calls[0]["answer_domain"] == "disease_info"

    calls.clear()
    health_answer = generator.generate(
        "Thẻ bảo hiểm y tế được cấp thế nào?",
        [health_hit],
        answer_domain="health_insurance_info",
        evidence_plan={
            "domain": "health_insurance_info",
            "source_type": "health_insurance",
            "entity": "thẻ bảo hiểm y tế",
            "answer_slot": "overview",
            "target_heading_paths": ["Điều 16"],
            "required_facts": ["cấp thẻ"],
            "confidence": 0.9,
        },
    )
    assert "Câu trả lời ban đầu" in health_answer
    assert calls == []


def test_evidence_brief_normalizes_llm_selection(monkeypatch):
    monkeypatch.setattr(
        evidence_brief,
        "call_mini",
        lambda *args, **kwargs: {
            "selected_indexes": [1, "2", 2, 99, "x"],
            "must_include_facts": ["định nghĩa", "định nghĩa", "đối tượng thường gặp"],
            "avoid_topics": ["điều trị"],
            "brief": "Chỉ dùng hai đoạn đầu.",
        },
    )
    hits = [
        Hit("intro", 1.0, "disease", "Basedow", "I. Đại cương", "basedow"),
        Hit("symptoms", 0.9, "disease", "Basedow", "II. Triệu chứng", "basedow"),
    ]

    brief = evidence_brief.build_evidence_brief(
        question="Basedow là gì?",
        hits=hits,
        evidence_plan=None,
        answer_domain="disease_info",
    )

    assert brief == {
        "selected_indexes": [1, 2],
        "must_include_facts": ["định nghĩa", "đối tượng thường gặp"],
        "avoid_topics": ["điều trị"],
        "brief": "Chỉ dùng hai đoạn đầu.",
    }


def test_answer_verifier_returns_repaired_answer(monkeypatch):
    monkeypatch.setattr(
        answer_verifier,
        "call_mini",
        lambda *args, **kwargs: {
            "needs_rewrite": True,
            "missing_required_facts": ["tính chất gia đình"],
            "unsupported_claims": ["biến chứng"],
            "repaired_answer": "Basedow có tính chất gia đình [1].",
        },
    )

    repaired = answer_verifier.repair_answer_with_evidence(
        question="Basedow là gì?",
        answer="Basedow có biến chứng tim.",
        evidence_text="[1] Basedow\nBệnh có tính chất gia đình.",
        evidence_plan={"required_facts": ["tính chất gia đình"]},
        answer_domain="disease_info",
    )

    assert repaired == "Basedow có tính chất gia đình [1]."


def test_answer_verifier_keeps_original_when_llm_returns_invalid(monkeypatch):
    monkeypatch.setattr(answer_verifier, "call_mini", lambda *args, **kwargs: None)

    original = "Basedow là bệnh cường giáp tự miễn [1]."
    repaired = answer_verifier.repair_answer_with_evidence(
        question="Basedow là gì?",
        answer=original,
        evidence_text="[1] Basedow\nBasedow là bệnh cường giáp tự miễn.",
        evidence_plan={"required_facts": ["bản chất bệnh"]},
        answer_domain="disease_info",
    )

    assert repaired == original


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


def test_generator_links_bachmai_source_pdf(monkeypatch):
    hit = Hit(
        text="Mày đay có thể gây sẩn phù.",
        score=1.0,
        source_type="disease",
        source_name="Mày đay",
        heading_path="I. ĐẠI CƯƠNG",
        source_slug="may_day",
        metadata={"chapter": "CHƯƠNG 11: DỊ ỨNG - MIỄN DỊCH LÂM SÀNG"},
    )
    fake_response = {
        "choices": [{"message": {"content": "Mày đay gây sẩn phù [1]."}}]
    }
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: fake_response)
        )
    )
    monkeypatch.setattr(generator, "PUBLIC_BASE_URL", "https://chat.example.vn")
    monkeypatch.setattr(generator, "get_openai", lambda: fake_client)

    answer = generator.generate("Mày đay là gì?", [hit])

    assert (
        "[Hướng dẫn chẩn đoán và điều trị - Bệnh viện Bạch Mai - "
        "CHƯƠNG 11: DỊ ỨNG - MIỄN DỊCH LÂM SÀNG - Mày đay]"
        "(https://chat.example.vn/sources/bachmai/may_day.pdf)"
    ) in answer


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


def test_extra_kwargs_omits_thinking_for_mistral_model_behind_proxy():
    # Mistral model routed through a non-Mistral host (e.g. local gateway):
    # the param must still be dropped, since detection by host alone misses it.
    assert mini.chat_completion_extra_kwargs(
        "http://localhost:20128/v1", model="mistral/mistral-small-latest"
    ) == {}


def test_extra_kwargs_keeps_thinking_for_non_mistral_model():
    kwargs = mini.chat_completion_extra_kwargs(
        "http://localhost:20128/v1", model="gpt-4.1-mini"
    )
    assert kwargs == {"extra_body": {"thinking": {"type": "disabled"}}}


def test_generator_returns_no_data_reply_without_retrieval_context():
    assert generator.generate("Bệnh hiếm XYZ điều trị thế nào?", []) == generator.NO_DATA_REPLY


def test_generator_extracts_doctor_handoff_flag_from_llm_json(monkeypatch):
    hits = [Hit("context", 1.0, "disease", "Bệnh cúm", "", "cum")]
    fake_response = {
        "choices": [{
            "message": {
                "content": (
                    '{"answer":"Bạn nên uống đủ nước [1].",'
                    '"doctor_handoff_recommended":true,'
                    '"doctor_specialty":"Hô hấp"}'
                )
            }
        }]
    }
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: fake_response)
        )
    )
    monkeypatch.setattr(generator, "get_openai", lambda: fake_client)

    answer, meta = generator.generate("Tôi bị sốt?", hits, return_meta=True)

    assert answer.startswith("Bạn nên uống đủ nước")
    assert meta["doctor_handoff_recommended"] is True
    assert meta["doctor_specialty"] == "Hô hấp"


def test_generator_marks_no_data_for_doctor_handoff():
    answer, meta = generator.generate("Bệnh hiếm XYZ điều trị thế nào?", [], return_meta=True)

    assert answer == generator.NO_DATA_REPLY
    assert meta["doctor_handoff_recommended"] is True


def test_generator_extracts_fenced_json_payload(monkeypatch):
    hits = [Hit("context", 1.0, "disease", "Bệnh cúm", "", "cum")]
    fake_response = {
        "choices": [{"message": {"content": '```json\n{"answer":"Nội dung trả lời [1].","doctor_handoff_recommended":true}\n```'}}]
    }
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: fake_response)
        )
    )
    monkeypatch.setattr(generator, "get_openai", lambda: fake_client)

    answer, meta = generator.generate("Tôi bị sốt?", hits, return_meta=True)

    assert answer.startswith("Nội dung trả lời")
    assert "```" not in answer
    assert meta["doctor_handoff_recommended"] is True


def test_generator_extracts_json_payload_with_language_prefix(monkeypatch):
    hits = [Hit("context", 1.0, "disease", "Bệnh cúm", "", "cum")]
    fake_response = {
        "choices": [{"message": {"content": 'json\n{"answer":"Nội dung trả lời [1].","doctor_handoff_recommended":true}'}}]
    }
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: fake_response)
        )
    )
    monkeypatch.setattr(generator, "get_openai", lambda: fake_client)

    answer, meta = generator.generate("Tôi bị sốt?", hits, return_meta=True)

    assert answer.startswith("Nội dung trả lời")
    assert "doctor_handoff_recommended" not in answer
    assert meta["doctor_handoff_recommended"] is True


def test_generator_extracts_malformed_json_with_raw_newlines(monkeypatch):
    hits = [Hit("context", 1.0, "disease", "Bệnh cúm", "", "cum")]
    fake_response = {
        "choices": [{"message": {"content": 'json\n{"answer":"Dòng 1\n\nDòng 2 [1]",\n"doctor_handoff_recommended":true}'}}]
    }
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: fake_response)
        )
    )
    monkeypatch.setattr(generator, "get_openai", lambda: fake_client)

    answer, meta = generator.generate("Tôi bị sốt?", hits, return_meta=True)

    assert answer.startswith("Dòng 1")
    assert "doctor_handoff_recommended" not in answer
    assert meta["doctor_handoff_recommended"] is True


def test_generator_renders_literal_escaped_newlines(monkeypatch):
    hits = [Hit("context", 1.0, "disease", "Bệnh cúm", "", "cum")]
    fake_response = {
        "choices": [{"message": {"content": '{"answer":"Dòng 1\\\\n\\\\nDòng 2 [1]","doctor_handoff_recommended":true}'}}]
    }
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: fake_response)
        )
    )
    monkeypatch.setattr(generator, "get_openai", lambda: fake_client)

    answer, meta = generator.generate("Tôi bị sốt?", hits, return_meta=True)

    assert "Dòng 1\n\nDòng 2" in answer
    assert "\\n" not in answer
    assert meta["doctor_handoff_recommended"] is True


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
