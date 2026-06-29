from __future__ import annotations

import pytest

from src.chat.guards.drug_policy import evaluate_drug_policy


def _analysis(
    *medications: str,
    intent: str = "pure_info",
    label: str = "informational",
    succeeded: bool = True,
) -> dict:
    return {
        "analysis_succeeded": succeeded,
        "turn": {"label": label, "intent": intent},
        "entities": {"symptoms": [], "medications": list(medications)},
        "context": {"references": []},
    }


def test_blocks_drug_class_that_does_not_resolve_to_an_otc_entry():
    decision = evaluate_drug_policy(
        "Cho tôi thông tin về thuốc kháng histamin H2",
        _analysis("thuốc kháng histamin H2"),
    )

    assert decision.is_drug_question is True
    assert decision.allowed is False
    assert decision.reason == "not_in_otc_list"


def test_blocks_unsafe_self_prescribing_non_otc_drug():
    decision = evaluate_drug_policy(
        "Tôi có nên tự dùng Amoxicillin không?",
        _analysis("Amoxicillin"),
    )

    assert decision.is_drug_question is True
    assert decision.allowed is False
    assert decision.reason == "unsafe_self_prescribing"


def test_blocks_mixed_otc_and_non_otc_drugs():
    decision = evaluate_drug_policy(
        "Paracetamol dùng cùng amoxicillin được không?",
        _analysis("Paracetamol", "amoxicillin", intent="contextual_drug_info"),
    )

    assert decision.is_drug_question is True
    assert decision.allowed is False


def test_allows_drug_in_otc_list():
    decision = evaluate_drug_policy(
        "Paracetamol có tác dụng gì?",
        _analysis("Paracetamol"),
    )

    assert decision.is_drug_question is True
    assert decision.allowed is True
    assert "Paracetamol" in decision.matched_otc_names


@pytest.mark.parametrize(
    "medication",
    (
        "canxi",
        "Calcium",
        "Calci (Canxi)",
        "bổ sung canxi",
        "có nên bổ sung calcium không",
    ),
)
def test_allows_calcium_spelling_variants_in_otc_list(medication):
    decision = evaluate_drug_policy(
        "Tôi bị loãng xương, có nên bổ sung canxi không?",
        _analysis(medication),
    )

    assert decision.is_drug_question is True
    assert decision.allowed is True
    assert "Calci" in decision.matched_otc_names


@pytest.mark.parametrize(
    ("question", "medication"),
    (
        ("Almagate dùng thế nào?", "Almagate"),
        ("Bạc sulfadiazin có chống chỉ định gì?", "Bạc sulfadiazin"),
        ("Miconazole có tác dụng phụ gì?", "Miconazole"),
        ("Sắt gluconat liều dùng ra sao?", "Sắt gluconat"),
    ),
)
def test_allows_factual_monograph_questions_for_prescribed_drugs(question, medication):
    decision = evaluate_drug_policy(question, _analysis(medication))

    assert decision.is_drug_question is True
    assert decision.allowed is True


def test_allows_prescription_context_for_non_otc_drug_information():
    decision = evaluate_drug_policy(
        "Bác sĩ kê Amoxicillin, thuốc này dùng thế nào?",
        _analysis("Amoxicillin"),
    )

    assert decision.is_drug_question is True
    assert decision.allowed is True


def test_blocks_explicit_strength_above_otc_limit():
    decision = evaluate_drug_policy(
        "Famotidin 40 mg có dùng được không?",
        _analysis("Famotidin", intent="contextual_drug_info"),
    )

    assert decision.is_drug_question is True
    assert decision.allowed is False
    assert decision.reason == "outside_otc_constraints"


def test_allows_explicit_strength_at_otc_limit():
    decision = evaluate_drug_policy(
        "Famotidin 20 mg uống thế nào?",
        _analysis("Famotidin", intent="contextual_drug_info"),
    )

    assert decision.allowed is True


def test_blocks_route_outside_otc_constraint():
    decision = evaluate_drug_policy(
        "Famotidin 20 mg tiêm tĩnh mạch thế nào?",
        _analysis("Famotidin", intent="contextual_drug_info"),
    )

    assert decision.allowed is False
    assert decision.reason == "outside_otc_constraints"


def test_analyzer_failure_fails_closed_for_explicit_drug_question():
    decision = evaluate_drug_policy(
        "Cho tôi thông tin về thuốc amoxicillin",
        _analysis(succeeded=False),
    )

    assert decision.is_drug_question is True
    assert decision.allowed is False
    assert decision.reason == "unresolved_drug"


def test_non_drug_medical_question_is_not_blocked():
    decision = evaluate_drug_policy(
        "Mày đay là bệnh gì?",
        _analysis(),
    )

    assert decision.is_drug_question is False
    assert decision.allowed is True
