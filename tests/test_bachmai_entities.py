from __future__ import annotations

import json
from types import SimpleNamespace

from src.processing.bachmai import entities as bachmai_entities


def test_entities_prompt_allows_multiple_questions_per_slot():
    prompt = bachmai_entities.SYSTEM_PROMPT

    assert '"onset": ["Câu hỏi' in prompt
    assert "mỗi key onset/severity/pattern/associated là array string" in prompt
    assert "Không gộp nhiều ý lâm sàng khác nhau vào một câu hỏi" in prompt


def test_collect_normalizes_symptom_detail_questions_to_lists(tmp_path, monkeypatch):
    work_dir = tmp_path / "work"
    entity_dir = tmp_path / "entities"
    work_dir.mkdir()
    (work_dir / "batch_id.txt").write_text("batch_123", encoding="utf-8")
    (work_dir / "mapping.json").write_text(
        json.dumps([{"custom_id": "benh_a", "file": "x"}], ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(bachmai_entities, "WORK_DIR", work_dir)
    monkeypatch.setattr(bachmai_entities, "ENTITY_OUT", entity_dir)
    monkeypatch.setattr(bachmai_entities, "fetch_results", lambda batch_id: [{
        "custom_id": "benh_a",
        "response": {
            "body": {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "disease_id": "ICD10:A00",
                            "name_vi": "Bệnh A",
                            "symptom_details": [{
                                "symptom_id": "symptom:S_abdominal_pain",
                                "name_vi": "Đau bụng",
                                "clarification_questions": {
                                    "onset": "Đau bụng bắt đầu từ khi nào?",
                                    "pattern": [
                                        "Đau bụng liên tục hay từng cơn?",
                                        "Đau bụng có liên quan bữa ăn không?",
                                    ],
                                },
                            }],
                        }, ensure_ascii=False)
                    }
                }]
            }
        },
    }])

    bachmai_entities.cmd_collect(SimpleNamespace())

    saved = json.loads((entity_dir / "benh_a.json").read_text(encoding="utf-8"))
    questions = saved["symptom_details"][0]["clarification_questions"]
    assert questions == {
        "onset": ["Đau bụng bắt đầu từ khi nào?"],
        "pattern": [
            "Đau bụng liên tục hay từng cơn?",
            "Đau bụng có liên quan bữa ăn không?",
        ],
    }
