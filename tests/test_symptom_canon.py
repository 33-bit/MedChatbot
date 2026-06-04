from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.processing import symptom_canon


def test_collect_symptoms_preserves_question_variants_by_slot(tmp_path, monkeypatch):
    disease_dir = tmp_path / "diseases"
    drug_dir = tmp_path / "drugs"
    disease_dir.mkdir()
    drug_dir.mkdir()
    for slug, onset, associated in (
        (
            "disease_a",
            "Sốt xuất hiện từ khi nào?",
            "Có kèm theo đau bụng hoặc tiêu chảy không?",
        ),
        (
            "disease_b",
            "Sốt có xuất hiện sau khi đi vùng dịch không?",
            "Có kèm theo phát ban hoặc đau khớp không?",
        ),
    ):
        (disease_dir / f"{slug}.json").write_text(
            json.dumps({
                "symptom_details": [{
                    "symptom_id": "symptom:S_fever",
                    "name_vi": "Sốt",
                    "name_en": "Fever",
                    "body_system": "Toàn thân",
                    "type": "objective",
                    "clarification_questions": {
                        "onset": onset,
                        "associated": associated,
                    },
                }],
            }, ensure_ascii=False),
            encoding="utf-8",
        )
    monkeypatch.setattr(symptom_canon, "DISEASE_ENTITY_DIR", disease_dir)
    monkeypatch.setattr(symptom_canon, "DRUG_ENTITY_DIR", drug_dir)

    catalog = symptom_canon.collect_symptoms()

    questions = catalog["symptom:S_fever"]["clarification_questions"]
    assert questions["onset"] == [
        "Sốt xuất hiện từ khi nào?",
        "Sốt có xuất hiện sau khi đi vùng dịch không?",
    ]
    assert questions["associated"] == [
        "Có kèm theo đau bụng hoặc tiêu chảy không?",
        "Có kèm theo phát ban hoặc đau khớp không?",
    ]


def test_normalize_question_map_splits_compound_questions():
    normalized = symptom_canon._normalize_question_map({
        "pattern": [
            "Đau liên tục hay từng cơn? Vị trí đau chính ở đâu (thượng vị, quanh rốn, hạ vị...)?",
        ],
    })

    assert normalized["pattern"] == [
        "Đau liên tục hay từng cơn?",
        "Vị trí đau chính ở đâu (thượng vị, quanh rốn, hạ vị...)?",
    ]


def test_system_prompt_forbids_multi_dimension_questions():
    assert "Mỗi câu hỏi chỉ được hỏi 1 ý lâm sàng" in symptom_canon.SYSTEM_PROMPT
    assert "không tạo một string chứa nhiều dấu ?" in symptom_canon.SYSTEM_PROMPT


def test_prepare_options_builds_llm_batch_requests(tmp_path, monkeypatch):
    symptom_dir = tmp_path / "symptoms"
    work_dir = tmp_path / "work"
    symptom_dir.mkdir()
    (symptom_dir / "S_diarrhea.json").write_text(
        json.dumps({
            "symptom_id": "symptom:S_diarrhea",
            "name_vi": "Tiêu chảy",
            "clarification_questions": {
                "onset": "Tiêu chảy bắt đầu từ khi nào?",
                "associated": "Có kèm theo nôn hoặc vàng da không?",
            },
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(symptom_canon, "SYMPTOM_OUT", symptom_dir)
    monkeypatch.setattr(symptom_canon, "OPTION_WORK_DIR", work_dir)
    monkeypatch.setattr(symptom_canon, "MODEL", "mistral-medium-3-5")

    symptom_canon.cmd_prepare_options(SimpleNamespace())

    records = [
        json.loads(line)
        for line in (work_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    body = records[0]["body"]
    assert body["model"] == "mistral-medium-3-5"
    system_prompt = body["messages"][0]["content"]
    user_payload = json.loads(body["messages"][1]["content"])
    assert "clarification_options" in system_prompt
    assert "clarification_selection_modes" in system_prompt
    assert "Không rõ" in system_prompt
    assert "Phân loại từng câu hỏi" in system_prompt
    assert "mode=multi" in system_prompt
    assert "Không tạo option kiểu 'Chỉ X + Y'" in system_prompt
    assert user_payload == [{
        "symptom_id": "symptom:S_diarrhea",
        "name_vi": "Tiêu chảy",
        "clarification_questions": {
            "onset": ["Tiêu chảy bắt đầu từ khi nào?"],
            "associated": ["Có kèm theo nôn hoặc vàng da không?"],
        },
    }]


def test_collect_options_merges_llm_options_into_symptom_files(tmp_path, monkeypatch):
    symptom_dir = tmp_path / "symptoms"
    work_dir = tmp_path / "work"
    symptom_dir.mkdir()
    work_dir.mkdir()
    (symptom_dir / "S_diarrhea.json").write_text(
        json.dumps({
            "symptom_id": "symptom:S_diarrhea",
            "name_vi": "Tiêu chảy",
            "clarification_questions": {
                "onset": "Tiêu chảy bắt đầu từ khi nào?",
                "associated": "Có kèm theo nôn hoặc vàng da không?",
            },
            "source_count": 24,
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    (work_dir / "batch_id.txt").write_text("batch_123", encoding="utf-8")
    monkeypatch.setattr(symptom_canon, "SYMPTOM_OUT", symptom_dir)
    monkeypatch.setattr(symptom_canon, "OPTION_WORK_DIR", work_dir)
    monkeypatch.setattr(symptom_canon, "fetch_results", lambda batch_id: [{
        "custom_id": "symptom_options_0000",
        "response": {
            "body": {
                "choices": [{
                    "message": {
                        "content": json.dumps([
                            {
                                "symptom_id": "symptom:S_diarrhea",
                                "clarification_options": {
                                    "presence": [
                                        "Có tiêu chảy",
                                        "Không tiêu chảy",
                                        "Không rõ",
                                        "Trả lời luôn",
                                    ],
                                    "onset": [
                                        "Dưới 6 giờ",
                                        "6-24 giờ",
                                        "Trên 24 giờ",
                                        "Không rõ",
                                        "Trả lời luôn",
                                    ],
                                    "associated": [
                                        "Có nôn",
                                        "Có vàng da",
                                        "Cả hai",
                                        "Không",
                                        "Không rõ",
                                        "Trả lời luôn",
                                    ],
                                },
                            }
                        ], ensure_ascii=False)
                    }
                }]
            }
        },
    }])

    symptom_canon.cmd_collect_options(SimpleNamespace())

    saved = json.loads((symptom_dir / "S_diarrhea.json").read_text(encoding="utf-8"))
    assert saved["source_count"] == 24
    assert saved["clarification_questions"]["onset"] == ["Tiêu chảy bắt đầu từ khi nào?"]
    assert saved["clarification_options"] == {
        "presence": ["Có tiêu chảy", "Không tiêu chảy", "Không rõ", "Trả lời luôn"],
        "onset": [["Dưới 6 giờ", "6-24 giờ", "Trên 24 giờ", "Không rõ", "Trả lời luôn"]],
        "associated": [["Có nôn", "Có vàng da", "Không", "Không rõ", "Trả lời luôn"]],
    }
    assert saved["clarification_selection_modes"] == {
        "presence": "single",
        "onset": ["single"],
        "associated": ["multi"],
    }


def test_collect_options_flattens_nested_llm_option_groups(tmp_path, monkeypatch):
    symptom_dir = tmp_path / "symptoms"
    work_dir = tmp_path / "work"
    symptom_dir.mkdir()
    work_dir.mkdir()
    (symptom_dir / "S_abdominal_distension.json").write_text(
        json.dumps({
            "symptom_id": "symptom:S_abdominal_distension",
            "name_vi": "Chướng bụng",
            "clarification_questions": {
                "pattern": [
                    "Chướng bụng xảy ra liên tục hay từng đợt? Có liên quan đến bữa ăn hoặc đại tiện không?",
                ],
            },
            "source_count": 7,
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    (work_dir / "batch_id.txt").write_text("batch_123", encoding="utf-8")
    monkeypatch.setattr(symptom_canon, "SYMPTOM_OUT", symptom_dir)
    monkeypatch.setattr(symptom_canon, "OPTION_WORK_DIR", work_dir)
    monkeypatch.setattr(symptom_canon, "fetch_results", lambda batch_id: [{
        "custom_id": "symptom_options_0000",
        "response": {
            "body": {
                "choices": [{
                    "message": {
                        "content": json.dumps([
                            {
                                "symptom_id": "symptom:S_abdominal_distension",
                                "clarification_options": {
                                    "presence": [
                                        "Có chướng bụng",
                                        "Không chướng bụng",
                                        "Trả lời luôn",
                                        "Không rõ",
                                    ],
                                    "pattern": [[
                                        "['Liên tục', 'Từng đợt', 'Không rõ', 'Trả lời luôn']",
                                        [
                                            "Sau bữa ăn",
                                            "Trước đại tiện",
                                            "Không liên quan",
                                            "Trả lời luôn",
                                        ],
                                        "Không rõ",
                                    ]],
                                },
                            }
                        ], ensure_ascii=False)
                    }
                }]
            }
        },
    }])

    symptom_canon.cmd_collect_options(SimpleNamespace())

    saved = json.loads(
        (symptom_dir / "S_abdominal_distension.json").read_text(encoding="utf-8")
    )
    assert saved["clarification_questions"]["pattern"] == [
        "Chướng bụng xảy ra liên tục hay từng đợt?",
        "Có liên quan đến bữa ăn hoặc đại tiện không?",
    ]
    assert saved["clarification_options"] == {
        "presence": [
            "Có chướng bụng",
            "Không chướng bụng",
            "Không rõ",
            "Trả lời luôn",
        ],
        "pattern": [
            ["Liên tục", "Từng đợt", "Không rõ", "Trả lời luôn"],
            [
                "Sau bữa ăn",
                "Trước đại tiện",
                "Không liên quan",
                "Không rõ",
                "Trả lời luôn",
            ],
        ],
    }
    assert saved["clarification_selection_modes"] == {
        "presence": "single",
        "pattern": ["single", "single"],
    }


def test_collect_options_preserves_llm_selection_modes(tmp_path, monkeypatch):
    symptom_dir = tmp_path / "symptoms"
    work_dir = tmp_path / "work"
    symptom_dir.mkdir()
    work_dir.mkdir()
    (symptom_dir / "S_rash.json").write_text(
        json.dumps({
            "symptom_id": "symptom:S_rash",
            "name_vi": "Phát ban",
            "clarification_questions": {
                "associated": [
                    "Có kèm sốt, đau khớp hoặc ngứa không?",
                    "Có xuất hiện sau dùng thuốc hoặc ăn thức ăn lạ không?",
                ],
            },
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    (work_dir / "batch_id.txt").write_text("batch_123", encoding="utf-8")
    monkeypatch.setattr(symptom_canon, "SYMPTOM_OUT", symptom_dir)
    monkeypatch.setattr(symptom_canon, "OPTION_WORK_DIR", work_dir)
    monkeypatch.setattr(symptom_canon, "fetch_results", lambda batch_id: [{
        "custom_id": "symptom_options_0000",
        "response": {
            "body": {
                "choices": [{
                    "message": {
                        "content": json.dumps([
                            {
                                "symptom_id": "symptom:S_rash",
                                "clarification_options": {
                                    "presence": ["Có phát ban", "Không phát ban"],
                                    "associated": [
                                        ["Có sốt", "Có đau khớp", "Có ngứa", "Không"],
                                        ["Sau dùng thuốc", "Sau ăn thức ăn lạ", "Không rõ"],
                                    ],
                                },
                                "clarification_selection_modes": {
                                    "presence": "single",
                                    "associated": ["multi", "single"],
                                },
                            }
                        ], ensure_ascii=False)
                    }
                }]
            }
        },
    }])

    symptom_canon.cmd_collect_options(SimpleNamespace())

    saved = json.loads((symptom_dir / "S_rash.json").read_text(encoding="utf-8"))
    assert saved["clarification_selection_modes"] == {
        "presence": "single",
        "associated": ["multi", "single"],
    }


def test_collect_options_removes_multi_select_combiner_noise(tmp_path, monkeypatch):
    symptom_dir = tmp_path / "symptoms"
    work_dir = tmp_path / "work"
    symptom_dir.mkdir()
    work_dir.mkdir()
    (symptom_dir / "S_headache.json").write_text(
        json.dumps({
            "symptom_id": "symptom:S_headache",
            "name_vi": "Đau đầu",
            "clarification_questions": {
                "associated": [
                    "Có kèm theo buồn nôn/nôn, chóng mặt hoặc rối loạn thị giác không?",
                ],
            },
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    (work_dir / "batch_id.txt").write_text("batch_123", encoding="utf-8")
    monkeypatch.setattr(symptom_canon, "SYMPTOM_OUT", symptom_dir)
    monkeypatch.setattr(symptom_canon, "OPTION_WORK_DIR", work_dir)
    monkeypatch.setattr(symptom_canon, "fetch_results", lambda batch_id: [{
        "custom_id": "symptom_options_0000",
        "response": {
            "body": {
                "choices": [{
                    "message": {
                        "content": json.dumps([
                            {
                                "symptom_id": "symptom:S_headache",
                                "clarification_options": {
                                    "presence": ["Có đau đầu", "Không đau đầu"],
                                    "associated": [[
                                        "Có nôn",
                                        "Có chóng mặt",
                                        "Có rối loạn thị giác",
                                        "Cả 3 triệu chứng",
                                        "Chỉ nôn + chóng mặt",
                                        "Chỉ nôn + thị giác",
                                        "Chỉ chóng mặt + thị giác",
                                        "Nhiều triệu chứng",
                                        "Không",
                                        "Không rõ",
                                        "Trả lời luôn",
                                    ]],
                                },
                                "clarification_selection_modes": {
                                    "presence": "single",
                                    "associated": ["multi"],
                                },
                            }
                        ], ensure_ascii=False)
                    }
                }]
            }
        },
    }])

    symptom_canon.cmd_collect_options(SimpleNamespace())

    saved = json.loads((symptom_dir / "S_headache.json").read_text(encoding="utf-8"))
    assert saved["clarification_options"]["associated"] == [[
        "Có nôn",
        "Có chóng mặt",
        "Có rối loạn thị giác",
        "Không",
        "Không rõ",
        "Trả lời luôn",
    ]]
    assert saved["clarification_selection_modes"]["associated"] == ["multi"]


def test_collect_results_fails_when_llm_omits_symptoms(tmp_path, monkeypatch):
    symptom_dir = tmp_path / "symptoms"
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "batch_id.txt").write_text("batch_123", encoding="utf-8")
    (work_dir / "symptom_catalog_raw.json").write_text(
        json.dumps({
            "symptom:S_fever": {"symptom_id": "symptom:S_fever"},
            "symptom:S_cough": {"symptom_id": "symptom:S_cough"},
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(symptom_canon, "SYMPTOM_OUT", symptom_dir)
    monkeypatch.setattr(symptom_canon, "WORK_DIR", work_dir)
    monkeypatch.setattr(symptom_canon, "fetch_results", lambda batch_id: [{
        "custom_id": "symptoms_0000",
        "response": {
            "body": {
                "choices": [{
                    "message": {
                        "content": json.dumps([
                            {
                                "symptom_id": "symptom:S_fever",
                                "name_vi": "Sốt",
                                "clarification_questions": {},
                            }
                        ], ensure_ascii=False)
                    }
                }]
            }
        },
    }])

    with pytest.raises(SystemExit, match="Thiếu 1 symptom"):
        symptom_canon.cmd_collect_results(SimpleNamespace())
