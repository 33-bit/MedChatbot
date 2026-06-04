from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

from src.processing.drugs import entities


def test_drug_entity_prompt_requests_structured_adr_entries():
    prompt = entities.SYSTEM_PROMPT

    assert '{"text": "ADR thường gặp", "symptom_id": "symptom:S_<slug>"}' in prompt
    assert '{"text": "ADR không phải triệu chứng"}' in prompt
    assert "Nếu ADR không phải triệu chứng hoặc không chắc symptom_id, chỉ giữ text" in prompt
    assert "ALLOWED_SYMPTOMS" in prompt


def _write_symptom(path, symptom_id, name_vi, name_en=""):
    path.write_text(
        json.dumps(
            {
                "symptom_id": symptom_id,
                "name_vi": name_vi,
                "name_en": name_en,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_build_user_prompt_includes_matching_symptom_candidates(tmp_path, monkeypatch):
    symptom_dir = tmp_path / "symptoms"
    symptom_dir.mkdir()
    _write_symptom(
        symptom_dir / "S_sore_throat.json",
        "symptom:S_sore_throat",
        "Đau họng",
        "Sore throat",
    )
    _write_symptom(
        symptom_dir / "S_oral_infection.json",
        "symptom:S_oral_infection",
        "Nhiễm trùng miệng",
        "Oral infection",
    )
    _write_symptom(symptom_dir / "S_pain.json", "symptom:S_pain", "Đau", "Pain")
    _write_symptom(symptom_dir / "S_cough.json", "symptom:S_cough", "Ho", "Cough")
    monkeypatch.setattr(entities, "SYMPTOM_DIR", symptom_dir, raising=False)

    prompt = entities.build_user_prompt(
        {
            "name": "Demo",
            "sections": [
                {
                    "heading": "Chỉ định",
                    "content": "Điều trị đau họng và nhiễm trùng khoang miệng.",
                }
            ],
        }
    )

    assert "=== ALLOWED_SYMPTOMS ===" in prompt
    assert "symptom:S_sore_throat" in prompt
    assert "symptom:S_oral_infection" in prompt
    assert "symptom:S_pain" not in prompt
    assert "symptom:S_cough" not in prompt


def test_build_user_prompt_prefers_canonical_symptom_id_over_legacy_duplicate(
    tmp_path, monkeypatch
):
    symptom_dir = tmp_path / "symptoms"
    symptom_dir.mkdir()
    _write_symptom(
        symptom_dir / "sore_throat.json",
        "symptom:sore_throat",
        "Đau họng",
        "Sore throat",
    )
    _write_symptom(
        symptom_dir / "S_sore_throat.json",
        "symptom:S_sore_throat",
        "Đau họng",
        "Sore throat",
    )
    monkeypatch.setattr(entities, "SYMPTOM_DIR", symptom_dir, raising=False)

    prompt = entities.build_user_prompt(
        {
            "name": "Demo",
            "sections": [
                {
                    "heading": "Chỉ định",
                    "content": "Dùng trong đau họng.",
                }
            ],
        }
    )

    assert "symptom:S_sore_throat" in prompt
    assert "symptom:sore_throat" not in prompt


def test_direct_mode_writes_entity_json_from_chat_completion(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "entities"
    input_dir.mkdir()
    (input_dir / "demo.json").write_text(
        json.dumps({
            "name": "Demo",
            "international_name": "Demo",
            "atc_codes": [],
            "drug_class": "",
            "sections": [],
        }),
        encoding="utf-8",
    )
    response_payload = {
        "drug_id": "drug:Demo",
        "generic_name_vi": "Demo",
        "adr_summary": {
            "common": [{"text": "Buồn nôn", "symptom_id": "symptom:S_nausea"}],
            "rare_serious": [{"text": "Tăng men gan"}],
        },
    }
    captured: dict[str, object] = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(response_payload, ensure_ascii=False)
                        )
                    )
                ]
            )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    monkeypatch.setattr(entities, "INPUT_DIR", input_dir)
    monkeypatch.setattr(entities, "ENTITY_OUT", output_dir)
    monkeypatch.setattr(entities, "make_openai_client", lambda: fake_client, raising=False)

    entities.cmd_direct(argparse.Namespace(limit=1))

    saved = json.loads((output_dir / "demo.json").read_text(encoding="utf-8"))
    assert saved == {**response_payload, "drug_slug": "demo"}
    assert captured["model"] == entities.MODEL
    assert captured["max_tokens"] == entities.BATCH_MAX_TOKENS
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["messages"][0] == {"role": "system", "content": entities.SYSTEM_PROMPT}
    assert captured["messages"][1]["role"] == "user"
