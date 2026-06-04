from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

from src.processing.drugs import adr_map


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def test_extract_adr_mentions_groups_unique_texts(tmp_path):
    drug_dir = tmp_path / "drugs"
    drug_dir.mkdir()
    _write_json(
        drug_dir / "aciclovir.json",
        {
            "adr_summary": {
                "common": [
                    {"text": "Nôn"},
                    {"text": "Buồn nôn", "symptom_id": "symptom:S_nausea"},
                ],
                "rare_serious": ["Sốc phản vệ"],
            }
        },
    )
    _write_json(
        drug_dir / "demo.json",
        {"adr_summary": {"common": [{"text": "nôn"}]}},
    )

    mentions = adr_map.extract_adr_mentions(drug_dir)
    grouped = adr_map.group_mentions(mentions)

    assert len(mentions) == 4
    assert grouped["non"]["text"] == "Nôn"
    assert grouped["non"]["count"] == 2
    assert grouped["non"]["occurrences"] == [
        {
            "drug_slug": "aciclovir",
            "bucket": "common",
            "index": 0,
            "existing_symptom_id": None,
        },
        {
            "drug_slug": "demo",
            "bucket": "common",
            "index": 0,
            "existing_symptom_id": None,
        },
    ]


def test_local_mappings_use_exact_and_safe_fuzzy_matches():
    catalog = [
        {
            "symptom_id": "symptom:S_vomiting",
            "name_vi": "Nôn",
            "name_en": "Vomiting",
        },
        {
            "symptom_id": "symptom:S_diarrhea",
            "name_vi": "Tiêu chảy",
            "name_en": "Diarrhea",
        },
    ]
    grouped = {
        "non": {"mention_key": "non", "text": "Nôn", "count": 3},
        "tieu chayy": {"mention_key": "tieu chayy", "text": "Tiêu chayy", "count": 1},
        "roi loan": {"mention_key": "roi loan", "text": "Rối loạn", "count": 1},
    }

    local_map, unresolved = adr_map.build_local_mappings(grouped, catalog)

    assert local_map["non"]["symptom_id"] == "symptom:S_vomiting"
    assert local_map["non"]["match_type"] == "exact"
    assert local_map["tieu chayy"]["symptom_id"] == "symptom:S_diarrhea"
    assert local_map["tieu chayy"]["match_type"] == "fuzzy"
    assert [item["mention_key"] for item in unresolved] == ["roi loan"]


def test_build_llm_messages_include_unresolved_texts_and_full_catalog():
    catalog = [
        {"symptom_id": "symptom:S_vomiting", "name_vi": "Nôn", "name_en": "Vomiting"},
        {"symptom_id": "symptom:S_rash", "name_vi": "Phát ban", "name_en": "Rash"},
    ]
    unresolved = [
        {"mention_key": "ban da", "text": "Ban da", "count": 2},
    ]

    messages = adr_map.build_llm_messages(unresolved, catalog)
    user_payload = json.loads(messages[1]["content"])

    assert "symptom_id" in messages[0]["content"]
    assert user_payload["unresolved_texts"] == unresolved
    assert user_payload["symptom_catalog"] == catalog


def test_apply_mappings_updates_drug_json(tmp_path):
    drug_dir = tmp_path / "drugs"
    drug_dir.mkdir()
    _write_json(
        drug_dir / "aciclovir.json",
        {
            "adr_summary": {
                "common": [
                    {"text": "Nôn"},
                    "Tiêu chảy",
                    {"text": "Không phải triệu chứng"},
                ]
            }
        },
    )
    mappings = {
        "non": {"symptom_id": "symptom:S_vomiting"},
        "tieu chay": {"symptom_id": "symptom:S_diarrhea"},
    }

    result = adr_map.apply_mappings_to_drugs(drug_dir, mappings, dry_run=False)

    saved = json.loads((drug_dir / "aciclovir.json").read_text(encoding="utf-8"))
    assert result == {"files_changed": 1, "entries_changed": 2}
    assert saved["adr_summary"]["common"] == [
        {"text": "Nôn", "symptom_id": "symptom:S_vomiting"},
        {"text": "Tiêu chảy", "symptom_id": "symptom:S_diarrhea"},
        {"text": "Không phải triệu chứng"},
    ]


def test_cmd_llm_writes_mapping_from_chat_completion(tmp_path, monkeypatch):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_json(
        work_dir / "unresolved.json",
        [{"mention_key": "ban da", "text": "Ban da", "count": 2}],
    )
    _write_json(
        work_dir / "symptom_catalog.json",
        [{"symptom_id": "symptom:S_rash", "name_vi": "Phát ban", "name_en": "Rash"}],
    )
    captured: dict[str, object] = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "mappings": [
                                        {
                                            "mention_key": "ban da",
                                            "symptom_id": "symptom:S_rash",
                                            "confidence": "high",
                                        }
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        )
                    )
                ]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setattr(adr_map, "WORK_DIR", work_dir)
    monkeypatch.setattr(adr_map, "make_openai_client", lambda: fake_client)

    adr_map.cmd_llm(argparse.Namespace(limit=0, chunk_size=10))

    saved = json.loads((work_dir / "llm_map.json").read_text(encoding="utf-8"))
    assert saved["ban da"]["symptom_id"] == "symptom:S_rash"
    assert captured["response_format"] == {"type": "json_object"}
