from __future__ import annotations

import json

from src.rag import kg_builder


def _write_json(path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_collect_graph_adds_adr_edges_only_for_entries_with_symptom_id(tmp_path, monkeypatch):
    disease_dir = tmp_path / "diseases"
    drug_dir = tmp_path / "drugs"
    symptom_dir = tmp_path / "symptoms"
    disease_dir.mkdir()
    drug_dir.mkdir()
    symptom_dir.mkdir()

    monkeypatch.setattr(kg_builder, "DISEASE_ENTITY_DIR", disease_dir)
    monkeypatch.setattr(kg_builder, "DRUG_ENTITY_DIR", drug_dir)
    monkeypatch.setattr(kg_builder, "SYMPTOM_ENTITY_DIR", symptom_dir)

    _write_json(
        drug_dir / "demo.json",
        {
            "drug_id": "drug:Demo",
            "generic_name_vi": "Demo",
            "adr_summary": {
                "common": [
                    {"text": "Buồn nôn", "symptom_id": "symptom:S_nausea"},
                    {"text": "Tăng men gan"},
                ],
                "rare_serious": [
                    {"text": "Sốc phản vệ", "symptom_id": "symptom:S_anaphylaxis"}
                ],
            },
        },
    )
    _write_json(
        symptom_dir / "nausea.json",
        {"symptom_id": "symptom:S_nausea", "name_vi": "Buồn nôn"},
    )
    _write_json(
        symptom_dir / "anaphylaxis.json",
        {"symptom_id": "symptom:S_anaphylaxis", "name_vi": "Sốc phản vệ"},
    )

    _, edges = kg_builder.collect_graph({})

    adr_edges = [e for e in edges if e["type"] == "MAY_CAUSE_ADR"]
    assert adr_edges == [
        {
            "src": "drug:Demo",
            "tgt": "symptom:S_nausea",
            "type": "MAY_CAUSE_ADR",
            "frequency": "common",
            "text": "Buồn nôn",
        },
        {
            "src": "drug:Demo",
            "tgt": "symptom:S_anaphylaxis",
            "type": "MAY_CAUSE_ADR",
            "frequency": "rare_serious",
            "text": "Sốc phản vệ",
        },
    ]
