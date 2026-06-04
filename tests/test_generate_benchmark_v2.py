from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path


def _load_benchmark_module():
    path = Path("eval/generators/generate_benchmark_v2.py").resolve()
    spec = importlib.util.spec_from_file_location("generate_benchmark_v2_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_benchmark_v2_test"] = module
    spec.loader.exec_module(module)
    return module


def test_v2_uses_independent_system_prompts():
    benchmark = _load_benchmark_module()

    assert set(benchmark.SYSTEM_PROMPTS) == {
        "disease_info",
        "drug_info",
        "symptom_triage",
    }
    assert "tên bệnh" in benchmark.SYSTEM_PROMPTS["disease_info"].lower()
    assert "tên thuốc" in benchmark.SYSTEM_PROMPTS["drug_info"].lower()
    assert "tác dụng không mong muốn" in benchmark.SYSTEM_PROMPTS["symptom_triage"].lower()


def test_build_single_doc_case_keeps_one_gold_document():
    benchmark = _load_benchmark_module()
    parsed = benchmark.SingleDocCase(
        question="Mẹ em được chẩn đoán viêm gan virus cấp tính. Bệnh này có dấu hiệu gì?",
        reference_answer="Viêm gan virus cấp tính có thể gây mệt mỏi, chán ăn, nước tiểu sẫm và vàng da.",
        supporting_heading_paths=["II. CHẨN ĐOÁN > 1. Lâm sàng"],
    )
    doc = benchmark.SourceDoc(
        title="Viêm gan virus cấp tính",
        path="/repo/outputs/bachmai/final/viem_gan_virus_cap_tinh.json",
        source_type="disease",
        source_slug="viem_gan_virus_cap_tinh",
        flat_rows=[
            {
                "path": "II. CHẨN ĐOÁN > 1. Lâm sàng",
                "content": "Mệt mỏi, nước tiểu sẫm, vàng da.",
            }
        ],
    )

    row = benchmark.build_single_doc_case(
        parsed,
        doc,
        category="disease_info",
        index=0,
    )

    assert row["category"] == "disease_info"
    assert row["source_docs"] == [
        {
            "title": "Viêm gan virus cấp tính",
            "path": "/repo/outputs/bachmai/final/viem_gan_virus_cap_tinh.json",
        }
    ]
    assert row["gold_heading_paths"] == ["II. CHẨN ĐOÁN > 1. Lâm sàng"]
    assert row["gold_chunks"] == [
        "disease:viem_gan_virus_cap_tinh:ii_chn_on_1_lm_sng"
    ]
    assert row["requires_citation"] is True


def test_build_symptom_triage_case_keeps_disease_and_drug_sources():
    benchmark = _load_benchmark_module()
    parsed = benchmark.SymptomTriageCase(
        question="Mẹ em vàng da, nước tiểu sẫm và chán ăn vài ngày nay. Có nguy hiểm không?",
        reference_answer=(
            "Các dấu hiệu này có thể liên quan bệnh gan mật, tan máu hoặc tác dụng "
            "không mong muốn của thuốc. Nên đi khám sớm và mang danh sách thuốc đang dùng."
        ),
        supporting_refs=[
            {
                "doc_key": "disease:viem_gan_virus_cap_tinh",
                "heading_path": "II. CHẨN ĐOÁN > 1. Lâm sàng",
            },
            {
                "doc_key": "drug:terbinafine",
                "heading_path": "Tác dụng không mong muốn",
            },
        ],
    )
    bundle = benchmark.TriageBundle(
        seed_symptoms=[
            {"id": "symptom:S_jaundice", "name": "Vàng da"},
            {"id": "symptom:S_dark_urine", "name": "Nước tiểu sẫm"},
        ],
        disease_docs=[
            benchmark.EvidenceDoc(
                title="Viêm gan virus cấp tính",
                path="/repo/outputs/bachmai/final/viem_gan_virus_cap_tinh.json",
                source_type="disease",
                source_slug="viem_gan_virus_cap_tinh",
                role="disease_candidate",
                evidence=["Vàng da, nước tiểu sẫm."],
                flat_rows=[
                    {
                        "path": "II. CHẨN ĐOÁN > 1. Lâm sàng",
                        "content": "Mệt mỏi, chán ăn, nước tiểu sẫm màu, vàng da.",
                    }
                ],
            )
        ],
        adr_drug_docs=[
            benchmark.EvidenceDoc(
                title="Terbinafine",
                path="/repo/outputs/otc_drugs/final_json/terbinafine.json",
                source_type="drug",
                source_slug="terbinafine",
                role="adr_candidate",
                evidence=["Vàng da", "Mệt mỏi"],
                flat_rows=[
                    {
                        "path": "Tác dụng không mong muốn",
                        "content": "Có thể gây vàng da, mệt mỏi, nước tiểu sẫm màu.",
                    }
                ],
            )
        ],
        red_flag_docs=[],
    )

    row = benchmark.build_symptom_triage_case(parsed, bundle, index=3)

    assert row is not None
    assert row["category"] == "symptom_triage"
    assert row["requires_citation"] is True
    assert row["source_docs"] == [
        {
            "title": "Viêm gan virus cấp tính",
            "path": "/repo/outputs/bachmai/final/viem_gan_virus_cap_tinh.json",
        },
        {
            "title": "Terbinafine",
            "path": "/repo/outputs/otc_drugs/final_json/terbinafine.json",
        },
    ]
    assert row["acceptable_source_docs"] == row["source_docs"]
    assert row["candidate_diseases"] == ["Viêm gan virus cấp tính"]
    assert row["candidate_adr_drugs"] == ["Terbinafine"]
    assert row["gold_heading_paths"] == [
        "II. CHẨN ĐOÁN > 1. Lâm sàng",
        "Tác dụng không mong muốn",
    ]
    assert row["gold_chunks"] == [
        "disease:viem_gan_virus_cap_tinh:ii_chn_on_1_lm_sng",
        "drug:terbinafine:tc_dng_khng_mong_mun",
    ]
    assert row["gold_supporting_refs"] == [
        {
            "doc_key": "disease:viem_gan_virus_cap_tinh",
            "heading_path": "II. CHẨN ĐOÁN > 1. Lâm sàng",
            "chunk_id": "disease:viem_gan_virus_cap_tinh:ii_chn_on_1_lm_sng",
        },
        {
            "doc_key": "drug:terbinafine",
            "heading_path": "Tác dụng không mong muốn",
            "chunk_id": "drug:terbinafine:tc_dng_khng_mong_mun",
        },
    ]
    assert "mention medication/ADR as possible cause" in row["expected_behavior"]


def test_format_triage_bundle_includes_adr_evidence():
    benchmark = _load_benchmark_module()
    bundle = benchmark.TriageBundle(
        seed_symptoms=[{"id": "symptom:S_rash", "name": "Phát ban"}],
        disease_docs=[],
        adr_drug_docs=[
            benchmark.EvidenceDoc(
                title="Ibuprofen",
                path="/repo/outputs/otc_drugs/final_json/ibuprofen.json",
                source_type="drug",
                source_slug="ibuprofen",
                role="adr_candidate",
                evidence=["Phát ban", "Mẩn ngứa"],
                flat_rows=[
                    {
                        "path": "Tác dụng không mong muốn",
                        "content": "Có thể gây phát ban, mẩn ngứa.",
                    }
                ],
            )
        ],
        red_flag_docs=[],
    )

    prompt_text = benchmark.format_triage_bundle_for_prompt(bundle)

    assert "Triệu chứng đầu vào" in prompt_text
    assert "Nguồn thuốc / ADR" in prompt_text
    assert "Ibuprofen" in prompt_text
    assert "Phát ban" in prompt_text
    assert "[doc_key: drug:ibuprofen | heading_path: Tác dụng không mong muốn]" in prompt_text


def test_run_sends_generation_requests_one_by_one(monkeypatch, tmp_path):
    benchmark = _load_benchmark_module()
    disease = benchmark.SourceDoc(
        title="Viêm gan virus cấp tính",
        path="/repo/outputs/bachmai/final/viem_gan_virus_cap_tinh.json",
        source_type="disease",
        source_slug="viem_gan_virus_cap_tinh",
        flat_rows=[],
    )
    drug = benchmark.SourceDoc(
        title="Terbinafine",
        path="/repo/outputs/otc_drugs/final_json/terbinafine.json",
        source_type="drug",
        source_slug="terbinafine",
        flat_rows=[],
    )
    bundle = benchmark.TriageBundle(
        seed_symptoms=[{"id": "symptom:S_jaundice", "name": "Vàng da"}],
        disease_docs=[],
        adr_drug_docs=[
            benchmark.EvidenceDoc(
                title="Terbinafine",
                path=drug.path,
                source_type="drug",
                source_slug="terbinafine",
                role="adr_candidate",
                evidence=["Vàng da"],
            )
        ],
        red_flag_docs=[],
    )

    active = 0
    max_active = 0
    call_order = []
    written_rows = []

    async def enter_call(label):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        call_order.append(label)
        active -= 1
        return {"id": label}

    async def fake_single_doc_case(sem, doc, category, index, content_char_budget):
        return await enter_call(f"{category}:{doc.title}")

    async def fake_triage_case(sem, bundle_arg, index):
        return await enter_call("symptom_triage")

    monkeypatch.setattr(benchmark, "collect_source_docs", lambda: ([disease], [drug]))
    monkeypatch.setattr(
        benchmark,
        "collect_seed_symptoms",
        lambda: [{"id": "symptom:S_jaundice", "name": "Vàng da"}],
    )
    monkeypatch.setattr(benchmark, "collect_triage_bundle", lambda seed, doc_index: bundle)
    monkeypatch.setattr(benchmark, "generate_single_doc_case", fake_single_doc_case)
    monkeypatch.setattr(benchmark, "generate_symptom_triage_case", fake_triage_case)
    monkeypatch.setattr(benchmark, "write_jsonl", lambda path, rows: written_rows.extend(rows))

    code = asyncio.run(
        benchmark.run(
            out_path=tmp_path / "benchmark.jsonl",
            seed=42,
            concurrency=99,
            content_char_budget=12000,
            symptom_triage_target=1,
            disease_limit=None,
            drug_limit=None,
        )
    )

    assert code == 0
    assert max_active == 1
    assert call_order == [
        "disease_info:Viêm gan virus cấp tính",
        "drug_info:Terbinafine",
        "symptom_triage",
    ]
    assert written_rows == [
        {"id": "disease_info:Viêm gan virus cấp tính"},
        {"id": "drug_info:Terbinafine"},
        {"id": "symptom_triage"},
    ]


def test_run_can_append_only_symptom_triage(monkeypatch, tmp_path):
    benchmark = _load_benchmark_module()
    out_path = tmp_path / "benchmark.jsonl"
    out_path.write_text('{"id":"existing","category":"disease_info"}\n', encoding="utf-8")
    disease = benchmark.SourceDoc(
        title="Viêm gan virus cấp tính",
        path="/repo/outputs/bachmai/final/viem_gan_virus_cap_tinh.json",
        source_type="disease",
        source_slug="viem_gan_virus_cap_tinh",
        flat_rows=[],
    )
    drug = benchmark.SourceDoc(
        title="Terbinafine",
        path="/repo/outputs/otc_drugs/final_json/terbinafine.json",
        source_type="drug",
        source_slug="terbinafine",
        flat_rows=[],
    )
    bundle = benchmark.TriageBundle(
        seed_symptoms=[{"id": "symptom:S_jaundice", "name": "Vàng da"}],
        disease_docs=[],
        adr_drug_docs=[
            benchmark.EvidenceDoc(
                title="Terbinafine",
                path=drug.path,
                source_type="drug",
                source_slug="terbinafine",
                role="adr_candidate",
                evidence=["Vàng da"],
            )
        ],
        red_flag_docs=[],
    )
    appended_rows = []
    triage_indexes = []

    async def fail_single_doc(*args, **kwargs):
        raise AssertionError("single-doc generation should be skipped")

    async def fake_triage_case(sem, bundle_arg, index):
        triage_indexes.append(index)
        return {"id": f"symptom_triage:{index}", "category": "symptom_triage"}

    monkeypatch.setattr(benchmark, "collect_source_docs", lambda: ([disease], [drug]))
    monkeypatch.setattr(
        benchmark,
        "collect_seed_symptoms",
        lambda: [{"id": "symptom:S_jaundice", "name": "Vàng da"}],
    )
    monkeypatch.setattr(benchmark, "collect_triage_bundle", lambda seed, doc_index: bundle)
    monkeypatch.setattr(benchmark, "generate_single_doc_case", fail_single_doc)
    monkeypatch.setattr(benchmark, "generate_symptom_triage_case", fake_triage_case)
    monkeypatch.setattr(benchmark, "append_jsonl", lambda path, rows: appended_rows.extend(rows))
    monkeypatch.setattr(
        benchmark,
        "write_jsonl",
        lambda path, rows: (_ for _ in ()).throw(AssertionError("write_jsonl should not be used")),
    )

    code = asyncio.run(
        benchmark.run(
            out_path=out_path,
            seed=42,
            concurrency=1,
            content_char_budget=12000,
            symptom_triage_target=1,
            disease_limit=None,
            drug_limit=None,
            categories=("symptom_triage",),
            append=True,
        )
    )

    assert code == 0
    assert triage_indexes == [1]
    assert appended_rows == [{"id": "symptom_triage:1", "category": "symptom_triage"}]


def test_evidence_doc_keeps_flat_rows_without_selecting_gold_chunks():
    benchmark = _load_benchmark_module()
    source = benchmark.SourceDoc(
        title="Terbinafine",
        path="/repo/outputs/otc_drugs/final_json/terbinafine.json",
        source_type="drug",
        source_slug="terbinafine",
        flat_rows=[
            {"path": "Chỉ định", "content": "Điều trị nấm da."},
            {
                "path": "Tác dụng không mong muốn",
                "content": "Có thể gây mệt mỏi, vàng da, nước tiểu sẫm màu.",
            },
        ],
    )

    doc = benchmark._evidence_doc(
        {"slug": "terbinafine", "name": "Terbinafine"},
        "drug",
        "adr_candidate",
        {("drug", "terbinafine"): source},
        ["vàng da", "nước tiểu sẫm"],
    )

    assert doc is not None
    assert doc.flat_rows == source.flat_rows
    assert not hasattr(doc, "heading_paths")
    assert not hasattr(doc, "snippets")


def test_invalid_triage_supporting_refs_are_rejected():
    benchmark = _load_benchmark_module()
    parsed = benchmark.SymptomTriageCase(
        question="Tôi bị vàng da và nước tiểu sẫm. Có nguy hiểm không?",
        reference_answer="Chưa thể chẩn đoán chắc chắn qua chat.",
        supporting_refs=[
            {
                "doc_key": "drug:terbinafine",
                "heading_path": "Không có trong prompt",
            }
        ],
    )
    bundle = benchmark.TriageBundle(
        seed_symptoms=[{"id": "symptom:S_jaundice", "name": "Vàng da"}],
        disease_docs=[],
        adr_drug_docs=[
            benchmark.EvidenceDoc(
                title="Terbinafine",
                path="/repo/outputs/otc_drugs/final_json/terbinafine.json",
                source_type="drug",
                source_slug="terbinafine",
                role="adr_candidate",
                evidence=["Vàng da"],
                flat_rows=[
                    {
                        "path": "Tác dụng không mong muốn",
                        "content": "Có thể gây vàng da.",
                    }
                ],
            )
        ],
        red_flag_docs=[],
    )

    assert benchmark.build_symptom_triage_case(parsed, bundle, index=0) is None
