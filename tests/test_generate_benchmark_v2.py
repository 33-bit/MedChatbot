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
    }
    assert "tên bệnh" in benchmark.SYSTEM_PROMPTS["disease_info"].lower()
    assert "tên thuốc" in benchmark.SYSTEM_PROMPTS["drug_info"].lower()


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

    monkeypatch.setattr(benchmark, "collect_source_docs", lambda: ([disease], [drug]))
    monkeypatch.setattr(benchmark, "generate_single_doc_case", fake_single_doc_case)
    monkeypatch.setattr(benchmark, "write_jsonl", lambda path, rows: written_rows.extend(rows))

    code = asyncio.run(
        benchmark.run(
            out_path=tmp_path / "benchmark.jsonl",
            concurrency=99,
            content_char_budget=12000,
            disease_limit=None,
            drug_limit=None,
        )
    )

    assert code == 0
    assert max_active == 1
    assert call_order == [
        "disease_info:Viêm gan virus cấp tính",
        "drug_info:Terbinafine",
    ]
    assert written_rows == [
        {"id": "disease_info:Viêm gan virus cấp tính"},
        {"id": "drug_info:Terbinafine"},
    ]


def test_run_can_append_only_drug_info(monkeypatch, tmp_path):
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
    appended_rows = []
    drug_indexes = []

    async def fake_single_doc_case(sem, doc, category, index, content_char_budget):
        assert category == "drug_info"
        drug_indexes.append(index)
        return {"id": f"drug_info:{index}", "category": "drug_info"}

    monkeypatch.setattr(benchmark, "collect_source_docs", lambda: ([disease], [drug]))
    monkeypatch.setattr(benchmark, "generate_single_doc_case", fake_single_doc_case)
    monkeypatch.setattr(benchmark, "append_jsonl", lambda path, rows: appended_rows.extend(rows))
    monkeypatch.setattr(
        benchmark,
        "write_jsonl",
        lambda path, rows: (_ for _ in ()).throw(AssertionError("write_jsonl should not be used")),
    )

    code = asyncio.run(
        benchmark.run(
            out_path=out_path,
            concurrency=1,
            content_char_budget=12000,
            disease_limit=None,
            drug_limit=None,
            categories=("drug_info",),
            append=True,
        )
    )

    assert code == 0
    assert drug_indexes == [1]
    assert appended_rows == [{"id": "drug_info:1", "category": "drug_info"}]
