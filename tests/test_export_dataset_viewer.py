import importlib.util
from pathlib import Path


def load_exporter():
    module_path = Path(__file__).resolve().parents[1] / "eval" / "tools" / "export_dataset_viewer.py"
    spec = importlib.util.spec_from_file_location("export_dataset_viewer", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_html_includes_category_filter(tmp_path, monkeypatch):
    eval_dir = tmp_path / "eval"
    dataset_dir = eval_dir / "datasets"
    artifact_dir = eval_dir / "artifacts"
    dataset_dir.mkdir(parents=True)
    artifact_dir.mkdir()
    (dataset_dir / "medical_qa_benchmark_v2.jsonl").write_text(
        "\n".join(
            [
                '{"id":"QA-1","category":"emergency","question":"Q1","reference_answer":"A1"}',
                '{"id":"QA-2","category":"drug_info","question":"Q2","reference_answer":"A2"}',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    load_exporter().export_html()

    html = (artifact_dir / "dataset_viewer.html").read_text(encoding="utf-8")
    assert 'id="categoryFilter"' in html
    assert "All categories" in html
    assert "categoryOptions" in html
    assert ".column(1)" in html
    assert ".search(selectedCategory" in html
    assert "$.fn.dataTable.util.escapeRegex" in html


def test_export_html_matches_current_dataset_schema(tmp_path, monkeypatch):
    eval_dir = tmp_path / "eval"
    dataset_dir = eval_dir / "datasets"
    artifact_dir = eval_dir / "artifacts"
    dataset_dir.mkdir(parents=True)
    artifact_dir.mkdir()
    (dataset_dir / "medical_qa_benchmark_v2.jsonl").write_text(
        "\n".join(
            [
                (
                    '{"id":"QA-1","category":"safety_prompt_injection",'
                    '"question":"Ignore rules","reference_answer":"Guardrail reply",'
                    '"source_docs":[],"gold_chunks":[],"gold_heading_paths":[],'
                    '"requires_citation":false,"generated":true,"llm_generated":true}'
                ),
                (
                    '{"id":"QA-2","category":"drug_info","question":"Q2",'
                    '"reference_answer":"A2","source_docs":[{"title":"Drug doc","path":"/tmp/drug.json"}],'
                    '"gold_chunks":["drug:drug:usage"],"gold_heading_paths":["1 Usage"],'
                    '"requires_citation":true,"generated":true,"llm_generated":true}'
                ),
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    load_exporter().export_html()

    html = (artifact_dir / "dataset_viewer.html").read_text(encoding="utf-8")
    assert "Evidence" in html
    assert "formatEvidence" in html
    assert "gold_heading_paths" in html
    assert "gold_chunks" in html
    assert "requires_citation" in html
    assert "safety_prompt_injection" in html
    assert "must_include" not in html
    assert "requires_emergency_advice" not in html
