import importlib.util
from pathlib import Path


def load_exporter():
    module_path = Path(__file__).resolve().parents[1] / "eval" / "tools" / "export_result_viewer.py"
    spec = importlib.util.spec_from_file_location("export_result_viewer", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_result_viewer_includes_expected_answer_from_dataset(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "eval" / "datasets"
    result_dir = tmp_path / "eval" / "results"
    artifact_dir = tmp_path / "eval" / "artifacts"
    dataset_dir.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    (dataset_dir / "medical_qa_benchmark_v2.jsonl").write_text(
        (
            '{"id":"QA-1","category":"disease_info","question":"Q1",'
            '"reference_answer":"Expected clinical answer."}'
        ),
        encoding="utf-8",
    )
    result_path = result_dir / "result.jsonl"
    result_path.write_text(
        (
            '{"case_id":"QA-1","category":"disease_info","question":"Q1",'
            '"answer":"Generated answer.","passed":true,"score":0.9}'
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    load_exporter().write_html(result_path, artifact_dir / "result_viewer.html")

    html = (artifact_dir / "result_viewer.html").read_text(encoding="utf-8")
    assert "Expected Answer" in html
    assert "Expected clinical answer." in html


def test_result_viewer_uses_wide_scrollable_table_layout(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "eval" / "datasets"
    result_dir = tmp_path / "eval" / "results"
    artifact_dir = tmp_path / "eval" / "artifacts"
    dataset_dir.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    (dataset_dir / "medical_qa_benchmark_v2.jsonl").write_text(
        '{"id":"QA-1","reference_answer":"Expected answer."}',
        encoding="utf-8",
    )
    result_path = result_dir / "result.jsonl"
    result_path.write_text(
        '{"case_id":"QA-1","question":"Question","answer":"Generated answer."}',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    load_exporter().write_html(result_path, artifact_dir / "result_viewer.html")

    html = (artifact_dir / "result_viewer.html").read_text(encoding="utf-8")
    assert "max-width: none;" in html
    assert "overflow-x: auto;" in html
    assert "min-width: 2100px;" in html
    assert ".question { width: 380px; }" in html
    assert ".expected { width: 430px; }" in html
    assert ".answer { width: 620px; }" in html
