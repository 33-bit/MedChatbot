from __future__ import annotations

import json
from types import SimpleNamespace

from src.processing import batch_api


def test_mistral_batch_records_omit_thinking(monkeypatch):
    monkeypatch.setattr(batch_api, "BASE_URL", "https://api.mistral.ai/v1", raising=False)

    record = batch_api.chat_completion_request(
        "req_1",
        "mistral-small-latest",
        [{"role": "user", "content": "hello"}],
    )
    prepared = batch_api._compat_record({
        "custom_id": "req_2",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": {"type": "disabled"},
        },
    })

    assert "thinking" not in record["body"]
    assert "thinking" not in prepared["body"]
    assert "model" not in prepared["body"]
    assert "method" not in prepared
    assert "url" not in prepared


def test_mistral_create_batch_uses_batch_jobs_endpoint(monkeypatch):
    calls: list[dict] = []

    def fake_request(method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"id": "batch_123"},
        )

    monkeypatch.setattr(batch_api, "BASE_URL", "https://api.mistral.ai/v1", raising=False)
    monkeypatch.setattr(batch_api, "LLM_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(batch_api, "httpx", SimpleNamespace(request=fake_request), raising=False)
    monkeypatch.setattr(
        batch_api,
        "make_openai_client",
        lambda: SimpleNamespace(
            batches=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(id="openai_batch"))
        ),
    )

    batch_id = batch_api.create_batch(
        "file_123",
        "symptom_options",
        model="mistral-small-latest",
    )

    assert batch_id == "batch_123"
    assert calls == [{
        "method": "POST",
        "url": "https://api.mistral.ai/v1/batch/jobs",
        "headers": {"Authorization": "Bearer test-key"},
        "timeout": 60.0,
        "json": {
            "input_files": ["file_123"],
            "endpoint": "/v1/chat/completions",
            "model": "mistral-small-latest",
            "metadata": {"description": "symptom_options"},
        },
    }]


def test_mistral_get_batch_uses_batch_jobs_endpoint(monkeypatch):
    calls: list[dict] = []

    def fake_request(method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"id": "batch_123", "status": "RUNNING"},
        )

    monkeypatch.setattr(batch_api, "BASE_URL", "https://api.mistral.ai/v1", raising=False)
    monkeypatch.setattr(batch_api, "LLM_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(batch_api, "httpx", SimpleNamespace(request=fake_request), raising=False)
    monkeypatch.setattr(
        batch_api,
        "make_openai_client",
        lambda: SimpleNamespace(
            batches=SimpleNamespace(retrieve=lambda batch_id: {"id": batch_id})
        ),
    )

    assert batch_api.get_batch("batch_123") == {"id": "batch_123", "status": "RUNNING"}
    assert calls == [{
        "method": "GET",
        "url": "https://api.mistral.ai/v1/batch/jobs/batch_123",
        "headers": {"Authorization": "Bearer test-key"},
        "timeout": 60.0,
    }]


def test_mistral_fetch_results_downloads_mistral_output_files(monkeypatch):
    calls: list[dict] = []

    def fake_request(method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        return SimpleNamespace(
            raise_for_status=lambda: None,
            text=json.dumps({
                "custom_id": "req_1",
                "response": {
                    "body": {
                        "choices": [{"message": {"content": "ok"}}],
                    },
                },
            }),
        )

    monkeypatch.setattr(batch_api, "BASE_URL", "https://api.mistral.ai/v1", raising=False)
    monkeypatch.setattr(batch_api, "LLM_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(batch_api, "httpx", SimpleNamespace(request=fake_request), raising=False)
    monkeypatch.setattr(
        batch_api,
        "get_batch",
        lambda batch_id: {
            "id": batch_id,
            "output_file": "out_123",
            "error_file": None,
        },
    )
    monkeypatch.setattr(
        batch_api,
        "make_openai_client",
        lambda: SimpleNamespace(files=SimpleNamespace(content=lambda file_id: "")),
    )

    assert batch_api.fetch_results("batch_123") == [{
        "custom_id": "req_1",
        "response": {
            "body": {
                "choices": [{"message": {"content": "ok"}}],
            },
        },
    }]
    assert calls == [{
        "method": "GET",
        "url": "https://api.mistral.ai/v1/files/out_123/content",
        "headers": {"Authorization": "Bearer test-key"},
        "timeout": 60.0,
    }]


def test_submit_batch_passes_request_model_to_mistral_create_batch(tmp_path, monkeypatch):
    jsonl_path = tmp_path / "requests.jsonl"
    jsonl_path.write_text(
        json.dumps({
            "custom_id": "req_1",
            "body": {
                "model": "mistral-small-latest",
                "messages": [{"role": "user", "content": "hello"}],
            },
        }) + "\n",
        encoding="utf-8",
    )
    seen: dict[str, str | None] = {}
    monkeypatch.setattr(batch_api, "BASE_URL", "https://api.mistral.ai/v1", raising=False)
    monkeypatch.setattr(
        batch_api,
        "make_openai_client",
        lambda: SimpleNamespace(
            files=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(id="file_123")
            )
        ),
    )

    def fake_create_batch(input_file_id, name="", model=None):
        seen["input_file_id"] = input_file_id
        seen["name"] = name
        seen["model"] = model
        return "batch_123"

    monkeypatch.setattr(batch_api, "create_batch", fake_create_batch)

    assert batch_api.submit_batch(jsonl_path, "symptom_options") == "batch_123"
    assert seen == {
        "input_file_id": "file_123",
        "name": "symptom_options",
        "model": "mistral-small-latest",
    }
