"""
batch_api.py
------------
Shared utilities for OpenAI Batch API jobs.

Workflow:
    1. build JSONL (OpenAI per-request records)
    2. upload + create batch → batch_id
    3. poll status
    4. fetch results
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BATCH_MAX_TOKENS, make_openai_client

CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"


def _plain(obj):
    """Convert SDK objects into plain Python containers."""
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except TypeError:
            return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def chat_completion_request(
    custom_id: str,
    model: str,
    messages: list[dict],
    *,
    system: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    response_format: dict | None = None,
    thinking_type: str = "disabled",
    reasoning_effort: str | None = None,
) -> dict:
    body_messages = []
    if system:
        body_messages.append({"role": "system", "content": system})
    body_messages.extend(messages)

    body = {
        "model": model,
        "messages": body_messages,
        "max_tokens": max_tokens or BATCH_MAX_TOKENS,
        "thinking": {"type": thinking_type},
    }
    if temperature is not None:
        body["temperature"] = temperature
    if response_format is not None:
        body["response_format"] = response_format
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": CHAT_COMPLETIONS_ENDPOINT,
        "body": body,
    }


def _compat_record(record: dict) -> dict:
    """Accept old in-repo records with `params` while writing OpenAI batch input."""
    if "body" in record:
        record.setdefault("method", "POST")
        record.setdefault("url", CHAT_COMPLETIONS_ENDPOINT)
        record["body"].setdefault("max_tokens", BATCH_MAX_TOKENS)
        record["body"].setdefault("thinking", {"type": "disabled"})
        return record

    params = record.get("params", {})
    messages = list(params.get("messages") or [])
    system = params.get("system")
    return chat_completion_request(
        custom_id=record["custom_id"],
        model=params["model"],
        messages=messages,
        system=system,
        max_tokens=params.get("max_tokens") or BATCH_MAX_TOKENS,
        temperature=params.get("temperature"),
        response_format=params.get("response_format"),
        thinking_type=params.get("thinking", {}).get("type", "disabled"),
        reasoning_effort=params.get("reasoning_effort"),
    )


def _prepare_upload_file(jsonl_path: Path) -> tuple[Path, int]:
    prepared_path = jsonl_path.with_name(f"{jsonl_path.stem}.openai.jsonl")
    requests = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            requests.append(_compat_record(json.loads(line)))

    write_jsonl(requests, prepared_path)
    return prepared_path, len(requests)


def create_batch(input_file_id: str, name: str = "") -> str:
    """Create an OpenAI Batch job and return its ID."""
    kwargs = {
        "input_file_id": input_file_id,
        "endpoint": CHAT_COMPLETIONS_ENDPOINT,
        "completion_window": "24h",
    }
    if name:
        kwargs["metadata"] = {"description": name}
    batch = make_openai_client().batches.create(**kwargs)
    return batch.id


def get_batch(batch_id: str) -> dict:
    batch = make_openai_client().batches.retrieve(batch_id)
    return _plain(batch)


def is_done(batch: dict) -> bool:
    return batch.get("status") in {"completed", "failed", "expired", "cancelled"}


def _file_text(file_response) -> str:
    text = getattr(file_response, "text", None)
    if isinstance(text, str):
        return text
    if callable(text):
        return text()
    if hasattr(file_response, "read"):
        data = file_response.read()
        return data.decode("utf-8") if isinstance(data, bytes) else str(data)
    return str(file_response)


def _normalize_result(item: dict) -> dict:
    body = (item.get("response") or {}).get("body") or {}
    if "choices" in body:
        return item

    content = body.get("output_text", "")
    if not content:
        for output in body.get("output", []) or []:
            if output.get("type") != "message":
                continue
            for block in output.get("content", []) or []:
                if block.get("type") == "output_text":
                    content += block.get("text", "")

    normalized_body = {
        "choices": [{
            "message": {"content": content},
        }]
    }
    return {
        "custom_id": item.get("custom_id"),
        "response": {"body": normalized_body},
        "error": item.get("error"),
    }


def fetch_results(batch_id: str) -> list[dict]:
    """Fetch all results and normalize to the collector shape:
        {"custom_id": ..., "response": {"body": {"choices": [...]}}}
    """
    client = make_openai_client()
    batch = get_batch(batch_id)
    file_ids = [batch.get("output_file_id"), batch.get("error_file_id")]
    normalized: list[dict] = []
    for file_id in filter(None, file_ids):
        content = _file_text(client.files.content(file_id))
        for line in content.splitlines():
            if line.strip():
                normalized.append(_normalize_result(json.loads(line)))
    return normalized


def write_jsonl(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def submit_batch(jsonl_path: Path, name: str) -> str:
    print(f"Loading {jsonl_path.name} ({jsonl_path.stat().st_size / 1024:.1f} KB) ...")
    upload_path, request_count = _prepare_upload_file(jsonl_path)
    print(f"  requests: {request_count}")
    client = make_openai_client()
    with open(upload_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"  input_file_id: {uploaded.id}")
    batch_id = create_batch(uploaded.id, name)
    print(f"  batch_id: {batch_id}")
    return batch_id


def poll_until_done(batch_id: str, interval: int = 60) -> dict:
    """Polling loop — dùng khi muốn chờ đồng bộ."""
    while True:
        batch = get_batch(batch_id)
        counts = batch.get("request_counts", {})
        print(
            f"  status={batch.get('status')} "
            f"completed={counts.get('completed', 0)} "
            f"failed={counts.get('failed', 0)} "
            f"total={counts.get('total', 0)}"
        )
        if is_done(batch):
            return batch
        time.sleep(interval)
