"""
bachmai_batch.py
----------------
Shared utilities để gọi xAI Batch API.

Workflow:
    1. build JSONL (custom_id + request body)
    2. upload file    → file_id
    3. create batch   → batch_id
    4. poll status
    5. download results

Docs: https://docs.x.ai/developers/advanced-api-usage/batch-api
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import XAI_API_KEY

API_BASE = "https://api.x.ai/v1"


def _headers() -> dict:
    return {"Authorization": f"Bearer {XAI_API_KEY}"}


def upload_file(jsonl_path: Path) -> str:
    """Upload JSONL → file_id."""
    with open(jsonl_path, "rb") as f:
        files = {"file": (jsonl_path.name, f, "application/jsonl")}
        r = httpx.post(f"{API_BASE}/files", headers=_headers(), files=files, timeout=120.0)
    r.raise_for_status()
    return r.json()["id"]


def create_batch(input_file_id: str, name: str) -> str:
    """Tạo batch từ file_id → batch_id."""
    r = httpx.post(
        f"{API_BASE}/batches",
        headers={**_headers(), "Content-Type": "application/json"},
        json={"name": name, "input_file_id": input_file_id},
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()["batch_id"]


def get_batch(batch_id: str) -> dict:
    r = httpx.get(f"{API_BASE}/batches/{batch_id}", headers=_headers(), timeout=30.0)
    r.raise_for_status()
    return r.json()


def is_done(batch: dict) -> bool:
    s = batch.get("state", {})
    pending = s.get("num_pending", 0)
    total = s.get("num_requests", 0)
    return total > 0 and pending == 0


def fetch_results(batch_id: str) -> list[dict]:
    """Fetch tất cả results (pagination) và normalize về shape:
        {"custom_id": ..., "response": {"body": {"choices": [...]}}}
    """
    raw: list[dict] = []
    token: str | None = None
    while True:
        params = {"limit": 100}
        if token:
            params["pagination_token"] = token
        r = httpx.get(
            f"{API_BASE}/batches/{batch_id}/results",
            headers=_headers(),
            params=params,
            timeout=60.0,
        )
        r.raise_for_status()
        payload = r.json()
        raw.extend(payload.get("results", []))
        token = payload.get("pagination_token")
        if not token:
            break

    normalized: list[dict] = []
    for item in raw:
        cid = item.get("custom_id") or item.get("batch_request_id")
        br = item.get("batch_result") or {}
        resp = br.get("response") or item.get("response") or {}
        completion = resp.get("chat_get_completion") or resp.get("body") or {}
        normalized.append({
            "custom_id": cid,
            "response": {"body": completion},
            "error": br.get("error") or item.get("error"),
        })
    return normalized


def write_jsonl(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def submit_batch(jsonl_path: Path, name: str) -> str:
    print(f"Uploading {jsonl_path.name} ({jsonl_path.stat().st_size / 1024:.1f} KB) ...")
    file_id = upload_file(jsonl_path)
    print(f"  file_id: {file_id}")
    batch_id = create_batch(file_id, name)
    print(f"  batch_id: {batch_id}")
    return batch_id


def poll_until_done(batch_id: str, interval: int = 60) -> dict:
    """Polling loop — dùng khi muốn chờ đồng bộ."""
    while True:
        batch = get_batch(batch_id)
        s = batch.get("state", {})
        print(
            f"  pending={s.get('num_pending', 0)} "
            f"success={s.get('num_success', 0)} "
            f"error={s.get('num_error', 0)} "
            f"total={s.get('num_requests', 0)}"
        )
        if is_done(batch):
            return batch
        time.sleep(interval)
