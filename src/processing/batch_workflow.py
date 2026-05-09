from __future__ import annotations

import json
from pathlib import Path

from src.processing.batch_api import get_batch, submit_batch

REQUESTS_FILE = "requests.jsonl"
BATCH_ID_FILE = "batch_id.txt"


def requests_path(work_dir: Path) -> Path:
    return work_dir / REQUESTS_FILE


def batch_id_path(work_dir: Path) -> Path:
    return work_dir / BATCH_ID_FILE


def submit_existing_requests(work_dir: Path, name: str) -> str:
    jsonl_path = requests_path(work_dir)
    if not jsonl_path.exists():
        raise SystemExit("Chưa có requests.jsonl. Chạy `prepare` trước.")
    batch_id = submit_batch(jsonl_path, name)
    path = batch_id_path(work_dir)
    path.write_text(batch_id, encoding="utf-8")
    print(f"Đã lưu batch_id → {path}")
    return batch_id


def read_batch_id(work_dir: Path) -> str:
    return batch_id_path(work_dir).read_text(encoding="utf-8").strip()


def print_batch_status(work_dir: Path) -> None:
    batch_id = read_batch_id(work_dir)
    print(json.dumps(get_batch(batch_id), ensure_ascii=False, indent=2))
