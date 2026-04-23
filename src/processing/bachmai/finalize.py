"""
bachmai_finalize.py
-------------------
Phase 5: Dùng xAI Batch API + LLM text để clean OCR + build JSON cuối
(đầy đủ nội dung, giữ thứ tự trước sau) cho mỗi bệnh.

Input per disease:
    metadata.json
    text_raw.txt
    visuals.json         (từ Phase 4)
    visuals_manifest.json

Output:
    outputs/bachmai/final/{disease_slug}.json

Commands:
    prepare    — build JSONL (1 request / bệnh)
    submit     — upload + tạo batch
    status     — xem status
    collect    — download results → final/{slug}.json

Usage:
    python -m src.processing.bachmai.finalize prepare --limit 5
    python -m src.processing.bachmai.finalize submit
    python -m src.processing.bachmai.finalize status
    python -m src.processing.bachmai.finalize collect
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL, OUTPUT_DIR
from src.processing.batch_api import (
    fetch_results,
    get_batch,
    submit_batch,
    write_jsonl,
)

RAW_BASE = OUTPUT_DIR / "bachmai" / "raw"
FINAL_DIR = OUTPUT_DIR / "bachmai" / "final"
WORK_DIR = OUTPUT_DIR / "bachmai" / "batch" / "finalize"

SYSTEM_PROMPT = """Bạn là công cụ xử lý tài liệu y khoa tiếng Việt.

Nhiệm vụ: Nhận TEXT thô (từ OCR của PDF) + DANH SÁCH BẢNG/HÌNH của một bệnh,
trả về JSON đầy đủ theo schema:

{
  "disease": "Tên bệnh",
  "chapter": "CHƯƠNG X: ...",
  "sections": [
    {"heading": "I. ĐẠI CƯƠNG", "content": "...", "subsections": [...]},
    ...
  ],
  "tables": [
    {"id": "bang_01", "title": "Nhan đề bảng (nếu có)", "position_hint": "đặt sau section I"}
  ],
  "figures": [
    {"id": "hinh_01", "title": "...", "position_hint": "..."}
  ]
}

Nguyên tắc:
- Giữ TOÀN BỘ nội dung text (không tóm tắt, không bỏ sót câu).
- Giữ THỨ TỰ trước sau như bản gốc.
- Sửa lỗi OCR rõ ràng (vd "Điểu" → "Điều", "cóng đóng" → "cộng đồng", mất dấu).
- Gộp các dòng bị wrap sai giữa câu, ngắt dòng ở đúng ranh giới đoạn.
- Xác định heading bằng số La Mã/số Ả Rập đầu dòng (I., II., 1., 1.1., ...).
- Subsections lồng đúng cấp (1., 1.1., 1.1.1., ...).
- Chèn placeholder [BẢNG bang_XX] hoặc [HÌNH hinh_XX] trong content TẠI vị trí xuất hiện trong bản gốc.
- Các dòng header/footer kiểu "testyhoc.vn", "CẨM NANG CHẨN ĐOÁN...", số trang — BỎ.
- Không thêm thông tin ngoài tài liệu.
- CHỈ trả về JSON hợp lệ, không có code fence ``` hoặc giải thích."""


def build_user_prompt(meta: dict, text_raw: str, visuals: dict) -> str:
    table_list = []
    for t in visuals.get("tables", []):
        title = t.get("title", "")
        desc = t.get("description") or t.get("markdown", "")
        table_list.append(f"[{t['id']}] (trang {t['page']}) {title}\n{desc}")
    figure_list = []
    for f in visuals.get("figures", []):
        title = f.get("title", "")
        desc = f.get("description", "")
        figure_list.append(f"[{f['id']}] (trang {f['page']}) {title}\n{desc}")

    parts = [
        f"BỆNH: {meta['disease']}",
        f"CHƯƠNG: {meta['chapter_title']}",
        "",
        "=== TEXT THÔ (text layer của PDF) ===",
        text_raw,
        "",
        "=== BẢNG TRONG BÀI ===",
        "\n\n".join(table_list) if table_list else "(không có)",
        "",
        "=== HÌNH TRONG BÀI ===",
        "\n\n".join(figure_list) if figure_list else "(không có)",
        "",
        "Trả về JSON theo đúng schema đã quy định.",
    ]
    return "\n".join(parts)


def build_request(disease_dir: Path) -> dict | None:
    slug = disease_dir.name
    meta = json.loads((disease_dir / "metadata.json").read_text(encoding="utf-8"))
    text_raw = (disease_dir / "text_raw.txt").read_text(encoding="utf-8")
    visuals_path = disease_dir / "visuals.json"
    visuals = json.loads(visuals_path.read_text(encoding="utf-8")) if visuals_path.exists() else {"tables": [], "figures": []}

    return {
        "custom_id": slug,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(meta, text_raw, visuals)},
            ],
            "response_format": {"type": "json_object"},
        },
    }


def find_disease_dirs(base: Path) -> list[Path]:
    return sorted(
        d for d in base.glob("chuong_*/*")
        if d.is_dir() and (d / "text_raw.txt").exists()
    )


def cmd_prepare(args) -> None:
    dirs = find_disease_dirs(args.input)
    if args.limit:
        dirs = dirs[: args.limit]

    requests = [r for d in dirs if (r := build_request(d))]

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = WORK_DIR / "requests.jsonl"
    write_jsonl(requests, jsonl_path)
    mapping = [{"custom_id": d.name, "disease_dir": str(d)} for d in dirs]
    (WORK_DIR / "mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Prepared {len(requests)} requests → {jsonl_path} "
          f"({jsonl_path.stat().st_size / 1024:.1f} KB)")


def cmd_submit(args) -> None:
    jsonl_path = WORK_DIR / "requests.jsonl"
    if not jsonl_path.exists():
        raise SystemExit("Chưa có requests.jsonl. Chạy `prepare` trước.")
    batch_id = submit_batch(jsonl_path, "bachmai_finalize")
    (WORK_DIR / "batch_id.txt").write_text(batch_id, encoding="utf-8")
    print(f"Đã lưu batch_id → {WORK_DIR / 'batch_id.txt'}")


def cmd_status(args) -> None:
    batch_id = (WORK_DIR / "batch_id.txt").read_text(encoding="utf-8").strip()
    print(json.dumps(get_batch(batch_id), ensure_ascii=False, indent=2))


JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_llm_json(text: str) -> dict | None:
    text = JSON_FENCE_RE.sub("", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
    return None


def cmd_collect(args) -> None:
    batch_id = (WORK_DIR / "batch_id.txt").read_text(encoding="utf-8").strip()
    mapping = {m["custom_id"]: m for m in json.loads((WORK_DIR / "mapping.json").read_text())}

    results = fetch_results(batch_id)
    print(f"Fetched {len(results)} results")
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    ok = bad = 0
    for r in results:
        cid = r.get("custom_id")
        m = mapping.get(cid)
        if not m:
            continue
        try:
            text = r["response"]["body"]["choices"][0]["message"]["content"]
        except Exception:
            text = ""
        parsed = parse_llm_json(text)
        if parsed is None:
            bad += 1
            (FINAL_DIR / f"{cid}.raw.txt").write_text(text, encoding="utf-8")
            print(f"  ! {cid}: JSON parse fail → saved .raw.txt")
            continue

        # Gắn metadata từ Phase 1 (pages, source_file, ...)
        meta = json.loads((Path(m["disease_dir"]) / "metadata.json").read_text())
        parsed.setdefault("disease", meta["disease"])
        parsed.setdefault("chapter", meta["chapter_title"])
        parsed["disease_slug"] = meta["disease_slug"]
        parsed["source_pages"] = [meta["start_page_in_book"], meta["end_page_in_book"]]
        parsed["source_file"] = meta["source_file"]

        (FINAL_DIR / f"{cid}.json").write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        ok += 1

    print(f"Đã lưu {ok} JSON ({bad} lỗi parse) → {FINAL_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["prepare", "submit", "status", "collect"])
    parser.add_argument("--input", type=Path, default=RAW_BASE)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    {"prepare": cmd_prepare, "submit": cmd_submit,
     "status": cmd_status, "collect": cmd_collect}[args.cmd](args)


if __name__ == "__main__":
    main()
