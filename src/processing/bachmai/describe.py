"""
bachmai_describe.py
-------------------
Phase 4: Dùng xAI Batch API + VLM để scan TẤT CẢ trang của 1 bệnh trong 1
request (multi-image), mô tả các bảng / hình xuất hiện trong bài.

Vì bảng/hình có thể kéo dài nhiều trang, gửi từng trang riêng lẻ sẽ không
xử lý được continuation. Gửi all-pages-in-one-request cho phép VLM nhận
diện đầy đủ phạm vi của mỗi bảng/hình.

Commands:
    prepare    — build JSONL (1 request / bệnh)
    submit     — upload + tạo batch
    status     — xem status
    collect    — download results → visuals.json per bệnh

Usage:
    python -m src.processing.bachmai.describe prepare --limit 5
    python -m src.processing.bachmai.describe submit
    python -m src.processing.bachmai.describe status
    python -m src.processing.bachmai.describe collect
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR, VISION_MODEL
from src.processing.batch_api import (
    fetch_results,
    get_batch,
    submit_batch,
    write_jsonl,
)

RAW_BASE = OUTPUT_DIR / "bachmai" / "raw"
WORK_DIR = OUTPUT_DIR / "bachmai" / "batch" / "describe"

DISEASE_PROMPT = """Đây là TẤT CẢ các trang của 1 bài bệnh trong sách y khoa tiếng Việt (đã scan).
Các ảnh được đánh số theo thứ tự: trang 1, trang 2, trang 3, ...

Hãy quét toàn bộ các trang và liệt kê MỌI bảng (Bảng N.N) và hình/sơ đồ (Hình N.N) xuất hiện trong bài.

QUAN TRỌNG: Bảng có thể kéo dài nhiều trang (trang tiếp theo không có tiêu đề
bảng, chỉ nối tiếp nội dung). Với bảng kéo dài, hãy GỘP thành 1 entry duy
nhất, mô tả đầy đủ TẤT CẢ các hàng từ đầu đến cuối (qua các trang).

Trả về JSON:
{
  "tables": [
    {
      "title": "Bảng 1.1. ...",
      "page": 2,
      "description": "Diễn giải văn xuôi tiếng Việt: bảng gồm các cột ..., liệt kê đầy đủ từng hàng với giá trị từng cột. Không rút gọn, không dùng 'v.v.', không bỏ hàng."
    }
  ],
  "figures": [
    {
      "title": "Hình 2.3. ...",
      "page": 5,
      "description": "Mô tả văn xuôi: sơ đồ/flowchart gồm những node nào, mũi tên nối gì với gì, chú thích ra sao."
    }
  ]
}

Nguyên tắc:
- "page": số trang (1-based) nơi bảng/hình bắt đầu xuất hiện.
- Nếu không có bảng/hình nào → trả về {"tables": [], "figures": []}.
- Bảng: DIỄN GIẢI bằng văn xuôi — "Hàng 1 (Cột A=..., Cột B=...). Hàng 2 (...). ...". KHÔNG dùng markdown table.
- Bảng kéo dài nhiều trang → 1 entry, "page" là trang bắt đầu, description bao phủ ĐẦY ĐỦ các hàng từ mọi trang.
- Hình/sơ đồ: mô tả chi tiết node/mũi tên/chú thích bằng văn xuôi.
- Giữ tiếng Việt có dấu đầy đủ.
- Không mô tả body text thông thường, chỉ quan tâm bảng + hình.
- Liệt kê theo đúng thứ tự xuất hiện từ trang 1 → trang cuối.
- CHỈ trả về JSON, không có code fence ``` hoặc giải thích.
"""


def encode_image(path: Path) -> str:
    b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_request(custom_id: str, image_paths: list[Path]) -> dict:
    content: list[dict] = [{"type": "text", "text": DISEASE_PROMPT}]
    for i, p in enumerate(image_paths, 1):
        content.append({"type": "text", "text": f"--- TRANG {i} ---"})
        content.append({"type": "image_url", "image_url": {"url": encode_image(p)}})
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": VISION_MODEL,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
        },
    }


def find_disease_dirs(base: Path) -> list[Path]:
    return sorted(
        d for d in base.glob("chuong_*/*")
        if d.is_dir() and (d / "pages_manifest.json").exists()
    )


def cmd_prepare(args) -> None:
    dirs = find_disease_dirs(args.input)
    if args.limit:
        dirs = dirs[: args.limit]

    requests: list[dict] = []
    mapping: list[dict] = []

    for d in dirs:
        pages = json.loads((d / "pages_manifest.json").read_text(encoding="utf-8"))
        image_paths = [d / p["path"] for p in pages if (d / p["path"]).exists()]
        if not image_paths:
            continue
        requests.append(build_request(d.name, image_paths))
        mapping.append({
            "custom_id": d.name,
            "disease_dir": str(d),
            "num_pages": len(image_paths),
        })

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = WORK_DIR / "requests.jsonl"
    write_jsonl(requests, jsonl_path)
    (WORK_DIR / "mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    size_mb = jsonl_path.stat().st_size / 1024 / 1024
    total_pages = sum(m["num_pages"] for m in mapping)
    print(f"Prepared {len(requests)} requests ({total_pages} pages total) "
          f"→ {jsonl_path} ({size_mb:.1f} MB)")


def cmd_submit(args) -> None:
    jsonl_path = WORK_DIR / "requests.jsonl"
    if not jsonl_path.exists():
        raise SystemExit("Chưa có requests.jsonl. Chạy `prepare` trước.")
    batch_id = submit_batch(jsonl_path, "bachmai_describe")
    (WORK_DIR / "batch_id.txt").write_text(batch_id, encoding="utf-8")
    print(f"Đã lưu batch_id → {WORK_DIR / 'batch_id.txt'}")


def cmd_status(args) -> None:
    batch_id = (WORK_DIR / "batch_id.txt").read_text(encoding="utf-8").strip()
    print(json.dumps(get_batch(batch_id), ensure_ascii=False, indent=2))


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"tables": [], "figures": [], "_raw": text}


def cmd_collect(args) -> None:
    batch_id = (WORK_DIR / "batch_id.txt").read_text(encoding="utf-8").strip()
    mapping = {m["custom_id"]: m for m in json.loads((WORK_DIR / "mapping.json").read_text())}

    results = fetch_results(batch_id)
    print(f"Fetched {len(results)} results")

    saved = 0
    for r in results:
        cid = r.get("custom_id")
        m = mapping.get(cid)
        if not m:
            continue
        try:
            text = r["response"]["body"]["choices"][0]["message"]["content"]
        except Exception:
            text = ""
        parsed = parse_json(text)

        tables: list[dict] = []
        for t in parsed.get("tables") or []:
            tables.append({
                "id": f"bang_{len(tables) + 1:02d}",
                "page": t.get("page"),
                "title": t.get("title", ""),
                "description": t.get("description") or t.get("markdown", ""),
            })
        figures: list[dict] = []
        for f in parsed.get("figures") or []:
            figures.append({
                "id": f"hinh_{len(figures) + 1:02d}",
                "page": f.get("page"),
                "title": f.get("title", ""),
                "description": f.get("description", ""),
            })

        out = Path(m["disease_dir"]) / "visuals.json"
        out.write_text(
            json.dumps({"tables": tables, "figures": figures}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        saved += 1
    print(f"Đã lưu visuals.json cho {saved} bệnh.")


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
