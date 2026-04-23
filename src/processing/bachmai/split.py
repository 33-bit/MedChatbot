"""
bachmai_split.py
----------------
Phase 1: Split PDF "Chẩn đoán & điều trị bệnh nội khoa - Bệnh viện Bạch Mai"
theo bookmark → mỗi bệnh 1 folder.

Output:
    outputs/bachmai/raw/chuong_XX_<chapter_slug>/<disease_slug>/
        ├── source.pdf            (các trang thuộc bệnh)
        └── metadata.json         (tên bệnh, chương, page range, ...)

Usage:
    python -m src.processing.bachmai.split
    python -m src.processing.bachmai.split --limit 5   # chỉ split 5 bệnh đầu
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

from pypdf import PdfReader, PdfWriter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DOCUMENTS_DIR, OUTPUT_DIR

PDF_PATH = DOCUMENTS_DIR / "Chẩn đoán & điều trị bệnh nội khoa - Bệnh viện Bạch Mai - testyhoc.vn .pdf"
OUTPUT_BASE = OUTPUT_DIR / "bachmai" / "raw"

CHAPTER_RE = re.compile(r"^CHƯƠNG\s+(\d+)\s*[:\-]\s*(.+)$", re.IGNORECASE)


def slugify(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:80]


def parse_outline(reader: PdfReader) -> list[dict]:
    """Walk outline → extract (chapter, disease, start_page)."""
    entries: list[dict] = []
    current_chapter: dict | None = None

    def walk(items) -> None:
        nonlocal current_chapter
        for item in items:
            if isinstance(item, list):
                walk(item)
                continue
            if not hasattr(item, "title"):
                continue
            title = item.title.strip()
            m = CHAPTER_RE.match(title)
            if m:
                num = int(m.group(1))
                name = m.group(2).strip()
                current_chapter = {
                    "chapter_num": num,
                    "chapter_title": title,
                    "chapter_slug": f"chuong_{num:02d}_{slugify(name)}",
                }
                continue
            if current_chapter is None:
                continue
            try:
                page_num = reader.get_destination_page_number(item)
            except Exception:
                continue
            entries.append({
                **current_chapter,
                "disease": title,
                "disease_slug": slugify(title),
                "start_page": page_num,  # 0-based
            })

    walk(reader.outline)
    return entries


def assign_end_pages(entries: list[dict], total_pages: int) -> None:
    entries.sort(key=lambda e: e["start_page"])
    for i, e in enumerate(entries):
        e["end_page"] = entries[i + 1]["start_page"] - 1 if i + 1 < len(entries) else total_pages - 1


def save_disease_pdf(reader: PdfReader, entry: dict, out_path: Path) -> None:
    writer = PdfWriter()
    for p in range(entry["start_page"], entry["end_page"] + 1):
        writer.add_page(reader.pages[p])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        writer.write(f)


def deduplicate_slugs(entries: list[dict]) -> None:
    """Nếu 2 bệnh cùng slug thì gắn suffix _2, _3, ..."""
    seen: dict[tuple[str, str], int] = {}
    for e in entries:
        key = (e["chapter_slug"], e["disease_slug"])
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            e["disease_slug"] = f"{e['disease_slug']}_{seen[key]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Chỉ split N bệnh đầu (0 = tất cả)")
    parser.add_argument("--pdf", type=Path, default=PDF_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_BASE)
    args = parser.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"Không tìm thấy {args.pdf}")

    print(f"Đang đọc {args.pdf.name} ...")
    reader = PdfReader(str(args.pdf))
    total_pages = len(reader.pages)
    print(f"Tổng số trang: {total_pages}")

    entries = parse_outline(reader)
    assign_end_pages(entries, total_pages)
    deduplicate_slugs(entries)
    print(f"Tìm thấy {len(entries)} bệnh trong {len({e['chapter_slug'] for e in entries})} chương\n")

    if args.limit:
        entries = entries[: args.limit]

    args.output.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []

    for i, e in enumerate(entries, 1):
        disease_dir = args.output / e["chapter_slug"] / e["disease_slug"]
        disease_dir.mkdir(parents=True, exist_ok=True)

        pdf_out = disease_dir / "source.pdf"
        save_disease_pdf(reader, e, pdf_out)

        meta = {
            "disease": e["disease"],
            "disease_slug": e["disease_slug"],
            "chapter_num": e["chapter_num"],
            "chapter_title": e["chapter_title"],
            "chapter_slug": e["chapter_slug"],
            "start_page_in_book": e["start_page"] + 1,
            "end_page_in_book": e["end_page"] + 1,
            "num_pages": e["end_page"] - e["start_page"] + 1,
            "source_file": str(args.pdf.relative_to(PROJECT_ROOT)),
        }
        (disease_dir / "metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        index.append(meta)

        print(f"  [{i:3d}/{len(entries)}] {e['chapter_slug']}/{e['disease_slug']} "
              f"(p{meta['start_page_in_book']}-{meta['end_page_in_book']}, "
              f"{meta['num_pages']} pages)")

    (args.output / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nĐã lưu {len(entries)} bệnh + index.json vào {args.output}")


if __name__ == "__main__":
    main()
