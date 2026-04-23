"""
bachmai_extract.py
------------------
Phase 2: Extract text từ source.pdf của mỗi bệnh bằng pypdf.

Output:
    {disease_dir}/text_raw.txt

Usage:
    python -m src.processing.bachmai.extract
    python -m src.processing.bachmai.extract --limit 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR

RAW_BASE = OUTPUT_DIR / "bachmai" / "raw"


def extract_pdf(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    return [p.extract_text() or "" for p in reader.pages]


def find_disease_dirs(base: Path) -> list[Path]:
    return sorted(
        d for d in base.glob("chuong_*/*")
        if d.is_dir() and (d / "source.pdf").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RAW_BASE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dirs = find_disease_dirs(args.input)
    if args.limit:
        dirs = dirs[: args.limit]

    print(f"Extracting text cho {len(dirs)} bệnh ...\n")

    for i, d in enumerate(dirs, 1):
        out = d / "text_raw.txt"
        if out.exists() and not args.overwrite:
            print(f"  [{i:3d}/{len(dirs)}] SKIP {d.name} (đã có text_raw.txt)")
            continue
        pages = extract_pdf(d / "source.pdf")
        body = "\n\n".join(f"=== PAGE {idx} ===\n{t.strip()}" for idx, t in enumerate(pages, 1))
        out.write_text(body, encoding="utf-8")
        chars = sum(len(t) for t in pages)
        print(f"  [{i:3d}/{len(dirs)}] {d.name} — {len(pages)} pages, {chars} chars")

    print(f"\nXong {len(dirs)} bệnh.")


if __name__ == "__main__":
    main()
