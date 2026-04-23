"""
bachmai_visuals.py
------------------
Phase 3: Render từng trang PDF của mỗi bệnh thành PNG.

(PDF gốc là scan + OCR layer, nên mỗi page là 1 ảnh raster lớn — không thể
detect table/figure riêng lẻ bằng pdfplumber. Giải pháp: render từng trang
rồi để VLM ở Phase 4 quét visuals trên từng trang.)

Output:
    {disease_dir}/pages/page_001.png
    {disease_dir}/pages_manifest.json

Usage:
    python -m src.processing.bachmai.visuals
    python -m src.processing.bachmai.visuals --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR

RAW_BASE = OUTPUT_DIR / "bachmai" / "raw"
RENDER_DPI = 180  # đủ để VLM đọc bảng/chữ nhỏ, < 200KB/page


def render_pages(pdf_path: Path, out_dir: Path, dpi: int) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            filename = f"page_{i:03d}.png"
            out_path = out_dir / filename
            page.to_image(resolution=dpi).save(str(out_path))
            entries.append({
                "page": i,
                "path": f"pages/{filename}",
                "size_kb": round(out_path.stat().st_size / 1024, 1),
            })
    return entries


def find_disease_dirs(base: Path) -> list[Path]:
    return sorted(
        d for d in base.glob("chuong_*/*")
        if d.is_dir() and (d / "source.pdf").exists()
    )


def cleanup_old_visuals(disease_dir: Path) -> None:
    """Xoá output cũ của phiên bản Phase 3 trước (tables/ figures/)."""
    import shutil
    for sub in ("tables", "figures"):
        p = disease_dir / sub
        if p.exists():
            shutil.rmtree(p)
    old_manifest = disease_dir / "visuals_manifest.json"
    if old_manifest.exists():
        old_manifest.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RAW_BASE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=RENDER_DPI)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cleanup", action="store_true", help="Xoá tables/ figures/ cũ")
    args = parser.parse_args()

    dirs = find_disease_dirs(args.input)
    if args.limit:
        dirs = dirs[: args.limit]

    print(f"Render pages cho {len(dirs)} bệnh (dpi={args.dpi}) ...\n")
    for i, d in enumerate(dirs, 1):
        if args.cleanup:
            cleanup_old_visuals(d)
        manifest_path = d / "pages_manifest.json"
        if manifest_path.exists() and not args.overwrite:
            print(f"  [{i:3d}/{len(dirs)}] SKIP {d.name}")
            continue
        entries = render_pages(d / "source.pdf", d / "pages", args.dpi)
        manifest_path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        total_kb = sum(e["size_kb"] for e in entries)
        print(f"  [{i:3d}/{len(dirs)}] {d.name} — {len(entries)} pages, {total_kb:.0f} KB")

    print(f"\nXong {len(dirs)} bệnh.")


if __name__ == "__main__":
    main()
