"""
drug_parse.py
-------------
Parse drug plain-text files (outputs/otc_drugs/final_data/*.txt) into
structured JSON documents (Layer 1).

Output: outputs/otc_drugs/final_json/{slug}.json

Usage:
    python -m src.processing.drugs.parse
    python -m src.processing.drugs.parse --limit 5
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

from src.config import OUTPUT_DIR

INPUT_DIR = OUTPUT_DIR / "otc_drugs" / "final_data"
OUTPUT_JSON_DIR = OUTPUT_DIR / "otc_drugs" / "final_json"

HEADER_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")
META_PATTERNS = {
    "international_name": re.compile(r"^Tên chung quốc tế:\s*(.+)", re.IGNORECASE),
    "atc_raw": re.compile(r"^Mã ATC:\s*(.+)", re.IGNORECASE),
    "drug_class": re.compile(r"^Loại thuốc:\s*(.+)", re.IGNORECASE),
}


def parse_atc(raw: str) -> list[str]:
    return [c.strip().rstrip(".") for c in re.split(r"[,;]\s*", raw) if c.strip()]


def parse_drug_text(text: str) -> dict:
    lines = text.split("\n")

    source_url = ""
    drug_name = ""
    meta: dict = {}
    sections: list[dict] = []
    current_stack: list[dict] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not source_url and line.startswith("Nguồn:"):
            source_url = line.replace("Nguồn:", "").strip()
            continue
        if line.startswith("=" * 10):
            continue

        if not drug_name and line and not any(m.match(line) for m in META_PATTERNS.values()) and not HEADER_RE.match(line):
            if not line.startswith("Nguồn:") and line != "":
                drug_name = line
                continue

        for key, pat in META_PATTERNS.items():
            m = pat.match(line)
            if m:
                meta[key] = m.group(1).strip()
                break
        else:
            hm = HEADER_RE.match(line)
            if hm:
                num = hm.group(1)
                title = hm.group(2).strip()
                depth = num.count(".")
                node = {
                    "number": num,
                    "heading": f"{num} {title}",
                    "content": "",
                    "subsections": [],
                }

                if depth == 0:
                    sections.append(node)
                    current_stack = [node]
                else:
                    while len(current_stack) > depth:
                        current_stack.pop()
                    if current_stack:
                        current_stack[-1]["subsections"].append(node)
                    else:
                        sections.append(node)
                    current_stack.append(node)
            else:
                if current_stack and line:
                    if current_stack[-1]["content"]:
                        current_stack[-1]["content"] += "\n" + line
                    else:
                        current_stack[-1]["content"] = line

    atc_codes = parse_atc(meta.get("atc_raw", ""))

    return {
        "drug_slug": "",
        "name": drug_name,
        "international_name": meta.get("international_name", "").rstrip("."),
        "atc_codes": atc_codes,
        "drug_class": meta.get("drug_class", "").rstrip("."),
        "source_url": source_url,
        "sections": sections,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    files = sorted(args.input.glob("*.txt"))
    if args.limit:
        files = files[: args.limit]

    args.output.mkdir(parents=True, exist_ok=True)

    ok = skip = err = 0
    for f in files:
        slug = f.stem
        out_path = args.output / f"{slug}.json"
        if out_path.exists() and not args.overwrite:
            skip += 1
            continue

        try:
            text = f.read_text(encoding="utf-8")
            doc = parse_drug_text(text)
            doc["drug_slug"] = slug

            out_path.write_text(
                json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            ok += 1
        except Exception as e:
            err += 1
            print(f"  ! {slug}: {e}")

    print(f"Parsed {ok} drugs ({skip} skipped, {err} errors) → {args.output}")


if __name__ == "__main__":
    main()
