#!/usr/bin/env python3
"""Normalize a benchmark dataset JSONL file.

Walks an existing benchmark JSONL and rewrites every row so:
- list-shaped fields that may have drifted (empty string instead of empty list,
  single string instead of one-item list, chat-message dicts instead of plain
  strings) are coerced back to the expected shape;
- empty / whitespace-only strings are dropped from list fields;
- `must_include_any` groups that ended up empty are removed;
- generic `must_not_include` keywords with high false-positive risk
  (e.g. `"chắc chắn"`, `"hiệu quả"`, `"an toàn"`) are stripped while
  multi-word forbidden phrases (`"chắc chắn bạn bị"`) are preserved;
- bare `safety` cases are reclassified to `safety_self_medication`
  (the category split was introduced after the early dataset);
- safety cases (`category` starts with `safety`) get their grounding
  fields (`requires_citation`, `gold_chunks`, `gold_heading_paths`,
  `supporting_heading_paths`) cleared per the dataset spec.

Usage:
    python3 eval/normalize_dataset.py --in eval/medical_qa_benchmark.jsonl \
                                      --out eval/medical_qa_benchmark.jsonl

Coercion rules (all idempotent):

- `turns`:
    [{"role":"user","content":"X"}, ...]   -> ["X", ...]
    [{"speaker":"...","text":"X"}, ...]    -> ["X", ...]
    [{"message":"X"}, ...]                 -> ["X", ...]
    None / already-string lists            -> passthrough

- list-of-string fields (`must_include`, `must_not_include`, `gold_chunks`,
  `gold_heading_paths`, `supporting_heading_paths`):
    "" or None                             -> []
    "kw"                                   -> ["kw"]
    ["a","","b"]                           -> ["a","b"]   (drop empty)

- group-of-strings field (`must_include_any`):
    "" or None                             -> []
    "kw"                                   -> [["kw"]]
    ["a","b"] (flat list)                  -> [["a","b"]] (one synonym group)
    [["a","b"], [], ["c"]]                 -> [["a","b"],["c"]]  (drop empty group)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Per-field coercion primitives
# ---------------------------------------------------------------------------


def coerce_turns(value: Any) -> Any:
    """Flatten chat-message dicts into plain strings."""
    if value is None:
        return value
    if not isinstance(value, list):
        return value
    coerced: list[str] = []
    for item in value:
        if isinstance(item, str):
            if item.strip():
                coerced.append(item)
        elif isinstance(item, dict):
            text = item.get("content") or item.get("text") or item.get("message") or ""
            if text and str(text).strip():
                coerced.append(str(text))
        else:
            text = str(item)
            if text.strip():
                coerced.append(text)
    return coerced


def coerce_string_list(value: Any) -> Any:
    """Coerce stray scalars into a list-of-strings; drop empty entries."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return value
    return [x for x in value if isinstance(x, str) and x.strip()]


def coerce_group_list(value: Any) -> Any:
    """Coerce stray shapes for the synonym-group field; drop empty groups."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [[value]] if value.strip() else []
    if not isinstance(value, list):
        return value
    if value and all(isinstance(x, str) for x in value):
        cleaned = [x for x in value if x.strip()]
        return [cleaned] if cleaned else []
    cleaned_groups: list[list[str]] = []
    for group in value:
        if not isinstance(group, list):
            continue
        cleaned_group = [x for x in group if isinstance(x, str) and x.strip()]
        if cleaned_group:
            cleaned_groups.append(cleaned_group)
    return cleaned_groups


# ---------------------------------------------------------------------------
# Content-level rules (separate from shape coercion)
# ---------------------------------------------------------------------------

# Single-word / overly generic phrases that produce false positives when used
# as `must_not_include`. Multi-word phrases containing these are preserved.
GENERIC_FORBIDDEN_TOKENS = {
    "chắc chắn",
    "hiệu quả",
    "an toàn",
    "tốt",
    "được",
    "phải",
}


def strip_generic_forbidden(value: Any) -> Any:
    """Drop overly generic single-token entries from `must_not_include`.

    Keeps entries that are multi-word *or* contain extra context
    (e.g. `"chắc chắn bạn bị"` is preserved while `"chắc chắn"` is dropped).
    """
    if not isinstance(value, list):
        return value
    return [
        kw for kw in value
        if isinstance(kw, str) and kw.strip()
        and kw.strip().casefold() not in GENERIC_FORBIDDEN_TOKENS
    ]


def normalize_safety_category(category: Any) -> Any:
    """Bare `safety` cases predate the sub-category split; remap them.

    All such cases in the current dataset are self-medication scenarios.
    Already-specific `safety_*` categories pass through.
    """
    if category == "safety":
        return "safety_self_medication"
    return category


def clear_safety_grounding(row: dict) -> dict:
    """Safety cases are not grounded in source docs; zero out grounding fields."""
    cat = row.get("category") or ""
    if not cat.startswith("safety"):
        return row
    out = dict(row)
    out["requires_citation"] = False
    for f in ("gold_chunks", "gold_heading_paths", "supporting_heading_paths"):
        if f in out:
            out[f] = []
    return out


# ---------------------------------------------------------------------------
# Row-level normalization
# ---------------------------------------------------------------------------

LIST_FIELDS = (
    "must_include",
    "must_not_include",
    "gold_chunks",
    "gold_heading_paths",
    "supporting_heading_paths",
)
GROUP_FIELDS = ("must_include_any",)
TURN_FIELDS = ("turns",)


def normalize_dataset_row(row: dict) -> dict:
    """Return a row with every drift-prone field coerced to its canonical shape."""
    out = dict(row)

    # 1. Shape coercion.
    for field in LIST_FIELDS:
        if field in out:
            out[field] = coerce_string_list(out[field])
    for field in GROUP_FIELDS:
        if field in out:
            out[field] = coerce_group_list(out[field])
    for field in TURN_FIELDS:
        if field in out:
            out[field] = coerce_turns(out[field])

    # 2. Content cleanup.
    if "must_not_include" in out:
        out["must_not_include"] = strip_generic_forbidden(out["must_not_include"])
    if "category" in out:
        out["category"] = normalize_safety_category(out["category"])
    out = clear_safety_grounding(out)

    return out


# ---------------------------------------------------------------------------
# Diff reporting
# ---------------------------------------------------------------------------


def _summarize_diffs(before: list[dict], after: list[dict]) -> dict:
    counters: Counter = Counter()
    for old, new in zip(before, after):
        if old.get("category") != new.get("category"):
            counters["category_renamed"] += 1
        for f in LIST_FIELDS + GROUP_FIELDS + TURN_FIELDS:
            if old.get(f) != new.get(f):
                counters[f"field_changed:{f}"] += 1
        if old.get("requires_citation") != new.get("requires_citation"):
            counters["requires_citation_cleared"] += 1
    return dict(counters)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="src", type=Path, required=True,
                        help="Input dataset JSONL path.")
    parser.add_argument("--out", dest="dst", type=Path, required=True,
                        help="Output path. Pass the same path as --in for in-place.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing.")
    args = parser.parse_args(argv)

    rows = _load_jsonl(args.src)
    fixed = [normalize_dataset_row(r) for r in rows]
    diffs = _summarize_diffs(rows, fixed)
    changed = sum(1 for old, new in zip(rows, fixed) if old != new)

    print(f"Total rows: {len(fixed)}")
    print(f"Rows changed: {changed}")
    if diffs:
        print("Per-field changes:")
        for k, v in sorted(diffs.items()):
            print(f"  {k}: {v}")

    if args.dry_run:
        print("(dry run — no file written)")
        return 0

    _write_jsonl(args.dst, fixed)
    print(f"Wrote: {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
