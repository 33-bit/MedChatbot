"""
adr_map.py
----------
Normalize drug ADR text entries to symptom catalog IDs after drug extraction.

Workflow:
    extract  — collect ADR texts, build local exact/fuzzy mappings, save unresolved
    llm      — map unresolved texts with LLM using the full compact symptom catalog
    apply    — write mapped symptom_id values back into drug entity JSON files

Usage:
    python -m src.processing.drugs.adr_map extract
    python -m src.processing.drugs.adr_map llm --limit 20 --chunk-size 5
    python -m src.processing.drugs.adr_map apply --dry-run
    python -m src.processing.drugs.adr_map apply
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BATCH_MAX_TOKENS, MODEL, OUTPUT_DIR, make_openai_client

DRUG_ENTITY_DIR = OUTPUT_DIR / "entities" / "drugs"
SYMPTOM_DIR = OUTPUT_DIR / "entities" / "symptoms"
WORK_DIR = OUTPUT_DIR / "entities" / "drugs" / "adr_mapping"
ADR_BUCKETS = ("common", "rare_serious")
FUZZY_THRESHOLD = 0.92
FUZZY_MARGIN = 0.03

LLM_SYSTEM_PROMPT = """Bạn là chuyên gia chuẩn hóa thuật ngữ y khoa.

Input JSON có:
- unresolved_texts: danh sách text ADR cần map, mỗi item có mention_key, text, count.
- symptom_catalog: toàn bộ danh mục symptom hiện có, mỗi item có symptom_id, name_vi, name_en.

Nhiệm vụ:
- Với mỗi unresolved text, chọn đúng symptom_id trong symptom_catalog nếu text là triệu chứng.
- Nếu text không phải triệu chứng, là xét nghiệm/bệnh/biến cố không map được, hoặc không chắc, để symptom_id null.
- Không tạo symptom_id mới. Chỉ dùng symptom_id có trong symptom_catalog.
- Ưu tiên match chính xác về nghĩa lâm sàng, không chọn symptom quá rộng nếu có symptom cụ thể hơn.

Trả về JSON object:
{
  "mappings": [
    {
      "mention_key": "...",
      "text": "...",
      "symptom_id": "symptom:S_..." hoặc null,
      "confidence": "high | medium | low | none",
      "reason": "ngắn gọn"
    }
  ]
}

CHỈ trả JSON, không code fence, không giải thích ngoài JSON."""


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.lower().replace("đ", "d")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_symptom_catalog(symptom_dir: Path = SYMPTOM_DIR) -> list[dict]:
    catalog = []
    for path in sorted(symptom_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            data = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        symptom_id = data.get("symptom_id")
        if not isinstance(symptom_id, str) or not symptom_id.startswith("symptom:"):
            continue
        catalog.append({
            "symptom_id": symptom_id,
            "name_vi": data.get("name_vi", ""),
            "name_en": data.get("name_en", ""),
        })
    return catalog


def _adr_item_text(item) -> tuple[str, str | None] | None:
    if isinstance(item, str):
        return item, None
    if isinstance(item, dict):
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            symptom_id = item.get("symptom_id")
            return text, symptom_id if isinstance(symptom_id, str) else None
    return None


def extract_adr_mentions(drug_dir: Path = DRUG_ENTITY_DIR) -> list[dict]:
    mentions = []
    for path in sorted(drug_dir.glob("*.json")):
        if path.parent.name == "adr_mapping":
            continue
        try:
            drug = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        adr_summary = drug.get("adr_summary") or {}
        if not isinstance(adr_summary, dict):
            continue

        for bucket in ADR_BUCKETS:
            values = adr_summary.get(bucket) or []
            if not isinstance(values, list):
                continue
            for idx, item in enumerate(values):
                parsed = _adr_item_text(item)
                if not parsed:
                    continue
                text, existing_symptom_id = parsed
                mentions.append({
                    "mention_key": normalize_text(text),
                    "text": text.strip(),
                    "drug_slug": path.stem,
                    "bucket": bucket,
                    "index": idx,
                    "existing_symptom_id": existing_symptom_id,
                })
    return mentions


def group_mentions(mentions: list[dict]) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    for mention in mentions:
        key = mention["mention_key"]
        if not key:
            continue
        group = grouped.setdefault(
            key,
            {
                "mention_key": key,
                "text": mention["text"],
                "count": 0,
                "existing_symptom_ids": [],
                "occurrences": [],
            },
        )
        group["count"] += 1
        existing_id = mention.get("existing_symptom_id")
        if existing_id and existing_id not in group["existing_symptom_ids"]:
            group["existing_symptom_ids"].append(existing_id)
        group["occurrences"].append({
            "drug_slug": mention["drug_slug"],
            "bucket": mention["bucket"],
            "index": mention["index"],
            "existing_symptom_id": existing_id,
        })
    return dict(sorted(grouped.items()))


def _catalog_terms(catalog: list[dict]) -> list[tuple[str, dict, str]]:
    terms = []
    for entry in catalog:
        for field in ("name_vi", "name_en"):
            value = entry.get(field)
            normalized = normalize_text(value) if isinstance(value, str) else ""
            if normalized:
                terms.append((normalized, entry, field))
    return terms


def _prefer_catalog_entry(entries: list[dict]) -> dict:
    return sorted(
        entries,
        key=lambda entry: (
            not str(entry.get("symptom_id", "")).startswith("symptom:S_"),
            len(normalize_text(entry.get("name_vi") or entry.get("name_en") or "")),
            entry.get("symptom_id", ""),
        ),
    )[0]


def build_local_mappings(
    grouped_mentions: dict[str, dict],
    catalog: list[dict],
    *,
    fuzzy_threshold: float = FUZZY_THRESHOLD,
    fuzzy_margin: float = FUZZY_MARGIN,
) -> tuple[dict[str, dict], list[dict]]:
    exact_index: dict[str, list[dict]] = {}
    terms_by_first: dict[str, list[tuple[str, dict, str]]] = {}
    terms = _catalog_terms(catalog)
    for term, entry, _ in terms:
        exact_index.setdefault(term, []).append(entry)
    for term, entry, field in terms:
        terms_by_first.setdefault(term[:1], []).append((term, entry, field))

    mappings: dict[str, dict] = {}
    unresolved = []
    for key, mention in grouped_mentions.items():
        exact_entries = exact_index.get(key)
        if exact_entries:
            entry = _prefer_catalog_entry(exact_entries)
            mappings[key] = {
                "mention_key": key,
                "text": mention["text"],
                "symptom_id": entry["symptom_id"],
                "confidence": "high",
                "match_type": "exact",
            }
            continue

        scored = []
        candidate_terms = terms_by_first.get(key[:1], [])
        for term, entry, field in candidate_terms:
            length_ratio = min(len(key), len(term)) / max(len(key), len(term))
            if length_ratio < 0.75:
                continue
            score = SequenceMatcher(None, key, term).ratio()
            scored.append((score, entry, field))
        scored.sort(key=lambda item: item[0], reverse=True)
        best = scored[0] if scored else None
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if best and best[0] >= fuzzy_threshold and best[0] - second_score >= fuzzy_margin:
            mappings[key] = {
                "mention_key": key,
                "text": mention["text"],
                "symptom_id": best[1]["symptom_id"],
                "confidence": "medium",
                "match_type": "fuzzy",
                "score": round(best[0], 3),
                "matched_field": best[2],
            }
            continue

        unresolved.append({
            "mention_key": key,
            "text": mention["text"],
            "count": mention["count"],
            "existing_symptom_ids": mention.get("existing_symptom_ids", []),
        })

    return mappings, unresolved


def build_llm_messages(unresolved: list[dict], catalog: list[dict]) -> list[dict]:
    payload = {
        "unresolved_texts": unresolved,
        "symptom_catalog": catalog,
    }
    return [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]


JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_llm_json(text: str) -> dict | None:
    text = JSON_FENCE_RE.sub("", text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def call_llm_mapping(unresolved: list[dict], catalog: list[dict]) -> dict[str, dict]:
    response = make_openai_client().chat.completions.create(
        model=MODEL,
        messages=build_llm_messages(unresolved, catalog),
        max_tokens=BATCH_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    parsed = parse_llm_json(response.choices[0].message.content or "")
    if not parsed or not isinstance(parsed.get("mappings"), list):
        raise RuntimeError("LLM mapping response did not contain mappings array")

    catalog_ids = {entry["symptom_id"] for entry in catalog}
    mappings = {}
    for item in parsed["mappings"]:
        if not isinstance(item, dict):
            continue
        key = item.get("mention_key")
        symptom_id = item.get("symptom_id")
        if not isinstance(key, str):
            continue
        if symptom_id is not None and symptom_id not in catalog_ids:
            continue
        mappings[key] = {
            "mention_key": key,
            "text": item.get("text", ""),
            "symptom_id": symptom_id,
            "confidence": item.get("confidence", "none"),
            "match_type": "llm",
            "reason": item.get("reason", ""),
        }
    return mappings


def _chunks(values: list[dict], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_mapping_files(work_dir: Path = WORK_DIR) -> dict[str, dict]:
    merged = {}
    for filename in ("local_map.json", "llm_map.json"):
        path = work_dir / filename
        if not path.exists():
            continue
        data = load_json(path)
        if isinstance(data, dict):
            merged.update(data)
    return merged


def apply_mappings_to_drugs(
    drug_dir: Path,
    mappings: dict[str, dict],
    *,
    dry_run: bool,
) -> dict[str, int]:
    files_changed = 0
    entries_changed = 0
    for path in sorted(drug_dir.glob("*.json")):
        if path.parent.name == "adr_mapping":
            continue
        try:
            drug = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        adr_summary = drug.get("adr_summary") or {}
        if not isinstance(adr_summary, dict):
            continue

        changed = False
        for bucket in ADR_BUCKETS:
            values = adr_summary.get(bucket) or []
            if not isinstance(values, list):
                continue
            for idx, item in enumerate(values):
                parsed = _adr_item_text(item)
                if not parsed:
                    continue
                text, _ = parsed
                mapping = mappings.get(normalize_text(text))
                symptom_id = mapping.get("symptom_id") if mapping else None
                if not symptom_id:
                    if isinstance(item, str):
                        values[idx] = {"text": item}
                        changed = True
                    continue

                next_item = dict(item) if isinstance(item, dict) else {"text": item}
                if next_item.get("symptom_id") == symptom_id:
                    continue
                next_item["symptom_id"] = symptom_id
                values[idx] = next_item
                entries_changed += 1
                changed = True

        if changed:
            files_changed += 1
            if not dry_run:
                write_json(path, drug)

    return {"files_changed": files_changed, "entries_changed": entries_changed}


def cmd_extract(args) -> None:
    mentions = extract_adr_mentions(DRUG_ENTITY_DIR)
    grouped = group_mentions(mentions)
    catalog = load_symptom_catalog(SYMPTOM_DIR)
    local_map, unresolved = build_local_mappings(
        grouped,
        catalog,
        fuzzy_threshold=args.fuzzy_threshold,
    )

    write_json(WORK_DIR / "mentions.json", mentions)
    write_json(WORK_DIR / "grouped_mentions.json", grouped)
    write_json(WORK_DIR / "symptom_catalog.json", catalog)
    write_json(WORK_DIR / "local_map.json", local_map)
    write_json(WORK_DIR / "unresolved.json", unresolved)

    print(f"ADR mentions: {len(mentions)}")
    print(f"Unique texts: {len(grouped)}")
    print(f"Local mappings: {len(local_map)}")
    print(f"Unresolved for LLM: {len(unresolved)}")
    print(f"Saved → {WORK_DIR}")


def cmd_llm(args) -> None:
    unresolved_path = WORK_DIR / "unresolved.json"
    catalog_path = WORK_DIR / "symptom_catalog.json"
    if not unresolved_path.exists() or not catalog_path.exists():
        raise SystemExit("Run `extract` first.")

    unresolved = load_json(unresolved_path)
    catalog = load_json(catalog_path)
    if args.limit:
        unresolved = unresolved[: args.limit]

    merged = load_json(WORK_DIR / "llm_map.json") if (WORK_DIR / "llm_map.json").exists() else {}
    for idx, chunk in enumerate(_chunks(unresolved, args.chunk_size), 1):
        print(f"LLM chunk {idx}: {len(chunk)} texts")
        merged.update(call_llm_mapping(chunk, catalog))
        write_json(WORK_DIR / "llm_map.json", merged)

    print(f"LLM mappings: {len(merged)} → {WORK_DIR / 'llm_map.json'}")


def cmd_apply(args) -> None:
    mappings = load_mapping_files(WORK_DIR)
    if not mappings:
        raise SystemExit("No mappings found. Run `extract` and optionally `llm` first.")
    result = apply_mappings_to_drugs(DRUG_ENTITY_DIR, mappings, dry_run=args.dry_run)
    suffix = " (dry-run)" if args.dry_run else ""
    print(f"Files changed: {result['files_changed']}{suffix}")
    print(f"ADR entries changed: {result['entries_changed']}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    extract = sub.add_parser("extract")
    extract.add_argument("--fuzzy-threshold", type=float, default=FUZZY_THRESHOLD)

    llm = sub.add_parser("llm")
    llm.add_argument("--limit", type=int, default=0)
    llm.add_argument("--chunk-size", type=int, default=5)

    apply = sub.add_parser("apply")
    apply.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    {"extract": cmd_extract, "llm": cmd_llm, "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    main()
