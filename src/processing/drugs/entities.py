"""
drug_entities.py
----------------
Extract structured drug entities (KG Layer 2) from parsed drug JSONs
using xAI Batch API.

Input:  outputs/otc_drugs/final_json/{slug}.json  (Layer 1 — document)
Output: outputs/entities/drugs/{slug}.json         (Layer 2 — KG entity)

Commands:
    prepare    — build JSONL (1 request / thuốc)
    submit     — upload + tạo batch
    status     — xem status
    collect    — download results → entity JSON per thuốc

Usage:
    python -m src.processing.drugs.entities prepare
    python -m src.processing.drugs.entities submit
    python -m src.processing.drugs.entities status
    python -m src.processing.drugs.entities collect
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

INPUT_DIR = OUTPUT_DIR / "otc_drugs" / "final_json"
ENTITY_OUT = OUTPUT_DIR / "entities" / "drugs"
WORK_DIR = OUTPUT_DIR / "otc_drugs" / "batch" / "entities"

SYSTEM_PROMPT = """Bạn là dược sĩ lâm sàng. Nhiệm vụ: phân tích bài thuốc (Dược thư Quốc gia VN)
và trích xuất JSON entity theo schema dưới đây.

SCHEMA:
{
  "drug_id": "drug:<GenericName>",
  "generic_name_vi": "Tên hoạt chất tiếng Việt",
  "generic_name_en": "English generic name",
  "coding": {"atc_code": "ATC:<code>"},
  "drug_class_vi": "Nhóm thuốc tiếng Việt",

  "indicated_for": {
    "disease_ids": ["ICD10:<code>"],
    "symptom_ids": ["symptom:S_<slug>"]
  },

  "dosage": {
    "adults": "Tóm tắt liều người lớn (1-3 câu)",
    "children": "Tóm tắt liều trẻ em (1-3 câu, nếu có)"
  },

  "contraindicated_in": {
    "disease_ids": ["ICD10:<code>"],
    "symptom_ids": []
  },

  "drug_interactions": [
    {
      "with_drug_id": "drug:<GenericName>",
      "severity": "minor | moderate | major",
      "mechanism_vi": "Cơ chế tương tác",
      "clinical_effect_vi": "Hậu quả lâm sàng",
      "management_vi": "Hướng xử trí"
    }
  ],

  "disease_interactions": [
    {
      "disease_id": "ICD10:<code>",
      "effect_vi": "Ảnh hưởng",
      "management_vi": "Hướng xử trí"
    }
  ],

  "adr_summary": {
    "common": ["ADR thường gặp"],
    "rare_serious": ["ADR hiếm gặp nghiêm trọng"]
  },

  "pregnancy_category": "Tóm tắt 1 câu (an toàn / thận trọng / chống chỉ định)",
  "breastfeeding": "Tóm tắt 1 câu"
}

QUY TẮC:
- drug_id: dùng tên hoạt chất tiếng Anh, CamelCase (vd: drug:Paracetamol, drug:Amoxicillin).
- atc_code: lấy mã ATC đầu tiên nếu có nhiều.
- indicated_for.disease_ids: mã ICD-10 cho các bệnh trong section Chỉ định.
- indicated_for.symptom_ids: triệu chứng thuốc trị (vd: S_fever, S_pain).
- drug_interactions: tối đa 10 tương tác quan trọng nhất (ưu tiên major/moderate).
- disease_interactions: bệnh nền cần thận trọng/chống chỉ định (vd: suy gan, suy thận).
- dosage: TÓM TẮT ngắn gọn, không copy nguyên văn.
- Trích xuất TỪ NỘI DUNG bài, KHÔNG bịa thêm.
- CHỈ trả về JSON, không code fence, không giải thích."""


def flatten_sections(sections: list[dict]) -> str:
    parts = []
    for s in sections:
        parts.append(f"\n### {s['heading']}\n{s.get('content', '')}")
        for sub in s.get("subsections", []):
            parts.append(f"\n#### {sub['heading']}\n{sub.get('content', '')}")
            for subsub in sub.get("subsections", []):
                parts.append(f"\n##### {subsub['heading']}\n{subsub.get('content', '')}")
    return "\n".join(parts)


def build_user_prompt(doc: dict) -> str:
    parts = [
        f"THUỐC: {doc['name']}",
        f"TÊN QUỐC TẾ: {doc.get('international_name', '')}",
        f"MÃ ATC: {', '.join(doc.get('atc_codes', []))}",
        f"NHÓM: {doc.get('drug_class', '')}",
        "",
        flatten_sections(doc.get("sections", [])),
    ]
    return "\n".join(parts)


def build_request(slug: str, doc: dict) -> dict:
    return {
        "custom_id": slug,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(doc)},
            ],
            "response_format": {"type": "json_object"},
        },
    }


def cmd_prepare(args) -> None:
    files = sorted(INPUT_DIR.glob("*.json"))
    if args.limit:
        files = files[: args.limit]

    requests = []
    mapping = []
    for f in files:
        slug = f.stem
        doc = json.loads(f.read_text(encoding="utf-8"))
        requests.append(build_request(slug, doc))
        mapping.append({"custom_id": slug, "file": str(f)})

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = WORK_DIR / "requests.jsonl"
    write_jsonl(requests, jsonl_path)
    (WORK_DIR / "mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Prepared {len(requests)} requests → {jsonl_path} "
          f"({jsonl_path.stat().st_size / 1024:.1f} KB)")


def cmd_submit(args) -> None:
    jsonl_path = WORK_DIR / "requests.jsonl"
    if not jsonl_path.exists():
        raise SystemExit("Chưa có requests.jsonl. Chạy `prepare` trước.")
    batch_id = submit_batch(jsonl_path, "drug_entities")
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
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def cmd_collect(args) -> None:
    batch_id = (WORK_DIR / "batch_id.txt").read_text(encoding="utf-8").strip()
    mapping = {m["custom_id"]: m for m in json.loads((WORK_DIR / "mapping.json").read_text())}

    results = fetch_results(batch_id)
    print(f"Fetched {len(results)} results")
    ENTITY_OUT.mkdir(parents=True, exist_ok=True)

    ok = bad = 0
    for r in results:
        cid = r.get("custom_id")
        if cid not in mapping:
            continue
        try:
            text = r["response"]["body"]["choices"][0]["message"]["content"]
        except Exception:
            text = ""

        parsed = parse_llm_json(text)
        if parsed is None:
            bad += 1
            (ENTITY_OUT / f"{cid}.raw.txt").write_text(text, encoding="utf-8")
            print(f"  ! {cid}: JSON parse fail")
            continue

        parsed["drug_slug"] = cid
        (ENTITY_OUT / f"{cid}.json").write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        ok += 1

    print(f"\nĐã lưu {ok} drug entity ({bad} parse fail) → {ENTITY_OUT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["prepare", "submit", "status", "collect"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    {"prepare": cmd_prepare, "submit": cmd_submit,
     "status": cmd_status, "collect": cmd_collect}[args.cmd](args)


if __name__ == "__main__":
    main()
