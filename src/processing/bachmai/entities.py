"""
bachmai_entities.py
-------------------
Extract structured entities (disease, symptom, drug refs) from final disease
JSONs using xAI Batch API.

Input:  outputs/bachmai/final/{slug}.json  (Layer 1 — document)
Output: outputs/entities/diseases/{slug}.json (Layer 2 — KG entity)

Commands:
    prepare    — build JSONL (1 request / bệnh)
    submit     — upload + tạo batch
    status     — xem status
    collect    — download results → entity JSON per bệnh

Usage:
    python -m src.processing.bachmai.entities prepare
    python -m src.processing.bachmai.entities submit
    python -m src.processing.bachmai.entities status
    python -m src.processing.bachmai.entities collect
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

FINAL_DIR = OUTPUT_DIR / "bachmai" / "final"
ENTITY_OUT = OUTPUT_DIR / "entities" / "diseases"
WORK_DIR = OUTPUT_DIR / "bachmai" / "batch" / "entities"

SYSTEM_PROMPT = """Bạn là chuyên gia y khoa. Nhiệm vụ: phân tích bài hướng dẫn chẩn đoán & điều trị
của 1 bệnh (tiếng Việt), trích xuất ra JSON entity theo schema dưới đây.

SCHEMA:
{
  "disease_id": "ICD10:<code>",
  "name_vi": "Tên bệnh tiếng Việt",
  "name_en": "English name",
  "coding": {"icd10": "<code>"},
  "classification": {
    "body_system": "Hệ cơ quan (vd: Tim mạch, Hô hấp, Truyền nhiễm, Thần kinh, ...)",
    "care_level": "primary | secondary | tertiary",
    "urgency": "routine | urgent | emergency"
  },
  "phases": [
    {
      "name": "Tên giai đoạn (nếu bệnh có diễn tiến theo giai đoạn)",
      "timing": "Thời gian (vd: Ngày 1-3)",
      "symptom_ids": ["symptom:S_<slug>"]
    }
  ],
  "symptoms": {
    "primary_ids": ["symptom:S_<slug>"],
    "red_flag_ids": ["symptom:S_<slug>"]
  },
  "treatment": {
    "first_line_drug_ids": ["drug:<generic_name>"],
    "contraindicated_drug_ids": ["drug:<generic_name>"],
    "when_to_see_doctor": "Mô tả ngắn khi nào cần đến viện",
    "home_care_summary_vi": "Tóm tắt chăm sóc tại nhà (nếu áp dụng)"
  },
  "hospitalization_criteria": {
    "patient_conditions": ["Điều kiện bệnh nhân cần nhập viện"],
    "comorbid_disease_ids": ["ICD10:<code>"]
  },
  "prevention_vi": ["Biện pháp phòng bệnh"],
  "symptom_details": [
    {
      "symptom_id": "symptom:S_<slug>",
      "name_vi": "Tên triệu chứng tiếng Việt",
      "name_en": "English name",
      "body_system": "Hệ cơ quan",
      "type": "objective | subjective",
      "clarification_questions": {
        "onset": "Câu hỏi về thời gian khởi phát",
        "severity": "Câu hỏi về mức độ",
        "pattern": "Câu hỏi về kiểu",
        "associated": "Câu hỏi về triệu chứng kèm theo"
      }
    }
  ]
}

QUY TẮC:
- disease_id: dùng mã ICD-10 phù hợp nhất. Nếu bệnh KHÔNG có mã ICD-10 rõ ràng
  (vd "Cấp cứu ngừng tuần hoàn cơ bản" là protocol chứ không phải bệnh), trả về
  disease_id = null → bệnh này sẽ bị bỏ qua.
- symptom slug: dùng tiếng Anh, snake_case, prefix "S_" (vd: S_fever, S_chest_pain).
- drug generic name: viết đúng tên hoạt chất (vd: Paracetamol, Amoxicillin).
- phases: CHỈ điền nếu bệnh thực sự có giai đoạn diễn tiến rõ ràng (vd sốt xuất huyết).
  Nếu không → phases = [].
- red_flag_ids: triệu chứng báo hiệu nặng / cần nhập viện.
- symptom_details: liệt kê đầy đủ MỌI symptom đã đề cập (cả primary + red_flag).
  Mỗi symptom có clarification_questions để chatbot hỏi follow-up.
- hospitalization_criteria: extract từ nội dung "Chỉ định nhập viện" hoặc suy luận
  từ context. comorbid_disease_ids: bệnh nền làm tăng nguy cơ.
- prevention_vi: extract từ section phòng bệnh. Nếu không có → [].
- home_care_summary_vi: chỉ áp dụng cho bệnh có thể tự chăm sóc tại nhà.
- Trích xuất TỪ NỘI DUNG bài viết, KHÔNG bịa thêm thông tin ngoài.
- CHỈ trả về JSON, không code fence, không giải thích."""


def build_user_prompt(doc: dict) -> str:
    sections_text = []
    for s in doc.get("sections", []):
        sections_text.append(f"\n### {s['heading']}\n{s.get('content', '')}")
        for sub in s.get("subsections", []):
            sections_text.append(f"\n#### {sub['heading']}\n{sub.get('content', '')}")
            for subsub in sub.get("subsections", []):
                sections_text.append(f"\n##### {subsub['heading']}\n{subsub.get('content', '')}")

    tables_text = ""
    for t in doc.get("tables", []):
        tables_text += f"\n[{t['id']}] {t.get('title', '')}\n{t.get('description', '')}\n"

    parts = [
        f"BỆNH: {doc['disease']}",
        f"CHƯƠNG: {doc.get('chapter', '')}",
        "\n".join(sections_text),
    ]
    if tables_text.strip():
        parts.append(f"\n=== BẢNG ===\n{tables_text}")

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
    files = sorted(FINAL_DIR.glob("*.json"))
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
    batch_id = submit_batch(jsonl_path, "bachmai_entities")
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

    ok = skipped = bad = 0
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

        if not parsed.get("disease_id"):
            skipped += 1
            print(f"  - {cid}: no ICD-10 → skipped")
            continue

        parsed["disease_slug"] = cid
        (ENTITY_OUT / f"{cid}.json").write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        ok += 1

    print(f"\nĐã lưu {ok} entity, {skipped} skipped (no ICD-10), {bad} parse fail → {ENTITY_OUT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["prepare", "submit", "status", "collect"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    {"prepare": cmd_prepare, "submit": cmd_submit,
     "status": cmd_status, "collect": cmd_collect}[args.cmd](args)


if __name__ == "__main__":
    main()
