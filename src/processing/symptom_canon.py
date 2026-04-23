"""
symptom_canon.py
----------------
Canonicalize symptoms from disease (and drug) entities into a unified
symptom catalog.

Steps:
  1. collect  — gather all symptom_details from disease entities,
                group by symptom_id, detect near-duplicates
  2. prepare  — build batch JSONL to canonicalize via LLM
  3. submit   — upload + create batch
  4. status   — check batch status
  5. collect_results — download results → canonical symptom JSONs

Usage:
    python -m src.processing.symptom_canon collect
    python -m src.processing.symptom_canon prepare
    python -m src.processing.symptom_canon submit
    python -m src.processing.symptom_canon status
    python -m src.processing.symptom_canon collect_results
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL, OUTPUT_DIR
from src.processing.batch_api import (
    fetch_results,
    get_batch,
    submit_batch,
    write_jsonl,
)

DISEASE_ENTITY_DIR = OUTPUT_DIR / "entities" / "diseases"
DRUG_ENTITY_DIR = OUTPUT_DIR / "entities" / "drugs"
SYMPTOM_OUT = OUTPUT_DIR / "entities" / "symptoms"
WORK_DIR = OUTPUT_DIR / "batch" / "symptom_canon"

TRIVIAL_MERGES = {
    "symptom:S_seizure": "symptom:S_seizures",
    "symptom:S_diaphore sis": "symptom:S_diaphoresis",
    "symptom:S_diaphoretic": "symptom:S_diaphoresis",
    "symptom:S_shortness_of_breath": "symptom:S_dyspnea",
    "symptom:S_itching": "symptom:S_pruritus",
    "symptom:S_joint_pain": "symptom:S_arthralgia",
    "symptom:S_runny_nose": "symptom:S_rhinorrhea",
    "symptom:S_skin_rash": "symptom:S_rash",
    "symptom:S_nausea_vomiting": "symptom:S_nausea_and_vomiting",
    "symptom:S_leg_swelling": "symptom:S_leg_edema",
    "symptom:S_local_warmth": "symptom:S_local_heat",
    "symptom:S_multiorgan_failure": "symptom:S_multi_organ_failure",
    "symptom:S_cardiac_arrhythmia": "symptom:S_arrhythmia",
    "symptom:S_muscle_fasciculations": "symptom:S_fasciculations",
    "symptom:S_psychotic_symptoms": "symptom:S_psychosis",
    "symptom:S_aggressive_behavior": "symptom:S_aggression",
    "symptom:S_nuchal_rigidity": "symptom:S_neck_stiffness",
    "symptom:S_finger_clubbing": "symptom:S_clubbing",
    "symptom:S_elevated_jvp": "symptom:S_elevated_jugular_venous_pressure",
    "symptom:S_reduced_glomerular_filtration_rate": "symptom:S_reduced_gfr",
    "symptom:S_elevated_serum_creatinine": "symptom:S_elevated_creatinine",
    "symptom:S_respiratory_weakness": "symptom:S_respiratory_muscle_weakness",
    "symptom:S_congestive_heart_failure": "symptom:S_heart_failure",
    "symptom:S_profuse_sweating": "symptom:S_sweating",
    "symptom:S_spinal_stiffness": "symptom:S_stiffness",
    "symptom:S_reduced_spinal_mobility": "symptom:S_reduced_mobility",
    "symptom:S_rigidity": "symptom:S_muscle_rigidity",
}


def collect_symptoms() -> dict[str, dict]:
    """Collect all symptom_details from disease entities, merge by canonical ID."""
    raw: dict[str, list[dict]] = defaultdict(list)

    for f in sorted(DISEASE_ENTITY_DIR.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for sd in d.get("symptom_details", []):
            sid = sd.get("symptom_id", "")
            if not sid:
                continue
            canonical = TRIVIAL_MERGES.get(sid, sid)
            raw[canonical].append(sd)

    if DRUG_ENTITY_DIR.exists():
        for f in sorted(DRUG_ENTITY_DIR.glob("*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            for sid in d.get("indicated_for", {}).get("symptom_ids", []):
                if sid and sid not in raw:
                    raw[TRIVIAL_MERGES.get(sid, sid)] = []

    catalog: dict[str, dict] = {}
    for sid, details_list in sorted(raw.items()):
        if not details_list:
            catalog[sid] = {
                "symptom_id": sid,
                "name_vi": "",
                "name_en": "",
                "body_system": "",
                "type": "",
                "clarification_questions": {},
                "source_count": 0,
            }
            continue

        name_vi_counter = Counter(d.get("name_vi", "") for d in details_list if d.get("name_vi"))
        name_en_counter = Counter(d.get("name_en", "") for d in details_list if d.get("name_en"))
        body_sys_counter = Counter(d.get("body_system", "") for d in details_list if d.get("body_system"))
        type_counter = Counter(d.get("type", "") for d in details_list if d.get("type"))

        best_q = {}
        for d in details_list:
            cq = d.get("clarification_questions", {})
            for k in ("onset", "severity", "pattern", "associated"):
                if cq.get(k) and k not in best_q:
                    best_q[k] = cq[k]

        catalog[sid] = {
            "symptom_id": sid,
            "name_vi": name_vi_counter.most_common(1)[0][0] if name_vi_counter else "",
            "name_en": name_en_counter.most_common(1)[0][0] if name_en_counter else "",
            "body_system": body_sys_counter.most_common(1)[0][0] if body_sys_counter else "",
            "type": type_counter.most_common(1)[0][0] if type_counter else "",
            "clarification_questions": best_q,
            "source_count": len(details_list),
        }

    return catalog


def cmd_collect(args) -> None:
    catalog = collect_symptoms()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    out = WORK_DIR / "symptom_catalog_raw.json"
    out.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    alias_out = WORK_DIR / "alias_map.json"
    alias_out.write_text(json.dumps(TRIVIAL_MERGES, ensure_ascii=False, indent=2), encoding="utf-8")

    has_details = sum(1 for v in catalog.values() if v["source_count"] > 0)
    no_details = sum(1 for v in catalog.values() if v["source_count"] == 0)
    print(f"Collected {len(catalog)} canonical symptoms "
          f"({has_details} with details, {no_details} ID-only)")
    print(f"Alias map: {len(TRIVIAL_MERGES)} trivial merges")
    print(f"Saved → {out}")


SYSTEM_PROMPT = """Bạn là chuyên gia y tế. Nhiệm vụ: chuẩn hóa danh mục triệu chứng y khoa.

Nhận input là danh sách triệu chứng (JSON array), mỗi item có:
- symptom_id, name_vi, name_en, body_system, type, clarification_questions

Nhiệm vụ:
1. Kiểm tra và sửa name_vi, name_en cho chính xác (thuật ngữ y khoa chuẩn).
2. Nếu name_vi hoặc name_en trống → điền vào dựa trên symptom_id.
3. Chuẩn hóa body_system thành 1 trong: Tim mạch, Hô hấp, Tiêu hóa, Thần kinh,
   Cơ xương khớp, Nội tiết, Huyết học, Thận - Tiết niệu, Da, Mắt, Tai Mũi Họng,
   Toàn thân, Tâm thần, Sinh dục, Miễn dịch.
4. Nếu clarification_questions thiếu bất kỳ key nào (onset/severity/pattern/associated)
   → bổ sung câu hỏi phù hợp bằng tiếng Việt.
5. Nếu type trống → điền "objective" hoặc "subjective".

Trả về JSON array đã chuẩn hóa, giữ nguyên thứ tự. CHỈ trả JSON, không giải thích."""

BATCH_SIZE = 50


def cmd_prepare(args) -> None:
    catalog_path = WORK_DIR / "symptom_catalog_raw.json"
    if not catalog_path.exists():
        raise SystemExit("Chưa có symptom_catalog_raw.json. Chạy `collect` trước.")

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    items = list(catalog.values())

    requests = []
    for i in range(0, len(items), BATCH_SIZE):
        batch_items = items[i : i + BATCH_SIZE]
        batch_id = f"symptoms_{i:04d}"
        requests.append({
            "custom_id": batch_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(batch_items, ensure_ascii=False)},
                ],
                "response_format": {"type": "json_object"},
            },
        })

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = WORK_DIR / "requests.jsonl"
    write_jsonl(requests, jsonl_path)
    print(f"Prepared {len(requests)} requests ({len(items)} symptoms, "
          f"batch size {BATCH_SIZE}) → {jsonl_path}")


def cmd_submit(args) -> None:
    jsonl_path = WORK_DIR / "requests.jsonl"
    if not jsonl_path.exists():
        raise SystemExit("Chưa có requests.jsonl. Chạy `prepare` trước.")
    batch_id = submit_batch(jsonl_path, "symptom_canon")
    (WORK_DIR / "batch_id.txt").write_text(batch_id, encoding="utf-8")
    print(f"Đã lưu batch_id → {WORK_DIR / 'batch_id.txt'}")


def cmd_status(args) -> None:
    batch_id = (WORK_DIR / "batch_id.txt").read_text(encoding="utf-8").strip()
    print(json.dumps(get_batch(batch_id), ensure_ascii=False, indent=2))


JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_llm_json(text: str):
    text = JSON_FENCE_RE.sub("", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return None


def cmd_collect_results(args) -> None:
    batch_id = (WORK_DIR / "batch_id.txt").read_text(encoding="utf-8").strip()
    results = fetch_results(batch_id)
    print(f"Fetched {len(results)} results")

    SYMPTOM_OUT.mkdir(parents=True, exist_ok=True)

    ok = bad = 0
    for r in results:
        cid = r.get("custom_id", "")
        try:
            text = r["response"]["body"]["choices"][0]["message"]["content"]
        except Exception:
            text = ""

        parsed = parse_llm_json(text)
        if parsed is None:
            bad += 1
            print(f"  ! {cid}: JSON parse fail")
            continue

        if isinstance(parsed, dict) and "symptoms" in parsed:
            parsed = parsed["symptoms"]
        if not isinstance(parsed, list):
            bad += 1
            print(f"  ! {cid}: expected array, got {type(parsed).__name__}")
            continue

        for item in parsed:
            sid = item.get("symptom_id", "")
            if not sid:
                continue
            slug = sid.replace("symptom:", "").strip()
            if not slug:
                continue
            out_path = SYMPTOM_OUT / f"{slug}.json"
            out_path.write_text(
                json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            ok += 1

    alias_src = WORK_DIR / "alias_map.json"
    if alias_src.exists():
        (SYMPTOM_OUT / "_alias_map.json").write_text(
            alias_src.read_text(encoding="utf-8"), encoding="utf-8"
        )

    print(f"\nĐã lưu {ok} symptom entities ({bad} batch parse fail) → {SYMPTOM_OUT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=[
        "collect", "prepare", "submit", "status", "collect_results",
    ])
    args = parser.parse_args()

    {
        "collect": cmd_collect,
        "prepare": cmd_prepare,
        "submit": cmd_submit,
        "status": cmd_status,
        "collect_results": cmd_collect_results,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
