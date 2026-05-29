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

from src.config import BATCH_MAX_TOKENS, MODEL, OUTPUT_DIR
from src.chat.diagnosis.clarification_options import (
    infer_selection_mode,
    normalize_options,
)
from src.processing.batch_api import (
    chat_completion_request,
    fetch_results,
    get_batch,
    submit_batch,
    write_jsonl,
)

DISEASE_ENTITY_DIR = OUTPUT_DIR / "entities" / "diseases"
DRUG_ENTITY_DIR = OUTPUT_DIR / "entities" / "drugs"
SYMPTOM_OUT = OUTPUT_DIR / "entities" / "symptoms"
WORK_DIR = OUTPUT_DIR / "batch" / "symptom_canon"
OPTION_WORK_DIR = OUTPUT_DIR / "batch" / "symptom_options"

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


QUESTION_SLOTS = ("onset", "severity", "pattern", "associated")


def _question_variants(value) -> list[str]:
    if not value:
        return []
    values = value if isinstance(value, list) else [value]
    questions: list[str] = []
    seen: set[str] = set()
    for raw in values:
        question = str(raw).strip()
        if not question or question in seen:
            continue
        seen.add(question)
        questions.append(question)
    return questions


def _normalize_question_map(raw_questions) -> dict[str, list[str]]:
    if not isinstance(raw_questions, dict):
        return {}
    normalized: dict[str, list[str]] = {}
    for slot in QUESTION_SLOTS:
        questions = _question_variants(raw_questions.get(slot))
        if questions:
            normalized[slot] = questions
    return normalized


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

        question_variants: dict[str, list[str]] = {}
        for d in details_list:
            cq = d.get("clarification_questions", {})
            for slot in QUESTION_SLOTS:
                for question in _question_variants(cq.get(slot) if isinstance(cq, dict) else None):
                    question_variants.setdefault(slot, [])
                    if question not in question_variants[slot]:
                        question_variants[slot].append(question)

        catalog[sid] = {
            "symptom_id": sid,
            "name_vi": name_vi_counter.most_common(1)[0][0] if name_vi_counter else "",
            "name_en": name_en_counter.most_common(1)[0][0] if name_en_counter else "",
            "body_system": body_sys_counter.most_common(1)[0][0] if body_sys_counter else "",
            "type": type_counter.most_common(1)[0][0] if type_counter else "",
            "clarification_questions": question_variants,
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
- symptom_id, name_vi, name_en, body_system, type, clarification_questions.
- clarification_questions là object; mỗi key onset/severity/pattern/associated
  có value là array câu hỏi lấy từ các bệnh có triệu chứng này.

Nhiệm vụ:
1. Kiểm tra và sửa name_vi, name_en cho chính xác (thuật ngữ y khoa chuẩn).
2. Nếu name_vi hoặc name_en trống → điền vào dựa trên symptom_id.
3. Chuẩn hóa body_system thành 1 trong: Tim mạch, Hô hấp, Tiêu hóa, Thần kinh,
   Cơ xương khớp, Nội tiết, Huyết học, Thận - Tiết niệu, Da, Mắt, Tai Mũi Họng,
   Toàn thân, Tâm thần, Sinh dục, Miễn dịch.
4. Với từng key trong clarification_questions:
   - Nếu các câu hỏi giống hoặc gần giống nhau, gộp thành 1 câu ngắn gọn.
   - Nếu các câu hỏi hỏi ý khác nhau về mặt lâm sàng, giữ thành nhiều câu.
   - Không gộp nhiều ý lâm sàng khác nhau vào một câu hỏi nhiều vế.
   - Mỗi value luôn là array string, kể cả khi chỉ còn 1 câu.
   - Tối đa 3 câu mỗi key, ưu tiên câu hỏi dễ trả lời qua chat.
   - Không dùng dấu nháy kép " bên trong nội dung câu hỏi; nếu cần nhấn mạnh
     một cụm từ, dùng dấu nháy đơn hoặc bỏ dấu nháy.
5. Nếu clarification_questions thiếu key onset/severity/pattern/associated
   → bổ sung 1 câu hỏi phù hợp bằng tiếng Việt dưới dạng array 1 phần tử.
6. Nếu type trống → điền "objective" hoặc "subjective".

Trả về JSON array đã chuẩn hóa, giữ nguyên thứ tự. CHỈ trả JSON, không giải thích."""

BATCH_SIZE = 5
OPTION_BATCH_SIZE = 5

OPTION_SYSTEM_PROMPT = """Bạn là chuyên gia y tế thiết kế nút trả lời nhanh cho chatbot y tế.

Input là JSON array các triệu chứng. Mỗi item có:
- symptom_id
- name_vi
- clarification_questions: object có thể gồm onset/severity/pattern/associated.
  Mỗi value là array câu hỏi tiếng Việt.

Nhiệm vụ:
1. Tạo field clarification_options cho từng symptom_id.
2. Tạo field clarification_selection_modes cho từng symptom_id.
3. clarification_options luôn có key "presence" và các key tương ứng với
   clarification_questions đang có.
4. presence là array string. Với onset/severity/pattern/associated:
   - Nếu clarification_questions[slot] có N câu hỏi, clarification_options[slot]
     phải là array có N phần tử.
   - Mỗi phần tử là array string options cho câu hỏi cùng index.
5. clarification_selection_modes có cùng shape:
   - presence luôn là "single".
   - Với onset/severity/pattern/associated, mỗi value là array có N phần tử.
   - Mỗi mode là "single" nếu người dùng chỉ nên chọn 1 option.
   - Mỗi mode là "multi" nếu nhiều option tích cực có thể cùng đúng.
6. Mỗi option array phải có "Không rõ" và "Trả lời luôn" ở cuối, trừ khi đã có sẵn.
7. Options phải ngắn gọn, dễ bấm trên điện thoại.
8. presence nên là lựa chọn xác nhận triệu chứng, ví dụ:
   ["Có sốt", "Không sốt", "Không rõ", "Trả lời luôn"].
9. Nếu câu hỏi hỏi nhiều triệu chứng liên quan, tạo lựa chọn riêng cho từng ý,
   ví dụ "Có kèm theo nôn hoặc vàng da không?" →
   ["Có nôn", "Có vàng da", "Không", "Không rõ", "Trả lời luôn"]
   và mode tương ứng là "multi". Không cần tạo option "Cả hai" khi mode là "multi".
10. "Không", "Không rõ", "Trả lời luôn" là lựa chọn loại trừ, không phải multi.
11. Không đưa chẩn đoán, thuốc, xét nghiệm hoặc lời khuyên điều trị vào options.
12. Không dùng dấu nháy kép " bên trong nội dung option.
13. Không đổi câu hỏi, không đổi symptom_id, không thêm giải thích.

Trả về JSON array đúng thứ tự input. Mỗi item chỉ gồm:
{
  "symptom_id": "...",
  "clarification_options": {
    "presence": [...],
    "onset": [[...], [...]],
    "severity": [[...]],
    "pattern": [[...]],
    "associated": [[...]]
  },
  "clarification_selection_modes": {
    "presence": "single",
    "onset": ["single", "single"],
    "severity": ["single"],
    "pattern": ["single"],
    "associated": ["multi"]
  }
}

CHỈ trả JSON, không markdown, không giải thích."""


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
        requests.append(chat_completion_request(
            batch_id,
            MODEL,
            [{
                "role": "user",
                "content": json.dumps(batch_items, ensure_ascii=False),
            }],
            system=SYSTEM_PROMPT,
            max_tokens=BATCH_MAX_TOKENS,
        ))

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = WORK_DIR / "requests.jsonl"
    write_jsonl(requests, jsonl_path)
    print(f"Prepared {len(requests)} requests ({len(items)} symptoms, "
          f"batch size {BATCH_SIZE}) → {jsonl_path}")


def _load_symptom_entities() -> list[dict]:
    items: list[dict] = []
    for path in sorted(SYMPTOM_OUT.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if item.get("symptom_id"):
            items.append(item)
    return items


def _option_prompt_item(item: dict) -> dict:
    return {
        "symptom_id": item.get("symptom_id", ""),
        "name_vi": item.get("name_vi", ""),
        "clarification_questions": _normalize_question_map(
            item.get("clarification_questions", {}) or {}
        ),
    }


def cmd_prepare_options(args) -> None:
    items = [_option_prompt_item(item) for item in _load_symptom_entities()]
    if not items:
        raise SystemExit(f"Không tìm thấy symptom entities trong {SYMPTOM_OUT}")

    requests = []
    for i in range(0, len(items), OPTION_BATCH_SIZE):
        batch_items = items[i : i + OPTION_BATCH_SIZE]
        batch_id = f"symptom_options_{i:04d}"
        requests.append(chat_completion_request(
            batch_id,
            MODEL,
            [{
                "role": "user",
                "content": json.dumps(batch_items, ensure_ascii=False),
            }],
            system=OPTION_SYSTEM_PROMPT,
            max_tokens=BATCH_MAX_TOKENS,
        ))

    OPTION_WORK_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OPTION_WORK_DIR / "requests.jsonl"
    write_jsonl(requests, jsonl_path)
    print(f"Prepared {len(requests)} option requests ({len(items)} symptoms, "
          f"batch size {OPTION_BATCH_SIZE}) → {jsonl_path}")


def cmd_submit(args) -> None:
    jsonl_path = WORK_DIR / "requests.jsonl"
    if not jsonl_path.exists():
        raise SystemExit("Chưa có requests.jsonl. Chạy `prepare` trước.")
    batch_id = submit_batch(jsonl_path, "symptom_canon")
    (WORK_DIR / "batch_id.txt").write_text(batch_id, encoding="utf-8")
    print(f"Đã lưu batch_id → {WORK_DIR / 'batch_id.txt'}")


def cmd_submit_options(args) -> None:
    jsonl_path = OPTION_WORK_DIR / "requests.jsonl"
    if not jsonl_path.exists():
        raise SystemExit("Chưa có option requests.jsonl. Chạy `prepare_options` trước.")
    batch_id = submit_batch(jsonl_path, "symptom_options")
    (OPTION_WORK_DIR / "batch_id.txt").write_text(batch_id, encoding="utf-8")
    print(f"Đã lưu batch_id → {OPTION_WORK_DIR / 'batch_id.txt'}")


def cmd_status(args) -> None:
    batch_id = (WORK_DIR / "batch_id.txt").read_text(encoding="utf-8").strip()
    print(json.dumps(get_batch(batch_id), ensure_ascii=False, indent=2))


def cmd_options_status(args) -> None:
    batch_id = (OPTION_WORK_DIR / "batch_id.txt").read_text(encoding="utf-8").strip()
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
    expected_ids: set[str] = set()
    catalog_path = WORK_DIR / "symptom_catalog_raw.json"
    if catalog_path.exists():
        raw_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        expected_ids = set(raw_catalog.keys())

    ok = bad = 0
    saved_ids: set[str] = set()
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
            item["clarification_questions"] = _normalize_question_map(
                item.get("clarification_questions", {}) or {}
            )
            slug = sid.replace("symptom:", "").strip()
            if not slug:
                continue
            out_path = SYMPTOM_OUT / f"{slug}.json"
            out_path.write_text(
                json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            saved_ids.add(sid)
            ok += 1

    alias_src = WORK_DIR / "alias_map.json"
    if alias_src.exists():
        (SYMPTOM_OUT / "_alias_map.json").write_text(
            alias_src.read_text(encoding="utf-8"), encoding="utf-8"
        )

    print(f"\nĐã lưu {ok} symptom entities ({bad} batch parse fail) → {SYMPTOM_OUT}")
    if expected_ids:
        missing_ids = sorted(expected_ids - saved_ids)
        if missing_ids:
            preview = ", ".join(missing_ids[:10])
            raise SystemExit(
                f"Thiếu {len(missing_ids)} symptom từ kết quả LLM: {preview}"
            )


def _symptom_path(symptom_id: str) -> Path:
    slug = symptom_id.replace("symptom:", "").strip()
    return SYMPTOM_OUT / f"{slug}.json"


def _normalized_option_map(
    raw_options: dict,
    questions: dict[str, list[str]],
) -> dict[str, list]:
    allowed_slots = {"presence", *questions.keys()}
    normalized: dict[str, list] = {}
    for slot, options in raw_options.items():
        if slot not in allowed_slots:
            continue
        if slot == "presence":
            labels = normalize_options(options)
            if labels:
                normalized[slot] = list(labels)
            continue

        question_count = len(questions.get(slot, []))
        if question_count == 0:
            continue
        option_groups = options if isinstance(options, list) else []
        if question_count == 1:
            groups = [option_groups] if option_groups else []
        elif option_groups and all(isinstance(group, list) for group in option_groups):
            groups = option_groups[:question_count]
        else:
            groups = [option_groups] if option_groups else []
        normalized_groups: list[list[str]] = []
        for group in groups:
            labels = normalize_options(group)
            if labels:
                normalized_groups.append(list(labels))
        if normalized_groups:
            normalized[slot] = normalized_groups
    return normalized


def _normalize_selection_mode(raw_mode, default: str) -> str:
    mode = str(raw_mode or "").strip().casefold()
    return mode if mode in {"single", "multi"} else default


def _raw_slot_mode(raw_modes: dict, slot: str, index: int):
    value = raw_modes.get(slot)
    if isinstance(value, list):
        if 0 <= index < len(value):
            return value[index]
        return None
    return value


def _normalized_selection_modes(
    raw_modes: dict,
    questions: dict[str, list[str]],
    options: dict[str, list],
) -> dict[str, list | str]:
    modes: dict[str, list | str] = {"presence": "single"}
    raw_modes = raw_modes if isinstance(raw_modes, dict) else {}
    for slot, slot_questions in questions.items():
        groups = options.get(slot, [])
        slot_modes: list[str] = []
        for index, question in enumerate(slot_questions):
            group_options: tuple[str, ...] = ()
            if isinstance(groups, list) and index < len(groups) and isinstance(groups[index], list):
                group_options = tuple(str(option) for option in groups[index])
            default = infer_selection_mode(slot, group_options, str(question))
            mode = _normalize_selection_mode(_raw_slot_mode(raw_modes, slot, index), default)
            slot_modes.append(default if default == "multi" else mode)
        if slot_modes:
            modes[slot] = slot_modes
    return modes


def _merge_option_payload(payload: dict) -> bool:
    symptom_id = payload.get("symptom_id", "")
    if not symptom_id:
        return False
    path = _symptom_path(symptom_id)
    if not path.exists():
        return False
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    questions = _normalize_question_map(current.get("clarification_questions", {}) or {})
    current["clarification_questions"] = questions
    raw_options = payload.get("clarification_options", {})
    if not isinstance(raw_options, dict):
        return False
    options = _normalized_option_map(raw_options, questions)
    if not options.get("presence"):
        return False
    current["clarification_options"] = options
    current["clarification_selection_modes"] = _normalized_selection_modes(
        payload.get("clarification_selection_modes", {}),
        questions,
        options,
    )
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def cmd_collect_options(args) -> None:
    batch_id = (OPTION_WORK_DIR / "batch_id.txt").read_text(encoding="utf-8").strip()
    results = fetch_results(batch_id)
    print(f"Fetched {len(results)} option results")

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
            if _merge_option_payload(item):
                ok += 1
            else:
                bad += 1
                print(f"  ! {cid}: option merge fail")

    print(f"\nĐã cập nhật clarification_options cho {ok} symptom entities ({bad} lỗi) → {SYMPTOM_OUT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=[
        "collect", "prepare", "submit", "status", "collect_results",
        "prepare_options", "submit_options", "options_status", "collect_options",
    ])
    args = parser.parse_args()

    {
        "collect": cmd_collect,
        "prepare": cmd_prepare,
        "submit": cmd_submit,
        "status": cmd_status,
        "collect_results": cmd_collect_results,
        "prepare_options": cmd_prepare_options,
        "submit_options": cmd_submit_options,
        "options_status": cmd_options_status,
        "collect_options": cmd_collect_options,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
