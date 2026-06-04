"""
drug_entities.py
----------------
Extract structured drug entities (KG Layer 2) from parsed drug JSONs
using OpenAI Batch API or direct chat completions.

Input:  outputs/otc_drugs/final_json/{slug}.json  (Layer 1 — document)
Output: outputs/entities/drugs/{slug}.json         (Layer 2 — KG entity)

Commands:
    prepare    — build JSONL (1 request / thuốc)
    submit     — upload + tạo batch
    status     — xem status
    collect    — download results → entity JSON per thuốc
    direct     — call chat completions one thuốc at a time

Usage:
    python -m src.processing.drugs.entities prepare
    python -m src.processing.drugs.entities submit
    python -m src.processing.drugs.entities status
    python -m src.processing.drugs.entities collect
    python -m src.processing.drugs.entities direct --limit 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BATCH_MAX_TOKENS, MODEL, OUTPUT_DIR, make_openai_client
from src.processing.batch_api import (
    chat_completion_request,
    fetch_results,
    get_batch,
    submit_batch,
    write_jsonl,
)

INPUT_DIR = OUTPUT_DIR / "otc_drugs" / "final_json"
ENTITY_OUT = OUTPUT_DIR / "entities" / "drugs"
SYMPTOM_DIR = OUTPUT_DIR / "entities" / "symptoms"
WORK_DIR = OUTPUT_DIR / "otc_drugs" / "batch" / "entities"
SYMPTOM_CANDIDATE_LIMIT = 20
_SYMPTOM_CATALOG_CACHE: tuple[Path, list[dict]] | None = None

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
    "common": [
      {"text": "ADR thường gặp", "symptom_id": "symptom:S_<slug>"},
      {"text": "ADR không phải triệu chứng"}
    ],
    "rare_serious": [
      {"text": "ADR hiếm gặp nghiêm trọng", "symptom_id": "symptom:S_<slug>"},
      {"text": "ADR không phải triệu chứng"}
    ]
  },

  "pregnancy_category": "Tóm tắt 1 câu (an toàn / thận trọng / chống chỉ định)",
  "breastfeeding": "Tóm tắt 1 câu"
}

QUY TẮC:
- drug_id: dùng tên hoạt chất tiếng Anh, CamelCase (vd: drug:Paracetamol, drug:Amoxicillin).
- atc_code: lấy mã ATC đầu tiên nếu có nhiều.
- indicated_for.disease_ids: mã ICD-10 cho các bệnh trong section Chỉ định.
- ALLOWED_SYMPTOMS trong user prompt là danh sách symptom_id được phép dùng.
- indicated_for.symptom_ids: CHỈ dùng symptom_id có trong ALLOWED_SYMPTOMS;
  nếu không có ứng viên phù hợp thì để [].
- adr_summary: mỗi ADR là object có text; thêm symptom_id nếu ADR là triệu chứng
  khớp với một symptom_id trong ALLOWED_SYMPTOMS.
- Nếu ADR không phải triệu chứng hoặc không chắc symptom_id, chỉ giữ text.
- Không tự tạo symptom_id ngoài ALLOWED_SYMPTOMS.
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


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.lower().replace("đ", "d")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_exact_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def load_symptom_catalog() -> list[dict]:
    global _SYMPTOM_CATALOG_CACHE

    if _SYMPTOM_CATALOG_CACHE and _SYMPTOM_CATALOG_CACHE[0] == SYMPTOM_DIR:
        return _SYMPTOM_CATALOG_CACHE[1]

    entries = []
    if SYMPTOM_DIR.exists():
        for path in sorted(SYMPTOM_DIR.glob("*.json")):
            if path.name.startswith("_"):
                continue
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            symptom_id = entry.get("symptom_id")
            if isinstance(symptom_id, str) and symptom_id.startswith("symptom:"):
                entries.append(entry)

    _SYMPTOM_CATALOG_CACHE = (SYMPTOM_DIR, entries)
    return entries


def _symptom_match_terms(entry: dict) -> list[str]:
    terms = []
    for key in ("name_vi", "name_en"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            terms.append(value)
    return terms


def _tokens_appear_near(term_tokens: list[str], document_tokens: list[str]) -> bool:
    if len(term_tokens) < 3 or any(len(token) < 4 for token in term_tokens):
        return False

    max_span = len(term_tokens) + 2
    for start, token in enumerate(document_tokens):
        if token != term_tokens[0]:
            continue

        matched = 1
        stop = min(len(document_tokens), start + max_span)
        for doc_token in document_tokens[start + 1 : stop]:
            if doc_token == term_tokens[matched]:
                matched += 1
                if matched == len(term_tokens):
                    return True
    return False


def _single_token_occurs(term_exact: str, document_exact_tokens: list[str]) -> bool:
    for idx, token in enumerate(document_exact_tokens):
        if token != term_exact:
            continue

        next_token = (
            document_exact_tokens[idx + 1]
            if idx + 1 < len(document_exact_tokens)
            else ""
        )
        if term_exact == "phù" and next_token == "hợp":
            continue
        return True
    return False


def _score_symptom_entry(
    entry: dict,
    document_norm: str,
    document_exact: str,
    document_tokens: list[str],
    document_exact_tokens: list[str],
) -> tuple[int, set[str]]:
    best_score = 0
    best_tokens: set[str] = set()
    padded_exact_document = f" {document_exact} "

    for term in _symptom_match_terms(entry):
        term_norm = _normalize_text(term)
        term_exact = _normalize_exact_text(term)
        if not term_norm:
            continue

        tokens = term_norm.split()
        token_set = set(tokens)
        if len(tokens) == 1:
            score = (
                700
                if _single_token_occurs(term_exact, document_exact_tokens)
                else 0
            )
        elif f" {term_exact} " in padded_exact_document:
            score = 1000 + len(term_norm)
        elif _tokens_appear_near(tokens, document_tokens):
            score = 850 + len(tokens) * 10 + len(term_norm)
        else:
            score = 0

        if score > best_score:
            best_score = score
            best_tokens = token_set

    symptom_id = entry.get("symptom_id", "")
    if (
        best_score
        and isinstance(symptom_id, str)
        and symptom_id.startswith("symptom:S_")
    ):
        best_score += 100

    return best_score, best_tokens


def select_symptom_candidates(text: str) -> list[dict]:
    document_norm = _normalize_text(text)
    if not document_norm:
        return []

    document_exact = _normalize_exact_text(text)
    document_tokens = document_norm.split()
    document_exact_tokens = document_exact.split()
    scored = []
    for entry in load_symptom_catalog():
        score, matched_tokens = _score_symptom_entry(
            entry,
            document_norm,
            document_exact,
            document_tokens,
            document_exact_tokens,
        )
        if score:
            scored.append((score, matched_tokens, entry))

    specific_token_sets = [
        matched_tokens for _, matched_tokens, _ in scored if len(matched_tokens) > 1
    ]

    filtered = []
    for score, matched_tokens, entry in scored:
        has_more_specific_match = any(
            matched_tokens < other_tokens for other_tokens in specific_token_sets
        )
        if has_more_specific_match:
            continue
        filtered.append((score, entry))

    by_name: dict[str, tuple[int, dict]] = {}
    for score, entry in filtered:
        key = _normalize_text(entry.get("name_vi") or entry.get("name_en") or "")
        if not key:
            key = entry["symptom_id"]
        current = by_name.get(key)
        if current is None or score > current[0]:
            by_name[key] = (score, entry)

    ranked = sorted(
        by_name.values(),
        key=lambda item: (
            item[0],
            len(
                _normalize_text(
                    item[1].get("name_vi") or item[1].get("name_en") or ""
                )
            ),
            item[1]["symptom_id"],
        ),
        reverse=True,
    )

    candidates = []
    for _, entry in ranked[:SYMPTOM_CANDIDATE_LIMIT]:
        candidate = {
            "symptom_id": entry["symptom_id"],
            "name_vi": entry.get("name_vi", ""),
        }
        name_en = entry.get("name_en")
        if name_en:
            candidate["name_en"] = name_en
        candidates.append(candidate)
    return candidates


def build_user_prompt(doc: dict) -> str:
    sections_text = flatten_sections(doc.get("sections", []))
    source_text = "\n".join(
        [
            doc.get("name", ""),
            doc.get("international_name", ""),
            doc.get("drug_class", ""),
            sections_text,
        ]
    )
    symptom_candidates = select_symptom_candidates(source_text)
    parts = [
        f"THUỐC: {doc['name']}",
        f"TÊN QUỐC TẾ: {doc.get('international_name', '')}",
        f"MÃ ATC: {', '.join(doc.get('atc_codes', []))}",
        f"NHÓM: {doc.get('drug_class', '')}",
        "",
        sections_text,
        "",
        "=== ALLOWED_SYMPTOMS ===",
        json.dumps(symptom_candidates, ensure_ascii=False, indent=2),
    ]
    return "\n".join(parts)


def build_request(slug: str, doc: dict) -> dict:
    return chat_completion_request(
        slug,
        MODEL,
        [{"role": "user", "content": build_user_prompt(doc)}],
        system=SYSTEM_PROMPT,
        max_tokens=BATCH_MAX_TOKENS,
        response_format={"type": "json_object"},
    )


def direct_completion_text(doc: dict) -> str:
    response = make_openai_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(doc)},
        ],
        max_tokens=BATCH_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


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


def cmd_direct(args) -> None:
    files = sorted(INPUT_DIR.glob("*.json"))
    if args.limit:
        files = files[: args.limit]

    ENTITY_OUT.mkdir(parents=True, exist_ok=True)
    ok = bad = 0
    for idx, f in enumerate(files, 1):
        slug = f.stem
        print(f"[{idx}/{len(files)}] {slug}")
        doc = json.loads(f.read_text(encoding="utf-8"))
        text = direct_completion_text(doc)
        parsed = parse_llm_json(text)
        if parsed is None:
            bad += 1
            (ENTITY_OUT / f"{slug}.raw.txt").write_text(text, encoding="utf-8")
            print(f"  ! {slug}: JSON parse fail")
            continue

        parsed["drug_slug"] = slug
        (ENTITY_OUT / f"{slug}.json").write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        ok += 1

    print(f"\nĐã lưu {ok} drug entity ({bad} parse fail) → {ENTITY_OUT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["prepare", "submit", "status", "collect", "direct"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    {"prepare": cmd_prepare, "submit": cmd_submit,
     "status": cmd_status, "collect": cmd_collect,
     "direct": cmd_direct}[args.cmd](args)


if __name__ == "__main__":
    main()
