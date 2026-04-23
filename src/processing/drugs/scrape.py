"""
Scrape detailed drug information from trungtamthuoc.com for all OTC drugs
listed in documents/otc_whitelist.json.

Pipeline:
    Step 1 — build_index:  Crawl A–Z + 09 pages → drug_slug_index.json
    Step 2 — map:          Algorithmic + LLM matching → otc_slug_mapping.json
    Step 3 — scrape:       Fetch each matched slug → outputs/otc_drugs/{slug}/

Usage:
    python scrape_otc_drugs.py build_index
    python scrape_otc_drugs.py map
    python scrape_otc_drugs.py scrape
    python scrape_otc_drugs.py all

Output structure per slug:
    outputs/otc_drugs/raw_data/{slug}/
        raw_text.txt              – drug text + OTC constraints at the end
        tables/bang_01.png        – screenshot of HTML table
        visual_descriptions.txt   – table metadata for LLM interpretation
"""

import json
import re
import sys
import time
import logging
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WHITELIST_PATH = PROJECT_ROOT / "documents" / "otc_whitelist.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "otc_drugs" / "raw_data"
INDEX_PATH = OUTPUT_DIR / "drug_slug_index.json"
MAPPING_PATH = OUTPUT_DIR / "otc_slug_mapping.json"
REPORT_PATH = OUTPUT_DIR / "scrape_report.json"

BASE_URL = "https://trungtamthuoc.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
KEYS = list("abcdefghijklmnopqrstuvwxyz") + ["09"]

SKIP_HEADINGS = {
    "các sản phẩm có chứa hoạt chất",
    "so sánh sản phẩm cùng hoạt chất",
    "tài liệu tham khảo",
}
PREAMBLE_PATTERNS = [
    r"Bài viết biên soạn(?:\s+dựa)?\s+theo.*?(?=Tên chung quốc tế|Mã ATC|Loại thuốc|1\s)",
]
INLINE_TAGS = {"a", "span", "b", "i", "em", "strong", "u", "sub", "sup", "abbr", "mark", "small"}

TABLE_HTML_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; background: white; }}
table {{ border-collapse: collapse; width: auto; }}
th, td {{ border: 1px solid #333; padding: 8px 12px; text-align: left; font-size: 14px; }}
th {{ background: #f0f0f0; font-weight: bold; }}
tr:nth-child(even) {{ background: #fafafa; }}
</style></head><body>{table}</body></html>"""

DELAY_BETWEEN_REQUESTS = 1.0
FUZZY_THRESHOLD = 0.82
LLM_BATCH_SIZE = 15  # unmatched entries per LLM call

# ---------------------------------------------------------------------------
#  Step 1 — Build slug index from A–Z pages
# ---------------------------------------------------------------------------


def build_index():
    """Crawl all A–Z + 09 pages and save {slug: name} to drug_slug_index.json."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index: dict[str, str] = {}

    log.info("Building slug index from %d pages...", len(KEYS))

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        for key in KEYS:
            url = f"{BASE_URL}/hoat-chat?key={key}"
            log.info("  Crawling key=%s ...", key)
            try:
                page.goto(url, wait_until="networkidle", timeout=15000)
                page.wait_for_selector('a[href*="/hoat-chat/"]', timeout=10000)
            except Exception:
                log.warning("    Timeout or error for key=%s, skipping", key)
                continue

            links = page.query_selector_all('a[href*="/hoat-chat/"]')
            count = 0
            for link in links:
                href = link.get_attribute("href") or ""
                text = link.inner_text().strip()
                m = re.match(r"^/hoat-chat/([\w-]+)$", href)
                if m and text:
                    slug = m.group(1)
                    index[slug] = text
                    count += 1

            log.info("    Found %d drugs for key=%s", count, key)
            time.sleep(0.5)

        browser.close()

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    log.info("Index saved: %d drugs → %s", len(index), INDEX_PATH)


# ---------------------------------------------------------------------------
#  Step 2 — Map whitelist ↔ slug index (algorithmic + LLM)
# ---------------------------------------------------------------------------

def normalize_for_match(name: str) -> str:
    """Lowercase, strip accents, collapse whitespace, remove punctuation."""
    name = name.lower().strip()
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    ascii_name = re.sub(r"[^\w\s-]", "", ascii_name)
    ascii_name = re.sub(r"\s+", " ", ascii_name).strip()
    return ascii_name


def _extract_alt_names(display_name: str) -> list[str]:
    """Extract alternate names from parenthetical text."""
    names = [normalize_for_match(display_name.split("(")[0])]
    for m in re.finditer(r"\(([^)]+)\)", display_name):
        for part in re.split(r"[;,/]", m.group(1)):
            n = normalize_for_match(part)
            if n and len(n) > 2:
                names.append(n)
    return names


def _algorithmic_match(
    whitelist: list[dict],
    index: dict[str, str],
) -> tuple[dict[int, list[str]], list[dict]]:
    """
    Returns:
        matched: {stt: [slug, ...]}
        unmatched: [whitelist entries that need LLM]
    """
    # Build multi-name lookup
    norm_index: dict[str, tuple[str, str]] = {}
    for slug, display_name in index.items():
        for alt in _extract_alt_names(display_name):
            if alt not in norm_index:
                norm_index[alt] = (slug, display_name)
        slug_norm = normalize_for_match(slug.replace("-", " "))
        if slug_norm not in norm_index:
            norm_index[slug_norm] = (slug, display_name)

    matched: dict[int, list[str]] = {}
    unmatched: list[dict] = []

    for drug in whitelist:
        stt = drug["stt"]
        queries = list(dict.fromkeys([
            normalize_for_match(drug["ten_normalized"]),
            normalize_for_match(drug["ten_ngan"]),
            normalize_for_match(drug["ten_hoat_chat"].split("(")[0]),
        ]))

        # Exact match
        found = False
        for q in queries:
            if q in norm_index:
                slug = norm_index[q][0]
                matched[stt] = [slug]
                found = True
                break

        if found:
            continue

        # Fuzzy
        best_score = 0.0
        best_slug = None
        for norm_name, (slug, _) in norm_index.items():
            for q in queries:
                score = SequenceMatcher(None, q, norm_name).ratio()
                if score > best_score:
                    best_score = score
                    best_slug = slug

        if best_score >= FUZZY_THRESHOLD and best_slug:
            matched[stt] = [best_slug]
        else:
            unmatched.append(drug)

    return matched, unmatched


def _llm_match_batch(
    entries: list[dict],
    slug_list_text: str,
) -> dict[int, list[str]]:
    """Use LLM to match a batch of unmatched entries to slugs (1-to-many)."""
    import sys as _sys
    if str(PROJECT_ROOT) not in _sys.path:
        _sys.path.insert(0, str(PROJECT_ROOT))

    from xai_sdk.chat import system, user, text
    from src.config import make_xai_client, MODEL

    client = make_xai_client()

    entries_text = "\n".join(
        f'  {e["stt"]}. "{e["ten_hoat_chat"]}" (tên ngắn: "{e["ten_ngan"]}")'
        for e in entries
    )

    prompt = f"""Bạn là chuyên gia dược lý. Dưới đây là danh sách các hoạt chất OTC cần tìm trang web tương ứng, và danh sách tất cả slug (đường dẫn) trên trungtamthuoc.com.

## Hoạt chất cần match:
{entries_text}

## Danh sách slug trên web (slug: tên hiển thị):
{slug_list_text}

## Yêu cầu:
- Với mỗi hoạt chất (theo STT), tìm TẤT CẢ slug phù hợp. Một mục OTC có thể ứng với nhiều slug.
  Ví dụ: "Vitamin nhóm B" → vitamin-b1, vitamin-b2, vitamin-b6, vitamin-b12, ...
  Ví dụ: "Kẽm oxid, Kẽm pyrithion, Kẽm Gluconat" → kem-oxid, kem-gluconat, ...
- Nếu không tìm thấy slug nào phù hợp, trả về mảng rỗng.
- CHỈ trả về JSON hợp lệ, không giải thích.

## Format output:
{{"results": {{"<stt>": ["slug1", "slug2", ...], ...}}}}"""

    chat = client.chat.create(
        model=MODEL,
        messages=[
            system("Bạn là chuyên gia dược lý Việt Nam. Chỉ trả về JSON hợp lệ, không markdown fence, không giải thích."),
            user(text(prompt)),
        ],
        max_tokens=4096,
        temperature=0.0,
    )
    raw = chat.sample().content.strip()

    # Parse JSON — handle possible markdown fences
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.error("LLM returned invalid JSON:\n%s", raw[:500])
        return {}

    result: dict[int, list[str]] = {}
    for stt_str, slugs in data.get("results", {}).items():
        stt = int(stt_str)
        if isinstance(slugs, list):
            result[stt] = [s for s in slugs if isinstance(s, str)]
    return result


def map_whitelist():
    """Match whitelist ↔ slug index. Save otc_slug_mapping.json."""
    with open(WHITELIST_PATH, encoding="utf-8") as f:
        whitelist = json.load(f)
    with open(INDEX_PATH, encoding="utf-8") as f:
        index = json.load(f)

    valid_slugs = set(index.keys())

    # Algorithmic matching
    algo_matched, unmatched = _algorithmic_match(whitelist, index)
    log.info("Algorithmic: %d matched, %d unmatched", len(algo_matched), len(unmatched))

    # LLM matching for unmatched (in batches)
    llm_matched: dict[int, list[str]] = {}
    if unmatched:
        slug_list_text = "\n".join(f"  {slug}: {name}" for slug, name in index.items())

        for batch_start in range(0, len(unmatched), LLM_BATCH_SIZE):
            batch = unmatched[batch_start:batch_start + LLM_BATCH_SIZE]
            log.info("LLM matching batch %d–%d / %d ...",
                     batch_start + 1, batch_start + len(batch), len(unmatched))
            try:
                batch_result = _llm_match_batch(batch, slug_list_text)
                # Validate slugs exist in index
                for stt, slugs in batch_result.items():
                    valid = [s for s in slugs if s in valid_slugs]
                    if valid:
                        llm_matched[stt] = valid
                        log.info("  [%d] → %d slugs", stt, len(valid))
            except Exception as e:
                log.error("LLM batch failed: %s", e)
            time.sleep(2)

    log.info("LLM: %d additional matches", len(llm_matched))

    # Merge results: build mapping entries
    all_matched = {**algo_matched, **llm_matched}

    mapping: list[dict] = []
    for drug in whitelist:
        stt = drug["stt"]
        slugs = all_matched.get(stt, [])
        mapping.append({
            "stt": stt,
            "ten_hoat_chat": drug["ten_hoat_chat"],
            "ten_ngan": drug["ten_ngan"],
            "ten_normalized": drug["ten_normalized"],
            "duong_dung_dang_bao_che": drug["duong_dung_dang_bao_che"],
            "dieu_kien_cu_the": drug.get("dieu_kien_cu_the", ""),
            "nhom": drug["nhom"],
            "source": drug["source"],
            "slugs": slugs,
            "match_type": "algo" if stt in algo_matched else ("llm" if stt in llm_matched else "none"),
        })

    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    total_matched = sum(1 for m in mapping if m["slugs"])
    total_unmatched = sum(1 for m in mapping if not m["slugs"])
    total_slugs = len(set(s for m in mapping for s in m["slugs"]))
    log.info("Mapping saved: %d entries matched, %d unmatched, %d unique slugs → %s",
             total_matched, total_unmatched, total_slugs, MAPPING_PATH)

    if total_unmatched:
        log.info("Unmatched drugs:")
        for m in mapping:
            if not m["slugs"]:
                log.info("  - [%d] %s", m["stt"], m["ten_ngan"])


# ---------------------------------------------------------------------------
#  Step 3 — Scrape
# ---------------------------------------------------------------------------

def screenshot_table(page, table_html: str, out_path: Path):
    html = TABLE_HTML_TEMPLATE.format(table=table_html)
    page.set_content(html)
    table_el = page.query_selector("table")
    if table_el:
        table_el.screenshot(path=str(out_path))


def extract_drug_content(soup: BeautifulSoup) -> tuple[str, list[str]]:
    """Returns (raw_text_with_placeholders, list_of_table_html)."""
    content_div = soup.find("div", class_="cs-content")
    if not content_div:
        content_div = soup.find("article")
    if not content_div:
        return "", []

    for tag in content_div.find_all(["script", "style", "iframe", "img"]):
        tag.decompose()

    for heading in content_div.find_all(["h2", "h3"]):
        heading_text = heading.get_text(strip=True).lower()
        if any(skip in heading_text for skip in SKIP_HEADINGS):
            for sibling in list(heading.find_next_siblings()):
                sibling.decompose()
            heading.decompose()
            break

    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)

    tables_html = []
    for table in content_div.find_all("table"):
        html_str = str(table)
        if html_str:
            tables_html.append(html_str)
            idx = len(tables_html)
            placeholder = soup.new_tag("p")
            placeholder.string = f"[BẢNG {idx}]"
            table.replace_with(placeholder)
        else:
            table.decompose()

    for tag in content_div.find_all(list(INLINE_TAGS)):
        tag.unwrap()
    content_div.smooth()

    for h in content_div.find_all(["h2", "h3", "h4"]):
        text = h.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        h.clear()
        h.string = text

    text = content_div.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)

    if title:
        text = f"{title}\n\n{text}"

    for pattern in PREAMBLE_PATTERNS:
        text = re.sub(pattern, "", text, count=1, flags=re.DOTALL)

    return text.strip(), tables_html


def build_otc_constraints(entries: list[dict]) -> str:
    """Build the OTC constraints section from matching whitelist entries."""
    lines = [
        "",
        "=" * 60,
        "THÔNG TIN OTC (Thuốc không kê đơn — TT 07/2017/TT-BYT)",
        "=" * 60,
    ]
    for entry in entries:
        lines.append("")
        lines.append(f"Tên hoạt chất: {entry['ten_hoat_chat']}")
        if entry["duong_dung_dang_bao_che"]:
            lines.append(f"Đường dùng, dạng bào chế: {entry['duong_dung_dang_bao_che']}")
        if entry.get("dieu_kien_cu_the"):
            lines.append(f"Quy định cụ thể: {entry['dieu_kien_cu_the']}")
        lines.append(f"Nhóm: {entry['nhom']}")
        lines.append(f"Nguồn: {entry['source']}")
    return "\n".join(lines)


def scrape_drug(slug: str) -> tuple[str, list[str], str] | None:
    """Fetch and extract drug content by slug."""
    url = f"{BASE_URL}/hoat-chat/{slug}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
    except requests.RequestException as e:
        log.error("Request failed for %s: %s", slug, e)
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    text, tables_html = extract_drug_content(soup)
    return text, tables_html, resp.url


def write_drug_output(folder: Path, name: str, raw_text: str, tables_html: list[str], page):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "raw_text.txt").write_text(raw_text, encoding="utf-8")

    if tables_html:
        tables_dir = folder / "tables"
        tables_dir.mkdir(exist_ok=True)
        for i, html in enumerate(tables_html, 1):
            try:
                screenshot_table(page, html, tables_dir / f"bang_{i:02d}.png")
            except Exception as e:
                log.warning("  Failed to screenshot table %d: %s", i, e)

    desc_lines = [f"# Visual Descriptions — {name}\n"]
    desc_lines.append("# Hướng dẫn: Điền mô tả text cho từng bảng bên dưới.")
    desc_lines.append("# Giữ nguyên ID và format. Viết mô tả sau dòng TEXT:\n")
    if tables_html:
        for i in range(1, len(tables_html) + 1):
            desc_lines.append(f"## BẢNG {i}")
            desc_lines.append("- TYPE: table")
            desc_lines.append(f"- IMAGE: tables/bang_{i:02d}.png")
            desc_lines.append("TEXT:\n\n")
    else:
        desc_lines.append("(Không có bảng)\n")
    (folder / "visual_descriptions.txt").write_text("\n".join(desc_lines), encoding="utf-8")


def scrape_all():
    """Read mapping, build slug→[otc_entries], scrape each unique slug."""
    with open(MAPPING_PATH, encoding="utf-8") as f:
        mapping = json.load(f)
    with open(INDEX_PATH, encoding="utf-8") as f:
        index = json.load(f)

    # Build slug → [otc entries] (reverse mapping)
    slug_to_entries: dict[str, list[dict]] = {}
    for entry in mapping:
        for slug in entry.get("slugs", []):
            slug_to_entries.setdefault(slug, []).append(entry)

    unique_slugs = sorted(slug_to_entries.keys())
    total = len(unique_slugs)
    log.info("Scraping %d unique slugs (from %d whitelist entries)",
             total, sum(1 for m in mapping if m["slugs"]))

    success_list = []
    fail_list = []
    skip_count = 0

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        for i, slug in enumerate(unique_slugs, 1):
            folder_name = slug  # folder = slug
            drug_folder = OUTPUT_DIR / folder_name
            web_name = index.get(slug, slug)

            raw_path = drug_folder / "raw_text.txt"
            if raw_path.exists() and raw_path.stat().st_size > 200:
                log.info("[%d/%d] SKIP (exists): %s", i, total, slug)
                success_list.append({"slug": slug, "status": "skipped"})
                skip_count += 1
                continue

            log.info("[%d/%d] Scraping: %s (%s)", i, total, slug, web_name)
            result = scrape_drug(slug)

            if result and result[0] and len(result[0]) >= 100:
                text, tables_html, source_url = result

                # Build header
                header = f"Nguồn: {source_url}\n{'=' * 60}\n\n"

                # Build OTC constraints footer
                otc_entries = slug_to_entries.get(slug, [])
                constraints = build_otc_constraints(otc_entries) if otc_entries else ""

                full_text = header + text + "\n" + constraints
                write_drug_output(drug_folder, web_name, full_text, tables_html, page)

                log.info("  → %d chars, %d tables, %d OTC entries → %s/",
                         len(full_text), len(tables_html), len(otc_entries), folder_name)
                success_list.append({
                    "slug": slug, "web_name": web_name, "status": "ok",
                    "chars": len(full_text), "tables": len(tables_html),
                    "otc_entries": len(otc_entries),
                })
            else:
                log.warning("  → FAILED (no content): %s", slug)
                fail_list.append({"slug": slug, "web_name": web_name, "reason": "no_content"})

            if i < total:
                time.sleep(DELAY_BETWEEN_REQUESTS)

        browser.close()

    # Add unmatched whitelist entries to fail list
    for m in mapping:
        if not m["slugs"]:
            fail_list.append({
                "slug": None,
                "ten_ngan": m["ten_ngan"],
                "stt": m["stt"],
                "reason": "no_match",
            })

    report = {
        "total_whitelist": len(mapping),
        "unique_slugs": len(unique_slugs),
        "scraped_ok": len([s for s in success_list if s["status"] == "ok"]),
        "skipped": skip_count,
        "failed": len(fail_list),
        "success": success_list,
        "fail": fail_list,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log.info("=" * 60)
    log.info("Done! OK: %d, Skipped: %d, Failed: %d",
             report["scraped_ok"], skip_count, len(fail_list))
    log.info("Report → %s", REPORT_PATH)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python scrape_otc_drugs.py {build_index|map|scrape|all}")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "build_index":
        build_index()
    elif cmd == "map":
        map_whitelist()
    elif cmd == "scrape":
        scrape_all()
    elif cmd == "all":
        build_index()
        map_whitelist()
        scrape_all()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
