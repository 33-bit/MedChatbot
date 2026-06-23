"""Extract chapters and articles from 22/VBHN-VPQH into structured JSON."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader

from src.config import OUTPUT_DIR, PROJECT_ROOT

DEFAULT_PDF = PROJECT_ROOT / "documents" / "health_insurance" / "22-VBHN-VPQH.pdf"
DEFAULT_OUTPUT = OUTPUT_DIR / "health_insurance" / "22-vbhn-vpqh.json"

DOCUMENT_NUMBER = "22/VBHN-VPQH"
DOCUMENT_TITLE = "Luật Bảo hiểm y tế"
ISSUED_DATE = "2025-02-26"
EXPECTED_CHAPTERS = 10
EXPECTED_ARTICLES = 57

_HEADER_RE = re.compile(r"^\s*CÔNG BÁO/Số.+?\s+\d+\s*$", re.IGNORECASE)
_CHAPTER_RE = re.compile(r"^\s*Chương\s+([IVXLCDM]+)\s*$", re.IGNORECASE)
_ARTICLE_RE = re.compile(r"^\s*Điều\s+(\d+[a-z]?)\.\s*(.*)$", re.IGNORECASE)
_LOWERCASE_START_RE = re.compile(r"^[a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]")


def _clean_article_title(title: str) -> str:
    return re.sub(r"(?<=\D)\d{1,3}$", "", title.strip()).rstrip()


@dataclass
class Article:
    chapter_number: str
    chapter_title: str
    article_number: str
    article_title: str
    body: str
    page_start: int
    page_end: int


def _clean_page_text(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.replace("\u00a0", " ").splitlines():
        if _HEADER_RE.match(raw_line):
            continue
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        lines.append(line)
    return lines


def _normalize_body(lines: list[str]) -> str:
    normalized: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if normalized and not previous_blank:
                normalized.append("")
            previous_blank = True
            continue
        normalized.append(line)
        previous_blank = False
    return "\n".join(normalized).strip()


def parse_pdf(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Health-insurance PDF not found: {path}")

    reader = PdfReader(str(path))
    articles: list[Article] = []
    chapters: list[dict] = []
    current_chapter_number = ""
    current_chapter_title = ""
    pending_chapter_title = False
    current: dict | None = None
    seen_article_numbers: set[str] = set()

    def finish_article(page_end: int) -> None:
        nonlocal current
        if current is None:
            return
        current["page_end"] = max(current["page_start"], page_end)
        current["body"] = _normalize_body(current.pop("body_lines"))
        articles.append(Article(**current))
        current = None

    for page_number, page in enumerate(reader.pages, start=1):
        lines = _clean_page_text(page.extract_text() or "")
        for line in lines:
            chapter_match = _CHAPTER_RE.match(line)
            if chapter_match:
                finish_article(page_number)
                current_chapter_number = chapter_match.group(1).upper()
                current_chapter_title = ""
                pending_chapter_title = True
                continue

            if pending_chapter_title:
                if not line:
                    continue
                current_chapter_title = line
                chapters.append({
                    "chapter_number": current_chapter_number,
                    "chapter_title": current_chapter_title,
                    "page_start": page_number,
                })
                pending_chapter_title = False
                continue

            article_match = _ARTICLE_RE.match(line)
            if article_match:
                article_number = article_match.group(1).lower()
                if article_number in seen_article_numbers:
                    if current is not None:
                        current["body_lines"].append(line)
                    continue
                finish_article(page_number)
                seen_article_numbers.add(article_number)
                current = {
                    "chapter_number": current_chapter_number,
                    "chapter_title": current_chapter_title,
                    "article_number": article_match.group(1),
                    "article_title": _clean_article_title(article_match.group(2)),
                    "body_lines": [],
                    "page_start": page_number,
                    "page_end": page_number,
                }
                continue

            if current is None:
                continue

            if (
                line
                and not current["body_lines"]
                and _LOWERCASE_START_RE.match(line)
                and not current["article_title"].endswith((".", ":", ";"))
            ):
                current["article_title"] = f"{current['article_title']} {line}".strip()
                continue
            current["body_lines"].append(line)

    finish_article(len(reader.pages))

    return {
        "document_number": DOCUMENT_NUMBER,
        "document_title": DOCUMENT_TITLE,
        "issued_date": ISSUED_DATE,
        "source_file": str(path),
        "page_count": len(reader.pages),
        "chapters": chapters,
        "articles": [asdict(article) for article in articles],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    document = parse_pdf(args.pdf)
    chapter_count = len(document["chapters"])
    article_count = len(document["articles"])
    if chapter_count != EXPECTED_CHAPTERS or article_count != EXPECTED_ARTICLES:
        raise SystemExit(
            "Unexpected document structure: "
            f"expected {EXPECTED_CHAPTERS} chapters/{EXPECTED_ARTICLES} articles, "
            f"got {chapter_count}/{article_count}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Health-insurance law: {chapter_count} chapters, "
        f"{article_count} articles -> {args.output}"
    )


if __name__ == "__main__":
    main()
