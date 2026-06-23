"""Resolve public citation identifiers to local source documents safely."""

from __future__ import annotations

import re
from pathlib import Path

from src.config import OUTPUT_DIR
from src.config import PROJECT_ROOT

BACHMAI_RAW_DIR = OUTPUT_DIR / "bachmai" / "raw"
HEALTH_INSURANCE_SOURCE_PDF = (
    PROJECT_ROOT / "documents" / "health_insurance" / "22-VBHN-VPQH.pdf"
)
_SOURCE_SLUG_RE = re.compile(r"^[a-z0-9_]{1,80}$")


def resolve_bachmai_source_pdf(
    source_slug: str,
    *,
    raw_dir: Path | None = None,
) -> Path | None:
    """Return the unique split PDF for a safe disease slug, if present."""
    if not _SOURCE_SLUG_RE.fullmatch(source_slug or ""):
        return None

    root = raw_dir or BACHMAI_RAW_DIR
    resolved_root = root.resolve()
    matches: list[Path] = []
    for candidate in root.glob(f"chuong_*/{source_slug}/source.pdf"):
        if not candidate.is_file():
            continue
        try:
            candidate.resolve().relative_to(resolved_root)
        except ValueError:
            continue
        matches.append(candidate)
    return matches[0] if len(matches) == 1 else None


def resolve_health_insurance_source_pdf(
    source_path: Path | None = None,
) -> Path | None:
    candidate = source_path or HEALTH_INSURANCE_SOURCE_PDF
    return candidate if candidate.is_file() else None
