from __future__ import annotations

from src.chat.retrieval.types import Hit

DRUG_URL_TEMPLATE = "https://trungtamthuoc.com/hoat-chat/{slug}"


def source_key(h: Hit) -> tuple:
    return (h.source_type, h.source_slug)


def dedupe(hits: list[Hit]) -> tuple[list[Hit], list[int]]:
    seen: dict[tuple, int] = {}
    unique: list[Hit] = []
    cite_idx: list[int] = []
    for h in hits:
        k = source_key(h)
        if k not in seen:
            unique.append(h)
            seen[k] = len(unique)
        cite_idx.append(seen[k])
    return unique, cite_idx


def drug_label(h: Hit) -> str:
    url = DRUG_URL_TEMPLATE.format(slug=h.source_slug)
    return f"Dược thư Quốc gia 2022 - [{h.source_name}]({url})"


def disease_label(h: Hit) -> str:
    chapter = (h.metadata or {}).get("chapter", "")
    return (
        "Hướng dẫn chẩn đoán và điều trị - Bệnh viện Bạch Mai - "
        f"{chapter} - {h.source_name}".rstrip(" -")
    )


def format_context(hits: list[Hit], cite_idx: list[int]) -> str:
    blocks = []
    for i, h in enumerate(hits):
        header = f"[{cite_idx[i]}] {h.source_name}"
        if h.heading_path:
            header += f" — {h.heading_path}"
        blocks.append(f"{header}\n{h.text}")
    return "\n\n---\n\n".join(blocks)


def format_sources(unique: list[Hit]) -> str:
    lines = []
    for i, h in enumerate(unique, 1):
        label = drug_label(h) if h.source_type == "drug" else disease_label(h)
        lines.append(f"[{i}] {label}")
    return "\n".join(lines)


def format_patient(patient: dict | None) -> str:
    if not patient:
        return ""
    parts = []
    if patient.get("symptoms"):
        lines = []
        for s in patient["symptoms"]:
            name = s.get("name", "")
            slots = [
                f"{k}: {s[k]}"
                for k in ("onset", "severity", "pattern", "associated")
                if s.get(k)
            ]
            line = f"- {name}"
            if slots:
                line += f" ({'; '.join(slots)})"
            lines.append(line)
        parts.append("Triệu chứng người bệnh:\n" + "\n".join(lines))
    if patient.get("medications"):
        parts.append("Thuốc đang dùng: " + ", ".join(patient["medications"]))
    if patient.get("candidate_diseases"):
        names = [d.get("name", "") for d in patient["candidate_diseases"][:5]]
        parts.append("Bệnh nghi ngờ (shortlist): " + ", ".join(filter(None, names)))
    return "\n\n".join(parts)
