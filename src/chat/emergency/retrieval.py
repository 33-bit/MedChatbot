"""Emergency-only retrieval.

Lightweight lexical retriever (overlap-based) over the filtered emergency
corpus. We intentionally avoid the full hybrid retriever here because:
  - emergency must not block the fast reply;
  - the emergency corpus is small (~hundreds of chunks);
  - a deterministic, low-latency matcher keeps the response predictable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.chat.emergency.corpus import (
    get_emergency_corpus_cached,
    is_emergency_chunk,
)
from src.chat.emergency.intents import (
    INTENT_SPECS,
    classify_emergency_intent,
    get_intent_spec,
    normalize_emergency_text,
)


# Generic term families used only when no emergency intent is detected.
_TERM_WEIGHTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "anaphylaxis",
        (
            "phan ve",
            "di ung",
            "sung moi",
            "sung luoi",
            "kho tho",
            "kho khe",
            "noi me day",
            "hai san",
            "epinephrine",
            "adrenalin",
        ),
    ),
    (
        "cardiac_arrest",
        (
            "ngung tho",
            "ngung tim",
            "ngung tuan hoan",
            "khong tho",
            "khong bat duoc mach",
            "khong dap",
            "bat tinh",
            "cpr",
            "hoi suc tim phoi",
            "aed",
        ),
    ),
    (
        "stroke",
        (
            "dot quy",
            "meo mieng",
            "noi ngong",
            "noi kho",
            "yeu liet",
            "liet nua nguoi",
            "te liet",
            "yeu nua nguoi",
            "tai bien mach mau nao",
        ),
    ),
    (
        "chest_pain",
        (
            "dau nguc",
            "that nguc",
            "lan len ham",
            "lan tay trai",
            "nhoi mau co tim",
            "hoi chung vanh",
        ),
    ),
    (
        "seizure",
        (
            "co giat",
            "dong kinh",
            "giat toan than",
            "trang thai dong kinh",
        ),
    ),
    (
        "co_poisoning",
        (
            "khi co",
            "carbon monoxide",
            "dot than",
            "than suoi",
            "phong kin",
            "ngo doc khi co",
        ),
    ),
    (
        "acute_poisoning",
        (
            "ngo doc",
            "nhiem doc",
            "uong nham",
            "an nham",
            "qua lieu",
            "thuoc ngu",
            "thuoc diet chuot",
            "hoa chat",
            "paracetamol qua lieu",
        ),
    ),
    (
        "organophosphate_poisoning",
        (
            "thuoc tru sau",
            "phospho huu co",
            "hoa chat tru sau",
            "thuoc sau",
            "tang tiet",
            "dong tu co",
            "co that phe quan",
        ),
    ),
    (
        "opioid_poisoning",
        (
            "opioid",
            "opiat",
            "heroin",
            "methadone",
            "fentanyl",
            "morphin",
            "codein",
            "thuoc phien",
            "tho cham",
            "dong tu co",
        ),
    ),
    (
        "snakebite",
        (
            "ran can",
            "ran doc",
            "ran ho mang",
            "ran luc",
            "ran cap nia",
            "vet can ran",
            "noc ran",
        ),
    ),
    (
        "shock",
        (
            "soc ",
            "soc nhiem khuan",
            "soc giam the tich",
            "tay chan lanh",
            "huyet ap tup",
            "mach nhanh nho",
        ),
    ),
    (
        "dengue_warning",
        (
            "sot xuat huyet",
            "giai doan nguy hiem",
            "dau hieu canh bao",
            "non nhieu",
            "dau bung nhieu",
            "lu du",
            "tay chan lanh",
            "chay mau kho can",
        ),
    ),
    (
        "abdominal",
        (
            "dau bung",
            "bung cung",
            "bung cung nhu go",
            "dau bung du doi",
        ),
    ),
    (
        "hypoglycemia",
        (
            "ha duong huyet",
            "duong huyet thap",
            "tut duong",
            "tuot duong",
            "insulin",
            "thuoc ha duong",
            "va mo hoi",
            "run tay",
        ),
    ),
    (
        "coma",
        (
            "hon me",
            "bat tinh",
            "mat y thuc",
            "khong goi day",
            "khong dap ung",
            "nam nghieng an toan",
        ),
    ),
    (
        "gi_bleeding",
        (
            "xuat huyet tieu hoa",
            "non ra mau",
            "non mau",
            "oi ra mau",
            "phan den",
            "di ngoai ra mau",
            "di cau ra mau",
        ),
    ),
    (
        "hypovolemic_shock",
        (
            "soc giam the tich",
            "soc mat mau",
            "mat mau nhieu",
            "chay mau nhieu",
            "huyet ap tut",
            "mach nho",
            "da lanh",
        ),
    ),
    (
        "dyspnea",
        (
            "kho tho",
            "kho tho cap",
            "suy ho hap",
            "hen suy",
            "con hen",
        ),
    ),
)


def _normalize(text: str) -> str:
    return normalize_emergency_text(text)


@dataclass(frozen=True)
class EmergencyHit:
    chunk_id: str
    source_slug: str
    source_name: str
    heading_path: str
    text: str
    score: float


def _score_chunk(
    query_norm: str,
    source_norm: str,
    heading_norm: str,
    text_norm: str,
    intent: str | None = None,
) -> float:
    """Score one emergency chunk for the detected intent."""
    spec = get_intent_spec(intent)
    if spec is not None:
        score = 0.0
        primary_source = any(term in source_norm for term in spec.primary_sources)
        secondary_source = any(term in source_norm for term in spec.secondary_sources)
        heading_hits = sum(1 for term in spec.heading_terms if term in heading_norm)
        body_hits = sum(1 for term in spec.body_terms if term in text_norm)

        if not primary_source and not secondary_source and heading_hits == 0 and body_hits == 0:
            return 0.0

        if primary_source:
            score += 12.0
        if secondary_source:
            score += 4.0
        score += min(heading_hits, 4) * 3.0
        score += min(body_hits, 6) * 0.75

        # Keep exact condition chapters above broad emergency chapters.
        if any(term in source_norm or term in heading_norm for term in spec.primary_sources):
            score += 4.0

        for other in INTENT_SPECS:
            if other.name == spec.name:
                continue
            other_source = any(term in source_norm for term in other.primary_sources)
            if other_source:
                score -= 8.0

        return max(score, 0.0)

    # Fallback: simple weighted overlap when the emergency type is unclear.
    score = 0.0
    for family, terms in _TERM_WEIGHTS:
        hit_query = any(t in query_norm for t in terms)
        if not hit_query:
            continue
        for term in terms:
            if term in heading_norm:
                score += 2.0
            if term in text_norm:
                score += 0.5
    if score <= 0.0:
        # Generic emergency corpus: count occurrences of generic terms
        # (cap cuu, xu tri cap cuu, hoi suc, ...) on the body.
        generic = ("cap cuu", "xu tri cap cuu", "hoi suc", "soc", "cap tinh")
        for g in generic:
            if g in text_norm:
                score += 0.05
    return score


def retrieve_emergency_aid(
    question: str,
    red_flags: Iterable[str] | str | None = None,
    *,
    corpus: list[dict] | None = None,
    top_k: int = 5,
) -> list[EmergencyHit]:
    """Return up to ``top_k`` emergency chunks, sorted by descending score."""
    if corpus is None:
        corpus = get_emergency_corpus_cached()
    flags_text = ""
    if red_flags:
        if isinstance(red_flags, str):
            flags_text = red_flags
        else:
            flags_text = " ".join(str(f) for f in red_flags)
    query = f"{question or ''} {flags_text}"
    query_norm = _normalize(query)
    if not query_norm:
        return []
    intent = classify_emergency_intent(question, red_flags)
    hits: list[EmergencyHit] = []
    for chunk in corpus:
        heading = chunk.get("heading_path", "")
        text = chunk.get("text", "")
        source = f"{chunk.get('source_slug', '')} {chunk.get('source_name', '')}"
        score = _score_chunk(
            query_norm,
            _normalize(source),
            _normalize(heading),
            _normalize(text),
            intent=intent,
        )
        if score <= 0.0:
            continue
        if not is_emergency_chunk(heading, text):
            continue
        hits.append(
            EmergencyHit(
                chunk_id=chunk["chunk_id"],
                source_slug=chunk.get("source_slug", ""),
                source_name=chunk.get("source_name", ""),
                heading_path=heading,
                text=text,
                score=score,
            )
        )
    hits.sort(key=lambda h: h.score, reverse=True)
    min_score = 5.0 if intent else 0.25
    if not hits or hits[0].score < min_score:
        return []
    return hits[:top_k]
