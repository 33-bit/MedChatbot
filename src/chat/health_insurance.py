"""Deterministic helpers for routing Vietnamese health-insurance queries."""

from __future__ import annotations

import re
import unicodedata


_EXPLICIT_TERMS = (
    "bao hiem y te",
    "bhyt",
    "luat bao hiem y te",
    "the bao hiem y te",
    "quy bao hiem y te",
)

_LEGAL_INSURANCE_TERMS = (
    "muc dong",
    "muc huong",
    "trai tuyen",
    "dung tuyen",
    "ho gia dinh",
    "cung chi tra",
    "khong duoc huong",
    "tham gia bao hiem",
    "the co gia tri",
    "chuyen co so",
    "chuyen vien",
    "chuyen nguoi benh",
    "ho so chuyen",
)

_LEGAL_CONTEXT_TERMS = (
    "nguoi lao dong",
    "nguoi su dung lao dong",
    "kham chua benh",
    "kham benh chua benh",
    "thanh toan",
    "chi tra",
    "tham gia",
    "the",
    "dong",
    "noi tru",
    "yeu cau chuyen mon",
    "giay to",
    "ho so",
)


def _normalize(text: str) -> str:
    text = (text or "").casefold().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_health_insurance_query(question: str) -> bool:
    """Return True for BHYT/legal health-insurance questions.

    This is intentionally conservative. It catches explicit BHYT wording and
    common legal-BHYT phrases that the LLM analyzer has misrouted, while
    avoiding a broad "insurance" override that could hide unrelated abuse or
    non-medical traffic.
    """
    text = _normalize(question)
    if not text:
        return False
    if any(term in text for term in _EXPLICIT_TERMS):
        return True
    if "bao hiem" not in text and not any(term in text for term in _LEGAL_INSURANCE_TERMS):
        return False
    return any(term in text for term in _LEGAL_INSURANCE_TERMS) and any(
        term in text for term in _LEGAL_CONTEXT_TERMS
    )


def expand_health_insurance_query(query: str) -> str:
    """Add compact legal anchors for BHYT retrieval.

    The original query stays first for reranking semantics; appended text only
    nudges dense/BM25 retrieval toward the law article likely needed.
    """
    text = _normalize(query)
    extras: list[str] = []
    if "ho gia dinh" in text and ("lan dau" in text or "the" in text or "gia tri" in text):
        extras.append(
            "khoản 5 Điều 12 nhóm tự đóng bảo hiểm y tế hộ gia đình; "
            "khoản 3 Điều 16 thẻ bảo hiểm y tế có giá trị sau 30 ngày"
        )
    if "muc dong" in text and (
        "nguoi lao dong" in text or "nguoi su dung lao dong" in text
    ):
        extras.append(
            "Điều 13 khoản 1 mức đóng người lao động người sử dụng lao động"
        )
    if "muc dong" in text and "ho gia dinh" in text:
        extras.append("Điều 13 khoản 4 khoản 6 mức đóng hộ gia đình")
    if "muc huong" in text or "cung chi tra" in text:
        extras.append("Điều 22 mức hưởng bảo hiểm y tế cùng chi trả")
    if "trai tuyen" in text or "khong dung noi dang ky" in text:
        extras.append("Điều 22 khoản 4 không đúng nơi đăng ký ban đầu trái tuyến")
    if "dung tuyen" in text:
        extras.append("Điều 22 mức hưởng đúng quy định khám chữa bệnh bảo hiểm y tế")
    if "khong duoc huong" in text or "khong duoc bao hiem" in text:
        extras.append("Điều 23 các trường hợp không được hưởng bảo hiểm y tế")
    if (
        "chuyen co so" in text
        or "chuyen vien" in text
        or "chuyen nguoi benh" in text
        or "ho so chuyen" in text
    ):
        extras.append(
            "Điều 28 khoản 3 thủ tục hồ sơ chuyển cơ sở khám chữa bệnh bảo hiểm y tế"
        )
    if not extras:
        return query
    return query + "\n" + "\n".join(extras)
