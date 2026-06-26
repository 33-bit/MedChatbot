"""Emergency-only corpus filter.

Many Bach Mai disease chapters contain Cấp cứu / Xử trí / Dấu hiệu cảnh báo
sub-sections even when the chapter is not a dedicated emergency protocol. To
keep the emergency RAG scope tight we filter the full disease_chunks.jsonl
by heading path and body text signals. The filter is intentionally
conservative: a chunk must match at least one heading signal OR at least one
text signal in the body to be kept. We do not rely only on literal "Cấp cứu"
headings.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable

# Heading-path signals. Matched against a normalized version of heading_path.
EMERGENCY_HEADING_SIGNALS: tuple[str, ...] = (
    "cap cuu",
    "xu tri cap cuu",
    "cap cuu ban dau",
    "hoi suc",
    "hoi suc tim phoi",
    "hoi suc tim mach",
    "ngan chan tu tuan hoan",
    "soc",
    "soc nhiem khuan",
    "soc giam the tich",
    "soc phan ve",
    "ngung tuan hoan",
    "ngung tho",
    "ngung tim",
    "dot quy",
    "dot quy nao",
    "tai bien mach mau nao",
    "kho tho",
    "kho tho cap",
    "suy ho hap",
    "hen suy",
    "hen phe quan",
    "phe quan phe viem",
    "phe quan co that",
    "dich vat duong tho",
    "dau nguc cap",
    "hoi chung vanh cap",
    "nhoi mau co tim cap",
    "ngo doc khi co",
    "ngo doc cap",
    "ran can",
    "ran doc can",
    "di ung thuoc",
    "diung thuoc nang",
    "phan ve",
    "so phan ve",
    "so mot",
    "benh non hien",
    "xuong khop",
    "bo nghiep",
    "tieu chay cap",
    "diarrhea",
    "nhiem khuan huyet",
    "dau bung cap",
    "viem ruot thua cap",
    "thu dam tac ruot",
    "loet da day ta trang",
    "thinh nhi",
    "thuy dam",
    "than kinh",
    "dong kinh",
    "co giat",
    "dot quy nh",
    "huyet ap thap",
    "sot xuat huyet",
    "sot ret",
    "dau hieu canh bao",
    "giai doan nguy hiem",
    "giai doan nguy kich",
    "giai doan nang",
    "cap cuu ngung",
    "cap cuu dot quy",
    "cap cuu kho tho",
    "cap cuu tim mach",
    "cap cuu ho hap",
    "cap cuu than kinh",
    "cap cuu nh",
    "cap cuu nhi",
    "cap cuu san",
    "cap cuu san khoa",
    "hoi suc so sinh",
    "hoi suc nh cap",
    "nhi cap cuu",
    "giai doan nguy",
    "giai doan bi",
    "khan cap",
    "nguy hiem",
    "tai bien",
    "sot cao co giat",
    "tac dong mach",
    "than kinh co giat",
    "than kinh dong kinh",
)

# Body-text signals. Matched against the chunk text (case-folded, no diacritics).
EMERGENCY_TEXT_SIGNALS: tuple[str, ...] = (
    "cap cuu",
    "xu tri cap cuu",
    "hoi suc",
    "soc ",
    "soc.",
    "soc,",
    "soc:",
    "soc\n",
    "soc)",
    "(soc",
    "ngung tuan hoan",
    "ngung tho",
    "ngung tim",
    "dot quy",
    "tai bien mach mau nao",
    "kho tho",
    "suy ho hap cap",
    "hen phe quan cap",
    "co that phe quan",
    "dich vat duong tho",
    "dau nguc cap",
    "hoi chung vanh cap",
    "nhoi mau co tim cap",
    "ngo doc khi co",
    "carbon monoxide",
    "ran can",
    "ran doc can",
    "di ung thuoc nang",
    "phan ve",
    "sot xuat huyet",
    "giai doan nguy hiem",
    "dau hieu canh bao",
    "truy dich cap",
    "khoi phuc tuan hoan",
    "epinephrine",
    "adrenalin",
    "hoi suc tim phoi",
    "cpr",
    "aed",
    "mach canh",
    "tho oxy",
    "trieu chung nang",
    "benh nhan nang",
    "tu vong",
    "bien chung nang",
    "suy ho hap",
    "suy tuan hoan",
    "suy da tang",
    "tuan hoan ngoai co the",
    "danh gia nhip tho",
    "kiem soat duong tho",
    "dung epinephrine",
    "khang histamin",
    "corticoid",
    "khang sinh",
    "chong soc",
    "sot ret ac tinh",
    "sot xuat huyet nang",
    "so nhiem khuan",
    "suy than cap",
    "suy gan cap",
    "tac mach phoi",
    "tac mach",
    "tai phat",
    "trieu chung nguy hiem",
    "phai nhap vien",
    "dau nguc khi nghi",
    "dau that nguc",
    "co that phe quan nang",
    "dot quy nhoi mau nao",
    "dot quy xuat huyet nao",
    "cap cuu tai bien",
    "hen suy",
    "con hen",
    "phu phoi cap",
    "phu nao",
    "xu tri khan cap",
    "dung thuoc khang sinh",
    "dung dich",
    "truyen dich",
    "dat noi khi quan",
    "thong khi nhan tao",
    "tho may",
    "nhoi mau co tim",
    "dau nguc dien hinh",
    "benh nhan phai nhap",
    "theo doi sat",
    "theo doi nhip tho mach",
    "benh nhan co bieu hien",
    "luong giac",
    "glasgow",
    "benh nhan lom o",
    "huyet ap tup",
    "mach nhanh nho",
    "tinh trang soc",
    "phuc hoi tuan hoan",
    "cap cuu ngoai vien",
    "cap cuu ban dau",
    "di ung",
    "di ung thuoc",
    "penicillin",
    "khang sinh nhom",
    "benh ly cap tinh",
    "cap tinh nang",
    "trieu chung cap",
    "dau hieu nang",
    "dau hieu nguy hiem",
    "tu vong cao",
    "nguy co tu vong",
    "de doa tinh mang",
    "khan cap ve y te",
    "huyet ap ket",
    "mach yeu",
    "trieu chung soc",
    "tinh trang kho tho",
    "kho tho khi nghi",
    "kho tho dien hinh",
    "kho tho dot ngot",
    "tho nhanh",
    "tho nong",
    "xam xit",
    "tai me",
    "me manh",
    "bat tinh",
    "ngat",
    "me hoac",
    "yeu liet nua nguoi",
    "liet nua nguoi",
    "liet mem",
    "liet cung",
    "tay chan lanh",
    "tay chan lanh am",
    "moi tim tai",
    "moi tim",
    "kho tho va dau nguc",
    "dau dau dot ngot",
    "non ra mau",
    "non mau",
    "di ngoai ra mau",
    "xa huyet ap",
    "xa truyen dich",
    "truyen dich tinh mach",
    "thay the the tich",
    "thay the huyet tuong",
    "huyet tuong",
    "ca phan tu",
    "ca cao phan tu",
    "nhiem toan",
    "toan chuyen hoa",
    "toan ho mau",
    "sinh hoc nang",
    "huyet hoc",
    "huyet ap cao dot bien",
    "tang huyet ap ac tinh",
    "tang huyet ap cap",
    "nhau bong non",
    "nhau tien dao",
    "vo tu cung",
    "bang huyet",
    "vo nhu",
    "dau vai",
    "nhoi mau co tim cap st",
    "dau that nguc dien hinh",
    "benh mach vanh",
    "benh tim thieu mau cuc bo",
    "con kich phat con hen",
    "phu phoi",
    "kho tho the",
    "kho tho kho khe",
    "kho khe",
    "tim tai",
    "moi tim tai",
    "tho rit",
    "tho ro",
    "am phe ran",
    "am phe ran am",
    "am phe ran nho",
    "thay doi y thuc",
    "trieu chung than kinh",
    "dong kinh lien tuc",
    "co giat lien tuc",
    "co giat keo dai",
    "benh dong kinh",
    "benh co giat",
    "trang thai dong kinh",
    "con dong kinh",
    "con co giat",
    "cap cuu dong kinh",
    "cap cuu co giat",
    "dot quy cap",
    "sot cap tinh",
    "sot rat cao",
    "sot tren 40",
    "co giat do sot",
    "giam dang nhiet do",
    "hon me",
    "huyet ap thap",
    "suy tim",
    "trieu chung suy tim",
    "suy tim cap",
    "suy tim mat bo",
    "cap cuu suy tim",
    "cap cuu benh nhan suy",
    "suy dinh duong",
    "mau",
    "thuoc doc",
    "thuoc qua lieu",
    "qua lieu",
    "ngo doc thuoc",
    "ngo doc ruou",
    "ngo doc methanol",
    "ngo doc phospho",
    "ngo doc thuoc tru sau",
    "ngo doc nam",
    "ngo doc hoa chat",
    "ngo doc kim loai",
    "ngo doc",
    "benh nhan nguy kich",
    "nhan dinh ban dau",
    "nhan dinh tinh trang",
    "nhan dinh cap cuu",
    "nhan dinh khan cap",
    "thuoc benh nhan",
    "theo doi benh nhan",
    "theo doi sat benh nhan",
    "nhap vien khan cap",
    "chuyen vien khan",
    "khan cap",
    "khan",
    "benh ly can nhap vien",
    "can nhap vien",
    "trieu chung cap cuu",
    "trieu chung khan cap",
    "dau hieu khan",
    "dau hieu de doa",
    "de doa tinh mang",
    "tinh mang",
)


def _normalize(text: str) -> str:
    """Lowercase + strip Vietnamese diacritics + collapse whitespace."""
    text = (text or "").casefold().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_emergency_chunk(
    heading_path: str,
    text: str,
    heading_signals: Iterable[str] = EMERGENCY_HEADING_SIGNALS,
    text_signals: Iterable[str] = EMERGENCY_TEXT_SIGNALS,
) -> bool:
    """Decide whether a chunk is part of the emergency corpus."""
    h_norm = _normalize(heading_path)
    t_norm = _normalize(text)
    for sig in heading_signals:
        if sig in h_norm:
            return True
    for sig in text_signals:
        if sig in t_norm:
            return True
    return False


# Heading tokens that strongly indicate a non-emergency chapter, used to
# short-circuit obvious false positives (e.g. "I. ĐẠI CƯƠNG" alone should not
# be kept just because "cap cuu" appears in the body once).
_EXCLUDE_TOKENS: tuple[str, ...] = (
    "tai lieu tham khao",
    "tltk",
    "sach tham khao",
)


def _is_junk_chunk(heading_path: str) -> bool:
    h_norm = _normalize(heading_path)
    return any(tok in h_norm for tok in _EXCLUDE_TOKENS)


def load_emergency_corpus(
    chunks_path: str | Path = "outputs/chunks/disease_chunks.jsonl",
) -> list[dict]:
    """Read the full disease chunk file and keep only emergency chunks."""
    chunks_path = Path(chunks_path)
    if not chunks_path.exists():
        return []
    kept: list[dict] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            heading = chunk.get("heading_path", "")
            text = chunk.get("text", "")
            if _is_junk_chunk(heading):
                continue
            if is_emergency_chunk(heading, text):
                kept.append(chunk)
    return kept


@lru_cache(maxsize=1)
def _load_emergency_corpus_cached(path_str: str) -> tuple[dict, ...]:
    """Cached version: stable across calls so a single emergency request
    doesn't re-read the JSONL each time."""
    return tuple(load_emergency_corpus(Path(path_str)))


def get_emergency_corpus_cached(
    chunks_path: str | Path = "outputs/chunks/disease_chunks.jsonl",
) -> list[dict]:
    return list(_load_emergency_corpus_cached(str(chunks_path)))
