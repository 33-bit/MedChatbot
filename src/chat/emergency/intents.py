"""Deterministic emergency intent classification."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass


def normalize_emergency_text(text: str) -> str:
    text = (text or "").casefold().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def red_flag_text(red_flags: Iterable[str] | str | None) -> str:
    if not red_flags:
        return ""
    if isinstance(red_flags, str):
        return red_flags
    return " ".join(str(flag) for flag in red_flags)


def emergency_query_text(
    question: str,
    red_flags: Iterable[str] | str | None = None,
) -> str:
    return normalize_emergency_text(f"{question or ''} {red_flag_text(red_flags)}")


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _has_all(text: str, terms: tuple[str, ...]) -> bool:
    return all(term in text for term in terms)


@dataclass(frozen=True)
class EmergencyIntentSpec:
    name: str
    primary_sources: tuple[str, ...]
    secondary_sources: tuple[str, ...] = ()
    heading_terms: tuple[str, ...] = ()
    body_terms: tuple[str, ...] = ()


INTENT_SPECS: tuple[EmergencyIntentSpec, ...] = (
    EmergencyIntentSpec(
        name="cardiac_arrest",
        primary_sources=(
            "cap cuu ngung tuan hoan co ban",
            "cap cuu ngung tuan hoan nang cao",
        ),
        heading_terms=(
            "ngung tuan hoan",
            "ep tim",
            "xu tri cap cuu",
            "hoi suc tim phoi",
        ),
        body_terms=(
            "ngung tuan hoan",
            "ep tim",
            "aed",
            "khong tho binh thuong",
            "mach canh",
            "hoi suc tim phoi",
        ),
    ),
    EmergencyIntentSpec(
        name="anaphylaxis",
        primary_sources=("cap cuu phan ve",),
        secondary_sources=("di ung thuoc",),
        heading_terms=("phan ve", "soc phan ve", "dieu tri", "tai cho"),
        body_terms=(
            "phan ve",
            "di ung",
            "noi me day",
            "sung moi",
            "sung luoi",
            "kho khe",
            "kho tho",
            "adrenalin",
            "epinephrine",
        ),
    ),
    EmergencyIntentSpec(
        name="stroke",
        primary_sources=("cap cuu dot quy", "nhoi mau nao"),
        secondary_sources=("dieu tri va phuc hoi chuc nang dot quy nao",),
        heading_terms=("dot quy", "dau hieu nghi ngo dot quy", "xu tri dot quy"),
        body_terms=(
            "dot quy",
            "meo mieng",
            "noi ngong",
            "yeu liet",
            "liet nua nguoi",
            "thoi diem khoi phat",
        ),
    ),
    EmergencyIntentSpec(
        name="chest_pain_acs",
        primary_sources=(
            "dau that nguc khong on dinh va nhoi mau co tim khong co st chenh len",
            "nhoi mau co tim cap co st chenh len",
        ),
        heading_terms=(
            "dau that nguc",
            "nhoi mau co tim",
            "hoi chung vanh",
            "truoc khi nhap vien",
        ),
        body_terms=(
            "dau nguc",
            "dau that nguc",
            "nhoi mau co tim",
            "hoi chung vanh",
            "truoc khi nhap vien",
            "khong tu lai xe",
        ),
    ),
    EmergencyIntentSpec(
        name="seizure",
        primary_sources=(
            "chan doan va xu tri trang thai dong kinh",
            "dinh nghia phan loai dong kinh",
        ),
        heading_terms=("trang thai dong kinh", "con dong kinh", "dong kinh"),
        body_terms=("co giat", "dong kinh", "trang thai dong kinh", "5 phut"),
    ),
    EmergencyIntentSpec(
        name="co_poisoning",
        primary_sources=("ngoc doc khi co",),
        secondary_sources=("tiep can benh nhan ngo doc cap",),
        heading_terms=("ngo doc khi co", "giam hap thu", "bien phap hoi suc"),
        body_terms=("ngo doc khi co", "carbon monoxide", "dot than", "phong kin", "oxy"),
    ),
    EmergencyIntentSpec(
        name="organophosphate_poisoning",
        primary_sources=("ngo doc cap hoa chat tru sau phospho huu co",),
        secondary_sources=("tiep can benh nhan ngo doc cap",),
        heading_terms=("phospho huu co", "han che hap thu", "hoi chung cuong cholin"),
        body_terms=(
            "thuoc tru sau",
            "phospho huu co",
            "cuong cholin",
            "tang tiet",
            "co that phe quan",
            "dua ngay benh nhan ra khoi",
            "coi bo quan ao",
        ),
    ),
    EmergencyIntentSpec(
        name="opioid_poisoning",
        primary_sources=("ngo doc opioid",),
        secondary_sources=("tiep can benh nhan ngo doc cap",),
        heading_terms=("ngo doc opioid", "naloxon", "dieu tri ho tro"),
        body_terms=(
            "opioid",
            "heroin",
            "methadone",
            "fentanyl",
            "hon me",
            "dong tu co",
            "tho cham",
            "ngung tho",
            "naloxon",
        ),
    ),
    EmergencyIntentSpec(
        name="acute_poisoning",
        primary_sources=("tiep can benh nhan ngo doc cap",),
        secondary_sources=(
            "ngo doc cap paracetamol",
            "ngo doc cap barbituric",
            "ngo doc nam",
            "ngo doc cap ethanol",
            "ngo doc cap hoa chat diet chuot",
        ),
        heading_terms=(
            "cap cuu ban dau",
            "han che hap thu",
            "chat doc qua duong ho hap",
            "chat doc qua duong tieu hoa",
            "da niem mac",
        ),
        body_terms=(
            "ngo doc",
            "nhiem doc",
            "qua lieu",
            "hoa chat",
            "thuoc",
            "chat doc",
            "bao bi",
            "duong tiep xuc",
            "thoi diem tiep xuc",
        ),
    ),
    EmergencyIntentSpec(
        name="snakebite",
        primary_sources=("ran doc can",),
        heading_terms=(
            "dieu tri truoc benh vien",
            "bang ep bat dong",
            "so cuu ran can",
            "cap cuu on dinh",
        ),
        body_terms=(
            "ran doc",
            "ran can",
            "noc doc",
            "bang ep bat dong",
            "huyet thanh khang noc",
            "liet co",
            "suy ho hap",
        ),
    ),
    EmergencyIntentSpec(
        name="severe_dyspnea",
        primary_sources=("cap cuu tinh trang kho tho",),
        secondary_sources=("suy tim cap va phu phoi cap",),
        heading_terms=("kho tho", "suy ho hap", "xu tri cap cuu", "khai thong duong tho"),
        body_terms=("kho tho", "suy ho hap", "tim tai", "khong nam duoc", "duong tho"),
    ),
    EmergencyIntentSpec(
        name="shock_sepsis",
        primary_sources=("cap cuu ban dau soc nhiem khuan", "soc nhiem khuan"),
        heading_terms=("soc nhiem khuan", "khoi phuc tuan hoan", "xu tri cap cuu"),
        body_terms=(
            "soc nhiem khuan",
            "huyet ap tup",
            "tay chan lanh",
            "lo mo",
            "nhiem trung",
            "thay the the tich",
        ),
    ),
    EmergencyIntentSpec(
        name="dengue_warning",
        primary_sources=("benh sot xuat huyet dengue",),
        heading_terms=("giai doan nguy hiem", "dau hieu canh bao", "sot xuat huyet"),
        body_terms=(
            "sot xuat huyet",
            "dengue",
            "dau hieu canh bao",
            "lu du",
            "dau bung",
            "tay chan lanh",
            "chay mau",
        ),
    ),
    EmergencyIntentSpec(
        name="hypoglycemia",
        primary_sources=("cap cuu ha duong huyet",),
        heading_terms=("cap cuu ha duong huyet", "neu benh nhan con tinh", "hon me"),
        body_terms=(
            "ha duong huyet",
            "duong huyet",
            "run tay",
            "va mo hoi",
            "doi con cao",
            "lo mo",
            "co giat",
            "hon me",
        ),
    ),
    EmergencyIntentSpec(
        name="coma_unconscious",
        primary_sources=("cap cuu hon me",),
        heading_terms=("hon me", "kiem soat chuc nang ho hap", "tu the"),
        body_terms=(
            "hon me",
            "mat y thuc",
            "khong dap ung",
            "nam nghieng an toan",
            "duong tho",
            "nguy co sac",
        ),
    ),
    EmergencyIntentSpec(
        name="gi_bleeding",
        primary_sources=(
            "cap cuu xuat huyet tieu hoa cao",
            "xuat huyet tieu hoa do loet da day ta trang",
        ),
        heading_terms=("xuat huyet tieu hoa", "bieu hien chay mau", "cac bien phap hoi suc"),
        body_terms=(
            "xuat huyet tieu hoa",
            "non mau",
            "phan den",
            "di cau ra mau",
            "mach nhanh",
            "huyet ap tut",
            "nhin an uong",
        ),
    ),
    EmergencyIntentSpec(
        name="hypovolemic_shock",
        primary_sources=("cap cuu soc giam the tich",),
        secondary_sources=("cap cuu xuat huyet tieu hoa cao",),
        heading_terms=("soc giam the tich", "soc mat mau", "hoi suc"),
        body_terms=(
            "soc giam the tich",
            "chay mau nghiem trong",
            "mat mau",
            "huyet ap tut",
            "mach nho",
            "da lanh",
            "roi loan y thuc",
        ),
    ),
    EmergencyIntentSpec(
        name="acute_abdomen",
        primary_sources=("cap cuu dau bung cap",),
        heading_terms=("dau bung cap", "huong xu tri", "cac buoc can lam ngay"),
        body_terms=("dau bung", "bung cung", "bung cung nhu go", "dau bung du doi"),
    ),
)

_SPEC_BY_NAME = {spec.name: spec for spec in INTENT_SPECS}


def get_intent_spec(intent: str | None) -> EmergencyIntentSpec | None:
    if not intent:
        return None
    return _SPEC_BY_NAME.get(intent)


def classify_emergency_intent(
    question: str,
    red_flags: Iterable[str] | str | None = None,
) -> str | None:
    text = emergency_query_text(question, red_flags)
    if not text:
        return None

    if _has_any(
        text,
        (
            "ngung tho",
            "ngung tim",
            "ngung tuan hoan",
            "khong tho binh thuong",
            "khong bat duoc mach",
        ),
    ) or (_has_any(text, ("khong dap", "bat tinh")) and _has_any(text, ("khong tho", "mach"))):
        return "cardiac_arrest"

    if _has_any(text, ("khi co", "ngo doc co", "carbon monoxide", "dot than", "than suoi", "phong kin")):
        return "co_poisoning"

    if _has_any(
        text,
        (
            "thuoc tru sau",
            "phospho huu co",
            "pphc",
            "hoa chat tru sau",
            "thuoc sau",
        ),
    ) or (
        _has_any(text, ("ngo doc", "uong nham", "hit phai", "tiep xuc"))
        and _has_any(text, ("tang tiet", "chay dai", "dong tu co", "co that phe quan"))
    ):
        return "organophosphate_poisoning"

    if _has_any(text, ("opioid", "opiat", "heroin", "methadone", "fentanyl", "morphin", "morphine", "codein", "thuoc phien")) and _has_any(
        text,
        ("hon me", "lo mo", "tho cham", "ngung tho", "dong tu co", "qua lieu", "ngat"),
    ):
        return "opioid_poisoning"

    if _has_any(text, ("ran can", "ran doc", "ran ho mang", "ran luc", "ran cap nia", "vet can ran", "noc ran")):
        return "snakebite"

    if _has_any(
        text,
        (
            "dot quy",
            "meo mieng",
            "noi ngong",
            "noi kho",
            "yeu liet",
            "liet nua nguoi",
            "yeu nua nguoi",
            "tai bien mach mau nao",
        ),
    ):
        return "stroke"

    if _has_any(text, ("ha duong huyet", "duong huyet thap", "tut duong", "tuot duong")) or (
        _has_any(text, ("dai thao duong", "tieu duong", "insulin", "thuoc ha duong"))
        and _has_any(text, ("va mo hoi", "run tay", "doi con cao", "lo mo", "hon me", "co giat"))
    ):
        return "hypoglycemia"

    if _has_any(text, ("co giat", "giat toan than", "dong kinh", "trang thai dong kinh")):
        return "seizure"

    allergy = _has_any(text, ("phan ve", "di ung", "noi me day", "me day", "sung moi", "sung luoi", "hai san"))
    breathing_or_swelling = _has_any(text, ("kho tho", "kho khe", "sung moi", "sung luoi", "nghet tho"))
    if "phan ve" in text or (allergy and breathing_or_swelling):
        return "anaphylaxis"

    if _has_any(
        text,
        ("dau nguc", "that nguc", "lan len ham", "lan tay trai", "nhoi mau co tim", "hoi chung vanh"),
    ):
        return "chest_pain_acs"

    if _has_any(text, ("sot xuat huyet", "dengue")) and _has_any(
        text,
        ("dau hieu canh bao", "giai doan nguy hiem", "lu du", "lo mo", "dau bung", "tay chan lanh", "chay mau"),
    ):
        return "dengue_warning"

    if _has_any(
        text,
        (
            "xuat huyet tieu hoa",
            "non ra mau",
            "non mau",
            "oi ra mau",
            "phan den",
            "di ngoai ra mau",
            "di cau ra mau",
        ),
    ):
        return "gi_bleeding"

    if _has_any(text, ("soc nhiem khuan", "nhiem trung huyet", "sepsis")) or (
        _has_any(text, ("sot cao", "nhiem trung", "nhiem trung tieu"))
        and _has_any(text, ("lo mo", "huyet ap tup", "tay chan lanh", "mach nhanh", "tho nhanh"))
    ):
        return "shock_sepsis"

    if _has_any(text, ("soc giam the tich", "soc mat mau", "mat mau nhieu", "chay mau nhieu", "chay mau khong cam")) or (
        _has_any(text, ("tieu chay nhieu", "non nhieu", "mat nuoc", "bong rong"))
        and _has_any(text, ("huyet ap tut", "tay chan lanh", "lo mo", "ngat", "mach nhanh"))
    ):
        return "hypovolemic_shock"

    if _has_any(text, ("kho tho du doi", "tim tai", "moi tim", "khong nam duoc", "suy ho hap", "tho rit")) or (
        "kho tho" in text and not allergy
    ):
        return "severe_dyspnea"

    if _has_any(text, ("dau bung du doi", "bung cung", "bung cung nhu go")) or _has_all(text, ("dau bung", "choang")):
        return "acute_abdomen"

    if _has_any(
        text,
        (
            "ngo doc",
            "nhiem doc",
            "uong nham",
            "an nham",
            "qua lieu",
            "thuoc ngu",
            "thuoc diet chuot",
            "hoa chat",
            "an nam doc",
            "methanol",
            "paracetamol qua lieu",
        ),
    ):
        return "acute_poisoning"

    if _has_any(text, ("hon me", "bat tinh", "mat y thuc", "khong goi day", "khong danh thuc", "khong dap ung")):
        return "coma_unconscious"

    return None
