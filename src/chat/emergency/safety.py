"""Deterministic safety post-check for emergency aid text.

The post-check enforces a small set of hard rules regardless of how the aid
block was produced, so a regression in retrieval or optional LLM generation
cannot produce an unsafe recommendation. It is intentionally surgical: it
patches a single bullet or inserts a fixed line, never rewrites the whole reply.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from src.chat.emergency.intents import classify_emergency_intent


def _norm(text: str) -> str:
    text = (text or "").casefold().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"\s+", " ", text).strip()
    return text


_INTENT_REQUIRED_RULES: dict[str, tuple[tuple[tuple[str, ...], str], ...]] = {
    "cardiac_arrest": (
        (
            ("ep tim",),
            "Trong lúc chờ 115, ép tim ngoài lồng ngực ngay nếu bệnh nhân không thở bình thường.",
        ),
        (("aed",), "Nhờ người xung quanh lấy AED nếu có sẵn và làm theo hướng dẫn của máy."),
        (
            ("khong cho", "tu tinh"),
            "Không chờ bệnh nhân tự tỉnh; tiếp tục ép tim cho đến khi nhân viên y tế tiếp nhận.",
        ),
    ),
    "seizure": (
        (
            ("thoi gian co giat",),
            "Theo dõi thời gian co giật; nếu kéo dài trên 5 phút hoặc chưa tỉnh lại, duy trì liên hệ 115.",
        ),
        (("nam nghieng",), "Đặt bệnh nhân nằm nghiêng an toàn và bảo vệ khỏi va đập."),
        (
            ("khong nhet",),
            "Không nhét bất cứ thứ gì vào miệng và không cố ghì bệnh nhân trong cơn co giật.",
        ),
    ),
    "stroke": (
        (("thoi diem khoi phat",), "Ghi lại chính xác thời điểm khởi phát triệu chứng."),
        (("khong cho", "an uong"), "Không cho bệnh nhân ăn uống trong lúc chờ 115."),
        (("khong", "tu dung thuoc"), "Không để bệnh nhân tự dùng thuốc trong lúc chờ 115."),
        (("khong", "tu lai xe"), "Không để bệnh nhân tự lái xe; chờ 115 đưa đến nơi có khả năng xử trí đột quỵ."),
    ),
    "chest_pain_acs": (
        (("hoi chung vanh",), "Đây là dấu hiệu nghi hội chứng vành cấp."),
        (("nghi", "tranh gang suc"), "Để bệnh nhân nghỉ, tránh gắng sức trong lúc chờ 115."),
        (("khong", "tu lai xe"), "Không để bệnh nhân tự lái xe hoặc tự chở đi cấp cứu."),
        (("khong cho con dau tu het",), "Không chờ cơn đau tự hết nếu đau ngực dữ dội hoặc kéo dài."),
    ),
    "co_poisoning": (
        (
            ("thoang khi",),
            "Chỉ đưa bệnh nhân ra nơi thoáng khí nếu an toàn; không quay lại phòng kín nhiễm khí.",
        ),
        (
            ("nguoi cuu ho", "hit khi doc"),
            "Tránh để người cứu hộ tiếp tục hít khí độc.",
        ),
        (
            ("khong tho", "ep tim"),
            "Nếu bệnh nhân không thở bình thường, bắt đầu ép tim ngoài lồng ngực trong lúc chờ 115.",
        ),
    ),
    "organophosphate_poisoning": (
        (
            ("thuoc tru sau",),
            "Đây là tình huống nghi ngộ độc thuốc trừ sâu phospho hữu cơ.",
        ),
        (
            ("coi bo", "quan ao"),
            "Cởi bỏ quần áo nhiễm hóa chất và rửa vùng da tiếp xúc bằng nước sạch.",
        ),
        (("rua", "nuoc sach"), "Rửa vùng da tiếp xúc hóa chất bằng nước sạch nếu làm được an toàn."),
        (("khong tu gay non",), "Không tự gây nôn hoặc cho uống thuốc giải độc tại nhà."),
    ),
    "opioid_poisoning": (
        (
            ("opioid", "tho cham"),
            "Đây là tình huống nghi ngộ độc opioid, có nguy cơ thở chậm hoặc ngừng thở.",
        ),
        (("theo doi", "nhip tho"), "Theo dõi nhịp thở, ý thức liên tục trong lúc chờ 115."),
        (
            ("khong tho", "ep tim"),
            "Nếu bệnh nhân không thở bình thường, bắt đầu ép tim ngoài lồng ngực trong lúc chờ 115.",
        ),
        (("nam nghieng",), "Nếu bệnh nhân còn thở, đặt nằm nghiêng an toàn để giảm nguy cơ sặc."),
    ),
    "acute_poisoning": (
        (("ngo doc cap",), "Đây là tình huống nghi ngộ độc cấp."),
        (
            ("bao bi",),
            "Giữ lại thuốc, hóa chất, thức ăn hoặc bao bì nghi gây độc để giao cho nhân viên y tế.",
        ),
        (
            ("khong tu gay non",),
            "Không tự gây nôn, cho uống thuốc, than hoạt hoặc mẹo dân gian nếu chưa được 115 hướng dẫn.",
        ),
    ),
    "snakebite": (
        (("ran doc can",), "Đây là tình huống nghi rắn độc cắn."),
        (("bat dong",), "Cho bệnh nhân nằm yên, hạn chế cử động và bất động chi bị cắn trong lúc chờ 115."),
        (("khong", "hut noc"), "Không rạch vết cắn, không hút nọc, không chườm đá, không tự garô."),
        (("khong", "garo"), "Không rạch vết cắn, không hút nọc, không chườm đá, không tự garô."),
    ),
    "anaphylaxis": (
        (("de doa tinh mang",), "Đây là dấu hiệu nghi phản vệ nặng, có thể đe dọa tính mạng."),
        (("di nguyen",), "Tránh tiếp xúc thêm với dị nguyên nghi ngờ nếu làm được an toàn."),
        (
            ("khong tu theo doi",),
            "Gọi 115 ngay; không tự theo dõi tại nhà.",
        ),
        (("theo doi", "nhip tho"), "Theo dõi nhịp thở, ý thức và để bệnh nhân ở tư thế dễ thở trong lúc chờ 115."),
        (
            ("thuc an", "thoi diem"),
            "Chuẩn bị thông tin về thức ăn, thuốc và thời điểm xuất hiện triệu chứng.",
        ),
    ),
    "severe_dyspnea": (
        (("tu the de tho",), "Để bệnh nhân ở tư thế dễ thở, nới lỏng quần áo trong lúc chờ 115."),
        (("khong", "tu di lai"), "Không để bệnh nhân tự đi lại hoặc tự lái xe."),
        (("theo doi", "nhip tho"), "Theo dõi nhịp thở, ý thức; báo ngay khi gọi 115 nếu bệnh nhân ngừng thở."),
    ),
    "shock_sepsis": (
        (("soc nhiem khuan",), "Đây là dấu hiệu nghi sốc nhiễm khuẩn hoặc suy tuần hoàn."),
        (("nam an toan",), "Để bệnh nhân nằm an toàn, giữ ấm nhẹ và theo dõi ý thức trong lúc chờ 115."),
        (("khong tu dung thuoc",), "Không tự dùng thuốc hoặc truyền dịch tại nhà."),
    ),
    "dengue_warning": (
        (("dau hieu canh bao",), "Báo với điều phối viên 115 các dấu hiệu cảnh báo như lừ đừ, đau bụng, tay chân lạnh."),
        (
            ("giai doan het sot", "nguy hiem"),
            "Giai đoạn hết sốt trong sốt xuất huyết Dengue có thể là giai đoạn nguy hiểm.",
        ),
        (("khong tu dung thuoc",), "Không tự dùng thuốc giảm đau kháng viêm; không trì hoãn gọi 115."),
    ),
    "hypoglycemia": (
        (("ha duong huyet",), "Đây là tình huống nghi hạ đường huyết cấp."),
        (
            ("con tinh", "nuot an toan"),
            "Nếu bệnh nhân còn tỉnh và nuốt an toàn, cho dùng đồ uống hoặc thức ăn có đường.",
        ),
        (("khong cho", "an uong"), "Không cho bệnh nhân ăn uống nếu lơ mơ, co giật hoặc hôn mê."),
        (
            ("insulin", "thuoc ha duong"),
            "Chuẩn bị thông tin insulin, thuốc hạ đường huyết và bữa ăn gần nhất.",
        ),
    ),
    "coma_unconscious": (
        (("hon me",), "Hôn mê hoặc mất ý thức là tình trạng cấp cứu."),
        (("nam nghieng",), "Nếu bệnh nhân còn thở, đặt nằm nghiêng an toàn để giảm nguy cơ tụt lưỡi, sặc."),
        (
            ("khong tho", "ep tim"),
            "Nếu bệnh nhân không thở bình thường, bắt đầu ép tim ngoài lồng ngực trong lúc chờ 115.",
        ),
        (("khong cho", "an uong"), "Không cho bệnh nhân ăn uống hoặc tự dùng thuốc."),
    ),
    "gi_bleeding": (
        (("xuat huyet tieu hoa",), "Đây là tình huống nghi xuất huyết tiêu hóa, có thể mất máu nặng."),
        (("khong an uong",), "Không ăn uống hoặc tự dùng thuốc cầm máu trong lúc chờ cấp cứu."),
        (
            ("nghieng dau",),
            "Để bệnh nhân nằm nghỉ, nghiêng đầu sang bên nếu buồn nôn hoặc nôn máu.",
        ),
        (
            ("luong mau",),
            "Chuẩn bị thông tin lượng máu nôn hoặc đi ngoài, thuốc đang dùng và bệnh nền.",
        ),
    ),
    "hypovolemic_shock": (
        (("soc giam the tich",), "Đây là tình huống nghi sốc giảm thể tích hoặc mất máu nặng."),
        (("nam an toan",), "Để bệnh nhân nằm an toàn, giữ ấm nhẹ và theo dõi ý thức trong lúc chờ 115."),
        (
            ("ep truc tiep",),
            "Nếu có chảy máu ngoài, ép trực tiếp lên vết thương bằng gạc hoặc khăn sạch.",
        ),
        (("khong tu truyen dich",), "Không tự truyền dịch hoặc cho ăn uống nếu lơ mơ, nôn nhiều."),
    ),
    "acute_abdomen": (
        (
            ("benh ly o bung cap",),
            "Đau bụng dữ dội kèm bụng cứng có thể là bệnh lý ổ bụng cấp.",
        ),
        (("khong tu dung thuoc",), "Không tự dùng thuốc giảm đau, không ăn uống trong lúc chờ được đánh giá cấp cứu."),
        (("nam yen",), "Nằm yên ở tư thế đỡ đau và theo dõi chóng mặt, vã mồ hôi, ngất."),
    ),
}

_CLINICAL_REPLACEMENT = (
    "Làm theo hướng dẫn trực tiếp của điều phối viên 115; không tự dùng thuốc, "
    "tiêm truyền hoặc thực hiện thủ thuật."
)
_TRANSPORT_REPLACEMENT = (
    "Không để bệnh nhân tự lái xe hoặc tự chở người bệnh; giữ liên lạc với 115 để được hướng dẫn."
)
_DOSE_RE = re.compile(r"\b\d+(?:[,.]\d+)?\s*(?:mg|mcg|ug|µg|g|ml|iu|ui)\s*/?\s*(?:kg|h|gio|phut)?\b", re.I)
_UNSAFE_CLINICAL_TERMS = (
    "truyen dich",
    "truyen tinh mach",
    "tiem tinh mach",
    "dat noi khi quan",
    "noi khi quan",
    "thong khi nhan tao",
    "tho may",
    "dat catheter",
    "hut dom",
    "hut dam",
    "hut dai",
    "hut dom dai",
    "adrenalin 0",
    "adrenaline 0",
    "epinephrine 0",
    "tiem adrenalin",
    "tiem adrenaline",
    "tiem epinephrine",
    "gay non",
    "kich thich non",
    "cho uong than hoat",
    "thuoc giai doc",
    "rach vet can",
    "hut noc",
    "chuom da",
    "garo",
    "ringer lactat",
    "natri clorid",
)
_UNSAFE_HOME_TERMS = (
    "theo doi tai nha",
    "tu theo doi tai nha",
    "cho theo doi o nha",
    "o nha theo doi",
)


@dataclass
class _SafetyItem:
    text: str
    norm: str
    required: bool = False
    low_priority: bool = False


def _bullet_norm(bullet: str) -> str:
    return _norm(bullet)


def _bullet_contains_phrase(bullet: str, phrase_norm: str) -> bool:
    return phrase_norm in _bullet_norm(bullet)


def _has_negation_before(text_norm: str, start: int) -> bool:
    prefix = text_norm[max(0, start - 35) : start]
    return any(word in prefix for word in ("khong", "dung", "tranh", "khong de", "khong nen"))


def _regex_unnegated(text_norm: str, pattern: str) -> bool:
    for match in re.finditer(pattern, text_norm):
        if not _has_negation_before(text_norm, match.start()):
            return True
    return False


def _has_unsafe_transport(text_norm: str) -> bool:
    if _regex_unnegated(text_norm, r"\btu lai xe\b"):
        return True
    if _regex_unnegated(text_norm, r"\btu cho\b"):
        return True
    if re.search(r"\bdua\b.{0,80}\bden khoa cap cuu\b", text_norm):
        return True
    return False


def _is_allowed_autoinjector(text_norm: str) -> bool:
    has_epinephrine = "epinephrine" in text_norm or "adrenaline" in text_norm or "adrenalin" in text_norm
    if not has_epinephrine:
        return False
    return "but tiem" in text_norm and ("ke san" in text_norm or "huong dan" in text_norm)


def _has_unsafe_clinical_instruction(text: str, text_norm: str) -> bool:
    if _DOSE_RE.search(text):
        return True
    if any(_regex_unnegated(text_norm, rf"\b{re.escape(term)}\b") for term in _UNSAFE_HOME_TERMS):
        return True
    for term in _UNSAFE_CLINICAL_TERMS:
        if _regex_unnegated(text_norm, rf"\b{re.escape(term)}\b"):
            if term.startswith("tiem ") and _is_allowed_autoinjector(text_norm):
                continue
            return True
    return False


def _sanitize_bullet(bullet: str) -> str:
    text = re.sub(r"\s+", " ", bullet).strip()
    text_norm = _norm(text)
    if _has_unsafe_clinical_instruction(text, text_norm):
        return _CLINICAL_REPLACEMENT
    if _has_unsafe_transport(text_norm):
        return _TRANSPORT_REPLACEMENT
    return text


def _is_low_priority_generic(text_norm: str) -> bool:
    return (
        "theo cac doan tai lieu cap cuu hien co" in text_norm
        or (
            "tu the an toan" in text_norm
            and "theo doi" in text_norm
            and "115" not in text_norm
        )
    )


def _append_item(
    out: list[_SafetyItem],
    seen_norm: set[str],
    text: str,
    *,
    required: bool = False,
) -> bool:
    safe_bullet = re.sub(r"\s+", " ", text).strip()
    safe_norm = _norm(safe_bullet)
    if safe_norm in seen_norm:
        if required:
            for item in out:
                if item.norm == safe_norm:
                    item.required = True
        return False
    out.append(
        _SafetyItem(
            text=safe_bullet,
            norm=safe_norm,
            required=required,
            low_priority=_is_low_priority_generic(safe_norm),
        )
    )
    seen_norm.add(safe_norm)
    return True


def _trim_items(items: list[_SafetyItem], limit: int = 5) -> list[str]:
    if len(items) <= limit:
        return [item.text for item in items]
    required_items = [item for item in items if item.required]
    if len(required_items) >= limit:
        keep = {id(item) for item in required_items[:limit]}
        return [item.text for item in items if id(item) in keep]

    slots = limit - len(required_items)
    medium_items = [
        item for item in items if not item.required and not item.low_priority
    ]
    low_items = [
        item for item in items if not item.required and item.low_priority
    ]
    selected = required_items + medium_items[:slots]
    slots = limit - len(selected)
    if slots > 0:
        selected.extend(low_items[:slots])
    keep = {id(item) for item in selected}
    return [item.text for item in items if id(item) in keep]


def apply_safety_post_check(
    bullets: list[str],
    question: str,
    red_flags: Iterable[str] | str | None = None,
) -> list[str]:
    """Mutates a copy of ``bullets`` to satisfy per-intent safety rules."""
    intent = classify_emergency_intent(question, red_flags)
    out: list[_SafetyItem] = []
    seen_norm: set[str] = set()
    for b in bullets:
        nb = _sanitize_bullet(b)
        if not nb:
            continue
        _append_item(out, seen_norm, nb)

    for required_phrases, safe_bullet in _INTENT_REQUIRED_RULES.get(intent or "", ()):
        matched = False
        for item in out:
            if all(_bullet_contains_phrase(item.text, p) for p in required_phrases):
                item.required = True
                matched = True
        if matched:
            continue
        _append_item(out, seen_norm, safe_bullet, required=True)

    return _trim_items(out, limit=5)
