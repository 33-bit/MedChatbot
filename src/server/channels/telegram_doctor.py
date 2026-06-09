"""
telegram_doctor.py
------------------
Doctor-connect Telegram handlers: menu, selection, accept/decline, relay.
"""

from __future__ import annotations

import contextlib
import logging
import time
import unicodedata
from dataclasses import dataclass

from src.chat.storage import doctors
from src.chat.storage import wallet
from src.chat.storage.seed_doctors import PAID_RATE_PER_MIN, SPECIALTIES
from src.chat.storage.session import load_session
from src.server.channels import telegram

log = logging.getLogger(__name__)

DOCTOR_PREFIX = "doctor:"
TIER_PREFIX = "doctor:tier:"
SPECIALTY_PREFIX = "doctor:specialty:"
SPECIALTIES_PREFIX = "doctor:specialties:"
PROFILE_PREFIX = "doctor:profile:"
PICK_PREFIX = "doctor:pick:"
ACCEPT_PREFIX = "doctor:accept:"
DECLINE_PREFIX = "doctor:decline:"
REFRESH_PREFIX = "doctor:refresh:"
EXTEND_PREFIX = "doctor:extend:"
WAIT_PREFIX = "doctor:wait:"
WLREFRESH_PREFIX = "doctor:wlrefresh:"
WLLEAVE_PREFIX = "doctor:wlleave:"
HANDOFF_ACCEPT = "doctor:handoff:accept"
HANDOFF_OTHER = "doctor:handoff:other"
HANDOFF_DECLINE = "doctor:handoff:decline"
CANCEL_CALLBACK = "doctor:cancel"
HANDOFF_TTL_SECONDS = 15 * 60


@dataclass
class DoctorHandoffContext:
    question: str
    bot_answer: str
    summary: str
    specialty_hint: str | None
    created_at: float


_HANDOFF_CONTEXTS: dict[str, DoctorHandoffContext] = {}

_SPECIALTY_KEYWORDS = {
    "Hô hấp": ("ho", "kho tho", "dau nguc", "hen", "pho quan", "phoi", "sot", "viem hong"),
    "Tai mũi họng": ("dau hong", "nghet mui", "so mui", "u tai", "dau tai", "khan tieng"),
    "Tiêu hóa": ("dau bung", "tieu chay", "tao bon", "non", "buon non", "da day", "tieu hoa"),
    "Da liễu": ("ngua", "phat ban", "noi man", "mun", "da", "di ung", "me day"),
    "Tim mạch": ("tim", "hoi hop", "huyet ap", "dau nguc", "mach", "choang"),
    "Thần kinh": ("dau dau", "chong mat", "te", "yeu liet", "co giat", "mat ngu"),
    "Cơ xương khớp": ("dau khop", "dau lung", "co xuong", "chan thuong", "sung khop"),
    "Sản phụ khoa": ("mang thai", "kinh nguyet", "am dao", "phu khoa", "san", "bau"),
    "Nhi khoa": ("tre", "be", "em be", "so sinh", "nhi"),
}


def _text_key(text: str) -> str:
    normalized = unicodedata.normalize("NFD", (text or "").strip().casefold())
    no_marks = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return " ".join(no_marks.split())


def _session_text(session, question: str, bot_answer: str) -> str:
    parts = [question, bot_answer]
    parts.extend(s.get("name", "") for s in session.symptoms)
    parts.extend(str(d.get("name") or d.get("name_vi") or d.get("disease") or "") for d in session.candidate_diseases)
    return " ".join(parts)


def _infer_specialty_hint(text: str) -> str | None:
    key = _text_key(text)
    for specialty, keywords in _SPECIALTY_KEYWORDS.items():
        if any(keyword in key for keyword in keywords):
            return specialty
    return None


def _normalize_specialty(specialty: str | None) -> str | None:
    key = _text_key(specialty or "")
    if not key:
        return None
    for canonical in SPECIALTIES:
        canonical_key = _text_key(canonical)
        if key == canonical_key or key in canonical_key or canonical_key in key:
            return canonical
    return None


def _specialty_slug(specialty: str) -> str:
    return _text_key(specialty).replace(" ", "_")


def _specialty_from_slug(rows: list[dict], slug: str) -> str | None:
    for row in rows:
        specialty = str(row.get("specialty") or "").strip()
        if specialty and _specialty_slug(specialty) == slug:
            return specialty
    return None


def _ordered_specialties(
    rows: list[dict],
    exclude: str | None = None,
) -> list[str]:
    excluded_key = _text_key(exclude or "")
    found: dict[str, str] = {}
    for row in rows:
        specialty = str(row.get("specialty") or "").strip()
        key = _text_key(specialty)
        if specialty and key and key != excluded_key:
            found.setdefault(key, specialty)

    ordered = []
    for specialty in SPECIALTIES:
        key = _text_key(specialty)
        if key in found:
            ordered.append(found.pop(key))
    ordered.extend(sorted(found.values(), key=_text_key))
    return ordered


def _format_items(items: list[str], empty: str) -> str:
    clean = [item for item in items if item]
    return ", ".join(clean[:5]) if clean else empty


def _build_handoff_summary(
    chat_id: int | str,
    question: str,
    bot_answer: str,
    specialty_hint: str | None = None,
) -> tuple[str, str | None]:
    session = load_session(f"tg:{chat_id}")
    symptoms = _format_items([s.get("name", "") for s in session.symptoms], "chưa ghi nhận rõ")
    medications = _format_items([str(m) for m in session.medications], "chưa ghi nhận")
    candidates = _format_items(
        [str(d.get("name") or d.get("name_vi") or d.get("disease") or "") for d in session.candidate_diseases],
        "chưa có",
    )
    recent = []
    for turn in session.conversation[-6:]:
        role = "Bệnh nhân" if turn.get("role") == "user" else "Bot"
        content = str(turn.get("content") or "").strip()
        if content:
            recent.append(f"  {role}: {content[:180]}")
    recent_text = "\n".join(recent[-6:]) or "  Chưa có hội thoại gần đây."
    specialty_hint = (
        _normalize_specialty(specialty_hint)
        or _infer_specialty_hint(_session_text(session, question, bot_answer))
    )
    summary = (
        "Tóm tắt từ bot:\n"
        f"- Câu hỏi gần nhất: {question}\n"
        f"- Triệu chứng đã ghi nhận: {symptoms}\n"
        f"- Thuốc liên quan: {medications}\n"
        f"- Bệnh/khả năng bot đang cân nhắc: {candidates}\n"
        "- Hội thoại gần đây:\n"
        f"{recent_text}\n\n"
        "Lưu ý: Tóm tắt tự động từ hội thoại bot, bác sĩ cần xác minh lại với bệnh nhân."
    )
    return summary, specialty_hint


def register_handoff_context(
    chat_id: int | str,
    question: str,
    bot_answer: str,
    specialty_hint: str | None = None,
) -> None:
    summary, specialty_hint = _build_handoff_summary(
        chat_id,
        question,
        bot_answer,
        specialty_hint,
    )
    _HANDOFF_CONTEXTS[str(chat_id)] = DoctorHandoffContext(
        question=question,
        bot_answer=bot_answer,
        summary=summary,
        specialty_hint=specialty_hint,
        created_at=time.time(),
    )


def _handoff_context(chat_id: int | str) -> DoctorHandoffContext | None:
    key = str(chat_id)
    context = _HANDOFF_CONTEXTS.get(key)
    if context is None:
        return None
    if time.time() - context.created_at > HANDOFF_TTL_SECONDS:
        _HANDOFF_CONTEXTS.pop(key, None)
        return None
    return context


def handoff_keyboard(chat_id: int | str | None = None) -> dict:
    context = _handoff_context(chat_id) if chat_id is not None else None
    specialty = context.specialty_hint if context else None
    if specialty:
        first_row = [
            {"text": f"🩺 {specialty}", "callback_data": HANDOFF_ACCEPT},
            {"text": "📁 Chuyên khoa khác", "callback_data": HANDOFF_OTHER},
        ]
    else:
        first_row = [
            {"text": "📁 Chọn chuyên khoa", "callback_data": HANDOFF_OTHER},
        ]
    return {
        "inline_keyboard": [
            first_row,
            [{"text": "Không cần", "callback_data": HANDOFF_DECLINE}],
        ]
    }


def _tier_keyboard(specialty: str | None = None) -> dict:
    suffix = f":{_specialty_slug(specialty)}" if specialty else ""
    return {
        "inline_keyboard": [
            [
                {"text": "Miễn phí", "callback_data": f"doctor:tier:free{suffix}"},
                {"text": "Trả phí", "callback_data": f"doctor:tier:paid{suffix}"},
            ],
            [{"text": "❌ Hủy", "callback_data": CANCEL_CALLBACK}],
        ]
    }


def _doctor_button_label(d: dict) -> str:
    parts = [d["name"]]
    if d.get("specialty"):
        parts.append(d["specialty"])
    if d["available"]:
        marker = "🟢"
        parts.append("rảnh")
    else:
        marker = "🟠"
        waiting = doctors.waitlist_count(d["id"])
        parts.append(f"đang bận · {waiting} đang chờ" if waiting else "đang bận")
    return f"{marker} " + " · ".join(parts)


def _doctor_list_keyboard(rows: list[dict], tier: str, specialty: str) -> dict:
    keyboard = []
    for d in rows:
        callback = f"doctor:profile:{d['id']}:{tier}:{_specialty_slug(specialty)}"
        keyboard.append([{"text": _doctor_button_label(d), "callback_data": callback}])
    keyboard.append([
        {"text": "🔄 Làm mới", "callback_data": f"doctor:refresh:{tier}:{_specialty_slug(specialty)}"},
        {"text": "⬅️ Quay lại", "callback_data": f"doctor:specialties:{tier}"},
    ])
    return {"inline_keyboard": keyboard}


def _doctor_profile_keyboard(doctor_id: int, tier: str, specialty_slug: str) -> dict:
    return {
        "inline_keyboard": [
            [{"text": "✅ Kết nối bác sĩ", "callback_data": f"doctor:pick:{doctor_id}"}],
            [
                {"text": "⬅️ Quay lại", "callback_data": f"doctor:specialty:{tier}:{specialty_slug}"},
                {"text": "❌ Hủy", "callback_data": CANCEL_CALLBACK},
            ],
        ]
    }


def _specialty_keyboard(tier: str, specialties: list[str]) -> dict:
    keyboard = [
        [{"text": f"📁 {specialty}", "callback_data": f"doctor:specialty:{tier}:{_specialty_slug(specialty)}"}]
        for specialty in specialties
    ]
    keyboard.append([
        {"text": "⬅️ Quay lại", "callback_data": "doctor:back"},
        {"text": "❌ Hủy", "callback_data": CANCEL_CALLBACK},
    ])
    return {"inline_keyboard": keyboard}


def _request_keyboard(consultation_id: int) -> dict:
    return {
        "inline_keyboard": [[
            {"text": "✅ Nhận tư vấn", "callback_data": f"doctor:accept:{consultation_id}"},
            {"text": "❌ Từ chối", "callback_data": f"doctor:decline:{consultation_id}"},
        ]]
    }


def _extend_keyboard(consultation_id: int) -> dict:
    return {
        "inline_keyboard": [[
            {"text": "➕ Gia hạn 15 phút", "callback_data": f"doctor:extend:{consultation_id}"},
        ]]
    }


def _busy_keyboard(doctor_id: int) -> dict:
    return {
        "inline_keyboard": [
            [{"text": "⏳ Vào hàng đợi", "callback_data": f"doctor:wait:{doctor_id}"}],
            [{"text": "⬅️ Quay lại", "callback_data": "doctor:back"}],
        ]
    }


def _waitlist_keyboard(doctor_id: int) -> dict:
    return {
        "inline_keyboard": [[
            {"text": "🔄 Làm mới", "callback_data": f"doctor:wlrefresh:{doctor_id}"},
            {"text": "🚪 Rời hàng đợi", "callback_data": f"doctor:wlleave:{doctor_id}"},
        ]]
    }


def _vnd(amount: int) -> str:
    """Format VND with dot thousands separators (Vietnamese convention)."""
    return f"{amount:,}".replace(",", ".")


def _tier_menu_text() -> str:
    free_min = doctors.FREE_SESSION_SECONDS // 60
    free_cd = doctors.FREE_COOLDOWN_SECONDS // 60
    block_min = doctors.PAID_BLOCK_SECONDS // 60
    pair_cd = doctors.PAID_PAIR_COOLDOWN_SECONDS // 60
    return (
        "👨‍⚕️ Kết nối bác sĩ\n\n"
        "Dịch vụ giúp bạn kết nối trực tiếp với bác sĩ theo chuyên khoa phù hợp "
        "để được tư vấn sâu hơn.\n\n"
        "Có 2 hình thức tư vấn:\n\n"
        "🎁 Miễn phí\n"
        "• Phiên tư vấn ngắn, phù hợp để trao đổi ban đầu.\n"
        f"• Mỗi phiên tối đa {free_min} phút.\n"
        f"• Sau khi kết thúc, chờ {free_cd} phút mới dùng lại lượt miễn phí.\n\n"
        "💳 Trả phí\n"
        "• Thời lượng dài hơn và có thể gia hạn khi cần.\n"
        f"• {_vnd(PAID_RATE_PER_MIN)}đ/phút, mỗi block {block_min} phút.\n"
        f"• Gần hết giờ có thể gia hạn; mỗi lần gia hạn phí tăng thêm "
        f"{_vnd(doctors.PAID_RATE_STEP_PER_MIN)}đ/phút.\n"
        f"• Sau khi kết thúc, chờ {pair_cd} phút mới kết nối lại đúng bác sĩ đó.\n\n"
        "Chọn hình thức tư vấn:"
    )


async def handle_doctor_command(chat_id: int | str) -> bool:
    await telegram.send_text(
        chat_id,
        _tier_menu_text(),
        inline_keyboard=_tier_keyboard(),
    )
    return True


async def _show_specialty_list(
    chat_id: int | str,
    message_id: int,
    tier: str,
    exclude_specialty: str | None = None,
) -> None:
    rows = doctors.list_doctors(tier)
    specialties = _ordered_specialties(rows, exclude=exclude_specialty)
    if not specialties and exclude_specialty:
        specialties = _ordered_specialties(rows)
    if not specialties:
        await telegram._edit_message_text(
            chat_id,
            message_id,
            "Hiện chưa có bác sĩ trong nhóm này.",
            _tier_keyboard(),
        )
        return
    title = "Chọn chuyên khoa:" if not exclude_specialty else "Chọn chuyên khoa khác:"
    await telegram._edit_message_text(
        chat_id,
        message_id,
        title,
        _specialty_keyboard(tier, specialties),
    )


async def _show_doctor_list(
    chat_id: int | str,
    message_id: int,
    tier: str,
    specialty_slug: str,
) -> None:
    rows = doctors.list_doctors(tier)
    specialty = _specialty_from_slug(rows, specialty_slug)
    if specialty is None:
        await telegram._edit_message_text(
            chat_id,
            message_id,
            "Không tìm thấy chuyên khoa này.",
            _tier_keyboard(),
        )
        return
    filtered = [row for row in rows if _specialty_slug(str(row.get("specialty") or "")) == specialty_slug]
    if not filtered:
        await _show_specialty_list(chat_id, message_id, tier, exclude_specialty=specialty)
        return
    await telegram._edit_message_text(
        chat_id,
        message_id,
        f"Chọn bác sĩ chuyên khoa {specialty}:",
        _doctor_list_keyboard(filtered, tier, specialty),
    )


def _doctor_profile_text(doctor: dict) -> str:
    experience = doctor.get("experience_years")
    experience_text = f"{experience} năm" if experience else "Chưa cập nhật"
    return (
        f"👨‍⚕️ {doctor['name']}\n"
        f"Chuyên khoa: {doctor.get('specialty') or 'Chưa cập nhật'}\n"
        f"Học vị: {doctor.get('degree') or 'Chưa cập nhật'}\n"
        f"Kinh nghiệm: {experience_text}\n"
        f"Công tác: {doctor.get('hospital') or 'Chưa cập nhật'}\n\n"
        "Giới thiệu:\n"
        f"{doctor.get('bio') or 'Chưa cập nhật'}"
    )


async def _show_doctor_profile(
    chat_id: int | str,
    message_id: int,
    doctor_id: int,
    tier: str,
    specialty_slug: str,
) -> None:
    doctor = doctors.get_doctor(doctor_id)
    if doctor is None or not doctor["active"]:
        await telegram._edit_message_text(chat_id, message_id, "Không tìm thấy bác sĩ này.")
        return
    await telegram._edit_message_text(
        chat_id,
        message_id,
        _doctor_profile_text(doctor),
        _doctor_profile_keyboard(doctor_id, tier, specialty_slug),
    )


async def _handle_pick(chat_id: int | str, message_id: int, doctor_id: int) -> None:
    debt = wallet.debt_status(f"tg:{chat_id}")
    if debt["in_debt"]:
        await telegram._edit_message_text(
            chat_id,
            message_id,
            f"Bạn đang nợ {debt['debt']:,} VND. Vui lòng thanh toán "
            f"{debt['payoff_amount']:,} VND (gồm 10% phí) bằng /paydebt trước khi kết nối bác sĩ.",
        )
        return
    if doctors.open_consultation_for_patient(chat_id) is not None:
        await telegram._edit_message_text(
            chat_id,
            message_id,
            "Bạn đang có phiên tư vấn đang chờ hoặc đang hoạt động. Dùng /end để kết thúc trước.",
        )
        return
    doctor = doctors.get_doctor(doctor_id)
    if doctor is None or not doctor["active"]:
        await telegram._edit_message_text(chat_id, message_id, "Không tìm thấy bác sĩ này.")
        return
    if doctor["tier"] == "free":
        cooldown = doctors.free_cooldown_remaining(chat_id)
        if cooldown > 0:
            await telegram._edit_message_text(
                chat_id,
                message_id,
                "Bạn vừa dùng tư vấn miễn phí. Vui lòng chờ "
                f"{_fmt_seconds(cooldown)} nữa hoặc chọn bác sĩ trả phí.",
            )
            return
    else:
        pair_cooldown = doctors.paid_pair_cooldown_remaining(chat_id, doctor_id)
        if pair_cooldown > 0:
            await telegram._edit_message_text(
                chat_id,
                message_id,
                "Bạn vừa kết thúc phiên với bác sĩ này. Vui lòng chờ "
                f"{_fmt_seconds(pair_cooldown)} nữa hoặc chọn bác sĩ khác.",
            )
            return
    if not doctor["available"]:
        await telegram._edit_message_text(
            chat_id,
            message_id,
            f"{doctor['name']} đang bận. Bạn có muốn vào hàng đợi không?",
            _busy_keyboard(doctor_id),
        )
        return
    try:
        consultation_id = doctors.create_consultation(chat_id, doctor, doctor["tier"])
    except ValueError:
        await telegram._edit_message_text(
            chat_id,
            message_id,
            "Bạn đang có phiên tư vấn đang chờ hoặc đang hoạt động. Dùng /end để kết thúc trước.",
        )
        return
    context = _handoff_context(chat_id)
    request_text = "Có bệnh nhân muốn tư vấn qua bot. Bạn có nhận không?"
    if context is not None:
        specialty_line = f"\nGợi ý chuyên khoa: {context.specialty_hint}" if context.specialty_hint else ""
        request_text = (
            "Có bệnh nhân muốn tư vấn sau khi hỏi bot. Bạn có nhận không?"
            f"{specialty_line}\n\n"
            f"{context.summary}"
        )
    try:
        await telegram.send_text(
            doctor["telegram_user_id"],
            request_text,
            inline_keyboard=_request_keyboard(consultation_id),
        )
    except Exception:
        log.exception("Failed to message doctor %s", doctor_id)
        doctors.end_consultation(consultation_id)
        await telegram._edit_message_text(chat_id, message_id, "Hiện không liên hệ được bác sĩ này.")
        return
    # The patient converted from waiter to active request; clear any waitlist
    # entry so it stops blocking promotion of the rest of the queue.
    doctors.leave_waitlist(doctor_id, chat_id)
    await telegram._edit_message_text(
        chat_id,
        message_id,
        f"Đã gửi yêu cầu đến {doctor['name']}. Vui lòng chờ bác sĩ xác nhận.",
    )


def _waitlist_text(doctor: dict | None, status: dict) -> str:
    name = doctor["name"] if doctor else "bác sĩ"
    eta = _fmt_seconds(status["estimated_wait_seconds"]) if status["estimated_wait_seconds"] else "sắp tới lượt"
    return (
        f"Bạn đang trong hàng đợi của {name}.\n"
        f"Vị trí: {status['position']}/{status['total_waiting']}\n"
        f"Thời gian chờ ước tính: {eta}."
    )


async def _handle_join_waitlist(chat_id: int | str, message_id: int, doctor_id: int) -> None:
    debt = wallet.debt_status(f"tg:{chat_id}")
    if debt["in_debt"]:
        await telegram._edit_message_text(
            chat_id,
            message_id,
            f"Bạn đang nợ {debt['debt']:,} VND. Vui lòng /paydebt trước khi vào hàng đợi.",
        )
        return
    if doctors.open_consultation_for_patient(chat_id) is not None:
        await telegram._edit_message_text(
            chat_id,
            message_id,
            "Bạn đang có phiên tư vấn đang chờ hoặc đang hoạt động. Dùng /end để kết thúc trước.",
        )
        return
    doctor = doctors.get_doctor(doctor_id)
    if doctor is None or not doctor["active"]:
        await telegram._edit_message_text(chat_id, message_id, "Không tìm thấy bác sĩ này.")
        return
    doctors.join_waitlist(doctor_id, chat_id, doctor["tier"])
    await _show_waitlist(chat_id, message_id, doctor_id)


async def _show_waitlist(chat_id: int | str, message_id: int, doctor_id: int) -> None:
    status = doctors.waitlist_status(doctor_id, chat_id)
    if status is None:
        await telegram._edit_message_text(
            chat_id,
            message_id,
            "Bạn không còn trong hàng đợi.",
        )
        return
    doctor = doctors.get_doctor(doctor_id)
    await telegram._edit_message_text(
        chat_id,
        message_id,
        _waitlist_text(doctor, status),
        _waitlist_keyboard(doctor_id),
    )


async def _handle_leave_waitlist(chat_id: int | str, message_id: int, doctor_id: int) -> None:
    doctors.leave_waitlist(doctor_id, chat_id)
    await telegram._edit_message_text(
        chat_id,
        message_id,
        "Bạn đã rời khỏi hàng đợi. Dùng /doctor để kết nối lại khi cần.",
    )


async def _handle_accept(chat_id: int | str, message_id: int, consultation_id: int) -> None:
    row = doctors.get_consultation(consultation_id)
    if row is None or row["status"] != "pending":
        await telegram.send_text(chat_id, "Yêu cầu tư vấn đã hết hạn.")
        return
    if int(chat_id) != int(row["doctor_chat_id"]):
        await telegram.send_text(chat_id, "Bạn không phải bác sĩ được chỉ định cho yêu cầu này.")
        return
    if not doctors.accept_consultation(consultation_id):
        await telegram.send_text(chat_id, "Bạn đang có phiên tư vấn khác hoặc yêu cầu đã hết hạn.")
        return
    row = doctors.get_consultation(consultation_id)
    await telegram._edit_message_reply_markup(chat_id, message_id, {"inline_keyboard": []})
    await telegram.send_text(
        row["patient_chat_id"],
        f"👨‍⚕️ {row['doctor_name']}: Xin chào, bác sĩ đã kết nối với bạn. "
        "Để tư vấn chính xác hơn, bạn cho bác sĩ biết:\n"
        "• Bạn bao nhiêu tuổi?\n"
        "• Giới tính của bạn?\n"
        "• Bạn đang gặp triệu chứng gì, bắt đầu từ khi nào, mức độ nặng nhẹ ra sao?\n\n"
        "Bạn có thể nhắn trực tiếp tại đây. Dùng /end để kết thúc.",
    )
    await telegram.send_text(
        row["doctor_chat_id"],
        "Đã kết nối với bệnh nhân. Gửi tin nhắn tại đây. Dùng /end để kết thúc.",
    )


async def _handle_decline(chat_id: int | str, message_id: int, consultation_id: int) -> None:
    row = doctors.get_consultation(consultation_id)
    if row is None or row["status"] != "pending":
        await telegram.send_text(chat_id, "Yêu cầu tư vấn đã hết hạn.")
        return
    if int(chat_id) != int(row["doctor_chat_id"]):
        await telegram.send_text(chat_id, "Bạn không phải bác sĩ được chỉ định cho yêu cầu này.")
        return
    doctors.decline_consultation(consultation_id)
    await telegram._edit_message_reply_markup(chat_id, message_id, {"inline_keyboard": []})
    await telegram.send_text(
        row["patient_chat_id"],
        f"{row['doctor_name']} đã từ chối yêu cầu tư vấn. Vui lòng chọn bác sĩ khác.",
    )
    await telegram.send_text(row["doctor_chat_id"], "Đã từ chối yêu cầu tư vấn.")


async def _handle_extend(chat_id: int | str, message_id: int, consultation_id: int) -> None:
    row = doctors.get_consultation(consultation_id)
    if row is None or row["status"] != "active":
        await telegram.send_text(chat_id, "Phiên tư vấn không còn hoạt động.")
        return
    if int(chat_id) != int(row["patient_chat_id"]):
        await telegram.send_text(chat_id, "Chỉ bệnh nhân mới gia hạn được phiên tư vấn.")
        return
    if not doctors.request_extension(consultation_id):
        await telegram.send_text(chat_id, "Không thể gia hạn phiên tư vấn này.")
        return
    renewed = doctors.get_consultation(consultation_id)
    with contextlib.suppress(Exception):
        await telegram._edit_message_reply_markup(chat_id, message_id, {"inline_keyboard": []})
    rate = renewed["rate_per_min"]
    await telegram.send_text(
        renewed["patient_chat_id"],
        f"Đã gia hạn thêm 15 phút. Phí mới: {rate:,} VND/phút.",
    )
    await telegram.send_text(
        renewed["doctor_chat_id"],
        "Bệnh nhân đã gia hạn thêm 15 phút.",
    )


async def handle_doctor_callback(callback_query: dict) -> bool:
    data = str(callback_query.get("data") or "")
    if not data.startswith(DOCTOR_PREFIX):
        return False
    callback_query_id = callback_query.get("id")
    message = callback_query.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    message_id = message.get("message_id")
    if chat_id is None:
        return True

    if data == HANDOFF_DECLINE:
        _HANDOFF_CONTEXTS.pop(str(chat_id), None)
        await telegram._edit_message_text(
            chat_id,
            int(message_id),
            "Đã bỏ qua kết nối bác sĩ.",
            {"inline_keyboard": []},
        )
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Đã bỏ qua.")
        return True
    if data == HANDOFF_ACCEPT:
        context = _handoff_context(chat_id)
        await telegram._edit_message_text(
            chat_id,
            int(message_id),
            _tier_menu_text(),
            _tier_keyboard(context.specialty_hint if context else None),
        )
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Chọn hình thức tư vấn.")
        return True
    if data == HANDOFF_OTHER:
        await telegram._edit_message_text(
            chat_id,
            int(message_id),
            _tier_menu_text(),
            _tier_keyboard(),
        )
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Chọn hình thức tư vấn.")
        return True
    if data == CANCEL_CALLBACK:
        _HANDOFF_CONTEXTS.pop(str(chat_id), None)
        await telegram._edit_message_text(
            chat_id,
            int(message_id),
            "Đã hủy kết nối bác sĩ.",
            {"inline_keyboard": []},
        )
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Đã hủy.")
        return True
    if data == "doctor:back":
        await telegram._edit_message_text(
            chat_id,
            int(message_id),
            _tier_menu_text(),
            _tier_keyboard(),
        )
        return True
    if data == "doctor:busy":
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Bác sĩ đang bận.")
        return True
    if data.startswith(TIER_PREFIX):
        payload = data[len(TIER_PREFIX):]
        tier, _, specialty_slug = payload.partition(":")
        if specialty_slug:
            await _show_doctor_list(chat_id, int(message_id), tier, specialty_slug)
        else:
            context = _handoff_context(chat_id)
            exclude = context.specialty_hint if context and context.specialty_hint else None
            await _show_specialty_list(chat_id, int(message_id), tier, exclude_specialty=exclude)
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Đã chọn nhóm.")
        return True
    if data.startswith(SPECIALTIES_PREFIX):
        tier = data[len(SPECIALTIES_PREFIX):]
        await _show_specialty_list(chat_id, int(message_id), tier)
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Đã chọn chuyên khoa.")
        return True
    if data.startswith(SPECIALTY_PREFIX):
        payload = data[len(SPECIALTY_PREFIX):]
        tier, _, specialty_slug = payload.partition(":")
        await _show_doctor_list(chat_id, int(message_id), tier, specialty_slug)
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Đã chọn chuyên khoa.")
        return True
    if data.startswith(PROFILE_PREFIX):
        payload = data[len(PROFILE_PREFIX):]
        doctor_id_text, _, rest = payload.partition(":")
        tier, _, specialty_slug = rest.partition(":")
        await _show_doctor_profile(chat_id, int(message_id), int(doctor_id_text), tier, specialty_slug)
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Thông tin bác sĩ.")
        return True
    if data.startswith(REFRESH_PREFIX):
        payload = data[len(REFRESH_PREFIX):]
        tier, _, specialty_slug = payload.partition(":")
        if specialty_slug:
            await _show_doctor_list(chat_id, int(message_id), tier, specialty_slug)
        else:
            context = _handoff_context(chat_id)
            exclude = context.specialty_hint if context and context.specialty_hint else None
            await _show_specialty_list(chat_id, int(message_id), tier, exclude_specialty=exclude)
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Đã làm mới.")
        return True
    if data.startswith(PICK_PREFIX):
        await _handle_pick(chat_id, int(message_id), int(data[len(PICK_PREFIX):]))
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Đã gửi yêu cầu.")
        return True
    if data.startswith(ACCEPT_PREFIX):
        await _handle_accept(chat_id, int(message_id), int(data[len(ACCEPT_PREFIX):]))
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Đã nhận.")
        return True
    if data.startswith(DECLINE_PREFIX):
        await _handle_decline(chat_id, int(message_id), int(data[len(DECLINE_PREFIX):]))
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Đã từ chối.")
        return True
    if data.startswith(EXTEND_PREFIX):
        await _handle_extend(chat_id, int(message_id), int(data[len(EXTEND_PREFIX):]))
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Đã gia hạn.")
        return True
    if data.startswith(WLREFRESH_PREFIX):
        await _show_waitlist(chat_id, int(message_id), int(data[len(WLREFRESH_PREFIX):]))
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Đã làm mới.")
        return True
    if data.startswith(WLLEAVE_PREFIX):
        await _handle_leave_waitlist(chat_id, int(message_id), int(data[len(WLLEAVE_PREFIX):]))
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Đã rời hàng đợi.")
        return True
    if data.startswith(WAIT_PREFIX):
        await _handle_join_waitlist(chat_id, int(message_id), int(data[len(WAIT_PREFIX):]))
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Đã vào hàng đợi.")
        return True
    return False


def _relay_target(chat_id: int | str, row: dict) -> tuple[int, str]:
    if int(chat_id) == int(row["patient_chat_id"]):
        return int(row["doctor_chat_id"]), "👤 Bệnh nhân"
    return int(row["patient_chat_id"]), f"👨‍⚕️ {row['doctor_name']}"


async def relay_message(chat_id: int | str, message: dict) -> bool:
    row = doctors.active_consultation_for_chat(chat_id)
    if row is None:
        return False
    target_chat_id, label = _relay_target(chat_id, row)

    text = message.get("text")
    if text:
        await telegram.send_text(target_chat_id, f"{label}: {text}")
        doctors.touch_activity(row["id"])
        return True

    photos = message.get("photo") or []
    if photos:
        file_id = photos[-1]["file_id"]
        caption = message.get("caption") or "ảnh"
        await telegram.send_photo(target_chat_id, file_id, f"{label}: {caption}")
        doctors.touch_activity(row["id"])
        return True

    voice = message.get("voice") or {}
    if voice.get("file_id"):
        await telegram.send_text(target_chat_id, f"{label} gửi tin nhắn thoại:")
        await telegram.send_voice(target_chat_id, voice["file_id"])
        doctors.touch_activity(row["id"])
        return True

    message_id = message.get("message_id")
    if message_id is not None:
        try:
            await telegram.send_text(target_chat_id, f"{label} gửi tin nhắn:")
            await telegram.copy_message(target_chat_id, chat_id, int(message_id))
        except Exception:
            log.exception("Không thể sao chép tin nhắn Telegram loại này.")
            await telegram.send_text(chat_id, "Loại tin nhắn này chưa được hỗ trợ trong phiên tư vấn.")
        doctors.touch_activity(row["id"])
        return True

    await telegram.send_text(chat_id, "Loại tin nhắn này chưa được hỗ trợ trong phiên tư vấn.")
    doctors.touch_activity(row["id"])
    return True


async def handle_end(chat_id: int | str) -> bool:
    row = doctors.active_consultation_for_chat(chat_id)
    if row is None:
        return False
    # Bill any unbilled whole minutes of the current paid block before closing.
    settlement = doctors.settle_block_minutes(row["id"])
    if settlement is not None:
        account_id, amount = settlement
        if amount > 0:
            wallet.debit(account_id, amount)
    doctors.end_consultation(row["id"])
    await telegram.send_text(
        row["patient_chat_id"],
        "Phiên tư vấn đã kết thúc. Bạn có thể tiếp tục hỏi bot hoặc dùng /doctor để kết nối lại.",
    )
    await telegram.send_text(row["doctor_chat_id"], "Phiên tư vấn đã kết thúc.")
    return True


def _fmt_seconds(seconds: float) -> str:
    mins = max(0, int(round(seconds / 60)))
    if mins <= 1:
        return "khoảng 1 phút"
    return f"khoảng {mins} phút"


async def _dispatch_warn(consultation: dict) -> None:
    expires_at = consultation.get("expires_at")
    remaining = _fmt_seconds((expires_at or 0) - time.time())
    await telegram.send_text(
        consultation["patient_chat_id"],
        f"Phiên tư vấn sắp hết thời gian (còn {remaining}).",
    )
    await telegram.send_text(
        consultation["doctor_chat_id"],
        f"Phiên tư vấn sắp hết thời gian (còn {remaining}).",
    )


async def _dispatch_ended_timeout(consultation: dict) -> None:
    await telegram.send_text(
        consultation["patient_chat_id"],
        "Phiên tư vấn đã kết thúc do hết thời gian. Dùng /doctor để kết nối lại.",
    )
    await telegram.send_text(
        consultation["doctor_chat_id"],
        "Phiên tư vấn đã kết thúc do hết thời gian.",
    )


def _dispatch_bill(effect: dict) -> None:
    account_id = effect.get("account_id")
    amount = effect.get("amount") or 0
    if account_id and amount > 0:
        wallet.debit(account_id, amount)


async def _dispatch_extend_offer(effect: dict) -> None:
    consultation = effect["consultation"]
    next_rate = effect.get("next_rate") or 0
    remaining = _fmt_seconds((consultation.get("expires_at") or 0) - time.time())
    await telegram.send_text(
        consultation["patient_chat_id"],
        (
            f"Phiên tư vấn sắp hết thời gian (còn {remaining}).\n"
            f"Gia hạn thêm 15 phút với phí {next_rate:,} VND/phút?"
        ),
        inline_keyboard=_extend_keyboard(consultation["id"]),
    )
    await telegram.send_text(
        consultation["doctor_chat_id"],
        f"Phiên tư vấn sắp hết thời gian (còn {remaining}).",
    )


async def _dispatch_waitlist_offer(entry: dict) -> None:
    doctor = doctors.get_doctor(entry["doctor_id"])
    name = doctor["name"] if doctor else "bác sĩ"
    keyboard = {
        "inline_keyboard": [[
            {"text": "✅ Kết nối ngay", "callback_data": f"doctor:pick:{entry['doctor_id']}"},
            {"text": "🚪 Rời hàng đợi", "callback_data": f"doctor:wlleave:{entry['doctor_id']}"},
        ]]
    }
    await telegram.send_text(
        entry["patient_chat_id"],
        f"{name} đã rảnh! Bấm Kết nối ngay để bắt đầu (lời mời có hạn vài phút).",
        inline_keyboard=keyboard,
    )


async def _dispatch_waitlist_offer_reminder(entry: dict) -> None:
    doctor = doctors.get_doctor(entry["doctor_id"])
    name = doctor["name"] if doctor else "bác sĩ"
    keyboard = {
        "inline_keyboard": [[
            {"text": "✅ Kết nối ngay", "callback_data": f"doctor:pick:{entry['doctor_id']}"},
            {"text": "🚪 Rời hàng đợi", "callback_data": f"doctor:wlleave:{entry['doctor_id']}"},
        ]]
    }
    await telegram.send_text(
        entry["patient_chat_id"],
        f"Nhắc bạn: {name} đang rảnh. Bấm Kết nối ngay trước khi lời mời hết hạn.",
        inline_keyboard=keyboard,
    )


async def _dispatch_waitlist_offer_expired(entry: dict) -> None:
    await telegram.send_text(
        entry["patient_chat_id"],
        "Lời mời kết nối đã hết hạn. Dùng /doctor để thử lại nếu bạn vẫn cần.",
    )


async def run_session_tick() -> None:
    """Process one time-driven sweep of doctor sessions.

    Called periodically by the app lifespan ticker. Pulls effects from the pure
    storage sweeps (sessions + waitlist) and performs the corresponding
    Telegram/wallet I/O. Each effect is isolated so one failure does not abort
    the rest of the tick.
    """
    try:
        effects = doctors.sweep_sessions()
    except Exception:
        log.exception("sweep_sessions failed")
        effects = []
    for effect in effects:
        kind = effect.get("kind")
        consultation = effect.get("consultation")
        try:
            if kind == "bill":
                _dispatch_bill(effect)
            elif kind == "warn" and consultation:
                await _dispatch_warn(consultation)
            elif kind == "extend_offer" and consultation:
                await _dispatch_extend_offer(effect)
            elif kind == "ended_timeout" and consultation:
                await _dispatch_ended_timeout(consultation)
        except Exception:
            log.exception("Failed to dispatch session effect kind=%s", kind)

    try:
        wl_effects = doctors.sweep_waitlist()
    except Exception:
        log.exception("sweep_waitlist failed")
        wl_effects = []
    for effect in wl_effects:
        kind = effect.get("kind")
        entry = effect.get("entry")
        try:
            if kind == "offer" and entry:
                await _dispatch_waitlist_offer(entry)
            elif kind == "offer_reminder" and entry:
                await _dispatch_waitlist_offer_reminder(entry)
            elif kind == "offer_expired" and entry:
                await _dispatch_waitlist_offer_expired(entry)
        except Exception:
            log.exception("Failed to dispatch waitlist effect kind=%s", kind)
