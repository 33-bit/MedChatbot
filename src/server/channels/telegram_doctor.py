"""
telegram_doctor.py
------------------
Doctor-connect Telegram handlers: menu, selection, accept/decline, relay.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import time
import unicodedata
from dataclasses import dataclass

from src.chat.profile import (
    build_medical_profile,
    ensure_subject,
    get_subject,
    list_subjects,
    migrate_owner_key,
)
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
SHARE_CONFIRM_PREFIX = "doctor:share:confirm:"
SHARE_CHOOSE_PREFIX = "doctor:share:choose:"
SHARE_SUBJECT_PREFIX = "doctor:share:subject:"
ACCEPT_PREFIX = "doctor:accept:"
DECLINE_PREFIX = "doctor:decline:"
REFRESH_PREFIX = "doctor:refresh:"
EXTEND_PREFIX = "doctor:extend:"
WAIT_PREFIX = "doctor:wait:"
WLREFRESH_PREFIX = "doctor:wlrefresh:"
WLLEAVE_PREFIX = "doctor:wlleave:"
ADMIN_PREFIX = "doctor:admin:"
ADMIN_MENU_CALLBACK = "doctor:admin:menu"
ADMIN_LIST_CALLBACK = "doctor:admin:list"
ADMIN_ADD_CALLBACK = "doctor:admin:add"
ADMIN_EDIT_CALLBACK = "doctor:admin:edit"
ADMIN_DELETE_CALLBACK = "doctor:admin:delete"
ADMIN_TIER_PREFIX = "doctor:admin:tier:"
ADMIN_SPECIALTY_PREFIX = "doctor:admin:specialty:"
ADMIN_PROFILE_PREFIX = "doctor:admin:profile:"
ADMIN_FIELDS_PREFIX = "doctor:admin:fields:"
ADMIN_FIELD_PREFIX = "doctor:admin:field:"
ADMIN_VALUE_PREFIX = "doctor:admin:value:"
ADMIN_CUSTOM_PREFIX = "doctor:admin:custom:"
ADMIN_CANCEL_EDIT_CALLBACK = "doctor:admin:cancel_edit"
ADMIN_DELETE_PREFIX = "doctor:admin:delete:"
ADMIN_DELETE_CONFIRM_PREFIX = "doctor:admin:delete_confirm:"
HANDOFF_ACCEPT = "doctor:handoff:accept"
HANDOFF_OTHER = "doctor:handoff:other"
HANDOFF_DECLINE = "doctor:handoff:decline"
CANCEL_CALLBACK = "doctor:cancel"
HANDOFF_TTL_SECONDS = 15 * 60

_RELATIONSHIP_LABELS = {
    "self": "Bạn",
    "father": "Bố",
    "mother": "Mẹ",
    "child": "Con",
    "spouse": "Vợ/Chồng",
    "relative": "Người thân",
}

_GENDER_LABELS = {
    "female": "Nữ",
    "male": "Nam",
    "other": "Khác",
    "unknown": "Không muốn trả lời",
}


@dataclass
class DoctorHandoffContext:
    question: str
    bot_answer: str
    summary: str
    specialty_hint: str | None
    created_at: float


@dataclass(frozen=True)
class AdminDoctorPendingEdit:
    user_id: int
    doctor_id: int
    field: str


_HANDOFF_CONTEXTS: dict[str, DoctorHandoffContext] = {}
_ADMIN_PENDING_EDITS: dict[str, AdminDoctorPendingEdit] = {}

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
            {"text": f"🩺 Gặp bác sĩ khoa {specialty}", "callback_data": HANDOFF_ACCEPT},
            {"text": "📁 Chọn khoa khác", "callback_data": HANDOFF_OTHER},
        ]
    else:
        first_row = [
            {"text": "📁 Chọn chuyên khoa", "callback_data": HANDOFF_OTHER},
        ]
    return {
        "inline_keyboard": [
            first_row,
            [{"text": "Không cần bác sĩ", "callback_data": HANDOFF_DECLINE}],
        ]
    }


def _tier_keyboard(specialty: str | None = None) -> dict:
    suffix = f":{_specialty_slug(specialty)}" if specialty else ""
    return {
        "inline_keyboard": [
            [{"text": "🎁 Tư vấn miễn phí", "callback_data": f"doctor:tier:free{suffix}"}],
            [{"text": "💳 Tư vấn trả phí", "callback_data": f"doctor:tier:paid{suffix}"}],
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


def _profile_owner_key(
    chat_id: int | str,
    telegram_user_id: int | str,
) -> str | None:
    from src.chat.security.identity import (
        derive_previous_owner_keys,
        derive_request_identity,
    )

    identity = derive_request_identity(
        "telegram",
        telegram_user_id,
        str(chat_id),
    )
    if not identity.owner_key:
        return None
    try:
        migrate_owner_key(
            identity.owner_key,
            derive_previous_owner_keys("telegram", telegram_user_id),
        )
    except Exception:
        log.exception("Doctor handoff owner-key rotation failed")
        return None
    return identity.owner_key


def _default_profile_subject_id(
    chat_id: int | str,
    telegram_user_id: int | str,
    owner_key: str,
) -> str:
    from src.chat.context.context_store import load_conversation_context
    from src.chat.security.identity import derive_request_identity

    identity = derive_request_identity("telegram", telegram_user_id, str(chat_id))
    state, _, available = load_conversation_context(identity.session_key, owner_key)
    if available and state.active_subject_id:
        if get_subject(owner_key, state.active_subject_id) is not None:
            return state.active_subject_id
        relationship_matches = [
            subject for subject in list_subjects(owner_key)
            if subject.get("relationship") == state.active_subject_id
        ]
        if len(relationship_matches) == 1:
            return str(relationship_matches[0]["subject_id"])
    return "self"


def _subject_display_name(subject: dict) -> str:
    name = str(subject.get("display_name") or "").strip()
    relationship = str(subject.get("relationship") or "").strip()
    return name or _RELATIONSHIP_LABELS.get(relationship, relationship or "Người được tư vấn")


def _profile_snapshot(owner_key: str, subject_id: str) -> str | None:
    subject = get_subject(owner_key, subject_id)
    if subject is None:
        return None
    profile = build_medical_profile(owner_key, subject_id)
    demo = profile.demographics
    relationship = _RELATIONSHIP_LABELS.get(
        demo.relationship or "",
        demo.relationship or "Chưa rõ",
    )
    lines = [
        f"Người được tư vấn: {_subject_display_name(subject)} ({relationship})",
        f"Tuổi: {demo.age if demo.age is not None else 'Chưa có thông tin'}",
        f"Giới tính: {_GENDER_LABELS.get(demo.gender or '', demo.gender or 'Chưa có thông tin')}",
    ]

    def add_section(
        heading: str,
        section: str,
        entries: list,
        formatter,
    ) -> None:
        if entries:
            lines.append(f"{heading}:")
            for entry in entries[:8]:
                confirmation = "đã xác nhận" if entry.confirmed else "chưa xác nhận"
                lines.append(f"- {formatter(entry)} ({confirmation})")
            if len(entries) > 8:
                lines.append(f"- Và {len(entries) - 8} mục khác")
            return
        state = profile.section_states.get(section)
        if state and state.status == "none_known":
            lines.append(f"{heading}: Đã xác nhận hiện không có")
        else:
            lines.append(f"{heading}: Chưa có thông tin")

    add_section(
        "Vấn đề sức khỏe",
        "problems",
        profile.problems,
        lambda entry: entry.condition,
    )
    add_section(
        "Dị ứng",
        "allergies",
        profile.allergies,
        lambda entry: (
            entry.agent
            + (f"; phản ứng: {', '.join(entry.reactions)}" if entry.reactions else "")
        ),
    )
    add_section(
        "Thuốc đang dùng",
        "medications",
        profile.medications,
        lambda entry: (
            entry.medication
            + (f"; liều: {entry.dosage_text}" if entry.dosage_text else "")
            + (f"; tần suất: {entry.frequency}" if entry.frequency else "")
        ),
    )
    pregnancy_labels = {
        "pregnant": "Đang mang thai",
        "not_pregnant": "Không mang thai",
        "true": "Đang mang thai",
        "false": "Không mang thai",
        "unknown": "Chưa rõ",
        "current": "Đang áp dụng",
    }
    add_section(
        "Thai kỳ",
        "pregnancy",
        profile.pregnancy,
        lambda entry: pregnancy_labels.get(entry.status.casefold(), entry.status),
    )
    snapshot = "\n".join(lines)
    if len(snapshot) > 2800:
        snapshot = snapshot[:2760].rstrip() + "\n- …Một số mục khác không hiển thị trong bản xem trước."
    return snapshot


def _profile_share_keyboard(doctor_id: int, subject_id: str) -> dict:
    return {
        "inline_keyboard": [
            [{
                "text": "✅ Đồng ý gửi và kết nối",
                "callback_data": f"{SHARE_CONFIRM_PREFIX}{doctor_id}:{subject_id}",
            }],
            [{
                "text": "👪 Đổi người được tư vấn",
                "callback_data": f"{SHARE_CHOOSE_PREFIX}{doctor_id}",
            }],
            [{"text": "❌ Hủy", "callback_data": CANCEL_CALLBACK}],
        ]
    }


async def _show_profile_share_confirmation(
    chat_id: int | str,
    message_id: int,
    telegram_user_id: int | str,
    doctor_id: int,
    subject_id: str,
) -> None:
    owner_key = _profile_owner_key(chat_id, telegram_user_id)
    if owner_key is None:
        await telegram._edit_message_text(
            chat_id,
            message_id,
            "Bot chưa thể mở hồ sơ sức khỏe lúc này. Vui lòng thử lại sau.",
        )
        return
    ensure_subject(owner_key, "self", relationship="self", display_name=None)
    snapshot = _profile_snapshot(owner_key, subject_id)
    doctor = doctors.get_doctor(doctor_id)
    if snapshot is None or doctor is None or not doctor["active"]:
        await telegram._edit_message_text(
            chat_id,
            message_id,
            "Không tìm thấy hồ sơ hoặc bác sĩ đã chọn. Vui lòng thử lại.",
        )
        return
    text = (
        "🔒 Xác nhận chia sẻ hồ sơ\n\n"
        f"Bác sĩ nhận thông tin: {doctor['name']}\n\n"
        f"{snapshot}\n\n"
        "Nếu bạn đồng ý, bot sẽ gửi đúng bản tóm tắt trên và phần hội thoại gần đây "
        "cho bác sĩ. Hãy chọn “Đổi người được tư vấn” nếu hồ sơ chưa đúng."
    )
    await telegram._edit_message_text(
        chat_id,
        message_id,
        text,
        _profile_share_keyboard(doctor_id, subject_id),
    )


async def _show_profile_subject_list(
    chat_id: int | str,
    message_id: int,
    telegram_user_id: int | str,
    doctor_id: int,
) -> None:
    owner_key = _profile_owner_key(chat_id, telegram_user_id)
    if owner_key is None:
        await telegram._edit_message_text(
            chat_id,
            message_id,
            "Bot chưa thể mở hồ sơ sức khỏe lúc này. Vui lòng thử lại sau.",
        )
        return
    ensure_subject(owner_key, "self", relationship="self", display_name=None)
    rows = []
    for subject in list_subjects(owner_key):
        subject_id = str(subject["subject_id"])
        relationship = _RELATIONSHIP_LABELS.get(
            str(subject.get("relationship") or ""),
            str(subject.get("relationship") or "Người thân"),
        )
        rows.append([{
            "text": f"👤 {_subject_display_name(subject)} · {relationship}",
            "callback_data": f"{SHARE_SUBJECT_PREFIX}{doctor_id}:{subject_id}",
        }])
    rows.append([{"text": "❌ Hủy", "callback_data": CANCEL_CALLBACK}])
    await telegram._edit_message_text(
        chat_id,
        message_id,
        "Ai là người cần bác sĩ tư vấn? Hãy chọn đúng hồ sơ trước khi gửi.",
        {"inline_keyboard": rows},
    )


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


ADMIN_MENU_TEXT = "🛠 Quản lý bác sĩ\n\nChọn tác vụ bên dưới."

ADMIN_ADD_USAGE = (
    "➕ Thêm bác sĩ\n\n"
    "Dùng lệnh:\n"
    "`/admin_doctor_add Tên | Chuyên khoa | free|paid | Giá/phút | Telegram user id | "
    "Học vị | Số năm kinh nghiệm | Bệnh viện | Giới thiệu`\n\n"
    "5 trường đầu là bắt buộc; các trường còn lại có thể bỏ trống."
)

ADMIN_EDIT_USAGE = (
    "✏️ Sửa bác sĩ\n\n"
    "Chọn bác sĩ từ danh sách, chọn trường cần sửa, rồi bấm giá trị gợi ý "
    "hoặc chọn nhập giá trị mới."
)

_ADMIN_ACTION_TITLES = {
    "view": "Danh sách bác sĩ",
    "edit": "Chọn bác sĩ cần sửa",
    "delete": "Chọn bác sĩ cần xóa",
}

_ADMIN_FIELD_LABELS = {
    "name": "Tên",
    "specialty": "Chuyên khoa",
    "tier": "Nhóm tư vấn",
    "price": "Giá/phút",
    "telegram_user_id": "Telegram user id",
    "active": "Trạng thái",
    "degree": "Học vị",
    "experience_years": "Kinh nghiệm",
    "hospital": "Bệnh viện",
    "bio": "Giới thiệu",
}

_ADMIN_DEGREE_OPTIONS = (
    ("ck1", "Bác sĩ chuyên khoa I"),
    ("msc", "Thạc sĩ, Bác sĩ"),
    ("ck2", "Bác sĩ chuyên khoa II"),
    ("phd", "Tiến sĩ, Bác sĩ"),
)

_ADMIN_HOSPITAL_OPTIONS = (
    ("bachmai", "Bệnh viện Bạch Mai"),
    ("daihocy", "Bệnh viện Đại học Y Hà Nội"),
    ("108", "Bệnh viện Trung ương Quân đội 108"),
    ("e", "Bệnh viện E"),
    ("vietduc", "Bệnh viện Hữu nghị Việt Đức"),
)


def _admin_doctors_keyboard() -> dict:
    return {
        "inline_keyboard": [
            [{"text": "📋 Danh sách bác sĩ", "callback_data": ADMIN_LIST_CALLBACK}],
            [{"text": "➕ Thêm bác sĩ", "callback_data": ADMIN_ADD_CALLBACK}],
            [{"text": "✏️ Sửa bác sĩ", "callback_data": ADMIN_EDIT_CALLBACK}],
            [{"text": "🗑 Xóa bác sĩ", "callback_data": ADMIN_DELETE_CALLBACK}],
            [{"text": "⬅️ Menu chính", "callback_data": "menu:main"}],
        ]
    }


def _admin_tier_keyboard(action: str) -> dict:
    return {
        "inline_keyboard": [
            [{"text": "🎁 Miễn phí", "callback_data": f"{ADMIN_TIER_PREFIX}{action}:free"}],
            [{"text": "💳 Trả phí", "callback_data": f"{ADMIN_TIER_PREFIX}{action}:paid"}],
            [{"text": "⬅️ Quay lại quản lý bác sĩ", "callback_data": ADMIN_MENU_CALLBACK}],
        ]
    }


def _admin_specialty_keyboard(action: str, tier: str, specialties: list[str]) -> dict:
    rows = [
        [{
            "text": specialty,
            "callback_data": f"{ADMIN_SPECIALTY_PREFIX}{action}:{tier}:{_specialty_slug(specialty)}",
        }]
        for specialty in specialties
    ]
    rows.append([{"text": "⬅️ Chọn nhóm tư vấn", "callback_data": _admin_action_root_callback(action)}])
    rows.append([{"text": "⬅️ Quản lý bác sĩ", "callback_data": ADMIN_MENU_CALLBACK}])
    return {"inline_keyboard": rows}


def _admin_doctor_callback(action: str, doctor_id: int) -> str:
    if action == "edit":
        return f"{ADMIN_FIELDS_PREFIX}{doctor_id}"
    if action == "delete":
        return f"{ADMIN_DELETE_PREFIX}{doctor_id}"
    return f"{ADMIN_PROFILE_PREFIX}{doctor_id}"


def _admin_doctor_list_keyboard(action: str, rows: list[dict], tier: str) -> dict:
    buttons = [
        [{
            "text": _admin_doctor_button_text(row),
            "callback_data": _admin_doctor_callback(action, int(row["id"])),
        }]
        for row in rows[:40]
    ]
    buttons.append([{"text": "⬅️ Chọn chuyên khoa", "callback_data": f"{ADMIN_TIER_PREFIX}{action}:{tier}"}])
    buttons.append([{"text": "⬅️ Quản lý bác sĩ", "callback_data": ADMIN_MENU_CALLBACK}])
    return {"inline_keyboard": buttons}


def _admin_back_keyboard() -> dict:
    return {
        "inline_keyboard": [
            [{"text": "⬅️ Quay lại quản lý bác sĩ", "callback_data": ADMIN_MENU_CALLBACK}]
        ]
    }


def _admin_delete_confirm_keyboard(doctor_id: int) -> dict:
    return {
        "inline_keyboard": [
            [
                {"text": "🗑 Xác nhận xóa", "callback_data": f"{ADMIN_DELETE_CONFIRM_PREFIX}{doctor_id}"},
                {"text": "❌ Hủy", "callback_data": ADMIN_DELETE_CALLBACK},
            ]
        ]
    }


def _is_admin_user(user_id: int | None) -> bool:
    return telegram._is_admin(user_id)


def _admin_doctor_summary(row: dict) -> str:
    status = "đang hoạt động" if row["active"] else "đã xóa"
    available = "rảnh" if row["available"] else "bận/ẩn"
    price = _vnd(int(row.get("price") or 0))
    return (
        f"#{row['id']} - {row['name']} | {row.get('specialty') or 'Chưa cập nhật'} | "
        f"{row['tier']} | {price}đ/phút | tg:{row['telegram_user_id']} | "
        f"{status}, {available}"
    )


def _admin_action_root_callback(action: str) -> str:
    if action == "edit":
        return ADMIN_EDIT_CALLBACK
    if action == "delete":
        return ADMIN_DELETE_CALLBACK
    return ADMIN_LIST_CALLBACK


def _admin_doctor_button_text(row: dict) -> str:
    status = "ẩn" if not row["active"] else "bận" if not row["available"] else "rảnh"
    return f"#{row['id']} {row['name']} ({status})"


def _admin_doctor_detail_text(row: dict) -> str:
    price = _vnd(int(row.get("price") or 0))
    active = "Đang hiển thị" if row["active"] else "Đã ẩn khỏi danh sách"
    available = "Rảnh" if row["available"] else "Bận/không khả dụng"
    experience = row.get("experience_years")
    experience_text = f"{experience} năm" if experience else "Chưa cập nhật"
    return (
        f"👨‍⚕️ #{row['id']} {row['name']}\n"
        f"Chuyên khoa: {row.get('specialty') or 'Chưa cập nhật'}\n"
        f"Nhóm tư vấn: {row['tier']}\n"
        f"Giá/phút: {price}đ\n"
        f"Telegram user id: {row['telegram_user_id']}\n"
        f"Trạng thái: {active}\n"
        f"Khả dụng: {available}\n"
        f"Học vị: {row.get('degree') or 'Chưa cập nhật'}\n"
        f"Kinh nghiệm: {experience_text}\n"
        f"Công tác: {row.get('hospital') or 'Chưa cập nhật'}\n\n"
        f"Giới thiệu:\n{row.get('bio') or 'Chưa cập nhật'}"
    )


def _admin_doctor_profile_keyboard(doctor_id: int) -> dict:
    return {
        "inline_keyboard": [
            [{"text": "✏️ Sửa thông tin", "callback_data": f"{ADMIN_FIELDS_PREFIX}{doctor_id}"}],
            [{"text": "🗑 Xóa bác sĩ", "callback_data": f"{ADMIN_DELETE_PREFIX}{doctor_id}"}],
            [{"text": "⬅️ Danh sách bác sĩ", "callback_data": ADMIN_LIST_CALLBACK}],
            [{"text": "⬅️ Quản lý bác sĩ", "callback_data": ADMIN_MENU_CALLBACK}],
        ]
    }


def _admin_field_value(row: dict, field: str) -> str:
    value = row.get(field)
    if field == "price":
        return f"{_vnd(int(value or 0))}đ/phút"
    if field == "active":
        return "Đang hiển thị" if value else "Đã ẩn"
    if field == "experience_years":
        return f"{value} năm" if value else "Chưa cập nhật"
    return str(value or "Chưa cập nhật")


def _admin_fields_text(row: dict) -> str:
    lines = [f"✏️ Sửa bác sĩ #{row['id']} - {row['name']}", "", "Chọn trường cần sửa:"]
    for field, label in _ADMIN_FIELD_LABELS.items():
        lines.append(f"{label}: {_admin_field_value(row, field)}")
    return "\n".join(lines)


def _admin_fields_keyboard(doctor_id: int) -> dict:
    rows = [
        [{"text": f"✏️ {label}", "callback_data": f"{ADMIN_FIELD_PREFIX}{doctor_id}:{field}"}]
        for field, label in _ADMIN_FIELD_LABELS.items()
    ]
    rows.append([{"text": "⬅️ Hồ sơ bác sĩ", "callback_data": f"{ADMIN_PROFILE_PREFIX}{doctor_id}"}])
    rows.append([{"text": "⬅️ Quản lý bác sĩ", "callback_data": ADMIN_MENU_CALLBACK}])
    return {"inline_keyboard": rows}


def _admin_custom_value_keyboard(doctor_id: int, field: str) -> dict:
    return {
        "inline_keyboard": [
            [{"text": "✍️ Nhập giá trị mới", "callback_data": f"{ADMIN_CUSTOM_PREFIX}{doctor_id}:{field}"}],
            [{"text": "⬅️ Chọn trường khác", "callback_data": f"{ADMIN_FIELDS_PREFIX}{doctor_id}"}],
        ]
    }


def _admin_field_options_keyboard(doctor_id: int, field: str) -> dict:
    rows: list[list[dict]] = []
    if field == "tier":
        rows.extend([
            [{"text": "🎁 Miễn phí", "callback_data": f"{ADMIN_VALUE_PREFIX}{doctor_id}:tier:free"}],
            [{"text": "💳 Trả phí", "callback_data": f"{ADMIN_VALUE_PREFIX}{doctor_id}:tier:paid"}],
        ])
    elif field == "active":
        rows.extend([
            [{"text": "✅ Hiển thị", "callback_data": f"{ADMIN_VALUE_PREFIX}{doctor_id}:active:1"}],
            [{"text": "🙈 Ẩn khỏi danh sách", "callback_data": f"{ADMIN_VALUE_PREFIX}{doctor_id}:active:0"}],
        ])
    elif field == "specialty":
        rows.extend([
            [{
                "text": specialty,
                "callback_data": f"{ADMIN_VALUE_PREFIX}{doctor_id}:specialty:{_specialty_slug(specialty)}",
            }]
            for specialty in SPECIALTIES
        ])
    elif field == "price":
        for amount in (0, PAID_RATE_PER_MIN, 5_000, 10_000, 20_000):
            rows.append([{
                "text": f"{_vnd(amount)}đ/phút",
                "callback_data": f"{ADMIN_VALUE_PREFIX}{doctor_id}:price:{amount}",
            }])
    elif field == "experience_years":
        for years in (1, 3, 5, 10, 15, 20):
            rows.append([{
                "text": f"{years} năm",
                "callback_data": f"{ADMIN_VALUE_PREFIX}{doctor_id}:experience_years:{years}",
            }])
    elif field == "degree":
        for code, label in _ADMIN_DEGREE_OPTIONS:
            rows.append([{
                "text": label,
                "callback_data": f"{ADMIN_VALUE_PREFIX}{doctor_id}:degree:{code}",
            }])
    elif field == "hospital":
        for code, label in _ADMIN_HOSPITAL_OPTIONS:
            rows.append([{
                "text": label,
                "callback_data": f"{ADMIN_VALUE_PREFIX}{doctor_id}:hospital:{code}",
            }])
    rows.append([{"text": "✍️ Nhập giá trị khác", "callback_data": f"{ADMIN_CUSTOM_PREFIX}{doctor_id}:{field}"}])
    rows.append([{"text": "⬅️ Chọn trường khác", "callback_data": f"{ADMIN_FIELDS_PREFIX}{doctor_id}"}])
    return {"inline_keyboard": rows}


def _admin_field_prompt(row: dict, field: str) -> str:
    label = _ADMIN_FIELD_LABELS[field]
    return (
        f"Đang sửa: {label}\n"
        f"Giá trị hiện tại: {_admin_field_value(row, field)}\n\n"
        "Chọn một giá trị bên dưới hoặc nhập giá trị mới."
    )


async def _admin_edit_or_send(
    chat_id: int | str,
    message_id: int | None,
    text: str,
    inline_keyboard: dict | None = None,
) -> None:
    if message_id is None:
        await telegram.send_text(chat_id, text, inline_keyboard=inline_keyboard)
        return
    await telegram._edit_message_text(chat_id, int(message_id), text, inline_keyboard)


def _command_word(text: str) -> str:
    return (text.strip().split(maxsplit=1)[0] if text.strip() else "").split("@", 1)[0].lower()


def _parse_amount(text: str) -> int:
    cleaned = text.strip().replace(".", "").replace(",", "")
    if not cleaned or not cleaned.isdigit():
        raise ValueError("Giá phải là số nguyên không âm.")
    return int(cleaned)


def _parse_positive_int(text: str, field_name: str) -> int:
    value = text.strip()
    if not value.isdigit():
        raise ValueError(f"{field_name} phải là số nguyên.")
    return int(value)


def _parse_tier(value: str) -> str:
    tier = value.strip().lower()
    if tier not in {"free", "paid"}:
        raise ValueError("Tier phải là free hoặc paid.")
    return tier


def _optional_text(value: str) -> str | None:
    value = value.strip()
    return value or None


def _optional_int(value: str, field_name: str) -> int | None:
    if not value.strip():
        return None
    return _parse_positive_int(value, field_name)


def _parse_active(value: str) -> bool:
    normalized = _text_key(value)
    if normalized in {"1", "true", "yes", "active", "on", "bat", "dang hoat dong"}:
        return True
    if normalized in {"0", "false", "no", "inactive", "off", "tat", "da xoa"}:
        return False
    raise ValueError("active phải là true/false hoặc 1/0.")


def _parse_admin_add_payload(text: str) -> dict:
    _, _, payload = text.strip().partition(" ")
    parts = [part.strip() for part in payload.split("|", 8)]
    if len(parts) < 5:
        raise ValueError("Thiếu thông tin. Cần ít nhất: Tên | Chuyên khoa | tier | Giá | Telegram user id.")
    parts.extend([""] * (9 - len(parts)))
    name = parts[0]
    if not name:
        raise ValueError("Tên bác sĩ không được để trống.")
    price = _parse_amount(parts[3])
    telegram_user_id = _parse_positive_int(parts[4], "Telegram user id")
    return {
        "name": name,
        "specialty": _optional_text(parts[1]),
        "tier": _parse_tier(parts[2]),
        "price": price,
        "telegram_user_id": telegram_user_id,
        "degree": _optional_text(parts[5]),
        "experience_years": _optional_int(parts[6], "Số năm kinh nghiệm"),
        "hospital": _optional_text(parts[7]),
        "bio": _optional_text(parts[8]),
    }


_ADMIN_EDIT_FIELD_ALIASES = {
    "name": "name",
    "specialty": "specialty",
    "tier": "tier",
    "price": "price",
    "telegram_user_id": "telegram_user_id",
    "telegram": "telegram_user_id",
    "tg": "telegram_user_id",
    "active": "active",
    "degree": "degree",
    "experience_years": "experience_years",
    "experience": "experience_years",
    "hospital": "hospital",
    "bio": "bio",
}


def _parse_admin_edit_value(field: str, value: str):
    if field == "tier":
        return _parse_tier(value)
    if field == "price":
        return _parse_amount(value)
    if field == "telegram_user_id":
        return _parse_positive_int(value, "Telegram user id")
    if field == "experience_years":
        return _parse_positive_int(value, "Số năm kinh nghiệm")
    if field == "active":
        return _parse_active(value)
    value = value.strip()
    if not value:
        raise ValueError(f"{field} không được để trống.")
    return value


def _admin_pending_key(chat_id: int | str, user_id: int | None) -> str:
    return f"{chat_id}:{user_id}"


def _admin_specialty_from_value(value: str) -> str:
    specialty = _specialty_from_slug([{"specialty": item} for item in SPECIALTIES], value)
    if specialty is None:
        raise ValueError("Không tìm thấy chuyên khoa này.")
    return specialty


def _admin_code_value(field: str, value: str):
    if field == "specialty":
        return _admin_specialty_from_value(value)
    if field == "degree":
        options = dict(_ADMIN_DEGREE_OPTIONS)
        if value not in options:
            raise ValueError("Không tìm thấy học vị này.")
        return options[value]
    if field == "hospital":
        options = dict(_ADMIN_HOSPITAL_OPTIONS)
        if value not in options:
            raise ValueError("Không tìm thấy bệnh viện này.")
        return options[value]
    return _parse_admin_edit_value(field, value)


def _apply_admin_doctor_update(doctor_id: int, field: str, value) -> dict:
    if field not in _ADMIN_FIELD_LABELS:
        raise ValueError("Trường sửa không hợp lệ.")
    if not doctors.update_doctor(doctor_id, **{field: value}):
        raise ValueError(f"Không tìm thấy bác sĩ #{doctor_id}.")
    doctor = doctors.get_doctor(doctor_id)
    if doctor is None:
        raise ValueError(f"Không tìm thấy bác sĩ #{doctor_id}.")
    return doctor


async def intercept_pending_admin_doctor_edit(
    chat_id: int | str,
    text: str,
    user_id: int | None,
) -> bool:
    pending = _ADMIN_PENDING_EDITS.get(_admin_pending_key(chat_id, user_id))
    if pending is None:
        return False
    if not _is_admin_user(user_id) or user_id != pending.user_id:
        return False
    try:
        value = _parse_admin_edit_value(pending.field, text)
        doctor = _apply_admin_doctor_update(pending.doctor_id, pending.field, value)
    except sqlite3.IntegrityError:
        await telegram.send_text(chat_id, "Không lưu được: Telegram user id đã được dùng cho bác sĩ khác.")
        return True
    except ValueError as exc:
        await telegram.send_text(chat_id, str(exc), inline_keyboard=_admin_custom_value_keyboard(
            pending.doctor_id, pending.field,
        ))
        return True
    _ADMIN_PENDING_EDITS.pop(_admin_pending_key(chat_id, user_id), None)
    await telegram.send_text(
        chat_id,
        f"Đã cập nhật {_ADMIN_FIELD_LABELS[pending.field]}.\n\n{_admin_doctor_detail_text(doctor)}",
        inline_keyboard=_admin_doctor_profile_keyboard(pending.doctor_id),
    )
    return True


def _parse_admin_edit_payload(text: str) -> tuple[int, dict]:
    _, _, payload = text.strip().partition(" ")
    doctor_id_text, _, fields_text = payload.strip().partition(" ")
    if not doctor_id_text or not fields_text:
        raise ValueError("Cú pháp: /admin_doctor_edit <id> field=value | field=value")
    doctor_id = _parse_positive_int(doctor_id_text, "Doctor id")
    changes = {}
    for raw_item in fields_text.split("|"):
        key, sep, value = raw_item.strip().partition("=")
        if not sep:
            raise ValueError("Mỗi trường sửa phải có dạng field=value.")
        field = _ADMIN_EDIT_FIELD_ALIASES.get(key.strip().lower())
        if field is None:
            raise ValueError(f"Không hỗ trợ field: {key.strip()}.")
        changes[field] = _parse_admin_edit_value(field, value)
    if not changes:
        raise ValueError("Chưa có trường nào để sửa.")
    return doctor_id, changes


def _parse_admin_delete_payload(text: str) -> int:
    _, _, payload = text.strip().partition(" ")
    doctor_id_text = payload.strip().split(maxsplit=1)[0] if payload.strip() else ""
    if not doctor_id_text:
        raise ValueError("Cú pháp: /admin_doctor_delete <id>")
    return _parse_positive_int(doctor_id_text, "Doctor id")


async def handle_admin_doctors_command(chat_id: int | str, user_id: int | None) -> bool:
    if not _is_admin_user(user_id):
        await telegram.send_text(chat_id, "Bạn không có quyền dùng chức năng này.")
        return True
    await telegram.send_text(chat_id, ADMIN_MENU_TEXT, inline_keyboard=_admin_doctors_keyboard())
    return True


async def handle_admin_doctor_command(
    chat_id: int | str,
    text: str,
    user_id: int | None,
) -> bool:
    if not _is_admin_user(user_id):
        await telegram.send_text(chat_id, "Bạn không có quyền dùng chức năng này.")
        return True
    cmd = _command_word(text)
    try:
        if cmd == "/admin_doctor_add":
            doctor_id = doctors.create_doctor(**_parse_admin_add_payload(text))
            await telegram.send_text(
                chat_id,
                f"Đã thêm bác sĩ #{doctor_id}.",
                inline_keyboard=_admin_doctors_keyboard(),
            )
            return True
        if cmd == "/admin_doctor_edit":
            doctor_id, changes = _parse_admin_edit_payload(text)
            if not doctors.update_doctor(doctor_id, **changes):
                await telegram.send_text(chat_id, f"Không tìm thấy bác sĩ #{doctor_id}.")
                return True
            await telegram.send_text(
                chat_id,
                f"Đã cập nhật bác sĩ #{doctor_id}.",
                inline_keyboard=_admin_doctors_keyboard(),
            )
            return True
        if cmd == "/admin_doctor_delete":
            doctor_id = _parse_admin_delete_payload(text)
            if not doctors.delete_doctor(doctor_id):
                await telegram.send_text(chat_id, f"Không tìm thấy bác sĩ #{doctor_id}.")
                return True
            await telegram.send_text(
                chat_id,
                f"Đã xóa bác sĩ #{doctor_id} khỏi danh sách đang hoạt động.",
                inline_keyboard=_admin_doctors_keyboard(),
            )
            return True
    except sqlite3.IntegrityError:
        await telegram.send_text(chat_id, "Không lưu được: Telegram user id đã được dùng cho bác sĩ khác.")
        return True
    except ValueError as exc:
        usage = ADMIN_ADD_USAGE if cmd == "/admin_doctor_add" else ADMIN_EDIT_USAGE
        if cmd == "/admin_doctor_delete":
            usage = "Cú pháp: `/admin_doctor_delete <id>`"
        await telegram.send_text(chat_id, f"{exc}\n\n{usage}")
        return True
    return False


def _admin_action_from_callback(data: str) -> str:
    if data == ADMIN_EDIT_CALLBACK:
        return "edit"
    if data == ADMIN_DELETE_CALLBACK:
        return "delete"
    return "view"


def _admin_rows_for_tier(tier: str) -> list[dict]:
    return [row for row in doctors.list_all_doctors(include_inactive=True) if row["tier"] == tier]


async def _show_admin_tiers(
    chat_id: int | str,
    message_id: int | None,
    action: str,
) -> None:
    title = _ADMIN_ACTION_TITLES.get(action, _ADMIN_ACTION_TITLES["view"])
    await _admin_edit_or_send(
        chat_id,
        message_id,
        f"{title}\n\nChọn nhóm tư vấn:",
        _admin_tier_keyboard(action),
    )


async def _show_admin_specialties(
    chat_id: int | str,
    message_id: int | None,
    action: str,
    tier: str,
) -> None:
    rows = _admin_rows_for_tier(tier)
    specialties = _ordered_specialties(rows)
    if not specialties:
        await _admin_edit_or_send(
            chat_id,
            message_id,
            "Chưa có bác sĩ trong nhóm này.",
            _admin_tier_keyboard(action),
        )
        return
    title = _ADMIN_ACTION_TITLES.get(action, _ADMIN_ACTION_TITLES["view"])
    await _admin_edit_or_send(
        chat_id,
        message_id,
        f"{title}\n\nChọn chuyên khoa:",
        _admin_specialty_keyboard(action, tier, specialties),
    )


async def _show_admin_doctor_list(
    chat_id: int | str,
    message_id: int | None,
    action: str,
    tier: str,
    specialty_slug: str,
) -> None:
    rows = _admin_rows_for_tier(tier)
    specialty = _specialty_from_slug(rows, specialty_slug)
    if specialty is None:
        await _admin_edit_or_send(
            chat_id,
            message_id,
            "Không tìm thấy chuyên khoa này.",
            _admin_tier_keyboard(action),
        )
        return
    filtered = [row for row in rows if _specialty_slug(str(row.get("specialty") or "")) == specialty_slug]
    title = _ADMIN_ACTION_TITLES.get(action, _ADMIN_ACTION_TITLES["view"])
    await _admin_edit_or_send(
        chat_id,
        message_id,
        f"{title}\n\nChọn bác sĩ chuyên khoa {specialty}:",
        _admin_doctor_list_keyboard(action, filtered, tier),
    )


async def _show_admin_doctor_profile(
    chat_id: int | str,
    message_id: int | None,
    doctor_id: int,
) -> None:
    doctor = doctors.get_doctor(doctor_id)
    if doctor is None:
        await _admin_edit_or_send(chat_id, message_id, f"Không tìm thấy bác sĩ #{doctor_id}.", _admin_doctors_keyboard())
        return
    await _admin_edit_or_send(
        chat_id,
        message_id,
        _admin_doctor_detail_text(doctor),
        _admin_doctor_profile_keyboard(doctor_id),
    )


async def _show_admin_fields(
    chat_id: int | str,
    message_id: int | None,
    doctor_id: int,
) -> None:
    doctor = doctors.get_doctor(doctor_id)
    if doctor is None:
        await _admin_edit_or_send(chat_id, message_id, f"Không tìm thấy bác sĩ #{doctor_id}.", _admin_doctors_keyboard())
        return
    await _admin_edit_or_send(
        chat_id,
        message_id,
        _admin_fields_text(doctor),
        _admin_fields_keyboard(doctor_id),
    )


async def _show_admin_field_options(
    chat_id: int | str,
    message_id: int | None,
    doctor_id: int,
    field: str,
) -> None:
    doctor = doctors.get_doctor(doctor_id)
    if doctor is None:
        await _admin_edit_or_send(chat_id, message_id, f"Không tìm thấy bác sĩ #{doctor_id}.", _admin_doctors_keyboard())
        return
    if field not in _ADMIN_FIELD_LABELS:
        await _admin_edit_or_send(chat_id, message_id, "Trường sửa không hợp lệ.", _admin_fields_keyboard(doctor_id))
        return
    await _admin_edit_or_send(
        chat_id,
        message_id,
        _admin_field_prompt(doctor, field),
        _admin_field_options_keyboard(doctor_id, field),
    )


async def _handle_admin_doctor_callback(
    data: str,
    chat_id: int | str,
    message_id: int | None,
    callback_query_id: str | None,
    user_id: int | None,
) -> bool:
    if not data.startswith(ADMIN_PREFIX):
        return False
    if not _is_admin_user(user_id):
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Bạn không có quyền dùng chức năng này.")
        return True
    if data == ADMIN_MENU_CALLBACK:
        await _admin_edit_or_send(chat_id, message_id, ADMIN_MENU_TEXT, _admin_doctors_keyboard())
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "")
        return True
    if data in {ADMIN_LIST_CALLBACK, ADMIN_EDIT_CALLBACK, ADMIN_DELETE_CALLBACK}:
        await _show_admin_tiers(chat_id, message_id, _admin_action_from_callback(data))
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Chọn nhóm tư vấn.")
        return True
    if data == ADMIN_ADD_CALLBACK:
        await _admin_edit_or_send(chat_id, message_id, ADMIN_ADD_USAGE, _admin_back_keyboard())
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Hướng dẫn thêm bác sĩ.")
        return True
    if data.startswith(ADMIN_TIER_PREFIX):
        payload = data[len(ADMIN_TIER_PREFIX):]
        action, _, tier = payload.partition(":")
        await _show_admin_specialties(chat_id, message_id, action, tier)
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Chọn chuyên khoa.")
        return True
    if data.startswith(ADMIN_SPECIALTY_PREFIX):
        payload = data[len(ADMIN_SPECIALTY_PREFIX):]
        action, _, rest = payload.partition(":")
        tier, _, specialty_slug = rest.partition(":")
        await _show_admin_doctor_list(chat_id, message_id, action, tier, specialty_slug)
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Chọn bác sĩ.")
        return True
    if data.startswith(ADMIN_PROFILE_PREFIX):
        await _show_admin_doctor_profile(chat_id, message_id, int(data[len(ADMIN_PROFILE_PREFIX):]))
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Thông tin bác sĩ.")
        return True
    if data.startswith(ADMIN_FIELDS_PREFIX):
        await _show_admin_fields(chat_id, message_id, int(data[len(ADMIN_FIELDS_PREFIX):]))
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Chọn trường cần sửa.")
        return True
    if data.startswith(ADMIN_FIELD_PREFIX):
        payload = data[len(ADMIN_FIELD_PREFIX):]
        doctor_id_text, _, field = payload.partition(":")
        await _show_admin_field_options(chat_id, message_id, int(doctor_id_text), field)
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Chọn giá trị mới.")
        return True
    if data.startswith(ADMIN_VALUE_PREFIX):
        payload = data[len(ADMIN_VALUE_PREFIX):]
        doctor_id_text, _, rest = payload.partition(":")
        field, _, raw_value = rest.partition(":")
        try:
            value = _admin_code_value(field, raw_value)
            doctor = _apply_admin_doctor_update(int(doctor_id_text), field, value)
            await _admin_edit_or_send(
                chat_id,
                message_id,
                f"Đã cập nhật {_ADMIN_FIELD_LABELS[field]}.\n\n{_admin_doctor_detail_text(doctor)}",
                _admin_doctor_profile_keyboard(int(doctor_id_text)),
            )
            if callback_query_id:
                await telegram._answer_callback_query(str(callback_query_id), "Đã cập nhật.")
        except sqlite3.IntegrityError:
            await _admin_edit_or_send(chat_id, message_id, "Không lưu được: Telegram user id đã được dùng cho bác sĩ khác.", _admin_doctors_keyboard())
            if callback_query_id:
                await telegram._answer_callback_query(str(callback_query_id), "Không lưu được.")
        except ValueError as exc:
            await _admin_edit_or_send(chat_id, message_id, str(exc), _admin_doctors_keyboard())
            if callback_query_id:
                await telegram._answer_callback_query(str(callback_query_id), "Không lưu được.")
        return True
    if data.startswith(ADMIN_CUSTOM_PREFIX):
        payload = data[len(ADMIN_CUSTOM_PREFIX):]
        doctor_id_text, _, field = payload.partition(":")
        doctor_id = int(doctor_id_text)
        if field not in _ADMIN_FIELD_LABELS or user_id is None:
            await _admin_edit_or_send(chat_id, message_id, "Không xử lý được trường này.", _admin_doctors_keyboard())
            return True
        _ADMIN_PENDING_EDITS[_admin_pending_key(chat_id, user_id)] = AdminDoctorPendingEdit(
            user_id=int(user_id),
            doctor_id=doctor_id,
            field=field,
        )
        await _admin_edit_or_send(
            chat_id,
            message_id,
            f"Nhập giá trị mới cho {_ADMIN_FIELD_LABELS[field]}.",
            {"inline_keyboard": [[{"text": "❌ Hủy", "callback_data": ADMIN_CANCEL_EDIT_CALLBACK}]]},
        )
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Nhập giá trị mới.")
        return True
    if data == ADMIN_CANCEL_EDIT_CALLBACK:
        _ADMIN_PENDING_EDITS.pop(_admin_pending_key(chat_id, user_id), None)
        await _admin_edit_or_send(chat_id, message_id, "Đã hủy sửa thông tin.", _admin_doctors_keyboard())
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Đã hủy.")
        return True
    if data.startswith(ADMIN_DELETE_CONFIRM_PREFIX):
        doctor_id = int(data[len(ADMIN_DELETE_CONFIRM_PREFIX):])
        if doctors.delete_doctor(doctor_id):
            text = f"Đã xóa bác sĩ #{doctor_id} khỏi danh sách đang hoạt động."
        else:
            text = f"Không tìm thấy bác sĩ #{doctor_id}."
        await _admin_edit_or_send(chat_id, message_id, text, _admin_doctors_keyboard())
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Đã xử lý.")
        return True
    if data.startswith(ADMIN_DELETE_PREFIX):
        doctor_id = int(data[len(ADMIN_DELETE_PREFIX):])
        doctor = doctors.get_doctor(doctor_id)
        if doctor is None or not doctor["active"]:
            await _admin_edit_or_send(chat_id, message_id, f"Không tìm thấy bác sĩ #{doctor_id}.", _admin_doctors_keyboard())
        else:
            await _admin_edit_or_send(
                chat_id,
                message_id,
                f"Xác nhận xóa bác sĩ này khỏi danh sách đang hoạt động?\n\n{_admin_doctor_summary(doctor)}",
                _admin_delete_confirm_keyboard(doctor_id),
            )
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Xác nhận xóa.")
        return True
    return True


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


async def _handle_pick(
    chat_id: int | str,
    message_id: int,
    doctor_id: int,
    *,
    profile_owner_key: str | None = None,
    subject_id: str | None = None,
) -> None:
    profile_snapshot = None
    if profile_owner_key is not None or subject_id is not None:
        if not profile_owner_key or not subject_id:
            await telegram._edit_message_text(
                chat_id,
                message_id,
                "Không xác định được hồ sơ cần gửi. Vui lòng chọn lại.",
            )
            return
        profile_snapshot = _profile_snapshot(profile_owner_key, subject_id)
        if profile_snapshot is None:
            await telegram._edit_message_text(
                chat_id,
                message_id,
                "Hồ sơ đã chọn không còn tồn tại. Vui lòng chọn lại.",
            )
            return
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
    if profile_snapshot is not None:
        request_text += (
            "\n\nHồ sơ sức khỏe người dùng đã xem và đồng ý chia sẻ:\n"
            f"{profile_snapshot}"
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
    telegram_user_id = (callback_query.get("from") or {}).get("id")
    if chat_id is None:
        return True

    if await _handle_admin_doctor_callback(
        data,
        chat_id,
        int(message_id) if message_id is not None else None,
        str(callback_query_id) if callback_query_id else None,
        telegram_user_id,
    ):
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
    if data.startswith(SHARE_CHOOSE_PREFIX):
        if telegram_user_id is None:
            await telegram._edit_message_text(
                chat_id, int(message_id), "Không xác định được người dùng. Vui lòng thử lại.",
            )
        else:
            await _show_profile_subject_list(
                chat_id,
                int(message_id),
                telegram_user_id,
                int(data[len(SHARE_CHOOSE_PREFIX):]),
            )
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Chọn người được tư vấn.")
        return True
    if data.startswith(SHARE_SUBJECT_PREFIX):
        payload = data[len(SHARE_SUBJECT_PREFIX):]
        doctor_id_text, _, subject_id = payload.partition(":")
        if telegram_user_id is None or not subject_id:
            await telegram._edit_message_text(
                chat_id, int(message_id), "Không xác định được hồ sơ. Vui lòng thử lại.",
            )
        else:
            await _show_profile_share_confirmation(
                chat_id,
                int(message_id),
                telegram_user_id,
                int(doctor_id_text),
                subject_id,
            )
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Đã chọn hồ sơ.")
        return True
    if data.startswith(SHARE_CONFIRM_PREFIX):
        payload = data[len(SHARE_CONFIRM_PREFIX):]
        doctor_id_text, _, subject_id = payload.partition(":")
        owner_key = (
            _profile_owner_key(chat_id, telegram_user_id)
            if telegram_user_id is not None else None
        )
        if owner_key is None or not subject_id:
            await telegram._edit_message_text(
                chat_id, int(message_id), "Không xác định được hồ sơ. Vui lòng thử lại.",
            )
        else:
            await _handle_pick(
                chat_id,
                int(message_id),
                int(doctor_id_text),
                profile_owner_key=owner_key,
                subject_id=subject_id,
            )
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Đã xác nhận chia sẻ.")
        return True
    if data.startswith(PICK_PREFIX):
        doctor_id = int(data[len(PICK_PREFIX):])
        if telegram_user_id is None:
            # Compatibility for internal callers and older test fixtures.
            await _handle_pick(chat_id, int(message_id), doctor_id)
        else:
            owner_key = _profile_owner_key(chat_id, telegram_user_id)
            subject_id = (
                _default_profile_subject_id(
                    chat_id, telegram_user_id, owner_key,
                )
                if owner_key is not None else "self"
            )
            await _show_profile_share_confirmation(
                chat_id,
                int(message_id),
                telegram_user_id,
                doctor_id,
                subject_id,
            )
        if callback_query_id:
            await telegram._answer_callback_query(str(callback_query_id), "Kiểm tra hồ sơ trước khi gửi.")
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
        row = doctors.open_consultation_for_patient(chat_id)
        if row is None:
            return False
    if row["status"] == "active":
        # Bill any unbilled whole minutes of the current paid block before closing.
        settlement = doctors.settle_block_minutes(row["id"])
        if settlement is not None:
            account_id, amount = settlement
            if amount > 0:
                wallet.debit(account_id, amount)
    doctors.end_consultation(row["id"])
    if row["status"] == "pending":
        await telegram.send_text(
            row["patient_chat_id"],
            "Yêu cầu tư vấn đã được hủy. Bạn có thể tiếp tục hỏi bot hoặc dùng /doctor để kết nối lại.",
        )
        await telegram.send_text(row["doctor_chat_id"], "Bệnh nhân đã hủy yêu cầu tư vấn.")
        return True
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
