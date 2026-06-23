"""Telegram inline-button UI for the IPS-inspired medical profile.

The only registered command is `/profile`. Callback prefix is `prof:`.
All tokens are one-shot, scoped to (chat_id, owner_key), and 10-minute TTL.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
import logging
import secrets
import time

from src import config as _config  # noqa: F401  (used by callers)
from src.chat.profile import (
    ALL_SECTIONS,
    ProfileFact,
    confirm_fact,
    count_active_facts,
    count_fact_lineage,
    delete_fact_with_lineage,
    delete_medical_profile,
    delete_subject_profile,
    ensure_subject,
    get_all_section_states,
    get_fact,
    get_section_state,
    get_subject,
    get_user_preference,
    list_facts_paginated,
    list_subjects,
    replace_fact,
    set_fact_inactive,
    set_fact_verification,
    set_section_status,
    set_user_preference,
    update_subject_demographics,
    write_profile_fact,
)
from src.chat.profile.ui_state import (
    clear_pending_profile_edit,
    consume_profile_token,
    invalidate_profile_sessions_for_owner,
    issue_profile_token,
    pop_pending_profile_edit,
    set_pending_profile_edit,
    track_profile_session,
)

logger = logging.getLogger(__name__)

PROFILE_CALLBACK_PREFIX = "prof:"

PRIVATE_ONLY_TEXT = (
    "Tính năng hồ sơ y tế chỉ dùng được trong cuộc trò chuyện riêng tư với bot."
)
NO_IDENTITY_TEXT = (
    "Bot chưa thể mở hồ sơ sức khỏe lúc này. Vui lòng thử lại sau hoặc liên hệ "
    "người hỗ trợ nếu lỗi vẫn tiếp diễn."
)
STALE_TEXT = "Thông tin đã thay đổi, hãy làm mới hồ sơ rồi thử lại."
EXPIRED_TEXT = "Lựa chọn này đã hết hạn."
GENERIC_RETRY_TEXT = "Không xử lý được yêu cầu này. Vui lòng thử lại sau."
INVALID_INPUT_TEXT = "Giá trị không hợp lệ. Vui lòng thử lại."
NESTED_REJECTED_TEXT = (
    "Trường này có cấu trúc phức tạp, bạn cần xóa rồi nhập lại thông tin mới."
)
OBSOLETE_COMMAND_TEXT = (
    "Lệnh này không còn được hỗ trợ. Dùng /profile để mở hồ sơ y tế cá nhân."
)

_FACT_FIELDS_TO_EDIT = ("entity_id", "value", "temporal_status")
_TEMPORAL_STATUS_CHOICES = ("current", "historical", "resolved", "unknown")
_TEMPORAL_STATUS_LABELS = {
    "current": "Đang áp dụng",
    "historical": "Trong quá khứ",
    "resolved": "Đã hết",
    "unknown": "Không rõ",
}
_FACT_FACT_TYPE_LABELS = {
    "medication_use": "Đang dùng thuốc",
    "allergy": "Dị ứng",
    "chronic_disease": "Bệnh nền",
    "age": "Tuổi",
    "sex": "Giới tính",
    "pregnancy_status": "Mang thai",
    "diagnosis": "Chẩn đoán",
    "symptom_state": "Triệu chứng",
    "symptom_history": "Tiền sử triệu chứng",
}
_RELATIONSHIP_LABELS = {
    "self": "Bạn",
    "father": "Bố bạn",
    "mother": "Mẹ bạn",
    "child": "Con bạn",
    "spouse": "Vợ/Chồng bạn",
    "relative": "Người thân",
}

_GENDER_LABELS = {
    "female": "Nữ",
    "male": "Nam",
    "other": "Khác",
    "unknown": "Không muốn trả lời",
}

_SECTION_LABELS = {
    "problems": "Vấn đề sức khỏe",
    "allergies": "Dị ứng",
    "medications": "Thuốc đang dùng",
    "pregnancy": "Thai kỳ",
}

_SECTION_GUIDANCE = {
    "problems": "Ghi các bệnh hoặc vấn đề sức khỏe đang cần theo dõi.",
    "allergies": "Ghi thuốc, thức ăn hoặc chất từng gây dị ứng.",
    "medications": "Ghi các thuốc đang dùng thường xuyên hoặc theo đơn.",
    "pregnancy": "Ghi tình trạng mang thai hiện tại nếu thông tin này phù hợp.",
}

_SECTION_STATUS_LABELS = {
    "unknown": "Chưa kiểm tra",
    "none_known": "Đã xác nhận không có",
    "has_entries": "Có thông tin",
}

_SECTION_FACT_TYPE_LABELS = {
    "problems": "Vấn đề",
    "allergies": "Dị ứng",
    "medications": "Thuốc",
    "pregnancy": "Thai kỳ",
}

DEFAULT_FACT_PAGE_SIZE = 5
DEFAULT_SUBJECT_PAGE_SIZE = 5


# ---------------------------------------------------------------------------
# Public dispatcher entry points
# ---------------------------------------------------------------------------


async def handle_profile_command(
    chat_id: int | str, user_id: int | str | None, chat_type: str,
) -> None:
    """`/profile` command — opens the medical-profile root directly."""
    from src.chat.security.identity import derive_request_identity, derive_previous_owner_keys

    if chat_type != "private":
        await _send(chat_id, PRIVATE_ONLY_TEXT)
        return
    if user_id is None:
        await _send(chat_id, NO_IDENTITY_TEXT)
        return

    identity = derive_request_identity("telegram", user_id, str(chat_id))
    if not identity.owner_key:
        await _send(chat_id, NO_IDENTITY_TEXT)
        return

    # Best-effort: rotate previous HMAC versions of this owner.
    try:
        from src.chat.profile.repository import migrate_owner_key
        migrate_owner_key(
            identity.owner_key,
            derive_previous_owner_keys("telegram", user_id),
        )
    except Exception:
        logger.exception("Profile owner-key rotation failed")

    # Always ensure a self subject exists, even when the profile is empty.
    ensure_subject(
        identity.owner_key, "self",
        relationship="self", display_name=None,
    )
    clear_pending_profile_edit(chat_id, identity.owner_key)

    await _render_profile_root(chat_id, identity.owner_key, "self")


async def handle_profile_callback(
    callback_query: dict,
    callback_user_id: int | str | None,
    chat_type: str,
) -> bool:
    """Top-level dispatcher for `prof:*` callback queries."""
    data = str(callback_query.get("data") or "")
    if not data.startswith(PROFILE_CALLBACK_PREFIX):
        return False

    cb_id = callback_query.get("id")
    message = callback_query.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")

    if chat_type != "private":
        if cb_id:
            await _answer_callback_query(str(cb_id), PRIVATE_ONLY_TEXT)
        if chat_id:
            await _send(chat_id, PRIVATE_ONLY_TEXT)
        return True
    if callback_user_id is None or chat_id is None:
        if cb_id:
            await _answer_callback_query(str(cb_id), GENERIC_RETRY_TEXT)
        return True

    try:
        return await _dispatch_profile_callback(
            data, chat_id, callback_user_id, str(cb_id) if cb_id else None,
        )
    except Exception:
        logger.exception("Profile manager callback failed")
        if cb_id:
            await _answer_callback_query(str(cb_id), GENERIC_RETRY_TEXT)
        if chat_id:
            await _send(chat_id, GENERIC_RETRY_TEXT)
        return True


async def intercept_pending_profile_edit(
    chat_id: int | str, user_id: int | str | None, text: str,
) -> bool:
    """Consume staged profile edits/demographics before the chatbot runs."""
    if user_id is None:
        return False
    from src.chat.security.identity import derive_request_identity
    identity = derive_request_identity("telegram", user_id, str(chat_id))
    if not identity.owner_key:
        return False
    if (text or "").strip().lower() == "/profile":
        clear_pending_profile_edit(chat_id, identity.owner_key)
        return False
    pending = pop_pending_profile_edit(chat_id, identity.owner_key)
    if pending is None:
        return False
    if (text or "").strip().lower() in {"hủy", "huy", "cancel"}:
        await _send(chat_id, "Đã hủy. Không có thông tin nào được thay đổi.")
        await _render_profile_root(
            chat_id, identity.owner_key, pending.get("subject_id", "self"),
        )
        return True
    kind = pending.get("kind")
    if kind == "subject_rename":
        await _apply_subject_rename(
            chat_id=chat_id,
            owner_key=identity.owner_key,
            subject_id=pending["subject_id"],
            expected_updated_at=pending.get("expected_updated_at"),
            new_display_name=text,
        )
        return True
    if kind == "fact_edit_field":
        await _stage_fact_edit_preview(
            chat_id=chat_id,
            owner_key=identity.owner_key,
            fact=get_fact(identity.owner_key, pending["profile_fact_id"]),
            field=pending["field"],
            raw_input=text,
        )
        return True
    if kind == "section_add_entry":
        section = pending["section"]
        subject_id = pending["subject_id"]
        _fact_type_for_section = {
            "problems": "chronic_disease",
            "allergies": "allergy",
            "medications": "medication_use",
            "pregnancy": "pregnancy_status",
        }
        fact_type = _fact_type_for_section.get(section)
        if fact_type is None:
            await _send(chat_id, GENERIC_RETRY_TEXT)
            return True
        name = (text or "").strip()
        if not name:
            set_pending_profile_edit(
                chat_id, identity.owner_key,
                payload={
                    "kind": "section_add_entry",
                    "section": section,
                    "subject_id": subject_id,
                },
            )
            await _send(chat_id, "Bạn chưa nhập tên. Vui lòng gửi lại, hoặc gửi chữ “hủy”.")
            return True
        value = {"name": name}
        if section in {"problems", "medications"}:
            value["status"] = True
        write_profile_fact(
            owner_id=identity.owner_key,
            subject_id=subject_id,
            fact_type=fact_type,
            section=section,
            value=value,
            entity_id=name,
        )
        invalidate_profile_sessions_for_owner(identity.owner_key)
        await _send(chat_id, f"✅ Đã thêm vào {_SECTION_LABELS[section]}: {name}")
        await _render_section_view(chat_id, identity.owner_key, subject_id, section)
        return True
    if kind == "subject_demographics":
        await _apply_subject_demographics_text(
            chat_id=chat_id,
            owner_key=identity.owner_key,
            subject_id=pending["subject_id"],
            field=pending["field"],
            text=text,
            expected_updated_at=pending.get("expected_updated_at"),
        )
        return True
    if kind == "subject_add_name":
        name = (text or "").strip()
        if not name or len(name) > 80:
            set_pending_profile_edit(
                chat_id, identity.owner_key,
                payload=pending,
            )
            await _send(chat_id, "Tên không hợp lệ. Vui lòng nhập từ 1 đến 80 ký tự.")
            return True
        subject_id = f"person_{secrets.token_hex(8)}"
        ensure_subject(
            identity.owner_key,
            subject_id,
            relationship=pending["relationship"],
            display_name=name,
        )
        invalidate_profile_sessions_for_owner(identity.owner_key)
        await _send(chat_id, f"✅ Đã thêm hồ sơ cho {name}.")
        await _render_profile_root(chat_id, identity.owner_key, subject_id)
        return True
    return False


# ---------------------------------------------------------------------------
# Internal dispatcher
# ---------------------------------------------------------------------------


async def _dispatch_profile_callback(
    data: str,
    chat_id: int | str,
    callback_user_id: int | str,
    callback_id: str | None,
) -> bool:
    from src.chat.security.identity import derive_request_identity

    payload = data[len(PROFILE_CALLBACK_PREFIX):]
    if ":" not in payload:
        if callback_id:
            await _answer_callback_query(callback_id, GENERIC_RETRY_TEXT)
        return True
    action, _, token = payload.partition(":")
    if not token:
        return True

    identity = derive_request_identity("telegram", callback_user_id, str(chat_id))
    if not identity.owner_key:
        if callback_id:
            await _answer_callback_query(callback_id, NO_IDENTITY_TEXT)
        return True

    body = consume_profile_token(
        token,
        expected_chat_id=chat_id,
        expected_owner_key=identity.owner_key,
    )
    if body is None:
        if callback_id:
            await _answer_callback_query(callback_id, EXPIRED_TEXT)
        await _send(chat_id, EXPIRED_TEXT)
        return True
    if callback_id:
        await _answer_callback_query(callback_id, "")

    scope = body["scope"]
    body_payload = body.get("payload") or {}

    if scope == "profile_root":
        await _render_profile_root(chat_id, identity.owner_key, body_payload["subject_id"])
    elif scope == "profile_settings":
        await _render_profile_settings(
            chat_id, identity.owner_key, body_payload["subject_id"],
        )
    elif scope == "profile_cancel_pending":
        clear_pending_profile_edit(chat_id, identity.owner_key)
        await _send(chat_id, "Đã hủy. Không có thông tin nào được thay đổi.")
        await _render_profile_root(
            chat_id, identity.owner_key, body_payload.get("subject_id", "self"),
        )
    elif scope == "subjects_page":
        await _render_subjects_page(
            chat_id, identity.owner_key, page=0,
            return_subject_id=body_payload.get("subject_id", "self"),
        )
    elif scope == "subject_add_menu":
        await _render_subject_add_menu(chat_id, identity.owner_key)
    elif scope == "subject_add_relationship":
        await _start_subject_add_name(
            chat_id, identity.owner_key, body_payload["relationship"],
        )
    elif scope == "subject_view":
        await _render_subject_detail(
            chat_id, identity.owner_key, body_payload["subject_id"],
        )
    elif scope == "subject_rename_confirm":
        ok = set_pending_profile_edit(
            chat_id, identity.owner_key,
            payload={
                "kind": "subject_rename",
                "subject_id": body_payload["subject_id"],
                "expected_updated_at": body_payload.get("expected_updated_at"),
            },
        )
        if not ok:
            await _send(chat_id, GENERIC_RETRY_TEXT)
        else:
            await _send(
                chat_id,
                "Hãy gửi tên bạn muốn hiển thị, ví dụ: Bác Lan.\n\n"
                "Gửi chữ “hủy” nếu bạn không muốn thay đổi.",
                inline_keyboard=_pending_cancel_keyboard(
                    chat_id, identity.owner_key, body_payload["subject_id"],
                ),
            )
    elif scope == "subject_delete_confirm":
        await _render_subject_delete_confirm(
            chat_id, identity.owner_key, body_payload["subject_id"],
        )
    elif scope == "subject_delete":
        await _execute_subject_delete(
            chat_id, identity.owner_key, body_payload["subject_id"],
            expected_updated_at=body_payload.get("expected_updated_at"),
        )
    elif scope == "facts_page":
        await _render_facts_page(
            chat_id, identity.owner_key, body_payload["subject_id"],
            page=int(body_payload.get("page", 0)),
        )
    elif scope == "fact_view":
        await _render_fact_detail(
            chat_id, identity.owner_key, body_payload["profile_fact_id"],
        )
    elif scope == "fact_edit_field_confirm":
        field = body_payload["field"]
        profile_fact_id = body_payload["profile_fact_id"]
        if field not in _FACT_FIELDS_TO_EDIT:
            await _send(chat_id, INVALID_INPUT_TEXT)
        else:
            ok = set_pending_profile_edit(
                chat_id, identity.owner_key,
                payload={
                    "kind": "fact_edit_field",
                    "profile_fact_id": profile_fact_id,
                    "field": field,
                    "expected_updated_at": body_payload.get("expected_updated_at"),
                },
            )
            if not ok:
                await _send(chat_id, GENERIC_RETRY_TEXT)
            else:
                await _send(
                    chat_id,
                    _fact_field_prompt(field, body_payload.get("current")),
                )
    elif scope == "fact_edit_apply":
        await _execute_fact_edit(
            chat_id, identity.owner_key, body_payload["profile_fact_id"],
            body_payload["field"], body_payload["new_value"],
            expected_updated_at=body_payload.get("expected_updated_at"),
        )
    elif scope == "fact_delete_confirm":
        await _render_fact_delete_confirm(
            chat_id, identity.owner_key, body_payload["profile_fact_id"],
        )
    elif scope == "fact_delete":
        await _execute_fact_delete(
            chat_id, identity.owner_key, body_payload["profile_fact_id"],
            expected_updated_at=body_payload.get("expected_updated_at"),
        )
    elif scope == "section_view":
        await _render_section_view(
            chat_id, identity.owner_key,
            body_payload["subject_id"], body_payload["section"],
        )
    elif scope == "section_set_state":
        await _apply_section_state(
            chat_id, identity.owner_key,
            body_payload["subject_id"], body_payload["section"],
            body_payload["status"],
        )
    elif scope == "section_add_entry":
        await _start_section_add_entry(
            chat_id, identity.owner_key,
            body_payload["subject_id"], body_payload["section"],
        )
    elif scope == "pregnancy_set":
        await _apply_pregnancy_status(
            chat_id, identity.owner_key, body_payload["subject_id"],
            body_payload["value"],
        )
    elif scope == "subject_demographics_confirm":
        await _start_subject_demographics(
            chat_id, identity.owner_key, body_payload["subject_id"],
        )
    elif scope == "subject_demographics_field":
        await _start_subject_demographics_field(
            chat_id, identity.owner_key, body_payload["subject_id"],
            body_payload["field"],
        )
    elif scope == "subject_gender_set":
        await _apply_subject_gender(
            chat_id, identity.owner_key, body_payload["subject_id"],
            body_payload["gender"],
            expected_updated_at=body_payload.get("expected_updated_at"),
        )
    elif scope == "profile_set_preference":
        await _apply_set_preference(
            chat_id, identity.owner_key,
            body_payload["preference"],
            bool(body_payload.get("enabled", False)),
            subject_id=body_payload.get("subject_id", "self"),
        )
    else:
        await _send(chat_id, GENERIC_RETRY_TEXT)
    return True


# ---------------------------------------------------------------------------
# Profile root
# ---------------------------------------------------------------------------


async def _render_profile_root(
    chat_id: int | str, owner_key: str, subject_id: str,
) -> None:
    subject = get_subject(owner_key, subject_id)
    if subject is None:
        await _send(chat_id, "Không tìm thấy chủ thể này.")
        return
    lines = [
        "🩺 **Hồ sơ sức khỏe**",
        "",
        f"Đang xem: **{_format_display_name(subject)}** "
        f"({_relationship_label(subject['relationship'])})",
        "",
        "Chọn một mục bên dưới để xem hoặc cập nhật. Thông tin bạn xác nhận "
        "có thể giúp bot trả lời phù hợp và an toàn hơn.",
        "",
        "Lưu ý: Đây là hồ sơ do bạn tự quản lý, không thay thế hồ sơ bệnh án của cơ sở y tế.",
    ]
    sections = get_all_section_states(owner_key, subject_id)
    rows: list[list[dict]] = []
    for section in ALL_SECTIONS:
        status = sections.get(section, "unknown")
        view_token = issue_profile_token(
            "section_view", chat_id=chat_id, owner_key=owner_key,
            payload={"subject_id": subject_id, "section": section},
        )
        rows.append([
            {
                "text": f"{_SECTION_LABELS[section]} · {_SECTION_STATUS_LABELS[status]}",
                "callback_data": f"{PROFILE_CALLBACK_PREFIX}section_view:{view_token}",
            }
        ])

    rows.append([
        {
            "text": "👤 Tên, ngày sinh và giới tính",
            "callback_data": (
                f"{PROFILE_CALLBACK_PREFIX}subject_demographics_confirm:"
                f"{issue_profile_token('subject_demographics_confirm', chat_id=chat_id, owner_key=owner_key, payload={'subject_id': subject_id, 'expected_updated_at': subject['updated_at']})}"
            ),
        }
    ])
    rows.append([
        {
            "text": "👪 Hồ sơ của tôi và người thân",
            "callback_data": (
                f"{PROFILE_CALLBACK_PREFIX}subjects_page:"
                f"{issue_profile_token('subjects_page', chat_id=chat_id, owner_key=owner_key, payload={'subject_id': subject_id})}"
            ),
        },
    ])
    settings_token = issue_profile_token(
        "profile_settings", chat_id=chat_id, owner_key=owner_key,
        payload={"subject_id": subject_id},
    )
    rows.append([{
        "text": "🔒 Cài đặt lưu và sử dụng hồ sơ",
        "callback_data": f"{PROFILE_CALLBACK_PREFIX}profile_settings:{settings_token}",
    }])

    await _send(
        chat_id,
        "\n".join(lines),
        inline_keyboard={"inline_keyboard": rows},
    )


async def _render_profile_settings(
    chat_id: int | str, owner_key: str, subject_id: str,
) -> None:
    storage = get_user_preference(owner_key, "storage")
    personalization = get_user_preference(owner_key, "personalization")
    subject = get_subject(owner_key, subject_id)

    def _state(value: bool | None) -> str:
        if value is None:
            return "Chưa chọn"
        return "Đang bật" if value else "Đang tắt"

    lines = [
        "🔒 **Cài đặt hồ sơ**",
        "",
        f"1. Lưu hồ sơ: **{_state(storage)}**",
        "Khi bật, thông tin bạn thêm hoặc xác nhận được giữ lại cho lần trò chuyện sau.",
        "",
        f"2. Dùng để trả lời phù hợp hơn: **{_state(personalization)}**",
        "Khi bật, bot có thể dùng hồ sơ đã lưu để điều chỉnh câu trả lời cho bạn.",
        "",
        "Bạn có thể tắt từng lựa chọn bất cứ lúc nào. Tắt không tự động xóa dữ liệu đã lưu.",
    ]
    rows: list[list[dict]] = []
    for preference, current, label in (
        ("storage", storage, "lưu hồ sơ"),
        ("personalization", personalization, "dùng hồ sơ khi trả lời"),
    ):
        enabled = current is not True
        token = issue_profile_token(
            "profile_set_preference", chat_id=chat_id, owner_key=owner_key,
            payload={
                "preference": preference,
                "enabled": enabled,
                "subject_id": subject_id,
            },
        )
        rows.append([{
            "text": f"{'✅ Bật' if enabled else '⛔ Tắt'} {label}",
            "callback_data": f"{PROFILE_CALLBACK_PREFIX}profile_set_preference:{token}",
        }])
    back_token = issue_profile_token(
        "profile_root", chat_id=chat_id, owner_key=owner_key,
        payload={"subject_id": subject_id},
    )
    rows.append([{
        "text": "🔙 Quay lại hồ sơ",
        "callback_data": f"{PROFILE_CALLBACK_PREFIX}profile_root:{back_token}",
    }])
    if subject is not None:
        delete_token = issue_profile_token(
            "subject_delete_confirm", chat_id=chat_id, owner_key=owner_key,
            payload={
                "subject_id": subject_id,
                "expected_updated_at": subject["updated_at"],
            },
        )
        rows.append([{
            "text": (
                "🗑 Xóa toàn bộ hồ sơ của tôi"
                if subject_id == "self" else "🗑 Xóa hồ sơ người này"
            ),
            "callback_data": f"{PROFILE_CALLBACK_PREFIX}subject_delete_confirm:{delete_token}",
        }])
    await _send(chat_id, "\n".join(lines), inline_keyboard={"inline_keyboard": rows})


# ---------------------------------------------------------------------------
# Subject list / detail
# ---------------------------------------------------------------------------


async def _render_subjects_page(
    chat_id: int | str, owner_key: str, *, page: int,
    return_subject_id: str = "self",
) -> None:
    subjects = list_subjects(owner_key)
    if not subjects:
        await _send(
            chat_id,
            "Chưa có chủ thể nào trong hồ sơ.",
            inline_keyboard={"inline_keyboard": [[
                {"text": "⚙️ Mở hồ sơ của bạn", "callback_data": (
                    f"{PROFILE_CALLBACK_PREFIX}profile_root:"
                    f"{issue_profile_token('profile_root', chat_id=chat_id, owner_key=owner_key, payload={'subject_id': 'self'})}"
                )},
            ]]},
        )
        return
    start = page * DEFAULT_SUBJECT_PAGE_SIZE
    end = start + DEFAULT_SUBJECT_PAGE_SIZE
    page_subjects = subjects[start:end]
    lines = [
        "👪 **Hồ sơ của tôi và người thân**",
        "",
        "Chọn một người để xem hồ sơ. Mỗi người có thông tin sức khỏe riêng.",
        "",
    ]
    for idx, subj in enumerate(page_subjects, start=start + 1):
        count = count_active_facts(owner_key, subj["subject_id"])
        lines.append(
            f"{idx}. {_format_display_name(subj)} "
            f"({_relationship_label(subj['relationship'])}) — "
            f"{count} thông tin"
        )
    rows: list[list[dict]] = []
    for subj in page_subjects:
        token = issue_profile_token(
            "profile_root", chat_id=chat_id, owner_key=owner_key,
            payload={"subject_id": subj["subject_id"]},
        )
        rows.append([
            {
                "text": f"📂 {_format_display_name(subj)}",
                "callback_data": f"{PROFILE_CALLBACK_PREFIX}profile_root:{token}",
            }
        ])
    add_token = issue_profile_token(
        "subject_add_menu", chat_id=chat_id, owner_key=owner_key, payload={},
    )
    rows.append([{
        "text": "➕ Thêm hồ sơ người thân",
        "callback_data": f"{PROFILE_CALLBACK_PREFIX}subject_add_menu:{add_token}",
    }])
    rows.append([
        {"text": "🔙 Quay lại", "callback_data": (
            f"{PROFILE_CALLBACK_PREFIX}profile_root:"
            f"{issue_profile_token('profile_root', chat_id=chat_id, owner_key=owner_key, payload={'subject_id': return_subject_id})}"
        )},
    ])
    await _send(chat_id, "\n".join(lines), inline_keyboard={"inline_keyboard": rows})


async def _render_subject_add_menu(chat_id: int | str, owner_key: str) -> None:
    lines = [
        "➕ **Thêm hồ sơ người thân**",
        "",
        "Người này có quan hệ gì với bạn? Hãy chọn một nút bên dưới.",
    ]
    rows: list[list[dict]] = []
    choices = (
        ("mother", "Mẹ"),
        ("father", "Bố"),
        ("spouse", "Vợ hoặc chồng"),
        ("child", "Con"),
        ("relative", "Người thân khác"),
    )
    for relationship, label in choices:
        token = issue_profile_token(
            "subject_add_relationship", chat_id=chat_id, owner_key=owner_key,
            payload={"relationship": relationship},
        )
        rows.append([{
            "text": label,
            "callback_data": f"{PROFILE_CALLBACK_PREFIX}subject_add_relationship:{token}",
        }])
    back_token = issue_profile_token(
        "subjects_page", chat_id=chat_id, owner_key=owner_key,
        payload={"subject_id": "self"},
    )
    rows.append([{
        "text": "🔙 Quay lại",
        "callback_data": f"{PROFILE_CALLBACK_PREFIX}subjects_page:{back_token}",
    }])
    await _send(chat_id, "\n".join(lines), inline_keyboard={"inline_keyboard": rows})


async def _start_subject_add_name(
    chat_id: int | str, owner_key: str, relationship: str,
) -> None:
    if relationship not in _RELATIONSHIP_LABELS or relationship == "self":
        await _send(chat_id, GENERIC_RETRY_TEXT)
        return
    ok = set_pending_profile_edit(
        chat_id, owner_key,
        payload={
            "kind": "subject_add_name",
            "relationship": relationship,
            "subject_id": "self",
        },
    )
    if not ok:
        await _send(chat_id, GENERIC_RETRY_TEXT)
        return
    await _send(
        chat_id,
        f"Hãy gửi tên của {_RELATIONSHIP_LABELS[relationship].lower()}.\n"
        "Ví dụ: Lan hoặc Bác Lan.\n\nGửi chữ “hủy” nếu bạn không muốn thêm.",
        inline_keyboard=_pending_cancel_keyboard(chat_id, owner_key, "self"),
    )


async def _render_subject_detail(
    chat_id: int | str, owner_key: str, subject_id: str,
) -> None:
    subject = get_subject(owner_key, subject_id)
    if subject is None:
        await _send(chat_id, "Không tìm thấy chủ thể này.")
        return
    active = count_active_facts(owner_key, subject_id)
    body_lines = [
        f"👤 **{_format_display_name(subject)}**",
        f"Quan hệ với bạn: {_relationship_label(subject['relationship'])}",
    ]
    if subject.get("birth_date"):
        body_lines.append(f"Ngày sinh: {_format_birth_date(subject['birth_date'])}")
    if subject.get("gender"):
        body_lines.append(
            f"Giới tính: {_GENDER_LABELS.get(subject['gender'], subject['gender'])}"
        )
    body_lines.append(f"Số thông tin sức khỏe đang lưu: {active}")
    await _send(chat_id, "\n".join(body_lines), inline_keyboard=_subject_detail_keyboard(
        chat_id=chat_id, owner_key=owner_key, subject=subject,
    ))
    await _render_facts_page(chat_id, owner_key, subject_id, page=0, header_lines=None)


async def _render_subject_delete_confirm(
    chat_id: int | str, owner_key: str, subject_id: str,
) -> None:
    subject = get_subject(owner_key, subject_id)
    if subject is None:
        await _send(chat_id, "Không tìm thấy chủ thể này.")
        return
    active = count_active_facts(owner_key, subject_id)
    confirm_token = issue_profile_token(
        "subject_delete", chat_id=chat_id, owner_key=owner_key,
        payload={"subject_id": subject_id, "expected_updated_at": subject["updated_at"]},
    )
    parent_token = issue_profile_token(
        "subject_view", chat_id=chat_id, owner_key=owner_key,
        payload={"subject_id": subject_id},
    )
    body = (
        f"❓ **Xác nhận xóa hồ sơ**\n\n"
        f"- Tên: {_format_display_name(subject)}\n"
        f"- Quan hệ: {_relationship_label(subject['relationship'])}\n"
        f"- Số thông tin sức khỏe sẽ bị xóa: {active}\n\n"
        "Sau khi xác nhận, dữ liệu không thể khôi phục. Nếu chưa chắc chắn, hãy chọn Hủy."
    )
    await _send(
        chat_id, body,
        inline_keyboard=_confirm_delete_keyboard(
            confirm_token, kind="subject", parent_token=parent_token,
        ),
    )


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


async def _render_section_view(
    chat_id: int | str, owner_key: str, subject_id: str, section: str,
) -> None:
    if section not in ALL_SECTIONS:
        await _send(chat_id, "Mục không hợp lệ.")
        return
    subject = get_subject(owner_key, subject_id)
    if subject is None:
        await _send(chat_id, "Không tìm thấy chủ thể này.")
        return
    facts = [f for f in list_facts_paginated(owner_key, subject_id) if f.section == section]
    status = get_section_state(owner_key, subject_id, section) or "unknown"
    lines = [
        f"📂 **{_SECTION_LABELS[section]}** — {_SECTION_STATUS_LABELS[status]}",
        "",
        _SECTION_GUIDANCE[section],
        "",
    ]
    if facts:
        for idx, fact in enumerate(facts, start=1):
            mark = "✅" if fact.verification_status == "confirmed" else "🟡"
            lines.append(f"{idx}. {mark} {_fact_summary(fact)}")
    else:
        if status == "none_known":
            lines.append("Bạn đã xác nhận hiện không có thông tin nào thuộc mục này.")
        else:
            lines.append("Chưa có thông tin. Hãy thêm một mục, hoặc xác nhận hiện không có.")
    if facts:
        lines.extend(["", "✅ Đã xác nhận · 🟡 Chưa xác nhận"])
    rows: list[list[dict]] = []
    add_token = issue_profile_token(
        "section_add_entry", chat_id=chat_id, owner_key=owner_key,
        payload={"subject_id": subject_id, "section": section},
    )
    rows.append([
        {"text": f"➕ Thêm {_SECTION_FACT_TYPE_LABELS[section].lower()}", "callback_data": f"{PROFILE_CALLBACK_PREFIX}section_add_entry:{add_token}"},
    ])
    none_token = issue_profile_token(
        "section_set_state", chat_id=chat_id, owner_key=owner_key,
        payload={"subject_id": subject_id, "section": section, "status": "none_known"},
    )
    unknown_token = issue_profile_token(
        "section_set_state", chat_id=chat_id, owner_key=owner_key,
        payload={"subject_id": subject_id, "section": section, "status": "unknown"},
    )
    rows.append([
        {"text": "✅ Hiện không có", "callback_data": f"{PROFILE_CALLBACK_PREFIX}section_set_state:{none_token}"},
    ])
    rows.append([
        {"text": "❓ Tôi chưa chắc", "callback_data": f"{PROFILE_CALLBACK_PREFIX}section_set_state:{unknown_token}"},
    ])
    for fact in facts:
        view_token = issue_profile_token(
            "fact_view", chat_id=chat_id, owner_key=owner_key,
            payload={"profile_fact_id": fact.profile_fact_id},
        )
        rows.append([
            {"text": f"📋 Mở: {fact.entity_id or fact.fact_type}", "callback_data": f"{PROFILE_CALLBACK_PREFIX}fact_view:{view_token}"},
        ])
    back_token = issue_profile_token(
        "profile_root", chat_id=chat_id, owner_key=owner_key,
        payload={"subject_id": subject_id},
    )
    rows.append([
        {"text": "🔙 Quay lại hồ sơ", "callback_data": f"{PROFILE_CALLBACK_PREFIX}profile_root:{back_token}"},
    ])
    await _send(chat_id, "\n".join(lines), inline_keyboard={"inline_keyboard": rows})


async def _apply_section_state(
    chat_id: int | str, owner_key: str, subject_id: str,
    section: str, status: str,
) -> None:
    if section not in ALL_SECTIONS or status not in ("unknown", "none_known"):
        await _send(chat_id, GENERIC_RETRY_TEXT)
        return
    if status == "none_known":
        has_entries = any(
            fact.section == section
            for fact in list_facts_paginated(owner_key, subject_id)
        )
        if has_entries:
            await _send(
                chat_id,
                "Mục này vẫn còn thông tin. Hãy xóa từng thông tin trước khi chọn “Hiện không có”.",
            )
            return
    set_section_status(owner_key, subject_id, section, status)
    invalidate_profile_sessions_for_owner(owner_key)
    await _send(
        chat_id,
        f"✅ Đã cập nhật {_SECTION_LABELS[section]}: {_SECTION_STATUS_LABELS[status]}.",
    )
    await _render_section_view(chat_id, owner_key, subject_id, section)


async def _start_section_add_entry(
    chat_id: int | str, owner_key: str, subject_id: str, section: str,
) -> None:
    if section not in ALL_SECTIONS:
        await _send(chat_id, GENERIC_RETRY_TEXT)
        return
    if section == "pregnancy":
        rows: list[list[dict]] = []
        for value, label in (
            ("pregnant", "Đang mang thai"),
            ("not_pregnant", "Không mang thai"),
            ("unknown", "Chưa rõ / không muốn trả lời"),
        ):
            token = issue_profile_token(
                "pregnancy_set", chat_id=chat_id, owner_key=owner_key,
                payload={"subject_id": subject_id, "value": value},
            )
            rows.append([{
                "text": label,
                "callback_data": f"{PROFILE_CALLBACK_PREFIX}pregnancy_set:{token}",
            }])
        back_token = issue_profile_token(
            "section_view", chat_id=chat_id, owner_key=owner_key,
            payload={"subject_id": subject_id, "section": section},
        )
        rows.append([{
            "text": "🔙 Quay lại",
            "callback_data": f"{PROFILE_CALLBACK_PREFIX}section_view:{back_token}",
        }])
        await _send(
            chat_id,
            "Hãy chọn tình trạng phù hợp. Nếu không chắc, chọn “Chưa rõ”.",
            inline_keyboard={"inline_keyboard": rows},
        )
        return
    ok = set_pending_profile_edit(
        chat_id, owner_key,
        payload={"kind": "section_add_entry", "section": section, "subject_id": subject_id},
    )
    if not ok:
        await _send(chat_id, GENERIC_RETRY_TEXT)
        return
    prompts = {
        "problems": "Hãy gửi tên một bệnh hoặc vấn đề sức khỏe.\nVí dụ: Tăng huyết áp",
        "allergies": "Hãy gửi tên một thuốc, thức ăn hoặc chất gây dị ứng.\nVí dụ: Penicillin",
        "medications": "Hãy gửi tên một thuốc đang dùng.\nVí dụ: Metformin",
    }
    await _send(
        chat_id,
        f"{prompts[section]}\n\nChỉ gửi một mục mỗi lần. Gửi chữ “hủy” nếu bạn không muốn thêm.",
        inline_keyboard=_pending_cancel_keyboard(chat_id, owner_key, subject_id),
    )


async def _apply_pregnancy_status(
    chat_id: int | str, owner_key: str, subject_id: str, value: str,
) -> None:
    if value not in {"pregnant", "not_pregnant", "unknown"}:
        await _send(chat_id, GENERIC_RETRY_TEXT)
        return
    for fact in list_facts_paginated(owner_key, subject_id):
        if fact.section == "pregnancy":
            set_fact_inactive(owner_key, fact.profile_fact_id, inactive=True)
    if value == "unknown":
        set_section_status(owner_key, subject_id, "pregnancy", "unknown")
        message = "✅ Đã ghi nhận: chưa rõ tình trạng thai kỳ."
    else:
        write_profile_fact(
            owner_id=owner_key,
            subject_id=subject_id,
            fact_type="pregnancy_status",
            section="pregnancy",
            value={"value": value},
            entity_id=value,
        )
        message = (
            "✅ Đã ghi nhận: đang mang thai."
            if value == "pregnant" else "✅ Đã ghi nhận: không mang thai."
        )
    invalidate_profile_sessions_for_owner(owner_key)
    await _send(chat_id, message)
    await _render_section_view(chat_id, owner_key, subject_id, "pregnancy")


# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------


async def _start_subject_demographics(
    chat_id: int | str, owner_key: str, subject_id: str,
) -> None:
    subject = get_subject(owner_key, subject_id)
    if subject is None:
        await _send(chat_id, "Không tìm thấy chủ thể.")
        return
    lines = [
        "👤 **Thông tin cơ bản**",
        "",
        f"Tên hiển thị: {subject.get('display_name') or 'Chưa nhập'}",
        f"Ngày sinh: {_format_birth_date(subject.get('birth_date')) if subject.get('birth_date') else 'Chưa nhập'}",
        f"Giới tính: {_GENDER_LABELS.get(subject.get('gender'), 'Chưa chọn')}",
        "",
        "Chọn đúng thông tin bạn muốn sửa. Bot sẽ hướng dẫn từng bước.",
    ]
    rows: list[list[dict]] = []
    for field, label in (
        ("display_name", "✏️ Sửa tên hiển thị"),
        ("birth_date", "🎂 Sửa ngày sinh"),
    ):
        token = issue_profile_token(
            "subject_demographics_field", chat_id=chat_id, owner_key=owner_key,
            payload={"subject_id": subject_id, "field": field},
        )
        rows.append([{
            "text": label,
            "callback_data": f"{PROFILE_CALLBACK_PREFIX}subject_demographics_field:{token}",
        }])
    for gender, label in _GENDER_LABELS.items():
        token = issue_profile_token(
            "subject_gender_set", chat_id=chat_id, owner_key=owner_key,
            payload={
                "subject_id": subject_id,
                "gender": gender,
                "expected_updated_at": subject["updated_at"],
            },
        )
        rows.append([{
            "text": f"Giới tính: {label}",
            "callback_data": f"{PROFILE_CALLBACK_PREFIX}subject_gender_set:{token}",
        }])
    back_token = issue_profile_token(
        "profile_root", chat_id=chat_id, owner_key=owner_key,
        payload={"subject_id": subject_id},
    )
    rows.append([{
        "text": "🔙 Quay lại hồ sơ",
        "callback_data": f"{PROFILE_CALLBACK_PREFIX}profile_root:{back_token}",
    }])
    await _send(chat_id, "\n".join(lines), inline_keyboard={"inline_keyboard": rows})


async def _start_subject_demographics_field(
    chat_id: int | str, owner_key: str, subject_id: str, field: str,
) -> None:
    if field not in {"display_name", "birth_date"}:
        await _send(chat_id, GENERIC_RETRY_TEXT)
        return
    subject = get_subject(owner_key, subject_id)
    if subject is None:
        await _send(chat_id, "Không tìm thấy hồ sơ người này.")
        return
    ok = set_pending_profile_edit(
        chat_id, owner_key,
        payload={
            "kind": "subject_demographics",
            "subject_id": subject_id,
            "field": field,
            "expected_updated_at": subject.get("updated_at"),
        },
    )
    if not ok:
        await _send(chat_id, GENERIC_RETRY_TEXT)
        return
    prompt = (
        "Hãy gửi tên muốn hiển thị, ví dụ: Cô Lan.\n"
        "Tên có thể dài tối đa 80 ký tự."
        if field == "display_name"
        else "Hãy gửi ngày sinh theo mẫu ngày/tháng/năm.\nVí dụ: 25/08/1955"
    )
    await _send(
        chat_id,
        f"{prompt}\n\nGửi chữ “hủy” nếu bạn không muốn thay đổi.",
        inline_keyboard=_pending_cancel_keyboard(chat_id, owner_key, subject_id),
    )


async def _apply_subject_demographics_text(
    chat_id: int | str, owner_key: str, subject_id: str, field: str, text: str,
    *, expected_updated_at: float | None,
) -> None:
    subject = get_subject(owner_key, subject_id)
    if subject is None:
        await _send(chat_id, "Không tìm thấy chủ thể.")
        return
    if expected_updated_at is not None and abs(
        float(subject["updated_at"]) - float(expected_updated_at)
    ) > 1e-6:
        await _send(chat_id, STALE_TEXT)
        return
    cleaned = (text or "").strip()
    kwargs: dict = {"expected_updated_at": expected_updated_at}
    retry_payload = {
        "kind": "subject_demographics",
        "subject_id": subject_id,
        "field": field,
        "expected_updated_at": expected_updated_at,
    }
    if field == "display_name":
        if not cleaned or len(cleaned) > 80:
            set_pending_profile_edit(chat_id, owner_key, payload=retry_payload)
            await _send(chat_id, "Tên không hợp lệ. Vui lòng nhập từ 1 đến 80 ký tự.")
            return
        kwargs["display_name"] = cleaned
    elif field == "birth_date":
        parsed = _parse_birth_date(cleaned)
        if parsed is None:
            set_pending_profile_edit(chat_id, owner_key, payload=retry_payload)
            await _send(
                chat_id,
                "Ngày sinh chưa đúng. Vui lòng nhập theo mẫu ngày/tháng/năm, ví dụ 25/08/1955.",
            )
            return
        kwargs["birth_date"] = parsed
    else:
        await _send(chat_id, GENERIC_RETRY_TEXT)
        return
    updated = update_subject_demographics(owner_key, subject_id, **kwargs)
    if updated is None:
        await _send(chat_id, STALE_TEXT)
        return
    invalidate_profile_sessions_for_owner(owner_key)
    await _send(chat_id, "✅ Đã cập nhật thông tin cơ bản.")
    await _start_subject_demographics(chat_id, owner_key, subject_id)


async def _apply_subject_gender(
    chat_id: int | str, owner_key: str, subject_id: str, gender: str,
    *, expected_updated_at: float | None,
) -> None:
    if gender not in _GENDER_LABELS:
        await _send(chat_id, GENERIC_RETRY_TEXT)
        return
    updated = update_subject_demographics(
        owner_key,
        subject_id,
        gender=gender,
        expected_updated_at=expected_updated_at,
    )
    if updated is None:
        await _send(chat_id, STALE_TEXT)
        return
    invalidate_profile_sessions_for_owner(owner_key)
    await _send(chat_id, f"✅ Đã cập nhật giới tính: {_GENDER_LABELS[gender]}.")
    await _start_subject_demographics(chat_id, owner_key, subject_id)


async def _apply_set_preference(
    chat_id: int | str, owner_key: str, preference: str, enabled: bool,
    *, subject_id: str,
) -> None:
    if preference not in {"storage", "personalization"}:
        await _send(chat_id, GENERIC_RETRY_TEXT)
        return
    set_user_preference(owner_key, preference, enabled)
    invalidate_profile_sessions_for_owner(owner_key)
    action = "bật" if enabled else "tắt"
    label = "lưu hồ sơ" if preference == "storage" else "dùng hồ sơ khi trả lời"
    await _send(chat_id, f"✅ Đã {action} {label}.")
    await _render_profile_settings(chat_id, owner_key, subject_id)


# ---------------------------------------------------------------------------
# Facts: page / detail / edit / delete
# ---------------------------------------------------------------------------


async def _render_facts_page(
    chat_id: int | str, owner_key: str, subject_id: str, *,
    page: int, header_lines: list[str] | None = None,
) -> None:
    subject = get_subject(owner_key, subject_id)
    if subject is None:
        await _send(chat_id, "Không tìm thấy chủ thể này.")
        return
    total = count_active_facts(owner_key, subject_id)
    if total == 0:
        lines = list(header_lines or [])
        lines.append(f"Không có thông tin nào đang hoạt động cho {_format_display_name(subject)}.")
        await _send(chat_id, "\n".join(lines) if lines else "Không có thông tin.")
        return
    facts = list_facts_paginated(
        owner_key, subject_id, page=page, page_size=DEFAULT_FACT_PAGE_SIZE,
    )
    if header_lines is None:
        lines = [
            f"Thông tin của {_format_display_name(subject)} ({total} đang hoạt động):",
            "",
        ]
    else:
        lines = list(header_lines) + [""]
    for idx, fact in enumerate(facts, start=page * DEFAULT_FACT_PAGE_SIZE + 1):
        lines.append(f"{idx}. {_fact_summary(fact)}")
    await _send(
        chat_id, "\n".join(lines),
        inline_keyboard=_facts_keyboard(
            chat_id=chat_id, owner_key=owner_key, subject_id=subject_id,
            facts=facts, page=page, page_size=DEFAULT_FACT_PAGE_SIZE, total=total,
        ),
    )


async def _render_fact_detail(
    chat_id: int | str, owner_key: str, profile_fact_id: str,
) -> None:
    fact = get_fact(owner_key, profile_fact_id)
    if fact is None:
        await _send(chat_id, "Không tìm thấy thông tin này.")
        return
    subject = get_subject(owner_key, fact.subject_id or "")
    lines = [
        f"📋 {_fact_summary(fact)}",
        "",
        *_safe_fact_lines(fact),
    ]
    if subject is not None:
        lines.append(
            f"Hồ sơ của: {_format_display_name(subject)} "
            f"({_relationship_label(subject['relationship'])})"
        )
    await _send(
        chat_id, "\n".join(lines),
        inline_keyboard=_fact_detail_keyboard(
            chat_id=chat_id, owner_key=owner_key, fact=fact,
        ),
    )


async def _render_fact_delete_confirm(
    chat_id: int | str, owner_key: str, profile_fact_id: str,
) -> None:
    fact = get_fact(owner_key, profile_fact_id)
    if fact is None:
        await _send(chat_id, "Không tìm thấy thông tin này.")
        return
    chain = count_fact_lineage(owner_key, profile_fact_id)
    confirm_token = issue_profile_token(
        "fact_delete", chat_id=chat_id, owner_key=owner_key,
        payload={"profile_fact_id": profile_fact_id, "expected_updated_at": fact.updated_at},
    )
    parent_token = issue_profile_token(
        "fact_view", chat_id=chat_id, owner_key=owner_key,
        payload={"profile_fact_id": profile_fact_id},
    )
    lineage_label = "1 bản ghi" if chain <= 1 else f"{chain} bản ghi"
    body = (
        f"❓ Xác nhận xóa thông tin\n\n"
        f"- {_fact_summary(fact)}\n"
        f"- Bao gồm cả các bản sửa đổi liên quan: {lineage_label}\n\n"
        "Hành động này không thể hoàn tác."
    )
    await _send(
        chat_id, body,
        inline_keyboard=_confirm_delete_keyboard(confirm_token, parent_token=parent_token, kind="fact"),
    )


async def _stage_fact_edit_preview(
    *, chat_id: int | str, owner_key: str, fact: ProfileFact | None,
    field: str, raw_input: str,
) -> None:
    if fact is None:
        await _send(chat_id, STALE_TEXT)
        return
    if field == "value" and isinstance(fact.value, dict) and any(
        isinstance(v, dict) for v in fact.value.values()
    ):
        await _send(chat_id, NESTED_REJECTED_TEXT)
        return
    parsed = _parse_field_input(field, raw_input)
    if parsed is None:
        await _send(chat_id, INVALID_INPUT_TEXT)
        return
    apply_token = issue_profile_token(
        "fact_edit_apply", chat_id=chat_id, owner_key=owner_key,
        payload={
            "profile_fact_id": fact.profile_fact_id,
            "field": field,
            "new_value": parsed,
            "expected_updated_at": fact.updated_at,
        },
    )
    parent_token = issue_profile_token(
        "fact_view", chat_id=chat_id, owner_key=owner_key,
        payload={"profile_fact_id": fact.profile_fact_id},
    )
    await _send(
        chat_id,
        _fact_edit_preview_text(fact, field, parsed),
        inline_keyboard=_edit_preview_keyboard(apply_token, parent_token),
    )


async def _apply_subject_rename(
    *, chat_id: int | str, owner_key: str, subject_id: str,
    expected_updated_at: float | None, new_display_name: str,
) -> None:
    subject = get_subject(owner_key, subject_id)
    if subject is None:
        await _send(chat_id, STALE_TEXT)
        return
    cleaned = (new_display_name or "").strip()
    if not cleaned or len(cleaned) > 80:
        await _send(chat_id, "Tên hiển thị không hợp lệ. Vui lòng nhập 1–80 ký tự.")
        return
    updated = update_subject_demographics(
        owner_key, subject_id, display_name=cleaned,
        expected_updated_at=expected_updated_at,
    )
    if updated is None:
        await _send(chat_id, STALE_TEXT)
        return
    invalidate_profile_sessions_for_owner(owner_key)
    await _send(chat_id, f"✅ Đã đổi tên hiển thị thành: {cleaned}")
    await _render_subject_detail(chat_id, owner_key, subject_id)


async def _execute_subject_delete(
    chat_id: int | str, owner_key: str, subject_id: str, *,
    expected_updated_at: float | None,
) -> None:
    current = get_subject(owner_key, subject_id)
    if current is None:
        await _send(chat_id, "Hồ sơ này không còn tồn tại.")
        return
    if (
        expected_updated_at is not None
        and abs(float(current["updated_at"]) - float(expected_updated_at)) > 1e-6
    ):
        await _send(chat_id, STALE_TEXT)
        return
    deleted = delete_subject_profile(owner_key, subject_id)
    invalidate_profile_sessions_for_owner(owner_key)
    await _send(chat_id, f"✅ Đã xóa hồ sơ và {deleted} thông tin sức khỏe liên quan.")
    if subject_id == "self":
        ensure_subject(owner_key, "self", relationship="self", display_name=None)
        await _render_profile_root(chat_id, owner_key, "self")
    else:
        await _render_subjects_page(chat_id, owner_key, page=0)


async def _execute_fact_edit(
    chat_id: int | str, owner_key: str, profile_fact_id: str,
    field: str, new_value, *, expected_updated_at: float | None,
) -> None:
    old = get_fact(owner_key, profile_fact_id)
    if old is None:
        await _send(chat_id, STALE_TEXT)
        return
    if field not in _FACT_FIELDS_TO_EDIT:
        await _send(chat_id, INVALID_INPUT_TEXT)
        return
    if field == "entity_id" and not isinstance(new_value, str):
        await _send(chat_id, INVALID_INPUT_TEXT)
        return
    replacement = ProfileFact(
        profile_fact_id=secrets.token_hex(16),
        owner_id=owner_key,
        subject_id=old.subject_id,
        section=old.section,
        fact_type=old.fact_type,
        entity_type=old.entity_type,
        entity_id=new_value if field == "entity_id" else old.entity_id,
        attribute=old.attribute,
        value=new_value if field == "value" else dict(old.value),
        temporal_status=new_value if field == "temporal_status" else old.temporal_status,
        confidence=old.confidence,
        verification_status=old.verification_status,
        source_kind="profile_edit",
        reporter_role=old.reporter_role,
        valid_from=old.valid_from,
        valid_until=old.valid_until,
        superseded_by=None,
        created_at=time.time(),
        updated_at=time.time(),
        coding_system=old.coding_system,
        coding_code=old.coding_code,
        coding_display=old.coding_display,
    )
    status, _new = replace_fact(
        owner_key, old.profile_fact_id, new_fact=replacement,
        expected_updated_at=expected_updated_at,
    )
    if status == "stale":
        await _send(chat_id, STALE_TEXT)
        return
    if status == "missing":
        await _send(chat_id, "Thông tin này không còn tồn tại.")
        return
    invalidate_profile_sessions_for_owner(owner_key)
    await _send(chat_id, f"✅ Đã cập nhật: {_fact_summary(replacement)}")
    await _render_fact_detail(chat_id, owner_key, replacement.profile_fact_id)


async def _execute_fact_delete(
    chat_id: int | str, owner_key: str, profile_fact_id: str, *,
    expected_updated_at: float | None,
) -> None:
    fact = get_fact(owner_key, profile_fact_id)
    if fact is None:
        await _send(chat_id, "Thông tin này không còn tồn tại.")
        return
    if (
        expected_updated_at is not None
        and abs(float(fact.updated_at) - float(expected_updated_at)) > 1e-6
    ):
        await _send(chat_id, STALE_TEXT)
        return
    subject_id = fact.subject_id
    removed = delete_fact_with_lineage(owner_key, profile_fact_id)
    invalidate_profile_sessions_for_owner(owner_key)
    await _send(
        chat_id,
        f"✅ Đã xóa thông tin và {removed} bản ghi lịch sử liên quan.",
    )
    if subject_id:
        await _render_subject_detail(chat_id, owner_key, subject_id)
    else:
        await _render_subjects_page(chat_id, owner_key, page=0)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _format_display_name(subject: dict) -> str:
    name = (subject.get("display_name") or "").strip()
    if name:
        return name
    return _relationship_label(subject.get("relationship") or subject.get("subject_id") or "")


def _relationship_label(relationship: str) -> str:
    return _RELATIONSHIP_LABELS.get(relationship, relationship or "?")


def _parse_birth_date(raw: str) -> str | None:
    text = (raw or "").strip()
    parsed: date | None = None
    for pattern in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(text, pattern).date()
            break
        except ValueError:
            continue
    if parsed is None:
        return None
    today = date.today()
    if parsed > today or (today - parsed).days > int(130 * 365.25):
        return None
    return parsed.isoformat()


def _format_birth_date(value: str) -> str:
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%d/%m/%Y")
    except (TypeError, ValueError):
        return value


def _fact_summary(fact: ProfileFact) -> str:
    fact_type = _FACT_FACT_TYPE_LABELS.get(fact.fact_type, fact.fact_type)
    value = fact.value or {}
    entity = fact.entity_id or value.get("name") or fact.coding_display or ""
    suffix = {
        "current": "",
        "historical": " — trong quá khứ",
        "resolved": " — đã hết",
        "unknown": " — chưa rõ thời điểm",
    }.get(fact.temporal_status, "")
    if fact.fact_type == "age":
        raw = value.get("value")
        if raw is not None:
            return f"{fact_type}: {raw}{suffix}"
    if fact.fact_type == "sex":
        raw = value.get("value")
        return f"{fact_type}: {_GENDER_LABELS.get(raw, raw or 'Chưa rõ')}{suffix}"
    if fact.fact_type == "pregnancy_status":
        pregnancy = {
            "pregnant": "Đang mang thai",
            "not_pregnant": "Không mang thai",
            "unknown": "Chưa rõ",
            True: "Đang mang thai",
            False: "Không mang thai",
        }.get(value.get("value"), value.get("name") or "Chưa rõ")
        return f"{fact_type}: {pregnancy}{suffix}"
    if fact.fact_type == "medication_use":
        verb = "Đã ngừng dùng" if value.get("status") is False else "Đang dùng"
        return f"{fact_type}: {verb} {entity}{suffix}"
    if fact.fact_type == "allergy":
        return f"{fact_type}: {entity}{suffix}"
    if fact.fact_type in {"symptom_state", "symptom_history", "chronic_disease", "diagnosis"}:
        flag = value.get("status")
        verb_map = {True: "đang có", False: "đã hết", None: ""}
        verb = verb_map.get(flag, "")
        if verb:
            return f"{fact_type}: {verb} {entity}{suffix}"
        return f"{fact_type}: {entity}{suffix}"
    return f"{fact_type}: {entity or value}{suffix}"


def _safe_fact_lines(fact: ProfileFact) -> list[str]:
    verification = {
        "confirmed": "Bạn đã xác nhận",
        "unconfirmed": "Chưa được bạn xác nhận",
        "refuted": "Đã đánh dấu không đúng",
        "entered_in_error": "Đã đánh dấu nhập nhầm",
    }.get(fact.verification_status, "Chưa rõ")
    return [
        f"Nhóm: {_SECTION_LABELS.get(fact.section or '', 'Thông tin sức khỏe')}",
        f"Tình trạng: {_TEMPORAL_STATUS_LABELS.get(fact.temporal_status, 'Chưa rõ')}",
        f"Xác nhận: {verification}",
    ]


def _fact_field_prompt(field: str, current) -> str:
    if field == "entity_id":
        return (
            "Hãy gửi tên hoặc nội dung mới. Ví dụ: Paracetamol.\n\n"
            "Gửi chữ “hủy” nếu bạn không muốn thay đổi."
        )
    if field == "temporal_status":
        return (
            "Hãy gửi một trong các lựa chọn sau: hiện tại, trong quá khứ, "
            "đã hết, hoặc không rõ.\n\nGửi chữ “hủy” nếu bạn không muốn thay đổi."
        )
    if field == "value":
        if isinstance(current, dict):
            if any(isinstance(v, (dict, list)) for v in current.values()):
                return NESTED_REJECTED_TEXT
            keys = ", ".join(sorted(current.keys()))
            return (
                "Nhập giá trị mới theo định dạng `key=value` trên mỗi dòng, "
                f"hiện có các khóa: {keys}. Trường hợp danh sách: viết cách nhau bằng dấu phẩy."
            )
        return "Nhập giá trị mới."
    return "Nhập giá trị mới."


def _fact_edit_preview_text(fact: ProfileFact, field: str, new_value) -> str:
    old_value = _field_value(fact, field)
    old_text = _format_field_for_preview(field, old_value)
    new_text = _format_field_for_preview(field, new_value)
    return (
        f"✏️ Xác nhận chỉnh sửa\n\n"
        f"Thông tin: {_fact_summary(fact)}\n"
        f"Trường: {_field_label(field)}\n"
        f"- Cũ: {old_text}\n"
        f"- Mới: {new_text}\n\n"
        "Bạn xác nhận thay đổi này chứ?"
    )


def _field_label(field: str) -> str:
    return {
        "entity_id": "Tên hoặc nội dung",
        "value": "Chi tiết",
        "temporal_status": "Tình trạng hiện tại",
    }.get(field, field)


def _field_value(fact: ProfileFact, field: str):
    if field == "entity_id":
        return fact.entity_id
    if field == "temporal_status":
        return fact.temporal_status
    if field == "value":
        return dict(fact.value)
    return None


def _format_field_for_preview(field: str, value) -> str:
    if field == "temporal_status":
        return f"{value} ({_TEMPORAL_STATUS_LABELS.get(value, value)})"
    if field == "value" and isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in sorted(value.items()))
    return str(value)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _parse_field_input(field: str, raw: str):
    text = (raw or "").strip()
    if field == "entity_id":
        if not text or len(text) > 120:
            return None
        return text
    if field == "temporal_status":
        lowered = text.lower()
        alias = {
            "hiện tại": "current",
            "đang áp dụng": "current",
            "quá khứ": "historical",
            "trong quá khứ": "historical",
            "đã hết": "resolved",
            "đã khỏi": "resolved",
            "không rõ": "unknown",
        }
        normalized = alias.get(lowered, lowered)
        if normalized not in _TEMPORAL_STATUS_CHOICES:
            return None
        return normalized
    if field == "value":
        value: dict = {}
        for line in text.splitlines():
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if not key:
                continue
            if "," in val and not (val.startswith("[") or val.startswith("{")):
                value[key] = [item.strip() for item in val.split(",") if item.strip()]
            else:
                value[key] = val
        if not value:
            return None
        return value
    return text or None


# ---------------------------------------------------------------------------
# Keyboards
# ---------------------------------------------------------------------------


def _subject_detail_keyboard(
    *, chat_id: int | str, owner_key: str, subject: dict,
) -> dict:
    rows: list[list[dict]] = []
    rename_token = issue_profile_token(
        "subject_rename_confirm", chat_id=chat_id, owner_key=owner_key,
        payload={"subject_id": subject["subject_id"], "expected_updated_at": subject["updated_at"]},
    )
    rows.append([
        {
            "text": "✏️ Đổi tên hiển thị",
            "callback_data": f"{PROFILE_CALLBACK_PREFIX}subject_rename_confirm:{rename_token}",
        }
    ])
    back_token = issue_profile_token(
        "profile_root", chat_id=chat_id, owner_key=owner_key,
        payload={"subject_id": subject["subject_id"]},
    )
    rows.append([
        {
            "text": "🔙 Quay lại hồ sơ",
            "callback_data": f"{PROFILE_CALLBACK_PREFIX}profile_root:{back_token}",
        }
    ])
    return {"inline_keyboard": rows}


def _pending_cancel_keyboard(
    chat_id: int | str, owner_key: str, subject_id: str,
) -> dict:
    token = issue_profile_token(
        "profile_cancel_pending", chat_id=chat_id, owner_key=owner_key,
        payload={"subject_id": subject_id},
    )
    return {
        "inline_keyboard": [[{
            "text": "❌ Hủy và quay lại hồ sơ",
            "callback_data": f"{PROFILE_CALLBACK_PREFIX}profile_cancel_pending:{token}",
        }]]
    }


def _facts_keyboard(
    *, chat_id: int | str, owner_key: str, subject_id: str,
    facts: list[ProfileFact], page: int, page_size: int, total: int,
) -> dict:
    rows: list[list[dict]] = []
    for fact in facts:
        view_token = issue_profile_token(
            "fact_view", chat_id=chat_id, owner_key=owner_key,
            payload={"profile_fact_id": fact.profile_fact_id},
        )
        rows.append([
            {
                "text": f"📋 {fact.entity_id or fact.fact_type}",
                "callback_data": f"{PROFILE_CALLBACK_PREFIX}fact_view:{view_token}",
            }
        ])
    nav: list[dict] = []
    if page > 0:
        prev_token = issue_profile_token(
            "facts_page", chat_id=chat_id, owner_key=owner_key,
            payload={"subject_id": subject_id, "page": page - 1},
        )
        nav.append({
            "text": "◀️ Trang trước",
            "callback_data": f"{PROFILE_CALLBACK_PREFIX}facts_page:{prev_token}",
        })
    if (page + 1) * page_size < total:
        next_token = issue_profile_token(
            "facts_page", chat_id=chat_id, owner_key=owner_key,
            payload={"subject_id": subject_id, "page": page + 1},
        )
        nav.append({
            "text": "Trang sau ▶️",
            "callback_data": f"{PROFILE_CALLBACK_PREFIX}facts_page:{next_token}",
        })
    if nav:
        rows.append(nav)
    back_token = issue_profile_token(
        "profile_root", chat_id=chat_id, owner_key=owner_key,
        payload={"subject_id": subject_id},
    )
    rows.append([{"text": "🔙 Quay lại hồ sơ", "callback_data": f"{PROFILE_CALLBACK_PREFIX}profile_root:{back_token}"}])
    return {"inline_keyboard": rows}


def _fact_detail_keyboard(
    *, chat_id: int | str, owner_key: str, fact: ProfileFact,
) -> dict:
    rows: list[list[dict]] = []
    edit_buttons = []
    for field in ("entity_id", "temporal_status"):
        token = issue_profile_token(
            "fact_edit_field_confirm", chat_id=chat_id, owner_key=owner_key,
            payload={
                "profile_fact_id": fact.profile_fact_id,
                "field": field,
                "expected_updated_at": fact.updated_at,
                "current": _field_value(fact, field),
            },
        )
        edit_buttons.append({
            "text": _field_label(field),
            "callback_data": f"{PROFILE_CALLBACK_PREFIX}fact_edit_field_confirm:{token}",
        })
    for i in range(0, len(edit_buttons), 2):
        rows.append(edit_buttons[i:i + 2])
    delete_token = issue_profile_token(
        "fact_delete_confirm", chat_id=chat_id, owner_key=owner_key,
        payload={"profile_fact_id": fact.profile_fact_id, "expected_updated_at": fact.updated_at},
    )
    rows.append([{
        "text": "🗑 Xóa thông tin",
        "callback_data": f"{PROFILE_CALLBACK_PREFIX}fact_delete_confirm:{delete_token}",
    }])
    back_token = issue_profile_token(
        "profile_root", chat_id=chat_id, owner_key=owner_key,
        payload={"subject_id": fact.subject_id},
    )
    rows.append([{
        "text": "🔙 Quay lại hồ sơ",
        "callback_data": f"{PROFILE_CALLBACK_PREFIX}profile_root:{back_token}",
    }])
    return {"inline_keyboard": rows}


def _edit_preview_keyboard(apply_token: str, parent_token: str) -> dict:
    return {
        "inline_keyboard": [
            [
                {
                    "text": "✅ Xác nhận",
                    "callback_data": f"{PROFILE_CALLBACK_PREFIX}fact_edit_apply:{apply_token}",
                },
                {
                    "text": "❌ Hủy",
                    "callback_data": f"{PROFILE_CALLBACK_PREFIX}fact_view:{parent_token}",
                },
            ]
        ]
    }


def _confirm_delete_keyboard(
    confirm_token: str, *, kind: str, parent_token: str | None = None,
) -> dict:
    action_scope = "subject_delete" if kind == "subject" else "fact_delete"
    if kind == "subject":
        if parent_token is None:
            raise ValueError("subject confirm needs parent_token")
        cancel = {
            "text": "❌ Hủy",
            "callback_data": f"{PROFILE_CALLBACK_PREFIX}subject_view:{parent_token}",
        }
    else:
        if parent_token is None:
            raise ValueError("fact confirm needs parent_token")
        cancel = {
            "text": "❌ Hủy",
            "callback_data": f"{PROFILE_CALLBACK_PREFIX}fact_view:{parent_token}",
        }
    return {
        "inline_keyboard": [
            [
                {
                    "text": "✅ Xóa vĩnh viễn",
                    "callback_data": f"{PROFILE_CALLBACK_PREFIX}{action_scope}:{confirm_token}",
                },
                cancel,
            ]
        ]
    }


# ---------------------------------------------------------------------------
# Indirection
# ---------------------------------------------------------------------------


async def _send(chat_id: int | str, text: str, *, inline_keyboard: dict | None = None) -> None:
    from src.server.channels import telegram
    await telegram.send_text(chat_id, text, inline_keyboard=inline_keyboard)


async def _answer_callback_query(callback_query_id: str, text: str) -> None:
    from src.server.channels import telegram
    await telegram._answer_callback_query(callback_query_id, text)
