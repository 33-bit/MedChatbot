"""
Telegram Bot webhook.

Setup:
  curl -F "url=https://<ngrok>/webhook/telegram" \\
       -F "secret_token=$TELEGRAM_WEBHOOK_SECRET" \\
       -F "drop_pending_updates=true" \\
       https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/setWebhook
"""

from __future__ import annotations

import asyncio
import contextlib
import html as _html
import logging
import re
import secrets
import time
import urllib.parse
from dataclasses import dataclass, field

import httpx
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request

from src.chat import answer_with_choices
from src.chat.mode_policy import mode_label, normalize_mode
from src.chat.replies import ChatReply, TECHNICAL_ERROR_REPLY
from src.chat.storage.feedback import create_feedback_request, record_feedback_rating
from src.chat.storage.session import clear_session, reserve_webhook_update
from src.chat.storage.wallet import (
    apply_payment,
    create_order,
    debt_status,
    get_balance,
    get_order,
    log_admin_credit,
    mark_order_paid,
    set_order_qr_message,
)
from src.config import (
    PUBLIC_BASE_URL,
    TELEGRAM_ADMIN_IDS,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_WEBHOOK_SECRET,
)
from src.server.payments.payos import create_payment as payos_create_payment
from src.server.payments.payos import extract_transfer_content as payos_extract_transfer_content
from src.server.channels import telegram_doctor

logger = logging.getLogger(__name__)
router = APIRouter()

TG_MAX_LEN = 4000  # safe under Telegram's 4096-byte limit
TG_HARD_MAX_BYTES = 4096
TYPING_REFRESH_SECONDS = 4.0
BOT_COMMANDS = [
    {"command": "help", "description": "📝 Cách đặt câu hỏi"},
    {"command": "mode", "description": "⚙️ Chọn chế độ trả lời"},
    {"command": "doctor", "description": "👨‍⚕️ Kết nối bác sĩ"},
    {"command": "end", "description": "⛔ Kết thúc tư vấn bác sĩ"},
    {"command": "topup", "description": "💰 Nạp tiền vào tài khoản"},
    {"command": "balance", "description": "💳 Xem số dư"},
    {"command": "paydebt", "description": "🧾 Thanh toán công nợ"},
    {"command": "new", "description": "🔄 Xóa ngữ cảnh và bắt đầu lượt mới"},
]
START_TEXT = """Xin chào! Tôi là trợ lý y tế.

Bạn có thể gửi câu hỏi về triệu chứng, bệnh lý hoặc thuốc không kê đơn. Tôi sẽ trả lời bằng tiếng Việt dựa trên tài liệu y khoa trong hệ thống.

Ví dụ:
- Tôi bị ho và sốt 2 ngày nay, nên làm gì?
- Phòng bệnh cúm như thế nào?
- Paracetamol dùng để làm gì?

Dùng /mode để chọn chế độ trả lời: Auto, Thông tin hoặc Chẩn đoán.

Nếu có triệu chứng nặng như khó thở, đau ngực dữ dội, lơ mơ, yếu liệt, co giật hoặc chảy máu nhiều, hãy đi cấp cứu ngay."""

HELP_TEXT = """Cách sử dụng:

- Mô tả rõ triệu chứng, thời gian bắt đầu, mức độ và bệnh nền nếu có.
- Với câu hỏi về thuốc, hãy gửi tên hoạt chất hoặc tên thuốc bạn muốn hỏi.
- Khi tôi hỏi thêm triệu chứng, bạn có thể trả lời ngắn gọn theo từng ý.
- Dùng /mode để chọn chế độ trả lời: Auto, Thông tin hoặc Chẩn đoán.

Lưu ý: Tôi hỗ trợ thông tin y tế, không thay thế bác sĩ. Nếu tình trạng nặng hoặc diễn tiến nhanh, hãy đi khám/cấp cứu."""

RATING_PROMPT = "Bạn đánh giá câu trả lời này từ 1 đến 5 nhé."
DOCTOR_OFFER_PROMPT = (
    "Với tình huống của bạn, bạn có muốn kết nối với bác sĩ chuyên khoa "
    "để được tư vấn sâu hơn không?"
)
MULTI_SELECT_DONE = "Xong"
MULTI_SELECT_PREFIX = "ms:"
MODE_CALLBACK_PREFIX = "mode:set:"
MODE_RETRY_PREFIX = "mode_retry:"
MULTI_SELECT_EXCLUSIVE_CHOICES = {"Không", "Không rõ", "Trả lời luôn"}
MULTI_SELECT_COMBINED_PREFIXES = ("Cả ", "Nhiều triệu chứng", "Chỉ 1-2")
ANSWER_NOW_CHOICE = "Trả lời luôn"
ANSWER_NOW_ICON = "⏭️"
ANSWER_NOW_ACK = "Đang tổng hợp câu trả lời..."
DONE_ICON = "✅"
SELECTED_ICON = "✓"
_INCOMING_ICON_PREFIXES = (f"{ANSWER_NOW_ICON} ", f"{DONE_ICON} ", f"{SELECTED_ICON} ")


def _decorate_choice_label(choice: str) -> str:
    if choice == ANSWER_NOW_CHOICE:
        return f"{ANSWER_NOW_ICON} {choice}"
    return choice


def _strip_choice_icon(text: str) -> str:
    for prefix in _INCOMING_ICON_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


@dataclass
class MultiSelectState:
    choices: tuple[str, ...]
    selected: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class ModeRetryState:
    question: str
    mode: str


_MULTI_SELECTS: dict[str, MultiSelectState] = {}
_CHAT_MODE_DEFAULTS: dict[str, str] = {}
_MODE_RETRIES: dict[str, ModeRetryState] = {}
# Chats (keyed by str(chat_id)) awaiting a typed custom top-up amount.
_TOPUP_PENDING: dict[str, bool] = {}

TOPUP_PRESETS = (10_000, 20_000, 50_000, 100_000, 200_000, 500_000)
TOPUP_MIN = 10_000
TOPUP_MAX = 10_000_000
TOPUP_CALLBACK_PREFIX = "topup:set:"
TOPUP_CUSTOM_CALLBACK = "topup:custom"
TOPUP_CANCEL_CALLBACK = "topup:cancel"
BALANCE_REFRESH_CALLBACK = "balance:refresh"
QR_IMAGE_BASE = "https://api.qrserver.com/v1/create-qr-code/"
VIETQR_IMAGE_BASE = "https://img.vietqr.io/image"


def _selection_confirmation_text(selection: str) -> str:
    return f"Người dùng chọn: {selection}"


def _send_url() -> str:
    return f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"


def _api_url(method: str) -> str:
    return f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


async def setup_bot_menu() -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN chưa cấu hình; bỏ qua thiết lập menu.")
        return
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(_api_url("setMyCommands"), json={"commands": BOT_COMMANDS})
    if r.status_code >= 400:
        logger.warning("Telegram setMyCommands failed → %s %s", r.status_code, r.text[:300])
    else:
        logger.info("Telegram command menu configured.")


async def _send_typing_action(chat_id: int | str) -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN chưa cấu hình; bỏ qua typing action.")
        return
    payload = {
        "chat_id": chat_id,
        "action": "typing",
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(_api_url("sendChatAction"), json=payload)
    logger.info("Telegram typing action → %s %s", r.status_code, r.text[:200])


async def _keep_typing(chat_id: int | str, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await _send_typing_action(chat_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Telegram typing action failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=TYPING_REFRESH_SECONDS)
        except asyncio.TimeoutError:
            pass


_CODE_FENCE_RE = re.compile(r"```([\s\S]*?)```")


def _format_non_code_markdown(text: str) -> str:
    s = _html.escape(text)
    # Escape first; these regexes introduce the only Telegram HTML tags.
    s = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", s)
    s = re.sub(r"^\s{0,3}#{1,6}\s*(.+)$", r"<b>\1</b>", s, flags=re.MULTILINE)
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*([^*\n]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"__([^_\n]+)__", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\*\w])\*([^*\n]+)\*(?!\w)", r"<i>\1</i>", s)
    return s


def _md_to_tg_html(text: str) -> str:
    """Convert a subset of Markdown to Telegram-compatible HTML."""
    parts: list[str] = []
    pos = 0
    for match in _CODE_FENCE_RE.finditer(text):
        parts.append(_format_non_code_markdown(text[pos:match.start()]))
        parts.append(f"<pre>{_html.escape(match.group(1))}</pre>")
        pos = match.end()
    parts.append(_format_non_code_markdown(text[pos:]))
    return "".join(parts)


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _byte_limited_prefix_len(text: str, limit: int) -> int:
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _utf8_len(text[:mid]) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _split_for_telegram(text: str, limit: int = TG_MAX_LEN) -> list[str]:
    if _utf8_len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while _utf8_len(remaining) > limit:
        prefix_len = _byte_limited_prefix_len(remaining, limit)
        cut = remaining.rfind("\n", 0, prefix_len + 1)
        if cut <= 0:
            cut = prefix_len
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


def _choice_keyboard(choices: list[str] | tuple[str, ...]) -> dict:
    rows = []
    for i in range(0, len(choices), 2):
        rows.append([{"text": _decorate_choice_label(choice)} for choice in choices[i:i + 2]])
    return {
        "keyboard": rows,
        "resize_keyboard": True,
        "one_time_keyboard": True,
    }


def _new_multi_select_token() -> str:
    return secrets.token_urlsafe(8)


def _multi_select_choices(choices: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        choice
        for choice in choices
        if not choice.startswith(MULTI_SELECT_COMBINED_PREFIXES)
    )


def _multi_choice_keyboard(
    token: str,
    choices: list[str] | tuple[str, ...],
    selected: set[int],
) -> dict:
    rows = []
    buttons = []
    for index, choice in enumerate(choices):
        if index in selected:
            label = f"{SELECTED_ICON} {choice}"
        elif choice == ANSWER_NOW_CHOICE:
            label = f"{ANSWER_NOW_ICON} {choice}"
        else:
            label = choice
        buttons.append({"text": label, "callback_data": f"{MULTI_SELECT_PREFIX}{token}:{index}"})
    for i in range(0, len(buttons), 2):
        rows.append(buttons[i:i + 2])
    rows.append([{"text": f"{DONE_ICON} {MULTI_SELECT_DONE}", "callback_data": f"{MULTI_SELECT_PREFIX}{token}:done"}])
    return {"inline_keyboard": rows}


def _menu_keyboard() -> dict:
    return {
        "inline_keyboard": [
            [
                {"text": "📝 Hướng dẫn", "callback_data": "cmd:/help"},
                {"text": "⚙️ Chế độ", "callback_data": "cmd:/mode"},
            ],
            [
                {"text": "👨‍⚕️ Bác sĩ", "callback_data": "cmd:/doctor"},
                {"text": "⛔ Kết thúc", "callback_data": "cmd:/end"},
            ],
            [
                {"text": "💰 Nạp tiền", "callback_data": "cmd:/topup"},
                {"text": "💳 Số dư", "callback_data": "cmd:/balance"},
            ],
            [
                {"text": "🧾 Công nợ", "callback_data": "cmd:/paydebt"},
                {"text": "🔄 Lượt mới", "callback_data": "cmd:/new"},
            ],
            [
                {"text": "🧠 Bộ nhớ", "callback_data": "menu:memory"},
            ],
            [
                {"text": "❌ Đóng", "callback_data": "cmd:close"},
            ],
        ]
    }


def _menu_memory_keyboard() -> dict:
    return {
        "inline_keyboard": [
            [
                {"text": "🧠 Xem trạng thái", "callback_data": "cmd:/memory"},
            ],
            [
                {"text": "✅ Bật bộ nhớ", "callback_data": "cmd:/memoryon"},
                {"text": "⛔ Tắt bộ nhớ", "callback_data": "cmd:/memoryoff"},
            ],
            [
                {"text": "🗑 Quên chủ thể", "callback_data": "cmd:/forget"},
                {"text": "🗑 Xóa toàn bộ", "callback_data": "cmd:/forgetall"},
            ],
            [
                {"text": "🔙 Quay lại", "callback_data": "menu:main"},
            ],
        ]
    }


def _mode_keyboard(selected_mode: str) -> dict:
    selected = normalize_mode(selected_mode)
    buttons = []
    for mode, label in (
        ("auto", "Auto"),
        ("information", "Thông tin"),
        ("diagnostic", "Chẩn đoán"),
    ):
        text = f"{SELECTED_ICON} {label}" if mode == selected else label
        buttons.append({"text": text, "callback_data": f"{MODE_CALLBACK_PREFIX}{mode}"})
    return {"inline_keyboard": [buttons]}


def _mode_retry_keyboard(mode: str, question: str) -> dict:
    token = secrets.token_urlsafe(8)
    normalized_mode = normalize_mode(mode)
    _MODE_RETRIES[token] = ModeRetryState(question=question, mode=normalized_mode)
    return {
        "inline_keyboard": [[
            {
                "text": f"Trả lời ở chế độ {mode_label(normalized_mode)}",
                "callback_data": f"{MODE_RETRY_PREFIX}{token}",
            }
        ]]
    }


async def send_text(
    chat_id: int | str,
    text: str,
    choices: list[str] | tuple[str, ...] = (),
    *,
    selection_mode: str = "single",
    inline_keyboard: dict | None = None,
) -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN chưa cấu hình; bỏ qua gửi tin.")
        return
    total_start = time.perf_counter()
    html = _md_to_tg_html(text)
    chunks = _split_for_telegram(html)
    async with httpx.AsyncClient(timeout=20.0) as client:
        for idx, chunk in enumerate(chunks, 1):
            payload = {
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            multi_token = ""
            if idx == len(chunks) and choices:
                if inline_keyboard is not None:
                    payload["reply_markup"] = inline_keyboard
                elif selection_mode == "multi":
                    choices = _multi_select_choices(choices)
                    multi_token = _new_multi_select_token()
                    payload["reply_markup"] = _multi_choice_keyboard(multi_token, choices, set())
                else:
                    payload["reply_markup"] = _choice_keyboard(choices)
            elif idx == len(chunks) and inline_keyboard is not None:
                payload["reply_markup"] = inline_keyboard
            elif idx == len(chunks):
                payload["reply_markup"] = {"remove_keyboard": True}
            chunk_start = time.perf_counter()
            r = await client.post(_send_url(), json=payload)
            if multi_token and r.status_code < 400:
                _MULTI_SELECTS[multi_token] = MultiSelectState(tuple(choices))
            logger.info("Telegram send → %s %s", r.status_code, r.text[:200])
            logger.info("Telegram timing stage=send_chunk chunk=%d/%d ms=%.1f status=%s chars=%d",
                        idx, len(chunks), _elapsed_ms(chunk_start), r.status_code, len(chunk))
            if r.status_code >= 400:
                # fallback: send as plain text so user still sees something
                for fallback_chunk in _split_for_telegram(chunk, TG_HARD_MAX_BYTES):
                    await client.post(_send_url(), json={"chat_id": chat_id, "text": fallback_chunk})
    logger.info("Telegram timing stage=send_total ms=%.1f chunks=%d chars=%d",
                _elapsed_ms(total_start), len(chunks), len(text))


async def _send_answer_now_ack(chat_id: int | str) -> int | None:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN chưa cấu hình; bỏ qua gửi tin.")
        return None
    payload = {
        "chat_id": chat_id,
        "text": ANSWER_NOW_ACK,
        "reply_markup": {"remove_keyboard": True},
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(_send_url(), json=payload)
    logger.info("Telegram answer-now ack → %s %s", r.status_code, r.text[:200])
    if r.status_code >= 400:
        return None
    try:
        return r.json().get("result", {}).get("message_id")
    except Exception:
        return None


async def send_photo(chat_id: int | str, photo_file_id: str, caption: str = "") -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN chưa cấu hình; bỏ qua gửi ảnh.")
        return
    payload = {"chat_id": chat_id, "photo": photo_file_id}
    if caption:
        payload["caption"] = caption
        payload["parse_mode"] = "HTML"
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(_api_url("sendPhoto"), json=payload)
    logger.info("Telegram sendPhoto(file_id) → %s %s", r.status_code, r.text[:200])


async def send_voice(chat_id: int | str, voice_file_id: str) -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN chưa cấu hình; bỏ qua gửi voice.")
        return
    payload = {"chat_id": chat_id, "voice": voice_file_id}
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(_api_url("sendVoice"), json=payload)
    logger.info("Telegram sendVoice → %s %s", r.status_code, r.text[:200])


async def copy_message(chat_id: int | str, from_chat_id: int | str, message_id: int | str) -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN chưa cấu hình; bỏ qua sao chép tin nhắn.")
        return
    payload = {
        "chat_id": chat_id,
        "from_chat_id": from_chat_id,
        "message_id": message_id,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(_api_url("copyMessage"), json=payload)
    logger.info("Telegram copyMessage → %s %s", r.status_code, r.text[:200])
    if r.status_code >= 400:
        raise RuntimeError(f"Telegram copyMessage failed: {r.status_code}")


def _rating_keyboard(token: str) -> dict:
    return {
        "inline_keyboard": [
            [
                {"text": str(score), "callback_data": f"rate:{token}:{score}"}
                for score in range(1, 6)
            ]
        ]
    }


def _parse_rating_callback(data: str) -> tuple[str, int] | None:
    parts = data.split(":")
    if len(parts) != 3 or parts[0] != "rate":
        return None
    try:
        rating = int(parts[2])
    except ValueError:
        return None
    if rating < 1 or rating > 5:
        return None
    return parts[1], rating


async def _send_rating_prompt(chat_id: int | str, token: str) -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN chưa cấu hình; bỏ qua gửi đánh giá.")
        return
    payload = {
        "chat_id": chat_id,
        "text": RATING_PROMPT,
        "reply_markup": _rating_keyboard(token),
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(_send_url(), json=payload)
    logger.info("Telegram rating prompt → %s %s", r.status_code, r.text[:200])


async def _answer_callback_query(callback_query_id: str, text: str) -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN chưa cấu hình; bỏ qua callback answer.")
        return
    payload = {
        "callback_query_id": callback_query_id,
        "text": text,
        "show_alert": False,
        "cache_time": 0,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(_api_url("answerCallbackQuery"), json=payload)
    logger.info("Telegram callback answer → %s %s", r.status_code, r.text[:200])


async def _edit_message_reply_markup(
    chat_id: int | str,
    message_id: int,
    reply_markup: dict,
) -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN chưa cấu hình; bỏ qua sửa reply_markup.")
        return
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "reply_markup": reply_markup,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(_api_url("editMessageReplyMarkup"), json=payload)
    logger.info("Telegram edit reply_markup → %s %s", r.status_code, r.text[:200])


async def _edit_message_text(
    chat_id: int | str,
    message_id: int,
    text: str,
    inline_keyboard: dict | None = None,
) -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN chưa cấu hình; bỏ qua sửa nội dung.")
        return
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
        "parse_mode": "HTML",
    }
    if inline_keyboard is not None:
        payload["reply_markup"] = inline_keyboard
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(_api_url("editMessageText"), json=payload)
    logger.info("Telegram edit text → %s %s", r.status_code, r.text[:200])


def _parse_multi_select_callback(data: str) -> tuple[str, str] | None:
    if not data.startswith(MULTI_SELECT_PREFIX):
        return None
    parts = data.split(":", 2)
    if len(parts) != 3 or not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2]


async def _handle_mode_callback(callback_query: dict) -> bool:
    data = str(callback_query.get("data") or "")
    if not data.startswith(MODE_CALLBACK_PREFIX):
        return False
    mode = normalize_mode(data[len(MODE_CALLBACK_PREFIX):])
    callback_query_id = callback_query.get("id")
    message = callback_query.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    message_id = message.get("message_id")
    if chat_id is None:
        if callback_query_id:
            await _answer_callback_query(str(callback_query_id), "Không xử lý được chế độ này.")
        return True

    _CHAT_MODE_DEFAULTS[str(chat_id)] = mode
    if message_id is not None:
        await _edit_message_reply_markup(chat_id, int(message_id), _mode_keyboard(mode))
    if callback_query_id:
        await _answer_callback_query(
            str(callback_query_id),
            f"Đã chọn chế độ {mode_label(mode)}.",
        )
    return True


async def _handle_mode_retry_callback(
    callback_query: dict,
    background_tasks: BackgroundTasks,
) -> bool:
    data = str(callback_query.get("data") or "")
    if not data.startswith(MODE_RETRY_PREFIX):
        return False
    token = data[len(MODE_RETRY_PREFIX):]
    state = _MODE_RETRIES.pop(token, None)
    callback_query_id = callback_query.get("id")
    if state is None:
        if callback_query_id:
            await _answer_callback_query(str(callback_query_id), "Lựa chọn này đã hết hạn.")
        return True

    message = callback_query.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    if chat_id is None:
        if callback_query_id:
            await _answer_callback_query(str(callback_query_id), "Không xử lý được lựa chọn này.")
        return True

    label = mode_label(state.mode)
    if callback_query_id:
        await _answer_callback_query(str(callback_query_id), f"Đang trả lời ở chế độ {label}.")
    await send_text(chat_id, f"Trả lời ở chế độ {label}: {state.question}")
    background_tasks.add_task(_answer_and_send, chat_id, state.question, state.mode)
    return True


async def _handle_multi_select_callback(
    callback_query: dict,
    background_tasks: BackgroundTasks,
) -> bool:
    parsed = _parse_multi_select_callback(str(callback_query.get("data") or ""))
    if parsed is None:
        return False
    token, action = parsed
    callback_query_id = callback_query.get("id")
    state = _MULTI_SELECTS.get(token)
    if state is None:
        if callback_query_id:
            await _answer_callback_query(str(callback_query_id), "Lựa chọn này đã hết hạn.")
        return True

    message = callback_query.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    message_id = message.get("message_id")
    if chat_id is None or message_id is None:
        if callback_query_id:
            await _answer_callback_query(str(callback_query_id), "Không xử lý được lựa chọn này.")
        return True

    if action == "done":
        if not state.selected:
            if callback_query_id:
                await _answer_callback_query(str(callback_query_id), "Bạn chọn ít nhất một ý nhé.")
            return True
        selected_text = ", ".join(
            state.choices[index] for index in sorted(state.selected)
        )
        _MULTI_SELECTS.pop(token, None)
        await _edit_message_reply_markup(chat_id, int(message_id), {"inline_keyboard": []})
        if callback_query_id:
            await _answer_callback_query(str(callback_query_id), "Đã chọn.")
        await send_text(chat_id, _selection_confirmation_text(selected_text))
        background_tasks.add_task(_answer_and_send, chat_id, selected_text)
        return True

    try:
        index = int(action)
    except ValueError:
        return True
    if index < 0 or index >= len(state.choices):
        return True

    choice = state.choices[index]
    if choice in MULTI_SELECT_EXCLUSIVE_CHOICES:
        _MULTI_SELECTS.pop(token, None)
        await _edit_message_reply_markup(chat_id, int(message_id), {"inline_keyboard": []})
        if callback_query_id:
            await _answer_callback_query(str(callback_query_id), "Đã chọn.")
        await send_text(chat_id, _selection_confirmation_text(choice))
        background_tasks.add_task(_answer_and_send, chat_id, choice)
        return True

    if index in state.selected:
        state.selected.remove(index)
    else:
        state.selected.add(index)
    await _edit_message_reply_markup(
        chat_id,
        int(message_id),
        _multi_choice_keyboard(token, state.choices, state.selected),
    )
    if callback_query_id:
        await _answer_callback_query(str(callback_query_id), "Đã cập nhật.")
    return True


async def _delete_rating_message(chat_id: int | str, message_id: int) -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN chưa cấu hình; bỏ qua xóa tin nhắn đánh giá.")
        return
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(_api_url("deleteMessage"), json=payload)
    logger.info("Telegram rating message deletion → %s %s", r.status_code, r.text[:200])


async def _handle_rating_callback(callback_query: dict) -> bool:
    parsed = _parse_rating_callback(str(callback_query.get("data") or ""))
    if parsed is None:
        return False
    token, rating = parsed
    saved = record_feedback_rating(token, rating)
    callback_query_id = callback_query.get("id")
    if callback_query_id:
        text = "Cảm ơn bạn đã đánh giá." if saved else "Đánh giá này không còn hiệu lực."
        await _answer_callback_query(str(callback_query_id), text)
    message = callback_query.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    message_id = message.get("message_id")
    if chat_id is not None and message_id is not None:
        await _delete_rating_message(chat_id, message_id)
    return True


async def _answer_and_send(chat_id: int | str, text: str, mode: str | None = None) -> None:
    total_start = time.perf_counter()
    session_id = f"tg:{chat_id}"
    answer_mode = normalize_mode(mode or _CHAT_MODE_DEFAULTS.get(str(chat_id)))
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(_keep_typing(chat_id, stop_typing))
    answer_now_ack_message_id: int | None = None
    try:
        if text == ANSWER_NOW_CHOICE:
            try:
                answer_now_ack_message_id = await _send_answer_now_ack(chat_id)
            except Exception:
                logger.exception("Telegram answer-now keyboard removal failed")
        if answer_mode == "auto":
            reply = await asyncio.to_thread(answer_with_choices, text, session_id=session_id)
        else:
            reply = await asyncio.to_thread(
                answer_with_choices,
                text,
                session_id=session_id,
                mode=answer_mode,
            )
    except Exception:
        logger.exception("Telegram answer failed")
        reply = ChatReply(TECHNICAL_ERROR_REPLY)
    finally:
        stop_typing.set()
        typing_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await typing_task

    try:
        inline_keyboard = (
            _mode_retry_keyboard(reply.suggest_mode, reply.retry_question or text)
            if reply.suggest_mode
            else None
        )
        if reply.doctor_offer:
            telegram_doctor.register_handoff_context(
                chat_id,
                text,
                reply.text,
                reply.doctor_specialty,
            )
        if inline_keyboard is None:
            await send_text(
                chat_id,
                reply.text,
                reply.choices,
                selection_mode=reply.selection_mode,
            )
        else:
            await send_text(
                chat_id,
                reply.text,
                reply.choices,
                selection_mode=reply.selection_mode,
                inline_keyboard=inline_keyboard,
            )
    except Exception:
        logger.exception("Telegram send failed")
    else:
        if answer_now_ack_message_id is not None:
            try:
                await _delete_message(chat_id, answer_now_ack_message_id)
            except Exception:
                logger.exception("Telegram answer-now ack deletion failed")
        if not reply.choices and not reply.suggest_mode:
            try:
                token = create_feedback_request(session_id, "telegram", str(chat_id), text, reply.text)
                await _send_rating_prompt(chat_id, token)
            except Exception:
                logger.exception("Telegram rating prompt failed")
        if reply.doctor_offer:
            try:
                await send_text(
                    chat_id,
                    DOCTOR_OFFER_PROMPT,
                    inline_keyboard=telegram_doctor.handoff_keyboard(chat_id),
                )
            except Exception:
                logger.exception("Telegram doctor offer failed")

    logger.info("Telegram timing stage=background_total ms=%.1f",
                _elapsed_ms(total_start))


def _command(text: str) -> str:
    first = text.strip().split(maxsplit=1)[0].lower()
    return first.split("@", 1)[0]


def _topup_keyboard() -> dict:
    rows = []
    buttons = [
        {"text": f"{amount:,} VND", "callback_data": f"{TOPUP_CALLBACK_PREFIX}{amount}"}
        for amount in TOPUP_PRESETS
    ]
    for i in range(0, len(buttons), 2):
        rows.append(buttons[i:i + 2])
    rows.append([{"text": "✏️ Nhập số khác", "callback_data": TOPUP_CUSTOM_CALLBACK}])
    return {"inline_keyboard": rows}


def _qr_image_url(qr_code: str) -> str:
    query = urllib.parse.urlencode({"size": "400x400", "data": qr_code})
    return f"{QR_IMAGE_BASE}?{query}"


def _vietqr_image_url(
    *,
    bank_bin: str,
    account_number: str,
    amount: int,
    content: str,
    account_name: str | None,
) -> str:
    """Branded VietQR image from PayOS's monitored account details.

    Uses PayOS's own bank bin/account and the FULL PayOS memo (`content`,
    including the unique reconciliation reference) as addInfo. Verified by
    decoding the rendered image: the encoded content/account/amount match
    PayOS's authoritative qr_code exactly, so money still reconciles.
    """
    params = {"amount": str(amount), "addInfo": content}
    if account_name:
        params["accountName"] = account_name
    query = urllib.parse.urlencode(params)
    return f"{VIETQR_IMAGE_BASE}/{bank_bin}-{account_number}-compact2.png?{query}"


def _topup_cancel_keyboard() -> dict:
    return {"inline_keyboard": [[{"text": "❌ Hủy", "callback_data": TOPUP_CANCEL_CALLBACK}]]}


async def _delete_message(chat_id: int | str, message_id: int) -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN chưa cấu hình; bỏ qua xóa tin nhắn.")
        return
    payload = {"chat_id": chat_id, "message_id": message_id}
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(_api_url("deleteMessage"), json=payload)
    logger.info("Telegram deleteMessage → %s %s", r.status_code, r.text[:200])


async def _delete_order_qr(order: dict) -> None:
    """Delete the QR message for a paid order, using the persisted message id.

    Works across restarts (id is on the order row) and for admin reconciles,
    deriving the chat from the order's tg:<chat_id> account_id.
    """
    qr_message_id = order.get("qr_message_id")
    account_id = order.get("account_id") or ""
    if qr_message_id is None or not account_id.startswith("tg:"):
        return
    qr_chat_id = account_id[len("tg:"):]
    try:
        await _delete_message(qr_chat_id, int(qr_message_id))
    except Exception:
        logger.exception("Failed to delete QR for order %s", order.get("order_code"))


async def _send_topup_qr(
    chat_id: int | str,
    photo_url: str,
    caption: str,
    inline_keyboard: dict | None = None,
) -> int | None:
    """Send the QR photo. Returns the Telegram message_id, or None."""
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN chưa cấu hình; bỏ qua gửi QR.")
        return None
    payload = {
        "chat_id": chat_id,
        "photo": photo_url,
        "caption": caption,
        "parse_mode": "HTML",
    }
    if inline_keyboard is not None:
        payload["reply_markup"] = inline_keyboard
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(_api_url("sendPhoto"), json=payload)
    logger.info("Telegram sendPhoto → %s %s", r.status_code, r.text[:200])
    try:
        return r.json().get("result", {}).get("message_id")
    except Exception:
        return None


async def _start_topup_payment(chat_id: int | str, amount: int) -> None:
    order_code = int(time.time() * 1000)
    description = f"NAPTIEN{order_code}"
    return_url = f"{PUBLIC_BASE_URL}/topup/return" if PUBLIC_BASE_URL else "https://example.com/return"
    cancel_url = f"{PUBLIC_BASE_URL}/topup/cancel" if PUBLIC_BASE_URL else "https://example.com/cancel"
    try:
        payment = payos_create_payment(
            order_code=order_code,
            amount=amount,
            description=description,
            return_url=return_url,
            cancel_url=cancel_url,
        )
    except Exception:
        logger.exception("PayOS create_payment failed")
        await send_text(chat_id, "Xin lỗi, không tạo được mã thanh toán. Bạn thử lại sau nhé.")
        return

    create_order(order_code, f"tg:{chat_id}", amount, payment.get("payment_link_id"))
    # PayOS embeds the full transfer memo (incl. its unique reconciliation
    # reference) in the qr_code. Use that exact content both for the branded
    # VietQR image and for what we show the user, so the two always agree and
    # the transfer reconciles.
    content = payos_extract_transfer_content(payment["qr_code"]) or description
    bank_bin = payment.get("bin")
    account_number = payment.get("account_number")
    if bank_bin and account_number:
        photo_url = _vietqr_image_url(
            bank_bin=bank_bin,
            account_number=account_number,
            amount=amount,
            content=content,
            account_name=payment.get("account_name"),
        )
    else:
        photo_url = _qr_image_url(payment["qr_code"])
    caption = (
        f"Quét mã VietQR để nạp <b>{amount:,} VND</b>.\n"
        f"Nội dung chuyển khoản: <code>{content}</code>\n\n"
        f"Hoặc mở liên kết thanh toán: {payment.get('checkout_url')}"
    )
    message_id = await _send_topup_qr(
        chat_id, photo_url, caption, _topup_cancel_keyboard()
    )
    if message_id is not None:
        # Persist so the QR can be deleted after payment even across restarts.
        set_order_qr_message(order_code, message_id)



async def _handle_topup_callback(callback_query: dict) -> bool:
    data = str(callback_query.get("data") or "")
    if not data.startswith("topup:"):
        return False
    callback_query_id = callback_query.get("id")
    message = callback_query.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    if chat_id is None:
        if callback_query_id:
            await _answer_callback_query(str(callback_query_id), "Không xử lý được lựa chọn này.")
        return True

    if data == TOPUP_CANCEL_CALLBACK:
        _TOPUP_PENDING.pop(str(chat_id), None)
        message_id = message.get("message_id")
        if message_id is not None:
            await _delete_message(chat_id, int(message_id))
        if callback_query_id:
            await _answer_callback_query(str(callback_query_id), "Đã hủy.")
        return True

    if data == TOPUP_CUSTOM_CALLBACK:
        _TOPUP_PENDING[str(chat_id)] = True
        if callback_query_id:
            await _answer_callback_query(str(callback_query_id), "Nhập số tiền bạn muốn nạp.")
        await send_text(
            chat_id,
            f"Bạn muốn nạp bao nhiêu? Nhập số tiền (VND), từ {TOPUP_MIN:,} đến {TOPUP_MAX:,}.\n"
            "Hoặc bấm Hủy để hỏi câu hỏi khác.",
            inline_keyboard=_topup_cancel_keyboard(),
        )
        return True

    if data.startswith(TOPUP_CALLBACK_PREFIX):
        try:
            amount = int(data[len(TOPUP_CALLBACK_PREFIX):])
        except ValueError:
            return True
        if callback_query_id:
            await _answer_callback_query(str(callback_query_id), "Đang tạo mã thanh toán…")
        await _start_topup_payment(chat_id, amount)
        return True

    return True


def _parse_topup_amount(text: str) -> int | None:
    digits = re.sub(r"[.,\s]", "", text.strip())
    if not digits.isdigit():
        return None
    return int(digits)


async def _handle_pending_topup_amount(chat_id: int | str, text: str) -> bool:
    if not _TOPUP_PENDING.get(str(chat_id)):
        return False
    amount = _parse_topup_amount(text)
    if amount is None or amount < TOPUP_MIN or amount > TOPUP_MAX:
        # Keep pending mode on so the user can simply re-type a valid amount.
        await send_text(
            chat_id,
            f"Số tiền không hợp lệ. Nhập số từ {TOPUP_MIN:,} đến {TOPUP_MAX:,} VND, "
            "hoặc bấm Hủy để hỏi câu hỏi khác.",
            inline_keyboard=_topup_cancel_keyboard(),
        )
        return True
    _TOPUP_PENDING.pop(str(chat_id), None)
    await _start_topup_payment(chat_id, amount)
    return True


def _is_admin(user_id: int | None) -> bool:
    return user_id is not None and user_id in TELEGRAM_ADMIN_IDS


def _parse_order_code(arg: str) -> int | None:
    """Extract an order_code from whatever the admin pasted.

    Accepts a bare number, the "NAPTIEN<code>" memo shown in the QR caption,
    or the full bank content "CS<ref> NAPTIEN<code>". Ignores zero-width and
    surrounding whitespace from copy-paste. Returns the last run of digits.
    """
    matches = re.findall(r"\d+", arg)
    if not matches:
        return None
    return int(matches[-1])


async def _handle_admin_paid(chat_id: int | str, text: str, user_id: int | None) -> None:
    """Admin-only: reconcile a stuck top-up order by crediting its balance.

    Used when a real payment succeeded at the bank but the PayOS webhook never
    credited it. Gated on the numeric-id allowlist; idempotent via the
    pending->paid transition so it can never double-credit.
    """
    if not _is_admin(user_id):
        await send_text(chat_id, "Bạn không có quyền dùng lệnh này.")
        return

    # Everything after the command word is the argument (may contain spaces,
    # e.g. a pasted "CS... NAPTIEN..." transfer content).
    _, _, arg = text.strip().partition(" ")
    order_code = _parse_order_code(arg)
    if order_code is None:
        await send_text(chat_id, "Cú pháp: /admin_paid <order_code>")
        return

    order = get_order(order_code)
    if order is None:
        await send_text(chat_id, f"Không tìm thấy đơn nạp {order_code}.")
        return

    if not mark_order_paid(order_code):
        await send_text(
            chat_id,
            f"Đơn {order_code} đã được xử lý trước đó (trạng thái: {order['status']}).",
        )
        return

    balance = apply_payment(order["account_id"], order["amount"])
    log_admin_credit(
        order_code=order_code,
        admin_user_id=user_id,
        account_id=order["account_id"],
        amount=order["amount"],
    )
    await send_text(
        chat_id,
        f"Đã cộng {order['amount']:,} VND cho {order['account_id']} "
        f"(đơn {order_code}). Số dư mới: {balance:,} VND.",
    )

    # Notify the target user and remove their stale QR, same as the webhook path.
    account_id = order["account_id"]
    if account_id.startswith("tg:"):
        target_chat = account_id[len("tg:"):]
        try:
            await send_text(
                target_chat,
                f"✅ Đã nhận thanh toán {order['amount']:,} VND.\n"
                f"Số dư hiện tại: {balance:,} VND.",
            )
        except Exception:
            logger.exception("Admin reconcile notify failed for %s", account_id)
    await _delete_order_qr(order)


def _balance_text(balance: int) -> str:
    return f"Số dư hiện tại của bạn: {balance:,} VND."


def _balance_keyboard() -> dict:
    return {"inline_keyboard": [[{"text": "🔄 Làm mới số dư", "callback_data": BALANCE_REFRESH_CALLBACK}]]}


async def _handle_balance_callback(callback_query: dict) -> bool:
    data = str(callback_query.get("data") or "")
    if data != BALANCE_REFRESH_CALLBACK:
        return False
    callback_query_id = callback_query.get("id")
    message = callback_query.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    message_id = message.get("message_id")
    if chat_id is None or message_id is None:
        if callback_query_id:
            await _answer_callback_query(str(callback_query_id), "Không làm mới được số dư.")
        return True
    balance = get_balance(f"tg:{chat_id}")
    await _edit_message_text(
        chat_id, int(message_id), _balance_text(balance), _balance_keyboard()
    )
    if callback_query_id:
        await _answer_callback_query(str(callback_query_id), "Đã làm mới.")
    return True


async def _handle_command(
    chat_id: int | str,
    text: str,
    chat_type: str = "private",
    user_id: int | None = None,
) -> bool:
    cmd = _command(text)
    if cmd == "/start":
        await send_text(chat_id, START_TEXT)
        return True
    if cmd == "/help":
        await send_text(chat_id, HELP_TEXT)
        return True
    if cmd == "/whoami":
        await send_text(chat_id, f"Telegram user id của bạn: <code>{user_id}</code>")
        return True
    if cmd == "/admin_paid":
        await _handle_admin_paid(chat_id, text, user_id)
        return True
    if cmd == "/menu":
        await send_text(
            chat_id,
            "📋 **Menu các lệnh hỗ trợ:**",
            inline_keyboard=_menu_keyboard(),
        )
        return True
    if cmd == "/mode":
        current = _CHAT_MODE_DEFAULTS.get(str(chat_id), "auto")
        await send_text(
            chat_id,
            (
                "Lựa chọn chế độ trả lời\n\n"
                "**Auto**: bot tự chọn cách trả lời phù hợp với câu hỏi.\n"
                "**Thông tin**: trả lời thông tin về bệnh, thuốc, phòng ngừa, điều trị và chăm sóc.\n"
                "**Chẩn đoán**: tư vấn triệu chứng, sàng lọc an toàn và hướng dẫn khi cần đi khám.\n\n"
                f"Chế độ hiện tại: **{mode_label(current)}**"
            ),
            inline_keyboard=_mode_keyboard(current),
        )
        return True
    if cmd == "/topup":
        if chat_type != "private":
            await send_text(chat_id, "Lệnh nạp tiền chỉ dùng được trong cuộc trò chuyện riêng tư với bot.")
            return True
        await send_text(
            chat_id,
            "Chọn số tiền muốn nạp, hoặc nhập số khác:",
            inline_keyboard=_topup_keyboard(),
        )
        return True
    if cmd == "/balance":
        balance = get_balance(f"tg:{chat_id}")
        await send_text(chat_id, _balance_text(balance), inline_keyboard=_balance_keyboard())
        return True
    if cmd == "/paydebt":
        status = debt_status(f"tg:{chat_id}")
        if not status["in_debt"]:
            await send_text(chat_id, "Bạn không có khoản nợ nào cần thanh toán.")
            return True
        await send_text(
            chat_id,
            f"Bạn đang nợ {status['debt']:,} VND. Thanh toán {status['payoff_amount']:,} VND "
            "(gồm 10% phí) để mở lại đầy đủ tính năng. Đang tạo mã thanh toán…",
        )
        await _start_topup_payment(chat_id, status["payoff_amount"])
        return True
    if cmd == "/doctor":
        return await telegram_doctor.handle_doctor_command(chat_id)
    if cmd == "/end":
        if await telegram_doctor.handle_end(chat_id):
            return True
        await send_text(chat_id, "Bạn không có phiên tư vấn bác sĩ đang hoạt động.")
        return True
    if cmd == "/new":
        clear_session(f"tg:{chat_id}")
        _CHAT_MODE_DEFAULTS.pop(str(chat_id), None)
        await send_text(chat_id, "Tôi đã xóa ngữ cảnh hội thoại hiện tại. Bạn có thể bắt đầu câu hỏi mới.")
        return True
    return False


@router.post("/webhook/telegram")
async def telegram_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> dict:
    if TELEGRAM_WEBHOOK_SECRET and x_telegram_bot_api_secret_token != TELEGRAM_WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret token")

    update = await request.json()
    update_id = update.get("update_id")
    if update_id is not None and not reserve_webhook_update("telegram", update_id):
        logger.info("Duplicate Telegram update ignored: %s", update_id)
        return {"ok": True}

    callback_query = update.get("callback_query") or {}
    if callback_query:
        if await _handle_mode_retry_callback(callback_query, background_tasks):
            return {"ok": True}
        if await _handle_mode_callback(callback_query):
            return {"ok": True}
        if await _handle_topup_callback(callback_query):
            return {"ok": True}
        if await _handle_balance_callback(callback_query):
            return {"ok": True}
        if await telegram_doctor.handle_doctor_callback(callback_query):
            return {"ok": True}
        if await _handle_multi_select_callback(callback_query, background_tasks):
            return {"ok": True}
        await _handle_rating_callback(callback_query)
        return {"ok": True}

    message = update.get("message") or {}
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    chat_type = chat.get("type", "private")
    user_id = (message.get("from") or {}).get("id")
    if not chat_id:
        return {"ok": True}

    text = message.get("text", "")

    # Commands take priority so /end can always escape an active consultation.
    if text.startswith("/") and await _handle_command(chat_id, text, chat_type, user_id):
        return {"ok": True}

    # Active doctor consultation intercepts text/photo/voice before chatbot.
    if await telegram_doctor.relay_message(chat_id, message):
        return {"ok": True}

    if not text:
        return {"ok": True}

    # Bot-wide ban: an unpaid debt past its grace window blocks the chatbot.
    # Commands above still run so the user can /paydebt, /topup, /balance.
    ban = debt_status(f"tg:{chat_id}")
    if ban["banned"]:
        await send_text(
            chat_id,
            f"Tài khoản của bạn đang bị khóa do nợ {ban['debt']:,} VND quá hạn. "
            f"Vui lòng thanh toán {ban['payoff_amount']:,} VND (gồm 10% phí) bằng /paydebt để mở khóa.",
        )
        return {"ok": True}

    if str(chat_id) in _TOPUP_PENDING:
        background_tasks.add_task(_handle_pending_topup_amount, chat_id, text)
        return {"ok": True}

    background_tasks.add_task(_answer_and_send, chat_id, _strip_choice_icon(text))
    return {"ok": True}
