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
import html as _html
import logging
import re
import time

import httpx
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request

from src.chat import answer_with_choices
from src.chat.replies import ChatReply, TECHNICAL_ERROR_REPLY
from src.chat.storage.feedback import create_feedback_request, record_feedback_rating
from src.chat.storage.session import clear_session, reserve_webhook_update
from src.config import TELEGRAM_BOT_TOKEN, TELEGRAM_WEBHOOK_SECRET

logger = logging.getLogger(__name__)
router = APIRouter()

TG_MAX_LEN = 4000  # safe under Telegram's 4096-byte limit
TG_HARD_MAX_BYTES = 4096
BOT_COMMANDS = [
    {"command": "start", "description": "Bắt đầu bot"},
    {"command": "help", "description": "Cách đặt câu hỏi"},
    {"command": "menu", "description": "Danh sách lệnh"},
    {"command": "new", "description": "Xóa ngữ cảnh và bắt đầu lượt mới"},
]
START_TEXT = """Xin chào! Tôi là trợ lý y tế.

Bạn có thể gửi câu hỏi về triệu chứng, bệnh lý hoặc thuốc không kê đơn. Tôi sẽ trả lời bằng tiếng Việt dựa trên tài liệu y khoa trong hệ thống.

Ví dụ:
- Tôi bị ho và sốt 2 ngày nay, nên làm gì?
- Phòng bệnh cúm như thế nào?
- Paracetamol dùng để làm gì?

Nếu có triệu chứng nặng như khó thở, đau ngực dữ dội, lơ mơ, yếu liệt, co giật hoặc chảy máu nhiều, hãy đi cấp cứu ngay."""

HELP_TEXT = """Cách sử dụng:

- Mô tả rõ triệu chứng, thời gian bắt đầu, mức độ và bệnh nền nếu có.
- Với câu hỏi về thuốc, hãy gửi tên hoạt chất hoặc tên thuốc bạn muốn hỏi.
- Khi tôi hỏi thêm triệu chứng, bạn có thể trả lời ngắn gọn theo từng ý.

Lưu ý: Tôi hỗ trợ thông tin y tế, không thay thế bác sĩ. Nếu tình trạng nặng hoặc diễn tiến nhanh, hãy đi khám/cấp cứu."""

MENU_TEXT = """Menu lệnh:

/start - Bắt đầu và xem hướng dẫn
/help - Cách đặt câu hỏi hiệu quả
/menu - Danh sách lệnh
/new - Xóa ngữ cảnh hội thoại hiện tại
"""
RATING_PROMPT = "Bạn đánh giá câu trả lời này từ 1 đến 5 nhé."


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


_CODE_FENCE_RE = re.compile(r"```([\s\S]*?)```")


def _format_non_code_markdown(text: str) -> str:
    s = _html.escape(text)
    # Escape first; these regexes introduce the only Telegram HTML tags.
    s = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", s)
    s = re.sub(r"^\s{0,3}#{1,6}\s*(.+)$", r"<b>\1</b>", s, flags=re.MULTILINE)
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*([^*\n]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"__([^_\n]+)__", r"<b>\1</b>", s)
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
        rows.append([{"text": choice} for choice in choices[i:i + 2]])
    return {
        "keyboard": rows,
        "resize_keyboard": True,
        "one_time_keyboard": True,
    }


async def send_text(chat_id: int | str, text: str, choices: list[str] | tuple[str, ...] = ()) -> None:
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
            if idx == len(chunks) and choices:
                payload["reply_markup"] = _choice_keyboard(choices)
            chunk_start = time.perf_counter()
            r = await client.post(_send_url(), json=payload)
            logger.info("Telegram send → %s %s", r.status_code, r.text[:200])
            logger.info("Telegram timing stage=send_chunk chunk=%d/%d ms=%.1f status=%s chars=%d",
                        idx, len(chunks), _elapsed_ms(chunk_start), r.status_code, len(chunk))
            if r.status_code >= 400:
                # fallback: send as plain text so user still sees something
                for fallback_chunk in _split_for_telegram(chunk, TG_HARD_MAX_BYTES):
                    await client.post(_send_url(), json={"chat_id": chat_id, "text": fallback_chunk})
    logger.info("Telegram timing stage=send_total ms=%.1f chunks=%d chars=%d",
                _elapsed_ms(total_start), len(chunks), len(text))


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


async def _answer_and_send(chat_id: int | str, text: str) -> None:
    total_start = time.perf_counter()
    session_id = f"tg:{chat_id}"
    try:
        reply = await asyncio.to_thread(answer_with_choices, text, session_id=session_id)
    except Exception:
        logger.exception("Telegram answer failed")
        reply = ChatReply(TECHNICAL_ERROR_REPLY)

    try:
        await send_text(chat_id, reply.text, reply.choices)
    except Exception:
        logger.exception("Telegram send failed")
    else:
        if not reply.choices:
            try:
                token = create_feedback_request(session_id, "telegram", str(chat_id), text, reply.text)
                await _send_rating_prompt(chat_id, token)
            except Exception:
                logger.exception("Telegram rating prompt failed")

    logger.info("Telegram timing stage=background_total ms=%.1f",
                _elapsed_ms(total_start))


def _command(text: str) -> str:
    first = text.strip().split(maxsplit=1)[0].lower()
    return first.split("@", 1)[0]


async def _handle_command(chat_id: int | str, text: str) -> bool:
    cmd = _command(text)
    if cmd == "/start":
        await send_text(chat_id, START_TEXT)
        return True
    if cmd == "/help":
        await send_text(chat_id, HELP_TEXT)
        return True
    if cmd == "/menu":
        await send_text(chat_id, MENU_TEXT)
        return True
    if cmd == "/new":
        clear_session(f"tg:{chat_id}")
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
        await _handle_rating_callback(callback_query)
        return {"ok": True}

    message = update.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    text = message.get("text", "")
    if not chat_id or not text:
        return {"ok": True}

    if text.startswith("/") and await _handle_command(chat_id, text):
        return {"ok": True}

    background_tasks.add_task(_answer_and_send, chat_id, text)
    return {"ok": True}
