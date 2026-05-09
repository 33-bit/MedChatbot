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

from src.chat import answer
from src.chat.replies import TECHNICAL_ERROR_REPLY
from src.chat.storage.session import clear_session, reserve_webhook_update
from src.config import TELEGRAM_BOT_TOKEN, TELEGRAM_WEBHOOK_SECRET

logger = logging.getLogger(__name__)
router = APIRouter()

TG_MAX_LEN = 4000  # safe under Telegram's 4096 limit
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


def _md_to_tg_html(text: str) -> str:
    """Convert a subset of Markdown to Telegram-compatible HTML."""
    s = _html.escape(text)
    s = re.sub(r"```([\s\S]*?)```", r"<pre>\1</pre>", s)
    s = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", s)
    s = re.sub(r"^\s{0,3}#{1,6}\s*(.+)$", r"<b>\1</b>", s, flags=re.MULTILINE)
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*([^*\n]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"__([^_\n]+)__", r"<b>\1</b>", s)
    return s


def _split_for_telegram(text: str, limit: int = TG_MAX_LEN) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


async def send_text(chat_id: int | str, text: str) -> None:
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
            chunk_start = time.perf_counter()
            r = await client.post(_send_url(), json=payload)
            logger.info("Telegram send → %s %s", r.status_code, r.text[:200])
            logger.info("Telegram timing stage=send_chunk chunk=%d/%d ms=%.1f status=%s chars=%d",
                        idx, len(chunks), _elapsed_ms(chunk_start), r.status_code, len(chunk))
            if r.status_code >= 400:
                # fallback: send as plain text so user still sees something
                await client.post(_send_url(), json={"chat_id": chat_id, "text": chunk})
    logger.info("Telegram timing stage=send_total ms=%.1f chunks=%d chars=%d",
                _elapsed_ms(total_start), len(chunks), len(text))


async def _answer_and_send(chat_id: int | str, text: str) -> None:
    total_start = time.perf_counter()
    try:
        stage_start = time.perf_counter()
        reply = await asyncio.to_thread(answer, text, session_id=f"tg:{chat_id}")
        logger.info("Telegram timing stage=answer ms=%.1f chars=%d",
                    _elapsed_ms(stage_start), len(reply))
        stage_start = time.perf_counter()
        await send_text(chat_id, reply)
        logger.info("Telegram timing stage=send ms=%.1f",
                    _elapsed_ms(stage_start))
    except Exception:
        logger.exception("Telegram handler error")
        try:
            await send_text(chat_id, TECHNICAL_ERROR_REPLY)
        except Exception:
            logger.exception("Telegram fallback send failed")
    finally:
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

    message = update.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    text = message.get("text", "")
    if not chat_id or not text:
        return {"ok": True}

    if text.startswith("/") and await _handle_command(chat_id, text):
        return {"ok": True}

    background_tasks.add_task(_answer_and_send, chat_id, text)
    return {"ok": True}
