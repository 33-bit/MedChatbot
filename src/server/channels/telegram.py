"""
Telegram Bot webhook.

Setup:
  curl -F "url=https://<ngrok>/webhook/telegram" \\
       -F "secret_token=$TELEGRAM_WEBHOOK_SECRET" \\
       https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/setWebhook
"""

from __future__ import annotations

import html as _html
import logging
import re

import httpx
from fastapi import APIRouter, Header, HTTPException, Request

from src.chat import answer
from src.config import TELEGRAM_BOT_TOKEN, TELEGRAM_WEBHOOK_SECRET

logger = logging.getLogger(__name__)
router = APIRouter()

TG_MAX_LEN = 4000  # safe under Telegram's 4096 limit


def _send_url() -> str:
    return f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"


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
    html = _md_to_tg_html(text)
    async with httpx.AsyncClient(timeout=20.0) as client:
        for chunk in _split_for_telegram(html):
            payload = {
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            r = await client.post(_send_url(), json=payload)
            logger.info("Telegram send → %s %s", r.status_code, r.text[:200])
            if r.status_code >= 400:
                # fallback: send as plain text so user still sees something
                await client.post(_send_url(), json={"chat_id": chat_id, "text": chunk})


@router.post("/webhook/telegram")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> dict:
    if TELEGRAM_WEBHOOK_SECRET and x_telegram_bot_api_secret_token != TELEGRAM_WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret token")

    update = await request.json()
    message = update.get("message") or update.get("edited_message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    text = message.get("text", "")
    if not chat_id or not text:
        return {"ok": True}

    try:
        reply = answer(text)
        await send_text(chat_id, reply)
    except Exception:
        logger.exception("Telegram handler error")
    return {"ok": True}
