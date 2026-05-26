"""
Zalo Bot Platform webhook.

Event payload (rút gọn) khi user nhắn tin bot:
{
  "ok": true,
  "result": {
    "event_name": "message.text.received",
    "message": {
      "chat": {"id": "<chat_id>"},
      "text": "..."
    }
  }
}
Docs: https://bot.zapps.me/docs/
"""

from __future__ import annotations

import hmac
import logging

import httpx
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request

from src.config import ZALO_BOT_TOKEN, ZALO_WEBHOOK_SECRET
from src.server.channels.common import answer_and_send

logger = logging.getLogger(__name__)
router = APIRouter()

ZALO_BOT_API_URL = "https://bot-api.zaloplatforms.com"
ZALO_MAX_TEXT_LEN = 2000


def _api_url(method: str) -> str:
    return f"{ZALO_BOT_API_URL}/bot{ZALO_BOT_TOKEN}/{method}"


def _split_for_zalo(text: str, limit: int = ZALO_MAX_TEXT_LEN) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit + 1)
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


async def send_text(chat_id: str, text: str) -> None:
    if not ZALO_BOT_TOKEN:
        logger.warning("ZALO_BOT_TOKEN chưa cấu hình; bỏ qua gửi tin.")
        return
    async with httpx.AsyncClient(timeout=20.0) as client:
        chunks = _split_for_zalo(text)
        for idx, chunk in enumerate(chunks, 1):
            payload = {"chat_id": chat_id, "text": chunk}
            r = await client.post(_api_url("sendMessage"), json=payload)
            logger.info(
                "Zalo send → %s %s chunk=%d/%d chars=%d",
                r.status_code,
                r.text[:200],
                idx,
                len(chunks),
                len(chunk),
            )


def _event_payload(body: dict) -> dict:
    result = body.get("result")
    return result if isinstance(result, dict) else body


async def _handle_event(body: dict, background_tasks: BackgroundTasks) -> None:
    result = _event_payload(body)
    event = result.get("event_name")
    if event != "message.text.received":
        return
    message = result.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    text = message.get("text", "")
    if not chat_id or not text:
        return
    background_tasks.add_task(
        answer_and_send,
        text,
        chat_id,
        f"zalo:{chat_id}",
        send_text,
        logger=logger,
        channel="Zalo",
    )


def _verify_secret(secret_token: str | None) -> None:
    if not ZALO_WEBHOOK_SECRET:
        return
    if not secret_token or not hmac.compare_digest(secret_token, ZALO_WEBHOOK_SECRET):
        raise HTTPException(status_code=403, detail="Invalid Zalo secret token")


@router.get("/webhook/zalo")
async def zalo_verify() -> dict:
    return {"ok": True}


@router.post("/webhook/zalo")
async def zalo_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_bot_api_secret_token: str | None = Header(default=None, alias="X-Bot-Api-Secret-Token"),
) -> dict:
    _verify_secret(x_bot_api_secret_token)
    body = await request.json()
    result = _event_payload(body)
    event = result.get("event_name")
    if event is None:
        logger.info(
            "Zalo event missing; body_keys=%s result_keys=%s",
            sorted(body.keys()),
            sorted((body.get("result") or {}).keys()) if isinstance(body.get("result"), dict) else [],
        )
    else:
        logger.info("Zalo event: %s", event)
    try:
        await _handle_event(body, background_tasks)
    except Exception:
        logger.exception("Zalo handler error")
    return {"ok": True}
