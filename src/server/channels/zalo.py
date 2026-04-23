"""
Zalo OA webhook.

Event payload (rút gọn) khi user nhắn tin OA:
{
  "event_name": "user_send_text",
  "sender": {"id": "<user_id>"},
  "recipient": {"id": "<oa_id>"},
  "message": {"text": "..."}
}
Docs: https://developers.zalo.me/docs/official-account/tin-nhan/gui-tin-tu-van
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Request

from src.chat import answer
from src.config import ZALO_OA_ACCESS_TOKEN

logger = logging.getLogger(__name__)
router = APIRouter()

ZALO_SEND_URL = "https://openapi.zalo.me/v3.0/oa/message/cs"


async def send_text(user_id: str, text: str) -> None:
    if not ZALO_OA_ACCESS_TOKEN:
        logger.warning("ZALO_OA_ACCESS_TOKEN chưa cấu hình; bỏ qua gửi tin.")
        return
    payload = {
        "recipient": {"user_id": user_id},
        "message": {"text": text},
    }
    headers = {"access_token": ZALO_OA_ACCESS_TOKEN, "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(ZALO_SEND_URL, json=payload, headers=headers)
        logger.info("Zalo send → %s %s", r.status_code, r.text[:200])


async def _handle_event(body: dict) -> None:
    event = body.get("event_name")
    if event != "user_send_text":
        return
    user_id = (body.get("sender") or {}).get("id")
    text = (body.get("message") or {}).get("text", "")
    if not user_id or not text:
        return
    reply = answer(text, session_id=f"zalo:{user_id}")
    await send_text(user_id, reply)


@router.get("/webhook/zalo")
async def zalo_verify() -> dict:
    # Zalo yêu cầu endpoint trả 200 cho health check khi cấu hình webhook.
    return {"ok": True}


@router.post("/webhook/zalo")
async def zalo_webhook(request: Request) -> dict:
    body = await request.json()
    logger.info("Zalo event: %s", body.get("event_name"))
    try:
        await _handle_event(body)
    except Exception:
        logger.exception("Zalo handler error")
    return {"ok": True}
