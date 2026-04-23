"""
Facebook Messenger webhook.

Docs: https://developers.facebook.com/docs/messenger-platform/webhooks
GET verify: hub.mode=subscribe & hub.verify_token=... & hub.challenge=<echo>
POST event: {object: "page", entry: [{messaging: [{sender, message: {text}}]}]}
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse

from src.chat import answer
from src.config import MESSENGER_PAGE_TOKEN, MESSENGER_VERIFY_TOKEN

logger = logging.getLogger(__name__)
router = APIRouter()

GRAPH_SEND_URL = "https://graph.facebook.com/v18.0/me/messages"


async def send_text(recipient_id: str, text: str) -> None:
    if not MESSENGER_PAGE_TOKEN:
        logger.warning("MESSENGER_PAGE_TOKEN chưa cấu hình; bỏ qua gửi tin.")
        return
    params = {"access_token": MESSENGER_PAGE_TOKEN}
    payload = {
        "recipient": {"id": recipient_id},
        "messaging_type": "RESPONSE",
        "message": {"text": text},
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(GRAPH_SEND_URL, params=params, json=payload)
        logger.info("Messenger send → %s %s", r.status_code, r.text[:200])


@router.get("/webhook/messenger")
async def messenger_verify(request: Request) -> PlainTextResponse:
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge", "")
    if mode == "subscribe" and token and token == MESSENGER_VERIFY_TOKEN:
        return PlainTextResponse(challenge)
    raise HTTPException(status_code=403, detail="Invalid verify token")


@router.post("/webhook/messenger")
async def messenger_webhook(request: Request) -> dict:
    body = await request.json()
    if body.get("object") != "page":
        return {"ok": True}

    for entry in body.get("entry", []):
        for evt in entry.get("messaging", []):
            sender_id = (evt.get("sender") or {}).get("id")
            text = (evt.get("message") or {}).get("text", "")
            if not sender_id or not text:
                continue
            try:
                reply = answer(text, session_id=f"fb:{sender_id}")
                await send_text(sender_id, reply)
            except Exception:
                logger.exception("Messenger handler error")
    return {"ok": True}
