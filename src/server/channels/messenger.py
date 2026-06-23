"""
Facebook Messenger webhook.

Docs: https://developers.facebook.com/docs/messenger-platform/webhooks
GET verify: hub.mode=subscribe & hub.verify_token=... & hub.challenge=<echo>
POST event: {object: "page", entry: [{messaging: [{sender, message: {text}}]}]}
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import PlainTextResponse

from src.config import MESSENGER_PAGE_TOKEN, MESSENGER_VERIFY_TOKEN
from src.chat.security.identity import derive_previous_owner_keys, derive_request_identity
from src.chat.profile.repository import migrate_owner_key
from src.server.channels.common import answer_and_send

logger = logging.getLogger(__name__)
router = APIRouter()

GRAPH_SEND_URL = "https://graph.facebook.com/v18.0/me/messages"


def _quick_replies(choices: list[str] | tuple[str, ...]) -> list[dict]:
    replies = []
    for choice in choices[:13]:
        title = choice[:20]
        replies.append({
            "content_type": "text",
            "title": title,
            "payload": f"choice:{title}",
        })
    return replies


async def send_text(recipient_id: str, text: str, choices: list[str] | tuple[str, ...] = ()) -> None:
    if not MESSENGER_PAGE_TOKEN:
        logger.warning("MESSENGER_PAGE_TOKEN chưa cấu hình; bỏ qua gửi tin.")
        return
    params = {"access_token": MESSENGER_PAGE_TOKEN}
    message = {"text": text}
    if choices:
        message["quick_replies"] = _quick_replies(choices)
    payload = {
        "recipient": {"id": recipient_id},
        "messaging_type": "RESPONSE",
        "message": message,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(GRAPH_SEND_URL, params=params, json=payload)
        logger.info("Messenger send status=%s", r.status_code)


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
async def messenger_webhook(request: Request, background_tasks: BackgroundTasks) -> dict:
    body = await request.json()
    if body.get("object") != "page":
        return {"ok": True}

    for entry in body.get("entry", []):
        for evt in entry.get("messaging", []):
            sender_id = (evt.get("sender") or {}).get("id")
            text = (evt.get("message") or {}).get("text", "")
            if not sender_id or not text:
                continue
            identity = derive_request_identity(
                "messenger",
                sender_id,
                sender_id,
            )
            if identity.owner_key:
                try:
                    migrate_owner_key(
                        identity.owner_key,
                        derive_previous_owner_keys("messenger", sender_id),
                    )
                except Exception:
                    logger.exception("Messenger owner-key rotation failed")
                    identity = type(identity)("", identity.session_key, False)
            background_tasks.add_task(
                answer_and_send,
                text,
                sender_id,
                identity.session_key,
                send_text,
                owner_id=identity.owner_key or None,
                logger=logger,
                channel="Messenger",
            )
    return {"ok": True}
