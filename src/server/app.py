"""
FastAPI entry.

Chạy local:
    uvicorn src.server.app:app --host 0.0.0.0 --port 8000 --reload

Expose public qua ngrok:
    ngrok http 8000
"""

from __future__ import annotations

import asyncio
import hashlib
import logging

from fastapi import FastAPI, Header, HTTPException, Query

from src.chat import answer, answer_with_meta
from src.chat.replies import TECHNICAL_ERROR_REPLY
from src.chat.retrieval.kg import ensure_fulltext_indexes
from src.chat.retrieval.preload import preload_retrieval_models
from src.config import CHAT_API_KEY
from src.server.channels import messenger, telegram, zalo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(title="Medical RAG Chatbot")
app.include_router(zalo.router)
app.include_router(telegram.router)
app.include_router(messenger.router)


@app.on_event("startup")
async def startup() -> None:
    await asyncio.to_thread(preload_retrieval_models)
    await asyncio.to_thread(ensure_fulltext_indexes)
    try:
        await telegram.setup_bot_menu()
    except Exception as exc:
        log.warning("Telegram command menu setup failed; continuing startup: %s", exc)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/chat")
async def chat_debug(
    body: dict,
    x_api_key: str | None = Header(default=None),
    include_meta: bool = Query(default=False),
) -> dict:
    """Authenticated chat endpoint.

    Headers: X-API-Key: <CHAT_API_KEY>
    Body:    {"question": "...", "session_id": "<per-user id>"}
    Query:   include_meta=1 attaches token usage, retrieved hits, and
             per-stage latency for evaluation use. Default off — channels
             and normal clients see the same {"answer": ...} shape.
    """
    if not CHAT_API_KEY:
        raise HTTPException(status_code=503, detail="Chat API disabled: CHAT_API_KEY not set")
    if x_api_key != CHAT_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    payload = body or {}
    question = payload.get("question", "")
    client_session_id = payload.get("session_id")
    if not isinstance(client_session_id, str) or not client_session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")

    session_key = f"{x_api_key}\0{client_session_id.strip()}"
    session_id = "api:" + hashlib.sha256(session_key.encode("utf-8")).hexdigest()[:32]
    if include_meta:
        try:
            reply, meta = answer_with_meta(question, session_id=session_id)
        except Exception:
            log.exception("Chat endpoint failed")
            return {"answer": TECHNICAL_ERROR_REPLY, "meta": {"error": "technical_error"}}
        return {"answer": reply, "meta": meta}
    try:
        reply = answer(question, session_id=session_id)
    except Exception:
        log.exception("Chat endpoint failed")
        reply = TECHNICAL_ERROR_REPLY
    return {"answer": reply}
