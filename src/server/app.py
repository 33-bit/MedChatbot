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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Query

from src.chat import answer, answer_with_meta
from src.chat.mode_policy import normalize_mode
from src.chat.replies import TECHNICAL_ERROR_REPLY
from src.chat.retrieval.kg import ensure_fulltext_indexes
from src.chat.retrieval.preload import preload_retrieval_models
from src.config import CHAT_API_KEY
from src.server.channels import messenger, telegram, zalo
from src.server.channels import telegram_doctor
from src.server.payments import router as payos_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# How often the doctor-session ticker runs (seconds). Drives time-limit
# enforcement, billing, near-timeout warnings, and waitlist promotion.
SESSION_TICK_INTERVAL_SECONDS = 30


async def startup() -> None:
    await asyncio.to_thread(preload_retrieval_models)
    await asyncio.to_thread(ensure_fulltext_indexes)
    try:
        await telegram.setup_bot_menu()
    except Exception as exc:
        log.warning("Telegram command menu setup failed; continuing startup: %s", exc)


async def _session_ticker(interval: float = SESSION_TICK_INTERVAL_SECONDS) -> None:
    """Periodic loop driving doctor-session time/billing enforcement.

    Runs until cancelled at shutdown. Each tick is wrapped so a single failure
    never kills the loop.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            await telegram_doctor.run_session_tick()
        except Exception:
            log.exception("Doctor session tick failed")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await startup()
    ticker = asyncio.create_task(_session_ticker())
    try:
        yield
    finally:
        ticker.cancel()
        try:
            await ticker
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Medical RAG Chatbot", lifespan=lifespan)
app.include_router(zalo.router)
app.include_router(telegram.router)
app.include_router(messenger.router)
app.include_router(payos_router.router)


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
    mode = normalize_mode(payload.get("mode"))
    client_session_id = payload.get("session_id")
    if not isinstance(client_session_id, str) or not client_session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")

    session_key = f"{x_api_key}\0{client_session_id.strip()}"
    session_id = "api:" + hashlib.sha256(session_key.encode("utf-8")).hexdigest()[:32]
    if include_meta:
        try:
            if mode == "auto":
                reply, meta = answer_with_meta(question, session_id=session_id)
            else:
                reply, meta = answer_with_meta(question, session_id=session_id, mode=mode)
        except Exception:
            log.exception("Chat endpoint failed")
            return {"answer": TECHNICAL_ERROR_REPLY, "meta": {"error": "technical_error"}}
        return {"answer": reply, "meta": meta}
    try:
        if mode == "auto":
            reply = answer(question, session_id=session_id)
        else:
            reply = answer(question, session_id=session_id, mode=mode)
    except Exception:
        log.exception("Chat endpoint failed")
        reply = TECHNICAL_ERROR_REPLY
    return {"answer": reply}
