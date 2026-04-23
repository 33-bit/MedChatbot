"""
FastAPI entry.

Chạy local:
    uvicorn src.server.app:app --host 0.0.0.0 --port 8000 --reload

Expose public qua ngrok:
    ngrok http 8000
"""

from __future__ import annotations

import hashlib
import logging

from fastapi import FastAPI, Header, HTTPException

from src.chat import answer
from src.config import CHAT_API_KEY
from src.server.channels import messenger, telegram, zalo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="Medical RAG Chatbot")
app.include_router(zalo.router)
app.include_router(telegram.router)
app.include_router(messenger.router)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/chat")
async def chat_debug(
    body: dict,
    x_api_key: str | None = Header(default=None),
) -> dict:
    """Authenticated chat endpoint.

    Headers: X-API-Key: <CHAT_API_KEY>
    Body:    {"question": "..."}  — session_id is derived from the key, not trusted from body.
    """
    if not CHAT_API_KEY:
        raise HTTPException(status_code=503, detail="Chat API disabled: CHAT_API_KEY not set")
    if x_api_key != CHAT_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    question = (body or {}).get("question", "")
    # Derive session_id from the API key so clients can't impersonate another session.
    # If you want per-client sessions, issue a distinct key per client.
    session_id = "api:" + hashlib.sha256(x_api_key.encode()).hexdigest()[:16]
    return {"answer": answer(question, session_id=session_id)}
