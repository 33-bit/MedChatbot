"""
FastAPI entry.

Chạy local:
    uvicorn src.server.app:app --host 0.0.0.0 --port 8000 --reload

Expose public qua ngrok:
    ngrok http 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from src.chat import answer
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
async def chat_debug(body: dict) -> dict:
    """Debug endpoint: POST {\"question\": \"...\"} → {\"answer\": \"...\"}"""
    question = (body or {}).get("question", "")
    return {"answer": answer(question)}
