"""
FastAPI entry.

Chạy local:
    uvicorn src.server.app:app --host 0.0.0.0 --port 8000 --reload

Expose public qua ngrok:
    ngrok http 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from src.chat import answer, answer_with_meta
from src.chat.mode_policy import normalize_mode
from src.chat.replies import TECHNICAL_ERROR_REPLY
from src.chat.security.identity import (
    RequestIdentity,
    derive_previous_owner_keys,
    derive_request_identity,
    validate_identity_configuration,
)
from src.chat.storage.durable_memory import migrate_owner_key
from src.chat.retrieval.kg import ensure_fulltext_indexes
from src.chat.retrieval.preload import preload_retrieval_models
from src.chat.storage.traces import get_chat_trace, list_chat_traces, save_chat_trace
from src.config import CHAT_API_KEY, CHAT_API_TENANT_ID
from src.server.channels import messenger, telegram, zalo
from src.server.channels import telegram_doctor
from src.server.payments import router as payos_router
from src.server.source_documents import resolve_bachmai_source_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# httpx includes full request URLs in INFO logs. Telegram embeds the bot token
# in that URL, so application logs must never emit normal HTTP request lines.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# How often the doctor-session ticker runs (seconds). Drives time-limit
# enforcement, billing, near-timeout warnings, and waitlist promotion.
SESSION_TICK_INTERVAL_SECONDS = 30
_DEBUG_CONSOLE_DIR = Path(__file__).parent / "static" / "debug_console"


async def startup() -> None:
    validate_identity_configuration()
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


async def _reminder_ticker(interval: float = 30.0) -> None:
    """Periodic loop driving Telegram medical reminders.

    Runs until cancelled at shutdown. Each tick is wrapped so a single failure
    never kills the loop.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            from src.server.channels import telegram
            await telegram.run_reminder_tick()
        except Exception:
            log.exception("Reminder session tick failed")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await startup()
    ticker = asyncio.create_task(_session_ticker())
    reminder_ticker = asyncio.create_task(_reminder_ticker())
    try:
        yield
    finally:
        ticker.cancel()
        reminder_ticker.cancel()
        try:
            await asyncio.gather(ticker, reminder_ticker, return_exceptions=True)
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


@app.get("/sources/bachmai/{source_slug}.pdf")
def bachmai_source_pdf(source_slug: str) -> FileResponse:
    pdf_path = resolve_bachmai_source_pdf(source_slug)
    if pdf_path is None:
        raise HTTPException(status_code=404, detail="Source PDF not found")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"{source_slug}.pdf",
        content_disposition_type="inline",
        headers={"Cache-Control": "public, max-age=86400"},
    )


def _require_chat_api_key(x_api_key: str | None) -> str:
    if not CHAT_API_KEY:
        raise HTTPException(status_code=503, detail="Chat API disabled: CHAT_API_KEY not set")
    if x_api_key != CHAT_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


def _scoped_api_session_id(api_key: str, client_session_id: str) -> str:
    del api_key
    return derive_request_identity(
        "api",
        None,
        client_session_id,
        tenant=CHAT_API_TENANT_ID or "api-default",
    ).session_key


def _api_request_identity(payload: dict) -> RequestIdentity:
    client_session_id = _require_client_session_id(payload)
    user_id = payload.get("user_id")
    stable_user_id = user_id if isinstance(user_id, str) and user_id.strip() else None
    # A tenant must be operationally stable and independent of API-key rotation.
    # Without one, temporary memory works but durable memory remains disabled.
    identity = derive_request_identity(
        "api",
        stable_user_id if CHAT_API_TENANT_ID else None,
        client_session_id,
        tenant=CHAT_API_TENANT_ID or "api-default",
    )
    if identity.owner_key and stable_user_id is not None:
        try:
            migrate_owner_key(
                identity.owner_key,
                derive_previous_owner_keys(
                    "api",
                    stable_user_id,
                    tenant=CHAT_API_TENANT_ID,
                ),
            )
        except Exception:
            log.exception("API owner-key rotation failed; durable memory disabled for request")
            return RequestIdentity("", identity.session_key, False)
    return identity


def _require_client_session_id(body: dict | None) -> str:
    client_session_id = (body or {}).get("session_id")
    if not isinstance(client_session_id, str) or not client_session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")
    return client_session_id.strip()


def _require_query_session_id(session_id: str | None) -> str:
    if not isinstance(session_id, str) or not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")
    return session_id.strip()


def _run_answer_with_meta(
    question: str,
    session_id: str,
    owner_id: str | None,
    mode: str,
) -> tuple[str, dict]:
    kwargs = {"session_id": session_id}
    if owner_id:
        kwargs["owner_id"] = owner_id
    if mode == "auto":
        return answer_with_meta(question, **kwargs)
    return answer_with_meta(question, mode=mode, **kwargs)


def _build_debug_trace(
    *,
    reply: str,
    meta: dict | None,
    question: str,
    session_id: str,
    internal_session_id: str,
    mode: str,
) -> dict:
    trace_meta = dict(meta or {})
    pipeline_trace_id = trace_meta.get("trace_id")
    trace_id = str(uuid.uuid4())
    if pipeline_trace_id:
        trace_meta["pipeline_trace_id"] = str(pipeline_trace_id)
    trace_meta["trace_id"] = trace_id
    created_at = time.time()
    return {
        "trace_id": trace_id,
        "session_id": session_id,
        "internal_session_id": internal_session_id,
        "mode": mode,
        "question": question,
        "answer": reply,
        "created_at": created_at,
        "meta": trace_meta,
    }


@app.get("/debug/chat-route", response_class=HTMLResponse)
def debug_chat_route_page() -> HTMLResponse:
    return HTMLResponse((_DEBUG_CONSOLE_DIR / "index.html").read_text("utf-8"))


@app.get("/debug/chat-route/static/{asset}")
def debug_chat_route_asset(asset: str) -> FileResponse:
    # Whitelist prevents path traversal.
    if asset not in {"app.js", "styles.css"}:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(_DEBUG_CONSOLE_DIR / asset)


@app.post("/debug/chat-route/run")
async def debug_chat_route_run(
    body: dict,
    x_api_key: str | None = Header(default=None),
) -> dict:
    api_key = _require_chat_api_key(x_api_key)
    payload = body or {}
    question = payload.get("question", "")
    mode = normalize_mode(payload.get("mode"))
    identity = _api_request_identity(payload)
    session_id = identity.session_key

    try:
        reply, meta = _run_answer_with_meta(
            question,
            session_id,
            identity.owner_key or None,
            mode,
        )
    except Exception:
        log.exception("Debug chat route run failed")
        reply = TECHNICAL_ERROR_REPLY
        meta = {"error": "technical_error"}
    trace = _build_debug_trace(
        reply=reply,
        meta=meta,
        question=question,
        session_id=session_id,
        internal_session_id=session_id,
        mode=mode,
    )
    try:
        saved = save_chat_trace(
            trace_id=trace["trace_id"],
            session_id=session_id,
            internal_session_id=session_id,
            mode=mode,
            question=question,
            answer=reply,
            meta=trace["meta"],
            created_at=trace["created_at"],
        )
    except Exception:
        log.exception("Debug trace persistence failed")
        return {"trace": trace, "warning": "trace_persistence_failed"}
    return {"trace": saved}


@app.post("/debug/chat-route/stream")
async def debug_chat_route_stream(
    body: dict,
    x_api_key: str | None = Header(default=None),
):
    api_key = _require_chat_api_key(x_api_key)
    payload = body or {}
    question = payload.get("question", "")
    mode = normalize_mode(payload.get("mode"))
    identity = _api_request_identity(payload)
    session_id = identity.session_key

    from src.chat import pipeline

    events: "queue.Queue" = queue.Queue()
    sentinel = object()
    result: dict = {}

    def _worker():
        pipeline._install_event_sink(events)
        try:
            reply, meta = _run_answer_with_meta(
                question,
                session_id,
                identity.owner_key or None,
                mode,
            )
        except Exception:
            log.exception("Debug chat route stream failed")
            reply, meta = TECHNICAL_ERROR_REPLY, {"error": "technical_error"}
        finally:
            pipeline._install_event_sink(None)
        result["reply"] = reply
        result["meta"] = meta
        events.put(sentinel)

    def _stream():
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        while True:
            item = events.get()
            if item is sentinel:
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        thread.join()

        trace = _build_debug_trace(
            reply=result["reply"],
            meta=result.get("meta"),
            question=question,
            session_id=session_id,
            internal_session_id=session_id,
            mode=mode,
        )
        try:
            saved = save_chat_trace(
                trace_id=trace["trace_id"],
                session_id=session_id,
                internal_session_id=session_id,
                mode=mode,
                question=question,
                answer=result["reply"],
                meta=trace["meta"],
                created_at=trace["created_at"],
            )
        except Exception:
            log.exception("Debug stream trace persistence failed")
            saved = trace
        yield f"data: {json.dumps({'type': 'done', 'trace': saved}, ensure_ascii=False)}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.get("/debug/chat-route/traces")
def debug_chat_route_traces(
    x_api_key: str | None = Header(default=None),
    session_id: str | None = Query(default=None),
    trace_id: str | None = Query(default=None),
    limit: int = Query(default=20),
) -> dict:
    api_key = _require_chat_api_key(x_api_key)
    client_session_id = _require_query_session_id(session_id)
    internal_session_id = _scoped_api_session_id(api_key, client_session_id)
    return {
        "traces": list_chat_traces(
            internal_session_id=internal_session_id,
            trace_id=trace_id,
            limit=limit,
        )
    }


@app.get("/debug/chat-route/traces/{trace_id}")
def debug_chat_route_trace_detail(
    trace_id: str,
    x_api_key: str | None = Header(default=None),
    session_id: str | None = Query(default=None),
) -> dict:
    api_key = _require_chat_api_key(x_api_key)
    client_session_id = _require_query_session_id(session_id)
    internal_session_id = _scoped_api_session_id(api_key, client_session_id)
    trace = get_chat_trace(trace_id, internal_session_id=internal_session_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {"trace": trace}


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
    api_key = _require_chat_api_key(x_api_key)

    payload = body or {}
    question = payload.get("question", "")
    mode = normalize_mode(payload.get("mode"))
    identity = _api_request_identity(payload)
    session_id = identity.session_key
    if include_meta:
        try:
            reply, meta = _run_answer_with_meta(
                question,
                session_id,
                identity.owner_key or None,
                mode,
            )
        except Exception:
            log.exception("Chat endpoint failed")
            return {"answer": TECHNICAL_ERROR_REPLY, "meta": {"error": "technical_error"}}
        return {"answer": reply, "meta": meta}
    try:
        answer_kwargs = {"session_id": session_id}
        if identity.owner_key:
            answer_kwargs["owner_id"] = identity.owner_key
        if mode == "auto":
            reply = answer(question, **answer_kwargs)
        else:
            reply = answer(question, mode=mode, **answer_kwargs)
    except Exception:
        log.exception("Chat endpoint failed")
        reply = TECHNICAL_ERROR_REPLY
    return {"answer": reply}
