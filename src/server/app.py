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
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse

from src.chat import answer, answer_with_meta
from src.chat.mode_policy import normalize_mode
from src.chat.replies import TECHNICAL_ERROR_REPLY
from src.chat.retrieval.kg import ensure_fulltext_indexes
from src.chat.retrieval.preload import preload_retrieval_models
from src.chat.storage.traces import get_chat_trace, list_chat_traces, save_chat_trace
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
DEBUG_CHAT_ROUTE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Chat Route Debug Console</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; }
    body { margin: 0; background: #111827; color: #e5e7eb; }
    main { max-width: 1200px; margin: 0 auto; padding: 24px; }
    h1, h2 { margin: 0 0 12px; }
    section { background: #1f2937; border: 1px solid #374151; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
    label { display: block; font-weight: 600; margin-bottom: 6px; }
    input, textarea, select, button { width: 100%; box-sizing: border-box; margin-bottom: 12px; padding: 10px; border-radius: 8px; border: 1px solid #4b5563; background: #111827; color: inherit; }
    button { cursor: pointer; background: #2563eb; border: 0; font-weight: 700; }
    button.secondary { background: #374151; }
    .grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
    .timeline-item, .trace-item { border: 1px solid #4b5563; border-radius: 8px; padding: 12px; margin-bottom: 10px; background: #111827; }
    .meta { color: #9ca3af; font-size: 14px; }
    pre { white-space: pre-wrap; word-break: break-word; background: #0b1220; padding: 12px; border-radius: 8px; overflow: auto; }
  </style>
</head>
<body>
  <main>
    <h1>Chat Route Debug Console</h1>
    <p class="meta">Live run endpoint: <code>/debug/chat-route/run</code> | Replay endpoint: <code>/debug/chat-route/traces</code></p>
    <div class="grid">
      <section>
        <h2>Live Run</h2>
        <label for="api-key">X-API-Key</label>
        <input id="api-key" type="password" autocomplete="off">
        <label for="session-id">Session ID</label>
        <input id="session-id" value="debug-user">
        <label for="mode">Mode</label>
        <select id="mode">
          <option value="auto">auto</option>
          <option value="information">information</option>
          <option value="diagnostic">diagnostic</option>
        </select>
        <label for="question">Question</label>
        <textarea id="question" rows="4">Tôi bị ho</textarea>
        <button id="run-button" type="button">Run</button>
        <div id="run-status" class="meta"></div>
      </section>
      <section>
        <h2>Replay</h2>
        <label for="filter-session-id">Session ID Filter</label>
        <input id="filter-session-id" value="debug-user">
        <label for="filter-trace-id">Trace ID Filter</label>
        <input id="filter-trace-id" placeholder="optional">
        <button id="list-button" class="secondary" type="button">Load Traces</button>
        <div id="trace-list"></div>
      </section>
    </div>
    <section>
      <h2>Trace Detail</h2>
      <div id="trace-summary" class="meta">No trace loaded.</div>
      <h3>Timeline</h3>
      <div id="timeline"></div>
      <h3>Retrieved Hits</h3>
      <div id="retrieved"></div>
      <h3>Usage</h3>
      <div id="usage"></div>
      <h3>Answer</h3>
      <pre id="answer"></pre>
      <h3>Raw JSON</h3>
      <pre id="raw-json"></pre>
    </section>
  </main>
  <script>
    const el = (id) => document.getElementById(id);
    const apiHeaders = () => {
      const apiKey = el("api-key").value;
      return apiKey ? { "Content-Type": "application/json", "X-API-Key": apiKey } : { "Content-Type": "application/json" };
    };

    function renderList(traces) {
      const root = el("trace-list");
      root.textContent = "";
      for (const trace of traces || []) {
        const item = document.createElement("div");
        item.className = "trace-item";
        const button = document.createElement("button");
        button.className = "secondary";
        button.textContent = `Open ${trace.trace_id}`;
        button.addEventListener("click", () => loadTrace(trace.trace_id));
        const title = document.createElement("div");
        title.textContent = `${trace.question} (${trace.route || "unknown"})`;
        const meta = document.createElement("div");
        meta.className = "meta";
        meta.textContent = `session=${trace.session_id} mode=${trace.mode} created_at=${trace.created_at ?? "n/a"} latency=${trace.latency_ms_total ?? "n/a"}ms`;
        item.append(title, meta, button);
        root.appendChild(item);
      }
      if (!root.childNodes.length) {
        root.textContent = "No traces found.";
      }
    }

    function renderCollection(rootId, values, formatter) {
      const root = el(rootId);
      root.textContent = "";
      for (const value of values || []) {
        const item = document.createElement("div");
        item.className = "timeline-item";
        item.textContent = formatter(value);
        root.appendChild(item);
      }
      if (!root.childNodes.length) {
        root.textContent = "None";
      }
    }

    function renderTrace(trace, warning) {
      const meta = trace.meta || {};
      el("trace-summary").textContent =
        `trace=${trace.trace_id} session=${trace.session_id} internal=${trace.internal_session_id} created_at=${trace.created_at} mode=${trace.mode} total=${meta.latency_ms_total ?? "n/a"} route=${meta.route_label || meta.outcome || "unknown"}${warning ? " warning=" + warning : ""}`;
      renderCollection("timeline", meta.timings, (item) => `${item.stage}: ${item.ms}ms ${JSON.stringify(item.fields || {})}`);
      renderCollection("retrieved", meta.retrieved, (item) => JSON.stringify(item));
      renderCollection("usage", meta.usage, (item) => JSON.stringify(item));
      el("answer").textContent = trace.answer || "";
      el("raw-json").textContent = JSON.stringify({ trace, warning }, null, 2);
    }

    async function loadTrace(traceId) {
      const response = await fetch(`/debug/chat-route/traces/${encodeURIComponent(traceId)}`, { headers: apiHeaders() });
      const data = await response.json();
      if (!response.ok) {
        el("run-status").textContent = data.detail || "Failed to load trace";
        return;
      }
      renderTrace(data.trace);
    }

    el("run-button").addEventListener("click", async () => {
      el("run-status").textContent = "Running...";
      const response = await fetch("/debug/chat-route/run", {
        method: "POST",
        headers: apiHeaders(),
        body: JSON.stringify({
          question: el("question").value,
          session_id: el("session-id").value,
          mode: el("mode").value,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        el("run-status").textContent = data.detail || "Run failed";
        return;
      }
      el("run-status").textContent = data.warning || "Run complete";
      renderTrace(data.trace, data.warning);
    });

    el("list-button").addEventListener("click", async () => {
      const params = new URLSearchParams();
      if (el("filter-session-id").value) params.set("session_id", el("filter-session-id").value);
      if (el("filter-trace-id").value) params.set("trace_id", el("filter-trace-id").value);
      const response = await fetch(`/debug/chat-route/traces?${params.toString()}`, { headers: apiHeaders() });
      const data = await response.json();
      if (!response.ok) {
        el("run-status").textContent = data.detail || "Failed to load traces";
        return;
      }
      renderList(data.traces);
    });
  </script>
</body>
</html>
"""


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


def _require_chat_api_key(x_api_key: str | None) -> str:
    if not CHAT_API_KEY:
        raise HTTPException(status_code=503, detail="Chat API disabled: CHAT_API_KEY not set")
    if x_api_key != CHAT_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


def _scoped_api_session_id(api_key: str, client_session_id: str) -> str:
    session_key = f"{api_key}\0{client_session_id}"
    return "api:" + hashlib.sha256(session_key.encode("utf-8")).hexdigest()[:32]


def _require_client_session_id(body: dict | None) -> str:
    client_session_id = (body or {}).get("session_id")
    if not isinstance(client_session_id, str) or not client_session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")
    return client_session_id.strip()


def _run_answer_with_meta(question: str, session_id: str, mode: str) -> tuple[str, dict]:
    if mode == "auto":
        return answer_with_meta(question, session_id=session_id)
    return answer_with_meta(question, session_id=session_id, mode=mode)


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
    trace_id = str(trace_meta.get("trace_id") or uuid.uuid4().hex)
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
def debug_chat_route_page() -> str:
    return DEBUG_CHAT_ROUTE_HTML


@app.post("/debug/chat-route/run")
async def debug_chat_route_run(
    body: dict,
    x_api_key: str | None = Header(default=None),
) -> dict:
    api_key = _require_chat_api_key(x_api_key)
    payload = body or {}
    question = payload.get("question", "")
    mode = normalize_mode(payload.get("mode"))
    client_session_id = _require_client_session_id(payload)
    session_id = _scoped_api_session_id(api_key, client_session_id)

    try:
        reply, meta = _run_answer_with_meta(question, session_id, mode)
    except Exception:
        log.exception("Debug chat route run failed")
        reply = TECHNICAL_ERROR_REPLY
        meta = {"error": "technical_error"}
    trace = _build_debug_trace(
        reply=reply,
        meta=meta,
        question=question,
        session_id=client_session_id,
        internal_session_id=session_id,
        mode=mode,
    )
    try:
        saved = save_chat_trace(
            trace_id=trace["trace_id"],
            session_id=client_session_id,
            internal_session_id=session_id,
            mode=mode,
            question=question,
            answer=reply,
            meta=trace["meta"],
        )
    except Exception:
        log.exception("Debug trace persistence failed")
        return {"trace": trace, "warning": "trace_persistence_failed"}
    return {"trace": saved}


@app.get("/debug/chat-route/traces")
def debug_chat_route_traces(
    x_api_key: str | None = Header(default=None),
    session_id: str | None = Query(default=None),
    trace_id: str | None = Query(default=None),
    limit: int = Query(default=20),
) -> dict:
    _require_chat_api_key(x_api_key)
    return {"traces": list_chat_traces(session_id=session_id, trace_id=trace_id, limit=limit)}


@app.get("/debug/chat-route/traces/{trace_id}")
def debug_chat_route_trace_detail(
    trace_id: str,
    x_api_key: str | None = Header(default=None),
) -> dict:
    _require_chat_api_key(x_api_key)
    trace = get_chat_trace(trace_id)
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
    client_session_id = _require_client_session_id(payload)
    session_id = _scoped_api_session_id(api_key, client_session_id)
    if include_meta:
        try:
            reply, meta = _run_answer_with_meta(question, session_id, mode)
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
