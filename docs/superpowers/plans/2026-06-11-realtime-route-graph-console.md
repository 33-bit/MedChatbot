# Realtime Route Graph Debug Console — Implementation Plan

**Date:** 2026-06-11
**Spec:** `docs/superpowers/specs/2026-06-10-realtime-route-graph-console-design.md`
**REQUIRED SUB-SKILL:** Use `superpowers:subagent-driven-development` to execute this plan.

## Goal

Upgrade `GET /debug/chat-route` so a developer can (1) watch the pipeline light up
node-by-node in realtime instead of waiting ~26s for the whole turn, and (2) read
the graph on a polished n8n-style canvas (icons, bezier edges, full restyle,
pan/zoom/drag). Move the frontend out of the Python triple-quoted string into
static files.

## Non-negotiable constraints

- **`answer(question, session_id="default")` stays byte-for-byte unchanged.** All
  channels use it. Streaming only touches the debug path `answer_with_meta`.
- **No medical-safety behavior changes** (guardrail verdicts, no-data replies,
  technical-error reply, emergency guidance).
- **The event sink is a no-op unless explicitly installed** by `/stream`. Every
  normal `answer()` call and every existing test must see zero difference.
- **`hybrid_search(query, top_k)` (the non-debug public path) stays unchanged.**
  Only `hybrid_search_with_debug` gains an optional callback.
- Per CLAUDE.md: run `gitnexus_impact({target, direction:"upstream"})` before
  editing any symbol; warn on HIGH/CRITICAL. Run `gitnexus_detect_changes()`
  before every commit. Never rename via find-and-replace — use `gitnexus_rename`.
- Use the **MedChatbot conda env**. No pip installs unless asked. No new frontend
  build system or JS framework.

## Architecture

```
POST /debug/chat-route/stream
  └─ spawn worker thread: answer_with_meta(question, session_id, mode)
       ├─ install thread-local event sink (queue.Queue) for this turn
       ├─ _log_timing(...) ──► _emit_node_event(stage, status, ms) ──► sink.put({node})
       └─ parallel retrieval (ThreadPoolExecutor, 2 workers)
            ├─ kg loader      ── sink passed EXPLICITLY across boundary ──► sink
            └─ hybrid loader  ── per-stage callback ──► sink (dense/sparse/fusion/rerank)
  └─ main thread drains queue ──► StreamingResponse (text/event-stream)
       ├─ data: {"type":"node","id":...,"status":...,"ms":...}\n\n   (one per event)
       └─ data: {"type":"done","trace":{...saved trace...}}\n\n      (terminal)

Client (fetch + ReadableStream, NOT EventSource — needs X-API-Key header)
  ├─ on node event: flip node skipped→success/error, pulse next as running
  └─ on done: render full graph + inspector from trace (identical to today)
```

**Why a queue + worker thread:** the pipeline is synchronous; the only way to emit
mid-flight is to run it off the response thread and bridge events through a
thread-safe queue. **Why explicit sink-passing into the executor:** the sink is
thread-local on the request worker; the retrieval `ThreadPoolExecutor` spawns
*new* threads where `_META_LOCAL.current` is `None`, so the sink object must be
handed in as an argument.

## Tech stack

- Python 3 / FastAPI / `StreamingResponse` / `queue.Queue` / `threading`.
- Existing: `_META_LOCAL` thread-local, `_log_timing` chokepoint, `_HybridSearchDebug`.
- Frontend: plain HTML/CSS/JS in `src/server/static/debug_console/`, served via
  `FileResponse`. No bundler.
- Tests: pytest. JS sanity: `node --check` on served bytes.

## Verification gate (run after EVERY task)

```bash
conda run -n MedChatbot python3 -m pytest \
  tests/test_pipeline.py tests/test_api.py \
  tests/test_storage_and_quota.py tests/test_retrieval_debug.py -v
conda run -n MedChatbot python3 -m compileall src eval
```

A task is not "done" until this gate is green and its own new test passes.

---

## Task 1 — Event sink primitive + `_emit_node_event`, wired into `_log_timing`

**Goal:** a thread-local event sink and one emission path. No-op when no sink is
installed, so `answer()` and every existing test are unchanged.

**Pre-edit:** `gitnexus_impact({target:"_log_timing", direction:"upstream"})` and
report blast radius. (Expect the orchestration helpers in `pipeline.py`.)

### 1a. Failing test

Add to `tests/test_pipeline.py`:

```python
def test_emit_node_event_is_noop_without_sink():
    from src.chat import pipeline

    # No sink installed on this thread -> must not raise, must not store.
    pipeline._META_LOCAL.current = None
    pipeline._emit_node_event("route", "ok", 1.5)  # no exception = pass


def test_emit_node_event_pushes_to_installed_sink():
    import queue as _queue
    from src.chat import pipeline

    sink: _queue.Queue = _queue.Queue()
    pipeline._install_event_sink(sink)
    try:
        pipeline._emit_node_event("kg_search", "ok", 12.0)
    finally:
        pipeline._install_event_sink(None)

    event = sink.get_nowait()
    assert event == {"type": "node", "id": "kg_search", "status": "ok", "ms": 12.0}


def test_log_timing_emits_node_event_when_sink_installed():
    import queue as _queue
    import time
    from src.chat import pipeline

    sink: _queue.Queue = _queue.Queue()
    pipeline._install_event_sink(sink)
    try:
        pipeline._log_timing("trace-x", "generate", time.perf_counter(), chars=10)
    finally:
        pipeline._install_event_sink(None)

    event = sink.get_nowait()
    assert event["type"] == "node"
    assert event["id"] == "generate"
    assert event["status"] == "ok"
    assert isinstance(event["ms"], float)
```

Run: `conda run -n MedChatbot python3 -m pytest tests/test_pipeline.py -k emit_node -q`
— expect failures (`_emit_node_event`, `_install_event_sink` undefined).

### 1b. Implement (in `src/chat/pipeline.py`)

The sink lives on the SAME thread-local as `meta`, on a separate attribute so it
survives independently of `_collect_graph`. Add near `_META_LOCAL` (line ~103):

```python
def _event_sink():
    """Return the active per-thread event sink, or None when not streaming."""
    return getattr(_META_LOCAL, "event_sink", None)


def _install_event_sink(sink) -> None:
    """Install (or clear with None) the per-thread streaming event sink."""
    _META_LOCAL.event_sink = sink


def _emit_node_event(stage: str, status: str, ms: float | None) -> None:
    """Push a compact node-progress event to the active sink, if one is installed.

    No-op when no sink is present (every normal answer() call and every test),
    so the public pipeline behavior is unchanged.
    """
    sink = _event_sink()
    if sink is None:
        return
    sink.put({
        "type": "node",
        "id": stage,
        "status": status,
        "ms": round(float(ms), 2) if ms is not None else None,
    })
```

Then have `_log_timing` emit after recording (line 179):

```python
def _log_timing(trace_id: str, stage: str, start: float, **fields) -> None:
    ms = elapsed_ms(start)
    _record_timing(stage, ms, fields)
    log_trace_timing(log, "pipeline", trace_id, stage, start, **fields)
    status = "error" if fields.get("failed") else "ok"
    _emit_node_event(stage, status, ms)
```

> Note: `_log_timing` currently calls `elapsed_ms(start)` indirectly via
> `_record_timing`/`log_trace_timing`. Compute `ms` once and pass it through to
> keep a single value. `log_trace_timing` takes `start` (not `ms`) — leave that
> signature as-is; only `_record_timing` and `_emit_node_event` use `ms`.
> Verify `_record_timing(stage, elapsed_ms(start), fields)` still receives the
> same value — it does, since `ms = elapsed_ms(start)`.

### 1c. Verify

- New tests pass.
- Full gate green (proves no-sink path unchanged for all existing pipeline tests).

### 1d. Commit

`gitnexus_detect_changes()` → confirm only `_log_timing` + new helpers affected.
Commit: `add streaming event sink primitive to pipeline`

---

## Task 2 — Carry the sink across the ThreadPoolExecutor boundary

**Goal:** coarse retrieval stages (`kg_search`, `hybrid_search`) that run in
worker threads still reach the sink. The sink is thread-local on the request
worker; executor threads can't see it — pass it explicitly.

**Pre-edit:** `gitnexus_impact` on `_load_kg_context`, `_load_hybrid_hits`,
`_call_with_elapsed`, and the `_retrieve_and_generate` block (the function
containing the `ThreadPoolExecutor` at line ~725). Report blast radius.

### 2a. Failing test

Add to `tests/test_pipeline.py`:

```python
def test_parallel_retrieval_emits_from_worker_threads(monkeypatch):
    import queue as _queue
    from src.chat import pipeline
    from src.chat.retrieval.types import Hit

    sink: _queue.Queue = _queue.Queue()

    def fake_kg_search(question):
        from src.chat.retrieval.kg import KgResult
        return KgResult(matched_entities=[], triples=[], chunks=[])

    def fake_hybrid(question, top_k):
        return [Hit(chunk_id="c1", source_type="disease", source_slug="s",
                    source_name="n", heading_path="h", text="t",
                    score=1.0, metadata={})]

    monkeypatch.setattr(pipeline, "kg_search", fake_kg_search)
    monkeypatch.setattr(pipeline, "hybrid_search", fake_hybrid)

    pipeline._install_event_sink(sink)
    try:
        pipeline._load_hybrid_hits("q", "trace-x")
    finally:
        pipeline._install_event_sink(None)

    ids = []
    while not sink.empty():
        ids.append(sink.get_nowait()["id"])
    assert "hybrid_search" in ids
```

> This proves the sink reaches the loader. Adjust the `Hit`/`KgResult`
> constructors to match the real dataclasses in `src/chat/retrieval/types.py`
> and `src/chat/retrieval/kg.py` — read them first; do not guess field names.

Run the test — expect failure if the sink isn't propagated (it currently relies
on thread-local, which IS visible when calling the loader directly, so this test
mainly locks behavior; the real boundary test follows in Task 4's stream test).

### 2b. Implement

The loaders already call `_log_timing`, which calls `_emit_node_event`. When the
loaders run **inside the executor**, `_event_sink()` returns `None` on the worker
thread. Fix: capture the sink on the request thread and re-install it on each
worker via a wrapper.

In the parallel block (line ~725), capture the sink before submitting and wrap
the loaders so each worker re-installs it:

```python
parent_sink = _event_sink()

def _with_sink(fn, *args):
    if parent_sink is not None:
        _install_event_sink(parent_sink)
    try:
        return _call_with_elapsed(fn, *args)
    finally:
        if parent_sink is not None:
            _install_event_sink(None)

with ThreadPoolExecutor(max_workers=2, thread_name_prefix="retrieval") as executor:
    kg_future = executor.submit(_with_sink, kg_loader, search_question, trace_id)
    hits_future = executor.submit(_with_sink, hits_loader, search_question, trace_id)
    ...
```

> Keep everything else in the block identical. `_with_sink` must return exactly
> what `_call_with_elapsed` returns (the `(result, elapsed)` tuple) so the
> downstream unpacking is unchanged.

### 2c. Verify + commit

Gate green. `gitnexus_detect_changes()`. Commit: `propagate event sink into parallel retrieval workers`

---

## Task 3 — Optional per-stage callback in retrieval service (sub-stage animation)

**Goal:** `dense`/`sparse`/`fusion`/`rerank` light up individually. Add an
optional callback to `hybrid_search_with_debug`; default `None` → unchanged.

**Pre-edit:** `gitnexus_impact` on `hybrid_search_with_debug`, `_run_hybrid_search`,
`hybrid_search`. Report blast radius. Confirm `hybrid_search` (non-debug) callers
are unaffected.

### 3a. Failing test

Add to `tests/test_retrieval_debug.py`:

```python
def test_hybrid_search_with_debug_invokes_substage_callback_in_order(monkeypatch):
    from src.chat.retrieval import service

    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(service, "dense_search", lambda q, top_k: [],
                        raising=False)
    # ... monkeypatch bm25_search, rrf_merge, rerank to return [] too;
    #     read service.py imports to patch at the right module path.

    def on_stage(stage: str, status: str, ms: float):
        calls.append((stage, status))

    service.hybrid_search_with_debug("q", top_k=3, on_stage=on_stage)

    assert [c[0] for c in calls] == ["dense", "sparse", "fusion", "rerank"]
    assert all(c[1] == "ok" for c in calls)


def test_hybrid_search_with_debug_without_callback_unchanged(monkeypatch):
    from src.chat.retrieval import service
    # same monkeypatching; call WITHOUT on_stage
    hits, debug = service.hybrid_search_with_debug("q", top_k=3)
    assert "timings_ms" in debug  # behaves exactly as today
```

> `dense_search` and `rerank` are imported lazily inside `_run_hybrid_search`
> (`from src.chat.retrieval.dense import dense_search`). Patch them on their
> source modules (`src.chat.retrieval.dense`, `src.chat.retrieval.rerank`), not
> on `service`. `bm25_search` and `rrf_merge` are module-level imports on
> `service` — patch those on `service`. Read the file to confirm before writing.

### 3b. Implement (in `src/chat/retrieval/service.py`)

Thread an optional `on_stage` callback through. `_run_hybrid_search` invokes it
after each stage's `_record_debug_timing`. The callback signature is
`(stage: str, status: str, ms: float) -> None`. Stage names: `dense`, `sparse`,
`fusion`, `rerank` (match the node ids in the graph, NOT the internal
`dense_search`/`sparse_search` timing keys — map them).

```python
def _run_hybrid_search(query, top_k, on_stage=None):
    ...
    dense_ms = _record_debug_timing(debug, "dense_search", stage_start)
    if on_stage is not None:
        on_stage("dense", "ok", dense_ms)
    ...
    # same after sparse -> on_stage("sparse", "ok", sparse_ms)
    # fusion -> on_stage("fusion", "ok", fusion_ms)
    # rerank -> on_stage("rerank", "ok", rerank_ms)
```

On failure, before re-raising, call `on_stage(<stage>, "error", ms)` (use the
node-id mapping). Keep the existing `_record_debug_failure` calls intact.

```python
def hybrid_search_with_debug(query, top_k=RERANK_TOP_K, on_stage=None):
    hits, debug = _run_hybrid_search(query, top_k, on_stage=on_stage)
    return hits, debug.as_dict()
```

**`hybrid_search` (line 183) is NOT touched** — it calls
`_run_hybrid_search(query, top_k)` with no callback (default `None`).

### 3c. Wire the pipeline to forward the callback → sink

In `src/chat/pipeline.py`, `_load_hybrid_hits_with_debug` (line ~526): build a
callback that forwards to `_emit_node_event` and pass it in.

```python
def _load_hybrid_hits_with_debug(question, trace_id):
    stage_start = time.perf_counter()

    def _on_stage(stage, status, ms):
        _emit_node_event(stage, status, ms)

    hits, retrieval_debug = hybrid_search_with_debug(
        question, top_k=RERANK_TOP_K, on_stage=_on_stage,
    )
    _log_timing(trace_id, "hybrid_search", stage_start, hits=len(hits))
    return hits, retrieval_debug
```

> Because Task 2 re-installs the sink on the retrieval worker, `_emit_node_event`
> here (running on that worker) reaches the sink correctly.

### 3d. Verify + commit

Gate green (existing `test_retrieval_debug.py` MUST still pass — proves
default-None path unchanged). `gitnexus_detect_changes()`.
Commit: `add optional per-substage callback to hybrid retrieval debug path`

---

## Task 4 — `POST /debug/chat-route/stream` SSE endpoint

**Goal:** run `answer_with_meta` on a worker thread, bridge events through a
`queue.Queue`, and stream them as `text/event-stream`, ending with a `done` event
carrying the full saved trace. `/run` stays as the non-streaming fallback.

**Pre-edit:** `gitnexus_impact` on `debug_chat_route_run`, `_build_debug_trace`,
`_run_answer_with_meta`. Report blast radius.

### 4a. Failing tests

Add to `tests/test_api.py`:

```python
def test_debug_chat_route_stream_requires_api_key(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")

    missing = client.post(
        "/debug/chat-route/stream",
        json={"question": "Tôi bị ho", "session_id": "debug-user"},
    )
    wrong = client.post(
        "/debug/chat-route/stream",
        headers={"X-API-Key": "wrong"},
        json={"question": "Tôi bị ho", "session_id": "debug-user"},
    )
    assert missing.status_code == 401
    assert wrong.status_code == 401


def test_debug_chat_route_stream_emits_nodes_then_done_and_persists(app_client, monkeypatch):
    import json
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")

    def fake_answer_with_meta(question, session_id="default", mode="auto"):
        # Drive two node events through the real sink, then return.
        from src.chat import pipeline
        import time
        pipeline._log_timing("t", "route", time.perf_counter(), label="informational")
        pipeline._log_timing("t", "generate", time.perf_counter(), chars=5)
        return "stream answer", {
            "trace_id": "trace-stream",
            "latency_ms_total": 9.0,
            "timings": [{"stage": "total", "ms": 9.0, "fields": {}}],
            "graph_nodes": [],
        }

    monkeypatch.setattr(app_module, "answer_with_meta", fake_answer_with_meta)

    with client.stream(
        "POST",
        "/debug/chat-route/stream",
        headers={"X-API-Key": "secret"},
        json={"question": "Tôi bị ho", "session_id": "debug-user", "mode": "information"},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = []
        for line in resp.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:"):].strip()))

    node_ids = [e["id"] for e in events if e["type"] == "node"]
    done = [e for e in events if e["type"] == "done"]
    assert "route" in node_ids and "generate" in node_ids
    assert len(done) == 1
    saved_trace = done[0]["trace"]
    assert saved_trace["answer"] == "stream answer"
    assert saved_trace["trace_id"]

    # Trace persisted -> replayable.
    replay = client.get(
        f"/debug/chat-route/traces/{saved_trace['trace_id']}?session_id=debug-user",
        headers={"X-API-Key": "secret"},
    )
    assert replay.status_code == 200


def test_debug_chat_route_stream_done_on_worker_exception(app_client, monkeypatch):
    import json
    from src.chat.replies import TECHNICAL_ERROR_REPLY
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")

    def broken(question, session_id="default", mode="auto"):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module, "answer_with_meta", broken)

    with client.stream(
        "POST",
        "/debug/chat-route/stream",
        headers={"X-API-Key": "secret"},
        json={"question": "x", "session_id": "debug-user"},
    ) as resp:
        events = []
        for line in resp.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:"):].strip()))

    done = [e for e in events if e["type"] == "done"]
    assert len(done) == 1
    assert done[0]["trace"]["answer"] == TECHNICAL_ERROR_REPLY
```

> `client.stream(...)` is Starlette `TestClient` (httpx) API. Confirm the
> conftest `app_client` yields an httpx-based `TestClient`; if it's the older
> `requests`-based client, use `resp = client.post(..., stream=True)` and
> `resp.iter_lines()` instead. Read `tests/conftest.py` first.

### 4b. Implement (in `src/server/app.py`)

Add imports: `import queue`, `import threading`, and
`from fastapi.responses import StreamingResponse` (check existing imports first).

```python
@app.post("/debug/chat-route/stream")
async def debug_chat_route_stream(
    body: dict,
    x_api_key: str | None = Header(default=None),
):
    api_key = _require_chat_api_key(x_api_key)
    payload = body or {}
    question = payload.get("question", "")
    mode = normalize_mode(payload.get("mode"))
    client_session_id = _require_client_session_id(payload)
    session_id = _scoped_api_session_id(api_key, client_session_id)

    from src.chat import pipeline

    events: "queue.Queue" = queue.Queue()
    _SENTINEL = object()
    result: dict = {}

    def _worker():
        pipeline._install_event_sink(events)
        try:
            reply, meta = _run_answer_with_meta(question, session_id, mode)
        except Exception:
            log.exception("Debug chat route stream failed")
            reply, meta = TECHNICAL_ERROR_REPLY, {"error": "technical_error"}
        finally:
            pipeline._install_event_sink(None)
        result["reply"] = reply
        result["meta"] = meta
        events.put(_SENTINEL)

    def _stream():
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        while True:
            item = events.get()
            if item is _SENTINEL:
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        thread.join()

        trace = _build_debug_trace(
            reply=result["reply"], meta=result.get("meta"),
            question=question, session_id=client_session_id,
            internal_session_id=session_id, mode=mode,
        )
        try:
            saved = save_chat_trace(
                trace_id=trace["trace_id"], session_id=client_session_id,
                internal_session_id=session_id, mode=mode, question=question,
                answer=result["reply"], meta=trace["meta"],
                created_at=trace["created_at"],
            )
        except Exception:
            log.exception("Debug stream trace persistence failed")
            saved = trace
        yield f"data: {json.dumps({'type': 'done', 'trace': saved}, ensure_ascii=False)}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")
```

> CRITICAL: install the sink **inside `_worker`** (the worker thread), not on the
> request thread — `answer_with_meta` runs there and Task 2 re-propagates it into
> the retrieval executor. `import json` is already used elsewhere in app.py;
> confirm it's imported at module top.

### 4c. Verify + commit

Gate green + the three new stream tests. `gitnexus_detect_changes()`.
Commit: `add streaming debug-route endpoint`

---

## Task 5 — Extract frontend to static files (no behavior change)

**Goal:** move the inline HTML/CSS/JS out of `DEBUG_CHAT_ROUTE_HTML` into
`src/server/static/debug_console/{index.html,app.js,styles.css}`, served via
`FileResponse`. Pure extraction — visuals/behavior identical to today.

### 5a. Update the served-page test

`tests/test_api.py::test_debug_chat_route_page_is_served` currently asserts inline
JS function names are present in the HTML response. After extraction the page only
references `app.js`. Change the assertions to:

```python
def test_debug_chat_route_page_is_served(app_client):
    client, _ = app_client
    response = client.get("/debug/chat-route")
    assert response.status_code == 200
    assert "Chat Route Debug Console" in response.text
    assert 'id="workflow-graph"' in response.text
    assert 'id="node-inspector"' in response.text
    assert "app.js" in response.text  # script is now external
```

Add a test that the JS asset is served and syntactically loadable:

```python
def test_debug_console_app_js_is_served(app_client):
    client, _ = app_client
    response = client.get("/debug/chat-route/static/app.js")
    assert response.status_code == 200
    assert "renderWorkflowGraph" in response.text
```

> The old `test_debug_chat_route_page_js_has_no_unterminated_string` (line 211)
> asserted on inline JS. After extraction this class of bug is gone; replace it
> with a `node --check` step in the verification, or repoint it to fetch
> `/debug/chat-route/static/app.js` and assert `.join("\n")` (now a real newline
> in a real JS file is correct, so flip the assertion). Simplest: delete that
> test and rely on `node --check` in the gate.

### 5b. Implement

1. Create `src/server/static/debug_console/index.html` — the markup from
   `DEBUG_CHAT_ROUTE_HTML`, with `<style>` replaced by
   `<link rel="stylesheet" href="/debug/chat-route/static/styles.css">` and
   `<script>` replaced by
   `<script src="/debug/chat-route/static/app.js"></script>`.
2. Create `styles.css` (the CSS) and `app.js` (the JS) — **paste verbatim**; the
   `\n` escape bug disappears because these are real files, not Python strings.
3. Serve them. Either mount StaticFiles or add explicit routes:

```python
from fastapi.responses import FileResponse
from pathlib import Path

_DEBUG_CONSOLE_DIR = Path(__file__).parent / "static" / "debug_console"

@app.get("/debug/chat-route", response_class=HTMLResponse)
def debug_chat_route_page() -> HTMLResponse:
    return HTMLResponse((_DEBUG_CONSOLE_DIR / "index.html").read_text("utf-8"))

@app.get("/debug/chat-route/static/{asset}")
def debug_chat_route_asset(asset: str):
    # whitelist to prevent path traversal
    if asset not in {"app.js", "styles.css"}:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(_DEBUG_CONSOLE_DIR / asset)
```

> SECURITY: the explicit whitelist avoids path traversal. Do NOT pass `asset`
> into the path without the membership check. Delete the now-unused
> `DEBUG_CHAT_ROUTE_HTML` constant once nothing references it (orphan cleanup —
> your own change made it dead).

### 5c. Verify + commit

Gate green + `node --check src/server/static/debug_console/app.js`.
`gitnexus_detect_changes()`. Commit: `extract debug console frontend to static files`

---

## Task 6 — Client-side stream consumption

**Goal:** the page calls `/stream` and lights nodes up live; falls back to `/run`
on stream failure.

### 6a. Implement (in `app.js`)

- Add a `runStream(question, sessionId, mode)` that POSTs to
  `/debug/chat-route/stream` with `fetch` (header `X-API-Key`), reads
  `response.body.getReader()`, decodes chunks, splits on `\n\n`, parses each
  `data:` line.
- On `{type:"node"}`: set that node's status class (`success`/`error`), and mark
  the next expected node `running` (client-side pulse using the known stage
  order).
- On `{type:"done"}`: call the existing `renderWorkflowGraph` + inspector render
  with `event.trace` — identical to today's end state.
- Wrap in try/catch; on any error (no `ReadableStream`, network), fall back to the
  existing `/run` fetch path so the console still works.
- Point the Run button at `runStream` (keep `/run` reachable as fallback only).

> No new test framework. Verify via Playwright smoke (manual, below). The Python
> tests already cover the server contract.

### 6b. Verify

- Gate green (unchanged server tests).
- Manual Playwright smoke (see Rollout) — nodes light up incrementally.

### 6c. Commit

`gitnexus_detect_changes()`. Commit: `consume stream client-side with live node lighting`

---

## Task 7 — Visual polish (icons, cards, edges, restyle)

**Goal:** n8n-style look. Pure CSS/JS/SVG in the static files — no server change.

### 7a. Implement

- **Icons:** inline SVG per node type (route=signpost, kg_search=graph,
  dense/sparse/fusion/rerank=magnifier variants, generate=robot, persist=database,
  input=form, total=flag). A small `NODE_ICONS` map in `app.js`.
- **Cards:** rounded node cards, status color accents (grey skipped / blue
  running-pulse / green success / red error), label + sublabel (`status · Xms`).
- **Edges:** swap straight lines for SVG bezier paths with arrowheads; clean
  branch out of `route` and merge into `generate`; a subtle bounding box grouping
  the retrieval branch (dense/sparse/fusion/rerank).
- **Restyle:** refreshed dark palette, restyled Live Run / Replay form panels,
  inspector typography, retrieval tables.

### 7b. Verify + commit

`node --check app.js`. Gate green. Manual visual check. `gitnexus_detect_changes()`.
Commit: `polish debug console nodes, edges, and palette`

---

## Task 8 — Pan / zoom / drag canvas

**Goal:** hand-rolled pan/zoom/drag on the absolute-positioned canvas. No library.

### 8a. Implement (in `app.js` + `styles.css`)

- Wrap the node layer in a `transform` container; track `{x, y, scale}` state.
- Drag-to-pan on empty canvas (pointerdown/move/up on the wrapper).
- Wheel-to-zoom (clamp scale, zoom toward cursor).
- Drag a node to reposition (pointer events on the node; update its stored
  position; redraw its connected edges).
- A "reset view" button to recenter.

### 8b. Verify + commit

`node --check`. Gate green. Manual smoke (pan, zoom, drag a node, edges follow).
`gitnexus_detect_changes()`. Commit: `add pan/zoom/drag to debug console canvas`

---

## Rollout & manual verification

After Task 8, end-to-end manual check (server in MedChatbot env):

```bash
conda run -n MedChatbot uvicorn src.server.app:app --host 0.0.0.0 --port 8000 --reload
```

Playwright smoke (macOS — do NOT use the `timeout` shell wrapper, it's absent):
1. Load `/debug/chat-route`, enter a question + session_id, set the API key.
2. Click Run → assert nodes flip skipped→running→success **incrementally** (not
   all at once), KG/retrieval sub-stages light individually, inspector populates
   on `done`.
3. Pan, zoom, drag a node; confirm edges follow.
4. Replay a saved trace; confirm legacy traces (no `graph_nodes`) still render via
   `buildLegacyGraphNodes`.

## Final verification (whole feature)

```bash
conda run -n MedChatbot python3 -m pytest \
  tests/test_pipeline.py tests/test_api.py \
  tests/test_storage_and_quota.py tests/test_retrieval_debug.py \
  tests/test_analyzer_and_generation.py -v
conda run -n MedChatbot python3 -m compileall src eval
node --check src/server/static/debug_console/app.js
```

Confirm: public `answer()` unchanged (channel tests pass), no medical-safety
behavior touched, all debug endpoints behind `CHAT_API_KEY`.

## Out of scope

- No change to `HYBRID_CANDIDATE_K` / `RERANK_TOP_K`.
- No streaming of the LLM answer text (only stage/node progress).
- No new frontend build system or JS framework.
