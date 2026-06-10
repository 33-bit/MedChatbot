# Realtime Route Graph Debug Console — Design

**Date:** 2026-06-10
**Status:** Draft (pending user spec review)

## Goal

Upgrade `/debug/chat-route` so a developer can (1) watch the pipeline execute in
realtime — each node lights up the moment its stage completes — and (2) read the
graph on a polished, n8n-style canvas with icons, refined edges, a full visual
restyle, and pan/zoom/drag.

This builds on the existing route-graph console (clickable nodes + inspector)
already on `main`. It does not change the pipeline's medical behavior or the
public `answer()` contract.

## Two parts

### Part 1 — Realtime streaming

**Constraint.** The pipeline runs synchronously to completion (~26s), accumulating
per-stage data in a thread-local `meta` dict. Every stage reports through one
chokepoint: `_log_timing(trace_id, stage, start, **fields)` in
`src/chat/pipeline.py`. The public `answer()` path (all channels) must stay
unchanged, and the existing `/debug/chat-route/run` endpoint stays as a
non-streaming fallback.

**Approach: SSE-style streaming over `fetch`.**

New endpoint `POST /debug/chat-route/stream` (same auth + session scoping as
`/run`):

1. Runs `answer_with_meta(question, session_id, mode)` on a worker thread.
2. A thread-local **event sink** (a `queue.Queue`) is installed for that turn.
   `_log_timing` gains an optional hook: when a sink is active on the current
   thread, it pushes a compact event `{"type":"node","id":<stage>,"status":<ok|error>,"ms":<float>}`
   after recording the timing as it does today. When no sink is active (every
   normal `answer()` call, every test), behavior is byte-for-byte unchanged.
3. The endpoint returns a `StreamingResponse` (media type `text/event-stream`)
   that drains the queue and yields one SSE `data:` line per event, then a final
   `{"type":"done","trace":{...}}` event carrying the full saved trace (same
   shape `/run` returns today, including `graph_nodes` for the inspector).
4. The trace is still persisted via `save_chat_trace` exactly as `/run` does, so
   Replay is unaffected.

**Client.** The page reads the stream with `fetch` + `ReadableStream` (not
`EventSource`, which can't send the `X-API-Key` header). On each `node` event it
flips that node from "skipped" to success/error and marks the *next* expected
node "running" (client-side pulse). On `done` it renders the full graph +
inspector from the trace, identical to today's end state.

**Stage→node mapping.** Coarse pipeline stages flow through `_log_timing` and
can light up live: `load_session`, `preflight`, `turn_analysis`, `route`,
`kg_search`, `parallel_retrieval`/`hybrid_search`, `generate`, `persist`,
`total`. Granularity decisions:

- **KG and hybrid retrieval run in parallel** (`parallel_retrieval`). They do not
  light up strictly one-before-the-other; both branches advance concurrently and
  the client may show both as "running" at once — which is the honest n8n-style
  depiction of parallel work.
- **The retrieval sub-stages** (`dense`/`sparse`/`fusion`/`rerank`) will animate
  individually. Today they are timed *inside*
  `src/chat/retrieval/service.py` into a `timings_ms` dict and not emitted live.
  To animate them, `hybrid_search_with_debug` gains an optional per-stage
  callback (default `None`); when provided, it invokes the callback as each of
  dense → sparse → fusion → rerank completes. The pipeline passes a callback that
  forwards to the same thread-local event sink used by `_log_timing`. When no
  callback is provided (every normal call and existing tests), `service.py`
  behaves exactly as today.
- `rewrite` has no own timing; it lights up together with `turn_analysis`.

Stages without a 1:1 node (diagnostic-only stages) are ignored for the live
animation but still appear in the final `graph_nodes`.

**Event sink as the shared primitive.** Rather than coupling the sink to
`_log_timing` alone, define a small module-level helper in `pipeline.py`:
`_emit_node_event(stage, status, ms)` that pushes to the active thread-local
sink if one is installed (no-op otherwise). `_log_timing` calls it for coarse
stages; the retrieval callback calls it for sub-stages. This keeps one emission
path and one "is a sink active?" check.

**Why this approach.** Zero change to `answer()`; reuses the single existing
chokepoint; old endpoint remains; one-directional push fits SSE (no WebSocket
lifecycle); no partial-trace store needed (vs polling).

### Part 2 — Visual polish

- **Node icons + cards:** inline SVG icon per node type (route=signpost,
  kg_search=graph, dense/sparse/fusion/rerank=magnifier variants,
  generate=robot, persist=database, input=form, total=flag). Rounded cards,
  status color accents (grey skipped / blue running-pulse / green success /
  red error), label + sublabel (status · duration).
- **Edges & layout:** smooth bezier edges with arrowheads, cleaner branch/merge
  routing out of `route` and into `generate`, and a subtle bounding box visually
  grouping the retrieval branch (dense/sparse/fusion/rerank).
- **Full page restyle:** refreshed dark palette, restyled Live Run / Replay form
  panels, inspector typography, and retrieval tables.
- **Pan/zoom/drag:** hand-rolled on the existing absolute-positioned canvas via a
  CSS `transform` wrapper + pointer events (drag-to-pan, wheel-to-zoom, drag a
  node to reposition). No library, honoring the "no frontend build system" rule.

## Frontend extraction

Move the inline HTML/CSS/JS out of the Python triple-quoted string
`DEBUG_CHAT_ROUTE_HTML` in `src/server/app.py` into static files:

```
src/server/static/debug_console/index.html
src/server/static/debug_console/app.js
src/server/static/debug_console/styles.css
```

Served via `FileResponse` (or a small `StaticFiles` mount) from the existing
`GET /debug/chat-route` route. **Why:** the streaming + polish work roughly
doubles the JS; keeping it in a Python string keeps re-introducing escape bugs
(a bare `\n` in `.join("\n")` already broke the entire script once). Static
files also let the browser cache JS/CSS normally.

## Compatibility & Security

- Public `answer()` response and all channel behavior unchanged.
- All debug endpoints (page, `/run`, `/stream`, `/traces`) stay behind
  `CHAT_API_KEY`; `/stream` reuses the same `_require_chat_api_key` +
  `_scoped_api_session_id` scoping as `/run`.
- Event sink is per-thread and only installed by `/stream`; absent it,
  `_log_timing` is unchanged — so tests and channels see no difference.
- Older saved traces (no `graph_nodes`) still render via the existing
  `buildLegacyGraphNodes` fallback.
- Streamed events carry only `id`/`status`/`ms` — no chunk bodies; the rich
  payloads arrive once in the final `done` trace, as today.

## Error Handling

- A stage that fails emits a `node` event with `status:"error"`; the client marks
  that node red and stops advancing the "running" marker.
- If the worker thread raises, the endpoint emits a final
  `{"type":"done","trace":{...,"error":"technical_error"}}` so the client always
  terminates cleanly. The `TECHNICAL_ERROR_REPLY` path is preserved.
- If streaming is unsupported or errors client-side, the page falls back to the
  existing non-streaming `/run` request.

## Verification

Tests:
- `_emit_node_event` pushes to an installed sink and is a no-op without one
  (proves `answer()` / `_log_timing` path unchanged).
- `hybrid_search_with_debug` invokes its optional per-stage callback once per
  sub-stage (dense/sparse/fusion/rerank) in order, and behaves identically when
  no callback is passed (existing `test_retrieval_debug.py` must still pass).
- `/debug/chat-route/stream` requires the API key, scopes the session, streams
  coarse-stage + retrieval-sub-stage `node` events followed by a terminal `done`
  event with a saved trace, and persists the trace.
- `GET /debug/chat-route` serves the static page (status 200, contains the graph
  container + key JS hooks).
- Older traces without `graph_nodes` still render (legacy fallback).
- Served JS is syntactically valid (guards the escape-bug class permanently).

Commands:
- `python3 -m pytest tests/test_pipeline.py tests/test_api.py tests/test_storage_and_quota.py tests/test_retrieval_debug.py -v`
- `python3 -m compileall src eval`
- `node --check` on the served `app.js` (or extracted static file)
- Playwright smoke: click Run, assert nodes light up incrementally and the
  inspector renders on completion (manual/optional, not in CI).

## Out of scope

- No change to retrieval limits (`HYBRID_CANDIDATE_K`, `RERANK_TOP_K`).
- No new frontend build system or JS framework.
- No streaming of the LLM answer text itself (only stage/node progress).

## Resolved decisions

- Realtime: nodes light up live (SSE over `fetch`), not just a progress bar.
- Granularity: retrieval sub-stages (dense/sparse/fusion/rerank) animate
  individually via an optional callback in `hybrid_search_with_debug`.
- Polish: all four — icons+cards, edges/layout, full restyle, pan/zoom/drag.
- Frontend lives in static files, not the Python string.
