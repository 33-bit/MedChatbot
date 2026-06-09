# Chat Route Debug Console — Design

**Date:** 2026-06-09
**Status:** Approved (design), pending spec review

## Goal

Add a developer-facing FastAPI debug page that visualizes how one chatbot turn
moves through the pipeline. A user can run a live query and see the request,
route choice, major calls, final response, retrieved context, token usage, and
time spent in each step. The same page can also replay previously saved trace
snapshots.

## Current System Findings

- `POST /chat?include_meta=1` already calls `answer_with_meta(...)` and returns
  `{"answer": ..., "meta": ...}`.
- `answer_with_meta(...)` currently captures `trace_id`, retrieved hits, LLM
  usage, retrieval/generator latency, and total latency.
- The pipeline already logs detailed stage timings through `_log_timing(...)`,
  but most stage timings are log-only and are not included in `meta`.
- SQLite `consultations` currently stores only `session_id`, `question`,
  `answer`, and `created_at`. That is not enough to replay a route visualization.
- Redis stores the active `PatientSession`, but it is TTL-based and should not
  be the replay source.

GitNexus impact checks before design:
- `src/chat/pipeline.py:answer_with_meta` upstream risk: LOW. Direct caller:
  `src/chat/__init__.py:answer_with_meta`; indirect callers include
  `src/server/app.py:chat_debug` and `eval/core.py:run_direct`.
- `src/server/app.py:chat_debug` upstream risk: LOW.
- `src/chat/storage/sqlite_profile.py:log_consultation` upstream risk: LOW.

## Scope

In scope:
- Add an authenticated debug page under FastAPI.
- Add live-run support that executes the real chatbot pipeline.
- Persist rich trace snapshots for replay.
- Browse and replay previous traces by `session_id` and `trace_id`.
- Render the route as a timeline with step duration and key inputs/outputs.
- Keep normal `/chat` behavior unchanged unless debug tracing is requested.

Out of scope:
- Public or unauthenticated access to trace data or live chatbot execution.
- Editing or deleting saved traces from the UI.
- Storing full source chunk text for every retrieved hit. The replay stores
  identifiers and concise metadata already returned in pipeline meta.
- Visualizing every internal helper call. The first version focuses on stable
  pipeline stages useful for debugging.
- Reconstructing rich traces for old `consultations` rows that were saved before
  this feature exists. Old rows can show only question, answer, and timestamp if
  surfaced later.

## Access Model

This is a developer tool protected by the existing `CHAT_API_KEY`.

- `GET /debug/chat-route` serves the HTML console.
- Debug API calls require the same `X-API-Key` header as `/chat`.
- The page has an API-key input stored only in browser memory for the session.
- If `CHAT_API_KEY` is unset, debug endpoints return 503, matching `/chat`.

Serving the HTML page itself can be unauthenticated because it contains no trace
data and cannot execute the chatbot without the API key. All data-bearing
endpoints remain authenticated. If stricter protection is needed later, the page
can also require an API key query or cookie, but that is not necessary for this
first version.

## API Design

### `GET /debug/chat-route`

Returns a static HTML page with inline CSS and JavaScript. The page talks to the
JSON endpoints below. Keeping it inline matches the existing eval viewer style
and avoids adding a frontend build system.

### `POST /debug/chat-route/run`

Request:

```json
{
  "question": "Tôi bị ho và sốt",
  "session_id": "debug-user",
  "mode": "auto"
}
```

Behavior:
- Validates `X-API-Key`.
- Scopes the public `session_id` the same way `/chat` does:
  `api:<sha256(api_key + "\\0" + session_id)[:32]>`.
- Calls `answer_with_meta(...)`.
- Persists the trace snapshot to SQLite.
- Returns the saved trace payload.

Response shape:

```json
{
  "trace": {
    "trace_id": "abcd1234",
    "session_id": "debug-user",
    "internal_session_id": "api:...",
    "mode": "auto",
    "question": "...",
    "answer": "...",
    "created_at": 1781020000.0,
    "meta": {
      "trace_id": "abcd1234",
      "timings": [
        {"stage": "load_session", "ms": 1.2, "fields": {}}
      ],
      "latency_ms_total": 1234.56,
      "retrieved": [],
      "usage": []
    }
  }
}
```

The public `session_id` is returned for usability. The internal scoped session id
is included because it is useful when matching server logs.

### `GET /debug/chat-route/traces`

Query parameters:
- `session_id`: optional public session id filter.
- `trace_id`: optional exact trace id filter.
- `limit`: default 20, max 100.

Returns a newest-first list of trace summaries:

```json
{
  "traces": [
    {
      "trace_id": "abcd1234",
      "session_id": "debug-user",
      "mode": "auto",
      "question": "...",
      "answer_preview": "...",
      "created_at": 1781020000.0,
      "latency_ms_total": 1234.56,
      "route": "informational"
    }
  ]
}
```

### `GET /debug/chat-route/traces/{trace_id}`

Returns the full stored trace payload for one trace id after API-key validation.

## Persistence

Add a new SQLite table rather than overloading the existing `consultations`
table:

```sql
CREATE TABLE IF NOT EXISTS chat_trace (
    trace_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    internal_session_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    meta_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_trace_session_created
    ON chat_trace(session_id, created_at);
```

New storage helpers in `src/chat/storage/traces.py`:
- `save_chat_trace(...) -> dict`
- `list_chat_traces(session_id: str | None, trace_id: str | None, limit: int) -> list[dict]`
- `get_chat_trace(trace_id: str) -> dict | None`

The storage module follows the existing SQLite pattern: `RLock`, `get_sqlite()`,
plain dict return values, and JSON serialized with `ensure_ascii=False`.

## Pipeline Trace Metadata

Extend metadata capture in `src/chat/pipeline.py` so debug traces can render a
complete timeline.

Current `_log_timing(...)` logs stage duration. It should also append the stage
to `meta["timings"]` when `_meta()` is active:

```python
{
    "stage": "turn_analysis",
    "ms": 37.4,
    "fields": {
        "verdict": "allow",
        "label": "informational",
        "intent": "disease_info",
        "mode": "auto"
    }
}
```

Keep the existing `latency_ms` map for eval compatibility. Add the richer
`timings` list without removing or renaming current meta fields.

Also store compact route-oriented fields in meta where already known:
- `mode`
- `intent`
- `suggest_mode`
- `retry_question`
- `route_label`
- total `outcome`

The debug view can infer most of the route from `timings`, but explicit
`route_label` and `outcome` make the UI clearer and reduce fragile parsing.

## UI Design

The page is a single static console with two modes.

Live run panel:
- API key input.
- Public session id input.
- Mode selector (`auto`, `information`, `diagnostic` using existing mode names).
- Query textarea.
- Run button.

Replay panel:
- Search by public session id or trace id.
- List recent traces with timestamp, latency, route, question preview.
- Selecting a trace renders the same detail view as a live run.

Trace detail view:
- Header: trace id, session id, mode, created time, total latency, route/outcome.
- Timeline: stage cards in execution order with duration and fields.
- Retrieval section: source type, source slug/name, heading path, chunk id, score.
- Usage section: model, prompt/completion/total tokens when present.
- Response section: final answer text.
- Raw JSON disclosure for debugging edge cases.

The HTML should escape user-controlled strings before rendering. No external JS
or CSS dependencies are required.

## Error Handling

- Missing API key or wrong API key: return the same status semantics as `/chat`.
- Missing `session_id` on live run: return 400.
- Empty question: allow the pipeline to return its existing "Bạn hãy đặt câu hỏi
  cụ thể nhé." response and save that trace.
- Pipeline exception: preserve current `answer_with_meta` behavior returning the
  technical reply with `meta["error"] = "technical_error"`, then save the trace.
- Trace save failure: return the live answer plus a debug warning field; do not
  fail the chatbot turn after a successful pipeline response.
- Unknown `trace_id`: return 404.

## Testing

Storage tests:
- Saving a trace persists `meta_json` and returns a full round-trip payload.
- Listing by session id is newest-first and respects `limit`.
- Lookup by trace id returns one trace or `None`.

Pipeline tests:
- `answer_with_meta(...)` includes a `timings` list with core stages such as
  `load_session`, `preflight`, `turn_analysis`, `route`, and `total`.
- Existing `latency_ms`, `retrieved`, and `usage` fields remain compatible.

API tests:
- Debug run rejects missing/wrong API keys.
- Debug run scopes session id the same way `/chat` does.
- Debug run returns and saves a trace.
- Trace list and detail endpoints require API key.
- Trace detail returns 404 for an unknown trace id.

HTML tests:
- The served page includes the live run form, replay controls, timeline section,
  and JavaScript calls to the debug endpoints.

Verification commands:
- `python -m pytest tests/test_storage_and_quota.py tests/test_api.py tests/test_pipeline.py`
- `python -m compileall src eval`

## Implementation Boundaries

Keep edits surgical:
- `src/chat/pipeline.py`: metadata timeline capture only.
- `src/chat/storage/traces.py`: new trace storage helper.
- `src/chat/storage/session.py`: re-export trace helpers only if useful.
- `src/chat/clients.py`: add `chat_trace` schema.
- `src/server/app.py`: add debug HTML and JSON endpoints.
- Tests under existing test files unless a new focused trace test file is
  cleaner.

Do not change normal channel webhook behavior. Telegram, Zalo, Messenger, and
existing `/chat` clients should keep receiving the same response shape unless
they explicitly use debug endpoints or `include_meta=1`.
