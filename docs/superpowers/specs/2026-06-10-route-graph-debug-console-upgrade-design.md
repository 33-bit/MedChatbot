# Route Graph Debug Console Upgrade - Design

**Date:** 2026-06-10
**Status:** Approved (design), pending user spec review

## Goal

Upgrade `/debug/chat-route` from a timeline-oriented page into a workflow-style
route graph. A developer can click a pipeline node and inspect its inputs,
outputs, latency, errors, and domain-specific artifacts.

## Approved UI

Use layout A: a left-to-right graph on the left and a persistent inspector on
the right.

Graph nodes represent:
- input
- preflight
- turn analysis
- query rewrite
- route decision
- KG search
- dense retrieval
- sparse retrieval
- RRF fusion
- rerank
- generation
- persistence
- total outcome

Clicking a node shows status, duration, input, output, relevant tables, and raw
JSON. Missing stages are omitted or greyed out. The existing timing list remains
available as secondary detail.

## Trace Data

Keep the existing `timings`, `retrieved`, `usage`, `route_label`, `outcome`, and
latency fields. Add:

- `graph_nodes`: UI-facing node records with id, label, status, milliseconds,
  input, output, and raw data.
- `rewrite_query`: original question, rewritten query, and confidence.
- `kg_context`: matched entities and related diseases, drugs, symptoms, adverse
  reactions, and relationships.
- `retrieval_debug`: retrieval query plus dense, sparse, fused, and reranked
  hit lists.

Each retrieval hit includes:
- rank and stage
- chunk id
- source type, slug, and name
- heading path
- stage-local score
- short text preview
- metadata

Store all candidates produced by the existing configured retrieval calls. Do
not increase `HYBRID_CANDIDATE_K` or `RERANK_TOP_K` for debugging.

## Retrieval Architecture

Keep `hybrid_search(...)` compatible for existing callers. Add a trace-aware
variant that returns both final hits and debug details for dense search, sparse
search, fusion, and rerank. Reuse the current retrieval functions rather than
duplicating retrieval logic.

Preserve the structured `KGContext` long enough to serialize it into trace
metadata before formatting it into prompt text. Record analyzer rewrite output
when turn analysis completes.

## Compatibility And Security

- Keep the normal `/chat` response unchanged.
- Keep all debug data behind `CHAT_API_KEY`.
- Keep replay scoped by public session id and derived internal session id.
- Older traces without new fields must still render.
- Store concise text previews rather than full chunk bodies.
- Keep inline HTML/CSS/JavaScript; do not add a frontend build system.

## Error Handling

A failed trace retains completed nodes and marks the failing node as `error`.
The inspector renders missing optional data as unavailable rather than failing.

## Verification

Tests must cover:
- dense, sparse, fused, and reranked debug candidates with ranks and scores
- rewrite and KG metadata in `answer_with_meta(...)`
- graph node data in debug API responses
- rendering older traces without the new fields
- unchanged session-scoped replay protection

Verification commands:
- `python3 -m pytest tests/test_pipeline.py tests/test_api.py tests/test_storage_and_quota.py -v`
- `python3 -m compileall src eval`

## Resolved Decisions

- Layout: graph plus right inspector.
- Retrieval detail: all candidates produced by configured limits.
- No unresolved product decisions remain.
