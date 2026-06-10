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
    main { max-width: 1600px; margin: 0 auto; padding: 24px; }
    h1, h2 { margin: 0 0 12px; }
    h3, h4 { margin: 16px 0 10px; }
    section { background: #1f2937; border: 1px solid #374151; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
    label { display: block; font-weight: 600; margin-bottom: 6px; }
    input, textarea, select, button { width: 100%; box-sizing: border-box; margin-bottom: 12px; padding: 10px; border-radius: 8px; border: 1px solid #4b5563; background: #111827; color: inherit; }
    button { cursor: pointer; background: #2563eb; border: 0; font-weight: 700; }
    button.secondary { background: #374151; }
    .grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
    .timeline-item, .trace-item { border: 1px solid #4b5563; border-radius: 8px; padding: 12px; margin-bottom: 10px; background: #111827; }
    .meta { color: #9ca3af; font-size: 14px; }
    pre { white-space: pre-wrap; word-break: break-word; background: #0b1220; padding: 12px; border-radius: 8px; overflow: auto; }
    .trace-area { display: grid; grid-template-columns: minmax(0, 2fr) minmax(320px, 1fr); gap: 16px; align-items: stretch; }
    .graph-panel, .inspector-panel { min-width: 0; border: 1px solid #374151; border-radius: 10px; background: #111827; }
    .graph-panel { overflow: auto; min-height: 580px; }
    #workflow-graph { position: relative; width: 1960px; height: 560px; }
    .graph-edges { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
    .graph-edge { fill: none; stroke: #64748b; stroke-width: 2; }
    .graph-node { position: absolute; width: 136px; min-height: 76px; margin: 0; padding: 10px; text-align: left; border: 2px solid #475569; background: #172033; box-shadow: 0 8px 18px rgba(0, 0, 0, .2); }
    .graph-node:hover, .graph-node.selected { border-color: #60a5fa; outline: 2px solid rgba(96, 165, 250, .2); }
    .graph-node.success { border-color: #16a34a; }
    .graph-node.error { border-color: #dc2626; }
    .graph-node.skipped { border-color: #64748b; opacity: .72; }
    .node-label, .node-detail { display: block; }
    .node-label { font-size: 13px; }
    .node-detail { margin-top: 8px; color: #cbd5e1; font-size: 11px; font-weight: 500; }
    .inspector-panel { padding: 16px; overflow: auto; max-height: 580px; }
    .inspector-header { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .status-pill { display: inline-block; border-radius: 999px; padding: 3px 8px; background: #334155; font-size: 12px; font-weight: 700; }
    .status-pill.success { background: #14532d; color: #bbf7d0; }
    .status-pill.error { background: #7f1d1d; color: #fecaca; }
    .status-pill.skipped { background: #334155; color: #cbd5e1; }
    .detail-card { margin-top: 12px; padding: 12px; border: 1px solid #334155; border-radius: 8px; background: #0b1220; }
    .detail-card h4 { margin-top: 0; }
    .detail-row { display: grid; grid-template-columns: 120px minmax(0, 1fr); gap: 10px; padding: 5px 0; border-bottom: 1px solid #1e293b; }
    .detail-row:last-child { border-bottom: 0; }
    .detail-label { color: #94a3b8; font-size: 12px; }
    .detail-value { min-width: 0; white-space: pre-wrap; word-break: break-word; }
    .empty-state { color: #94a3b8; font-style: italic; }
    .table-wrap { overflow: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { padding: 7px; text-align: left; vertical-align: top; border-bottom: 1px solid #334155; }
    th { color: #93c5fd; }
    td.preview { min-width: 220px; white-space: normal; }
    @media (max-width: 980px) {
      main { padding: 14px; }
      .trace-area { grid-template-columns: 1fr; }
      .inspector-panel { max-height: none; }
    }
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
      <h3>Workflow</h3>
      <div class="trace-area">
        <div class="graph-panel" aria-label="Workflow graph canvas">
          <div id="workflow-graph"></div>
        </div>
        <aside id="node-inspector" class="inspector-panel" aria-live="polite">
          <h3>Node Inspector</h3>
          <div class="empty-state">Select a workflow node to inspect it.</div>
        </aside>
      </div>
      <h3>Timing List</h3>
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
    const GRAPH_LAYOUT = {
      input: [20, 240],
      load_session: [180, 240],
      preflight: [340, 240],
      turn_analysis: [500, 240],
      rewrite: [660, 240],
      route: [820, 240],
      entity_ingest: [980, 40],
      kg_search: [1140, 40],
      dense_search: [980, 220],
      sparse_search: [980, 400],
      fusion: [1140, 310],
      rerank: [1300, 310],
      generate: [1460, 220],
      generation: [1460, 220],
      persist: [1620, 220],
      total: [1780, 220],
    };
    const WORKFLOW_EDGES = [
      ["input", "load_session"],
      ["load_session", "preflight"],
      ["preflight", "turn_analysis"],
      ["turn_analysis", "rewrite"],
      ["rewrite", "route"],
      ["route", "entity_ingest"],
      ["entity_ingest", "kg_search"],
      ["route", "dense_search"],
      ["route", "sparse_search"],
      ["dense_search", "fusion"],
      ["sparse_search", "fusion"],
      ["fusion", "rerank"],
      ["kg_search", "generate"],
      ["kg_search", "generation"],
      ["rerank", "generate"],
      ["rerank", "generation"],
      ["generate", "persist"],
      ["generation", "persist"],
      ["persist", "total"],
    ];
    let activeGraph = { nodes: new Map(), nodeElements: new Map(), meta: {}, trace: null };

    const isObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
    const asArray = (value) => Array.isArray(value) ? value : [];
    const isAvailable = (value) => value !== null && value !== undefined && value !== "";
    const durationText = (value) => typeof value === "number" ? `${value}ms` : "n/a";

    function appendEmpty(root, message = "Unavailable") {
      const empty = document.createElement("div");
      empty.className = "empty-state";
      empty.textContent = message;
      root.appendChild(empty);
    }

    function appendDetailRow(root, label, value) {
      const row = document.createElement("div");
      row.className = "detail-row";
      const name = document.createElement("div");
      name.className = "detail-label";
      name.textContent = label;
      const detail = document.createElement("div");
      detail.className = "detail-value";
      detail.textContent = isAvailable(value) ? String(value) : "Unavailable";
      row.append(name, detail);
      root.appendChild(row);
    }

    function appendJsonCard(root, title, value) {
      const card = document.createElement("div");
      card.className = "detail-card";
      const heading = document.createElement("h4");
      heading.textContent = title;
      card.appendChild(heading);
      if (!isAvailable(value)) {
        appendEmpty(card);
      } else {
        const block = document.createElement("pre");
        block.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
        card.appendChild(block);
      }
      root.appendChild(card);
    }

    function buildLegacyGraphNodes(meta, trace) {
      const timings = asArray(meta && meta.timings);
      const nodes = timings.filter(isObject).map((timing, index) => {
        const stage = String(timing.stage || `stage_${index + 1}`);
        const fields = isObject(timing.fields) ? timing.fields : {};
        return {
          id: stage,
          label: stage.replaceAll("_", " "),
          status: fields.failed ? "error" : "success",
          ms: typeof timing.ms === "number" ? timing.ms : null,
          input: null,
          output: fields,
          raw: timing,
        };
      });
      if (!nodes.length) {
        nodes.push({
          id: "total",
          label: "Total",
          status: meta && meta.error ? "error" : "success",
          ms: meta && typeof meta.latency_ms_total === "number" ? meta.latency_ms_total : null,
          input: trace ? { question: trace.question, mode: trace.mode } : null,
          output: meta ? { outcome: meta.outcome, route_label: meta.route_label } : null,
          raw: meta || {},
        });
      }
      return nodes;
    }

    function graphPosition(node, index) {
      if (GRAPH_LAYOUT[node.id]) return GRAPH_LAYOUT[node.id];
      return [20 + (index % 11) * 160, 470];
    }

    function svgElement(name) {
      return document.createElementNS("http://www.w3.org/2000/svg", name);
    }

    function drawGraphEdges(root, edges) {
      const svg = svgElement("svg");
      svg.classList.add("graph-edges");
      svg.setAttribute("aria-hidden", "true");
      const defs = svgElement("defs");
      const marker = svgElement("marker");
      marker.setAttribute("id", "graph-arrow");
      marker.setAttribute("markerWidth", "8");
      marker.setAttribute("markerHeight", "8");
      marker.setAttribute("refX", "7");
      marker.setAttribute("refY", "4");
      marker.setAttribute("orient", "auto");
      const arrow = svgElement("path");
      arrow.setAttribute("d", "M 0 0 L 8 4 L 0 8 z");
      arrow.setAttribute("fill", "#64748b");
      marker.appendChild(arrow);
      defs.appendChild(marker);
      svg.appendChild(defs);
      for (const [fromId, toId] of edges) {
        const from = activeGraph.nodeElements.get(fromId);
        const to = activeGraph.nodeElements.get(toId);
        if (!from || !to) continue;
        const x1 = from.offsetLeft + from.offsetWidth;
        const y1 = from.offsetTop + from.offsetHeight / 2;
        const x2 = to.offsetLeft;
        const y2 = to.offsetTop + to.offsetHeight / 2;
        const midpoint = x1 + Math.max(24, (x2 - x1) / 2);
        const path = svgElement("path");
        path.classList.add("graph-edge");
        path.setAttribute("d", `M ${x1} ${y1} C ${midpoint} ${y1}, ${midpoint} ${y2}, ${x2} ${y2}`);
        path.setAttribute("marker-end", "url(#graph-arrow)");
        svg.appendChild(path);
      }
      root.prepend(svg);
    }

    function renderWorkflowGraph(trace) {
      const root = el("workflow-graph");
      root.textContent = "";
      const meta = isObject(trace && trace.meta) ? trace.meta : {};
      const suppliedNodes = asArray(meta.graph_nodes).filter(isObject);
      const nodes = suppliedNodes.length ? suppliedNodes : buildLegacyGraphNodes(meta, trace);
      activeGraph = { nodes: new Map(), nodeElements: new Map(), meta, trace };
      nodes.forEach((node, index) => {
        const normalized = {
          id: String(node.id || `node_${index + 1}`),
          label: String(node.label || node.id || `Node ${index + 1}`),
          status: ["success", "error", "skipped"].includes(node.status) ? node.status : "skipped",
          ms: typeof node.ms === "number" ? node.ms : null,
          input: node.input,
          output: node.output,
          raw: node.raw,
        };
        const button = document.createElement("button");
        const position = graphPosition(normalized, index);
        button.type = "button";
        button.className = `graph-node ${normalized.status}`;
        button.style.left = `${position[0]}px`;
        button.style.top = `${position[1]}px`;
        button.dataset.nodeId = normalized.id;
        const label = document.createElement("span");
        label.className = "node-label";
        label.textContent = normalized.label;
        const detail = document.createElement("span");
        detail.className = "node-detail";
        detail.textContent = `${normalized.status} | ${durationText(normalized.ms)}`;
        button.append(label, detail);
        button.addEventListener("click", () => selectGraphNode(normalized.id));
        activeGraph.nodes.set(normalized.id, normalized);
        activeGraph.nodeElements.set(normalized.id, button);
        root.appendChild(button);
      });
      const edges = suppliedNodes.length
        ? WORKFLOW_EDGES
        : nodes.slice(1).map((node, index) => [String(nodes[index].id), String(node.id)]);
      drawGraphEdges(root, edges);
      if (nodes.length) selectGraphNode(String(nodes[0].id));
    }

    function renderRewriteDetails(root, node, meta) {
      const rewrite = isObject(node.output) ? node.output : (isObject(meta.rewrite_query) ? meta.rewrite_query : null);
      const card = document.createElement("div");
      card.className = "detail-card";
      const heading = document.createElement("h4");
      heading.textContent = "Rewrite Detail";
      card.appendChild(heading);
      if (!rewrite) {
        appendEmpty(card);
      } else {
        appendDetailRow(card, "Original", rewrite.original);
        appendDetailRow(card, "Rewritten", rewrite.rewritten);
        appendDetailRow(card, "Confidence", rewrite.confidence ?? rewrite.confident);
      }
      root.appendChild(card);
    }

    function renderKgDetails(root, node, meta) {
      const kg = isObject(node.output) ? node.output : (isObject(meta.kg_context) ? meta.kg_context : null);
      const card = document.createElement("div");
      card.className = "detail-card";
      const heading = document.createElement("h4");
      heading.textContent = "Knowledge Graph Detail";
      card.appendChild(heading);
      if (!kg) {
        appendEmpty(card);
      } else {
        const groups = [
          ["Matched entities", kg.matched_entities],
          ["Related diseases", kg.related_diseases],
          ["Related drugs", kg.related_drugs],
          ["Related symptoms", kg.related_symptoms],
          ["Adverse reactions", kg.related_adrs ?? kg.adverse_reactions],
          ["Relationships", kg.relationships],
        ];
        for (const [label, values] of groups) {
          const formatted = asArray(values).map((value) => typeof value === "string" ? value : JSON.stringify(value)).join("\n");
          appendDetailRow(card, label, formatted || "Empty");
        }
      }
      root.appendChild(card);
    }

    function renderRetrievalTable(root, candidates) {
      const card = document.createElement("div");
      card.className = "detail-card";
      const heading = document.createElement("h4");
      heading.textContent = "Retrieval Candidates";
      card.appendChild(heading);
      const rows = asArray(candidates).filter(isObject);
      if (!rows.length) {
        appendEmpty(card, "No candidates available.");
        root.appendChild(card);
        return;
      }
      const wrap = document.createElement("div");
      wrap.className = "table-wrap";
      const table = document.createElement("table");
      const head = document.createElement("thead");
      const headRow = document.createElement("tr");
      for (const label of ["Rank", "Chunk ID", "Source", "Heading", "Score", "Preview"]) {
        const th = document.createElement("th");
        th.textContent = label;
        headRow.appendChild(th);
      }
      head.appendChild(headRow);
      table.appendChild(head);
      const body = document.createElement("tbody");
      for (const candidate of rows) {
        const sourceParts = [candidate.source_type, candidate.source_name, candidate.source_slug].filter(isAvailable);
        const values = [
          candidate.rank,
          candidate.chunk_id,
          sourceParts.join(" / "),
          candidate.heading_path ?? candidate.heading,
          candidate.score,
          candidate.preview ?? candidate.text_preview ?? candidate.text,
        ];
        const row = document.createElement("tr");
        values.forEach((value, index) => {
          const cell = document.createElement("td");
          if (index === 5) cell.className = "preview";
          cell.textContent = isAvailable(value) ? String(value) : "Unavailable";
          row.appendChild(cell);
        });
        body.appendChild(row);
      }
      table.appendChild(body);
      wrap.appendChild(table);
      card.appendChild(wrap);
      root.appendChild(card);
    }

    function selectGraphNode(nodeId) {
      const node = activeGraph.nodes.get(nodeId);
      if (!node) return;
      for (const [id, button] of activeGraph.nodeElements) {
        button.classList.toggle("selected", id === nodeId);
      }
      const root = el("node-inspector");
      root.textContent = "";
      const title = document.createElement("h3");
      title.textContent = node.label;
      const header = document.createElement("div");
      header.className = "inspector-header";
      const status = document.createElement("span");
      status.className = `status-pill ${node.status}`;
      status.textContent = node.status;
      const duration = document.createElement("span");
      duration.className = "meta";
      duration.textContent = `Duration: ${durationText(node.ms)}`;
      header.append(status, duration);
      root.append(title, header);
      if (node.id === "rewrite") renderRewriteDetails(root, node, activeGraph.meta);
      if (node.id === "kg_search") renderKgDetails(root, node, activeGraph.meta);
      const retrievalKeys = {
        dense_search: "dense_hits",
        sparse_search: "sparse_hits",
        fusion: "fused_hits",
        rerank: "reranked_hits",
      };
      if (retrievalKeys[node.id]) {
        const debug = isObject(activeGraph.meta.retrieval_debug) ? activeGraph.meta.retrieval_debug : {};
        const candidates = Array.isArray(node.output) ? node.output : debug[retrievalKeys[node.id]];
        renderRetrievalTable(root, candidates);
      }
      appendJsonCard(root, "Input", node.input);
      appendJsonCard(root, "Output", node.output);
      appendJsonCard(root, "Raw JSON", node.raw);
    }

    function renderList(traces) {
      const root = el("trace-list");
      root.textContent = "";
      for (const trace of asArray(traces).filter(isObject)) {
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
      for (const value of asArray(values)) {
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
      const safeTrace = isObject(trace) ? trace : {};
      const meta = isObject(safeTrace.meta) ? safeTrace.meta : {};
      el("trace-summary").textContent =
        `trace=${safeTrace.trace_id ?? "n/a"} session=${safeTrace.session_id ?? "n/a"} internal=${safeTrace.internal_session_id ?? "n/a"} created_at=${safeTrace.created_at ?? "n/a"} mode=${safeTrace.mode ?? "n/a"} total=${meta.latency_ms_total ?? "n/a"} route=${meta.route_label || meta.outcome || "unknown"}${warning ? " warning=" + warning : ""}`;
      renderWorkflowGraph(safeTrace);
      renderCollection("timeline", meta.timings, (item) => {
        const timing = isObject(item) ? item : {};
        return `${timing.stage ?? "unknown"}: ${timing.ms ?? "n/a"}ms ${JSON.stringify(timing.fields || {})}`;
      });
      renderCollection("retrieved", meta.retrieved, (item) => JSON.stringify(item));
      renderCollection("usage", meta.usage, (item) => JSON.stringify(item));
      el("answer").textContent = safeTrace.answer || "";
      el("raw-json").textContent = JSON.stringify({ trace: safeTrace, warning }, null, 2);
    }

    async function loadTrace(traceId) {
      const params = new URLSearchParams();
      params.set("session_id", el("filter-session-id").value);
      const response = await fetch(`/debug/chat-route/traces/${encodeURIComponent(traceId)}?${params.toString()}`, { headers: apiHeaders() });
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


def _require_query_session_id(session_id: str | None) -> str:
    if not isinstance(session_id, str) or not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")
    return session_id.strip()


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
            created_at=trace["created_at"],
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
