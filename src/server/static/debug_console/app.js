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
    const svgIcon = (body) => `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;
    const NODE_ICONS = {
      _default: svgIcon('<circle cx="12" cy="12" r="3"></circle>'),
      input: svgIcon('<path d="M12 20h9"></path><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L8 18l-4 1 1-4z"></path>'),
      load_session: svgIcon('<circle cx="12" cy="12" r="8"></circle><path d="M12 8v4l3 2"></path>'),
      preflight: svgIcon('<path d="M12 3l7 3v5c0 4.4-3 7.5-7 9-4-1.5-7-4.6-7-9V6z"></path><path d="M9 12l2 2 4-4"></path>'),
      turn_analysis: svgIcon('<rect x="6" y="6" width="12" height="12" rx="2"></rect><path d="M9 1.5v2M15 1.5v2M9 20.5v2M15 20.5v2M1.5 9h2M1.5 15h2M20.5 9h2M20.5 15h2"></path><circle cx="12" cy="12" r="2"></circle>'),
      rewrite: svgIcon('<path d="M4 20h16"></path><path d="M14 4l6 6L9 21l-5 1 1-5z"></path>'),
      route: svgIcon('<path d="M12 3v18"></path><path d="M12 7H6L4 5l2-2h6"></path><path d="M12 13h6l2 2-2 2h-6"></path>'),
      entity_ingest: svgIcon('<path d="M20.6 13.4l-7.2 7.2a2 2 0 0 1-2.8 0l-7-7A2 2 0 0 1 3 12.2V5a2 2 0 0 1 2-2h7.2a2 2 0 0 1 1.4.6l7 7a2 2 0 0 1 0 2.8z"></path><circle cx="7.5" cy="7.5" r="1.2"></circle>'),
      kg_search: svgIcon('<circle cx="5" cy="6" r="2.4"></circle><circle cx="19" cy="7" r="2.4"></circle><circle cx="12" cy="18" r="2.4"></circle><path d="M7 7l3.6 9M16.8 8.6L13.4 16.6M7.2 6.4L16.4 6.8"></path>'),
      dense_search: svgIcon('<circle cx="10.5" cy="10.5" r="6"></circle><path d="M20 20l-5.2-5.2"></path>'),
      sparse_search: svgIcon('<circle cx="10.5" cy="10.5" r="6"></circle><path d="M20 20l-5.2-5.2"></path><path d="M8 10.5h5M10.5 8v5"></path>'),
      fusion: svgIcon('<path d="M4 4c5 0 5 8 8 8M4 20c5 0 5-8 8-8"></path><path d="M12 12h8"></path><path d="M17 9l3 3-3 3"></path>'),
      rerank: svgIcon('<path d="M5 18V7M5 7l-3 3M5 7l3 3"></path><path d="M11 6h9M11 11h6M11 16h3"></path>'),
      generate: svgIcon('<rect x="5" y="8" width="14" height="11" rx="2"></rect><path d="M12 4v4M9 1.5h6"></path><circle cx="9.5" cy="13" r="1.1"></circle><circle cx="14.5" cy="13" r="1.1"></circle>'),
      generation: svgIcon('<rect x="5" y="8" width="14" height="11" rx="2"></rect><path d="M12 4v4M9 1.5h6"></path><circle cx="9.5" cy="13" r="1.1"></circle><circle cx="14.5" cy="13" r="1.1"></circle>'),
      persist: svgIcon('<ellipse cx="12" cy="6" rx="7" ry="3"></ellipse><path d="M5 6v6c0 1.7 3.1 3 7 3s7-1.3 7-3V6"></path><path d="M5 12v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6"></path>'),
      total: svgIcon('<path d="M5 21V4"></path><path d="M5 4h11l-2 3 2 3H5"></path>'),
    };

    let activeGraph = { nodes: new Map(), nodeElements: new Map(), meta: {}, trace: null, edges: [] };

    let viewState = { x: 16, y: 16, scale: 0.75 };
    const clamp = (value, lo, hi) => Math.min(hi, Math.max(lo, value));

    function applyViewTransform() {
      const root = el("workflow-graph");
      if (!root) return;
      root.style.transform = `translate(${viewState.x}px, ${viewState.y}px) scale(${viewState.scale})`;
    }

    function resetView() {
      viewState = { x: 16, y: 16, scale: 0.75 };
      applyViewTransform();
    }

    function redrawEdges() {
      const root = el("workflow-graph");
      if (!root) return;
      const existingEdges = root.querySelector(".graph-edges");
      if (existingEdges) existingEdges.remove();
      const existingGroup = root.querySelector(".graph-group");
      if (existingGroup) existingGroup.remove();
      drawGraphEdges(root, activeGraph.edges || []);
      drawRetrievalGroup(root);
    }

    const STREAM_TO_NODE = {
      dense: ["dense_search"],
      sparse: ["sparse_search"],
      generate: ["generate", "generation"],
    };
    const SKELETON_NODE_IDS = [
      "input", "load_session", "preflight", "turn_analysis", "rewrite", "route",
      "entity_ingest", "kg_search", "dense_search", "sparse_search", "fusion",
      "rerank", "generate", "persist", "total",
    ];

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
      arrow.setAttribute("fill", "#5b6b85");
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

    const RETRIEVAL_GROUP_IDS = ["dense_search", "sparse_search", "fusion", "rerank"];

    function drawRetrievalGroup(root) {
      // Bounds the retrieval cluster; derived from GRAPH_LAYOUT so it tracks layout changes.
      const PAD = 24;
      const NODE_W = 136;
      const NODE_H = 80;
      const coords = RETRIEVAL_GROUP_IDS.map((id) => GRAPH_LAYOUT[id]);
      if (coords.some((c) => !c)) return;
      const xs = coords.map((c) => c[0]);
      const ys = coords.map((c) => c[1]);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const left = minX - PAD;
      const top = minY - PAD;
      const width = maxX + NODE_W + PAD - left;
      const height = maxY + NODE_H + PAD - top;
      const box = document.createElement("div");
      box.className = "graph-group";
      box.setAttribute("aria-hidden", "true");
      box.style.left = `${left}px`;
      box.style.top = `${top}px`;
      box.style.width = `${width}px`;
      box.style.height = `${height}px`;
      const tag = document.createElement("span");
      tag.className = "graph-group-label";
      tag.textContent = "retrieval";
      box.appendChild(tag);
      root.prepend(box);
    }

    function renderWorkflowGraph(trace) {
      const root = el("workflow-graph");
      root.textContent = "";
      const meta = isObject(trace && trace.meta) ? trace.meta : {};
      const suppliedNodes = asArray(meta.graph_nodes).filter(isObject);
      const nodes = suppliedNodes.length ? suppliedNodes : buildLegacyGraphNodes(meta, trace);
      activeGraph = { nodes: new Map(), nodeElements: new Map(), meta, trace, edges: [] };
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
        const button = buildGraphNodeButton(normalized, index);
        activeGraph.nodes.set(normalized.id, normalized);
        activeGraph.nodeElements.set(normalized.id, button);
        root.appendChild(button);
      });
      const edges = suppliedNodes.length
        ? WORKFLOW_EDGES
        : nodes.slice(1).map((node, index) => [String(nodes[index].id), String(node.id)]);
      activeGraph.edges = edges;
      drawGraphEdges(root, edges);
      drawRetrievalGroup(root);
      if (nodes.length) selectGraphNode(String(nodes[0].id));
      resetView();
    }

    function buildGraphNodeButton(node, index) {
      const button = document.createElement("button");
      const position = graphPosition(node, index);
      button.type = "button";
      button.className = `graph-node ${node.status}`;
      button.style.left = `${position[0]}px`;
      button.style.top = `${position[1]}px`;
      button.dataset.nodeId = node.id;
      const head = document.createElement("span");
      head.className = "node-head";
      const icon = document.createElement("span");
      icon.className = "node-icon";
      icon.innerHTML = NODE_ICONS[node.id] || NODE_ICONS._default;
      const label = document.createElement("span");
      label.className = "node-label";
      label.textContent = node.label;
      head.append(icon, label);
      const detail = document.createElement("span");
      detail.className = "node-detail";
      detail.textContent = `${node.status} | ${durationText(node.ms)}`;
      button.append(head, detail);
      let dragging = false;
      let moved = false;
      let dragStartX = 0;
      let dragStartY = 0;
      let origLeft = 0;
      let origTop = 0;
      button.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        event.stopPropagation();
        dragging = true;
        moved = false;
        dragStartX = event.clientX;
        dragStartY = event.clientY;
        origLeft = parseFloat(button.style.left) || 0;
        origTop = parseFloat(button.style.top) || 0;
        try { button.setPointerCapture(event.pointerId); } catch (e) {}
      });
      button.addEventListener("pointermove", (event) => {
        if (!dragging) return;
        const dxScreen = event.clientX - dragStartX;
        const dyScreen = event.clientY - dragStartY;
        if (!moved && Math.hypot(dxScreen, dyScreen) < 4) return;
        moved = true;
        const nextLeft = origLeft + dxScreen / viewState.scale;
        const nextTop = origTop + dyScreen / viewState.scale;
        button.style.left = `${nextLeft}px`;
        button.style.top = `${nextTop}px`;
        GRAPH_LAYOUT[node.id] = [nextLeft, nextTop];
        redrawEdges();
      });
      const endNodeDrag = (event) => {
        if (!dragging) return;
        dragging = false;
        try { button.releasePointerCapture(event.pointerId); } catch (e) {}
        if (!moved) selectGraphNode(node.id);
      };
      button.addEventListener("pointerup", endNodeDrag);
      button.addEventListener("pointercancel", endNodeDrag);
      return button;
    }

    function renderSkeletonGraph() {
      const root = el("workflow-graph");
      root.textContent = "";
      activeGraph = { nodes: new Map(), nodeElements: new Map(), meta: {}, trace: null, edges: [] };
      SKELETON_NODE_IDS.forEach((id, index) => {
        const node = {
          id,
          label: id.replaceAll("_", " "),
          status: "skipped",
          ms: null,
          input: null,
          output: null,
          raw: null,
        };
        const button = buildGraphNodeButton(node, index);
        activeGraph.nodes.set(id, node);
        activeGraph.nodeElements.set(id, button);
        root.appendChild(button);
      });
      activeGraph.edges = WORKFLOW_EDGES;
      drawGraphEdges(root, WORKFLOW_EDGES);
      drawRetrievalGroup(root);
      if (SKELETON_NODE_IDS.length) selectGraphNode(SKELETON_NODE_IDS[0]);
      resetView();
    }

    function markNodeStatus(streamId, status, ms) {
      const targets = STREAM_TO_NODE[streamId] || [streamId];
      const displayStatus = status === "error" ? "error" : "success";
      const normalizedMs = typeof ms === "number" ? ms : null;
      for (const id of targets) {
        const button = activeGraph.nodeElements.get(id);
        if (!button) continue;
        button.className = `graph-node ${displayStatus}${button.classList.contains("selected") ? " selected" : ""}`;
        const detail = button.querySelector(".node-detail");
        if (detail) detail.textContent = `${displayStatus} | ${durationText(normalizedMs)}`;
        const node = activeGraph.nodes.get(id);
        if (node) {
          node.status = displayStatus;
          node.ms = normalizedMs;
        }
      }
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

    async function runOnce() {
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
    }

    async function runStream() {
      el("run-status").textContent = "Running...";
      renderSkeletonGraph();
      let response;
      try {
        response = await fetch("/debug/chat-route/stream", {
          method: "POST",
          headers: apiHeaders(),
          body: JSON.stringify({
            question: el("question").value,
            session_id: el("session-id").value,
            mode: el("mode").value,
          }),
        });
      } catch (err) {
        return runOnce();
      }
      if (!response.ok || !response.body || !window.ReadableStream) {
        if (response.status === 401 || response.status === 503) {
          let detail = "Run failed";
          try { detail = (await response.json()).detail || detail; } catch (e) {}
          el("run-status").textContent = detail;
          return;
        }
        return runOnce();
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let doneTrace = null;
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let sep;
          while ((sep = buffer.indexOf("\n\n")) !== -1) {
            const frame = buffer.slice(0, sep);
            buffer = buffer.slice(sep + 2);
            const line = frame.split("\n").find((l) => l.startsWith("data:"));
            if (!line) continue;
            let evt;
            try { evt = JSON.parse(line.slice(5).trim()); } catch (e) { continue; }
            if (evt.type === "node") {
              markNodeStatus(evt.id, evt.status, evt.ms);
            } else if (evt.type === "done") {
              doneTrace = evt.trace;
            }
          }
        }
      } catch (err) {
        el("run-status").textContent = "Stream interrupted";
        return;
      }
      if (doneTrace) {
        el("run-status").textContent = "Run complete";
        renderTrace(doneTrace);
      } else {
        el("run-status").textContent = "Run complete (no trace)";
      }
    }

    el("run-button").addEventListener("click", () => { runStream(); });

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


    function setupCanvasInteraction() {
      const panel = document.querySelector(".graph-panel");
      if (!panel) return;
      const resetButton = document.createElement("button");
      resetButton.type = "button";
      resetButton.className = "graph-reset";
      resetButton.textContent = "Reset view";
      resetButton.addEventListener("pointerdown", (event) => event.stopPropagation());
      resetButton.addEventListener("click", () => resetView());
      panel.appendChild(resetButton);

      let panning = false;
      let startX = 0;
      let startY = 0;
      let startPanX = 0;
      let startPanY = 0;
      panel.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        if (event.target.closest(".graph-node, .graph-reset")) return;
        panning = true;
        startX = event.clientX;
        startY = event.clientY;
        startPanX = viewState.x;
        startPanY = viewState.y;
        panel.classList.add("grabbing");
        try { panel.setPointerCapture(event.pointerId); } catch (e) {}
      });
      panel.addEventListener("pointermove", (event) => {
        if (!panning) return;
        viewState.x = startPanX + (event.clientX - startX);
        viewState.y = startPanY + (event.clientY - startY);
        applyViewTransform();
      });
      const endPan = (event) => {
        if (!panning) return;
        panning = false;
        panel.classList.remove("grabbing");
        try { panel.releasePointerCapture(event.pointerId); } catch (e) {}
      };
      panel.addEventListener("pointerup", endPan);
      panel.addEventListener("pointercancel", endPan);
      panel.addEventListener("wheel", (event) => {
        event.preventDefault();
        const rect = panel.getBoundingClientRect();
        const cx = event.clientX - rect.left;
        const cy = event.clientY - rect.top;
        const factor = event.deltaY < 0 ? 1.1 : 0.9;
        const newScale = clamp(viewState.scale * factor, 0.3, 2.0);
        const k = newScale / viewState.scale;
        viewState.x = cx - k * (cx - viewState.x);
        viewState.y = cy - k * (cy - viewState.y);
        viewState.scale = newScale;
        applyViewTransform();
      }, { passive: false });
    }

    setupCanvasInteraction();
    applyViewTransform();
