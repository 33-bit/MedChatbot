from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_RESULT = Path("eval/results/local-rag-health_insurance_info-results-20260623-rerun.jsonl")
DEFAULT_OUTPUT = Path("eval/artifacts/health_insurance_info.html")
DEFAULT_DATASET = Path("eval/datasets/medical_qa_benchmark_v2.jsonl")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_summary(path: Path) -> dict[str, Any] | None:
    summary_path = path.with_suffix(".summary.json")
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_expected_answers(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    expected: dict[str, str] = {}
    for row in load_jsonl(path):
        case_id = row.get("id")
        answer = row.get("reference_answer")
        if case_id and answer:
            expected[str(case_id)] = str(answer)
    return expected


def with_expected_answers(rows: list[dict[str, Any]], dataset_path: Path) -> list[dict[str, Any]]:
    expected_answers = load_expected_answers(dataset_path)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        case_id = item.get("case_id") or item.get("id")
        expected_answer = (
            item.get("expected_answer")
            or item.get("reference_answer")
            or expected_answers.get(str(case_id), "")
        )
        item["expected_answer"] = expected_answer
        enriched.append(item)
    return enriched


def write_html(result_path: Path, output_path: Path) -> None:
    rows = with_expected_answers(load_jsonl(result_path), DEFAULT_DATASET)
    summary = load_summary(result_path)
    payload = {
        "source": str(result_path),
        "dataset": str(DEFAULT_DATASET),
        "rows": rows,
        "summary": summary,
    }
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Evaluation Result Viewer</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --line: #d9dee7;
      --text: #162033;
      --muted: #647084;
      --good: #0f766e;
      --bad: #b42318;
      --warn: #b25e09;
      --chip: #eef2f7;
      --focus: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: rgba(247, 248, 250, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 16px 20px 14px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 20px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    .source {{
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    main {{
      padding: 18px 20px 32px;
      max-width: none;
      margin: 0 auto;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      min-height: 70px;
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
    }}
    .metric .value {{
      font-size: 19px;
      font-weight: 700;
    }}
    .filters {{
      display: grid;
      grid-template-columns: minmax(260px, 1fr) 140px 140px 140px 150px;
      gap: 10px;
      margin: 12px 0 16px;
    }}
    input, select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 9px 10px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }}
    input:focus, select:focus {{
      border-color: var(--focus);
      outline: 2px solid rgba(37, 99, 235, 0.14);
    }}
    .table-wrap {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      min-width: 2100px;
      border-collapse: collapse;
      table-layout: fixed;
    }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{
      background: #eef2f7;
      color: #2a3445;
      text-align: left;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    tbody tr:hover {{ background: #fbfcfe; }}
    .id {{ width: 210px; overflow-wrap: anywhere; }}
    .score {{ width: 86px; }}
    .status {{ width: 88px; }}
    .latency {{ width: 120px; }}
    .retrieval {{ width: 165px; }}
    .question {{ width: 380px; }}
    .expected {{ width: 430px; }}
    .answer {{ width: 620px; }}
    .badge {{
      display: inline-block;
      padding: 2px 7px;
      border-radius: 999px;
      background: var(--chip);
      font-size: 12px;
      font-weight: 650;
      white-space: nowrap;
    }}
    .pass {{ color: var(--good); background: #dff7f2; }}
    .fail {{ color: var(--bad); background: #fee4e2; }}
    .warn {{ color: var(--warn); background: #fff0d6; }}
    .muted {{ color: var(--muted); }}
    .answer-text, .expected-text, .question-text {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      max-height: 165px;
      overflow: auto;
    }}
    details {{
      margin-top: 8px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fbfcfe;
    }}
    summary {{
      cursor: pointer;
      padding: 7px 9px;
      color: #334155;
      font-weight: 650;
    }}
    pre {{
      margin: 0;
      padding: 0 9px 9px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: #27364a;
    }}
    .kv {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 6px;
      padding: 0 9px 9px;
    }}
    .kv-item {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 6px 8px;
      min-width: 0;
    }}
    .kv-key {{
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 2px;
      overflow-wrap: anywhere;
    }}
    .kv-value {{
      font: 12px/1.35 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      overflow-wrap: anywhere;
    }}
    .mini {{
      display: grid;
      gap: 3px;
      margin-top: 5px;
      color: var(--muted);
      font-size: 12px;
    }}
    .source-list {{
      max-height: 220px;
      overflow: auto;
      padding: 0 9px 9px;
    }}
    .source-list div {{
      padding: 4px 0;
      border-bottom: 1px solid #edf0f5;
      overflow-wrap: anywhere;
      font: 12px/1.35 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    .empty {{
      padding: 22px;
      text-align: center;
      color: var(--muted);
    }}
    @media (max-width: 1100px) {{
      .summary {{ grid-template-columns: repeat(2, minmax(130px, 1fr)); }}
      .filters {{ grid-template-columns: 1fr 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Evaluation Result Viewer</h1>
    <div class="source" id="source"></div>
  </header>
  <main>
    <section class="summary" id="summary"></section>
    <section class="filters">
      <input id="search" type="search" placeholder="Search case, question, expected answer, answer">
      <select id="statusFilter">
        <option value="all">All statuses</option>
        <option value="pass">Pass</option>
        <option value="fail">Fail</option>
      </select>
      <select id="routingFilter">
        <option value="all">All routing</option>
        <option value="normal">Normal RAG</option>
        <option value="triage">Triage-like</option>
        <option value="rejected">Rejected</option>
        <option value="no-retrieval">No retrieval</option>
      </select>
      <select id="scoreFilter">
        <option value="all">All scores</option>
        <option value="lt50">&lt; 0.50</option>
        <option value="lt70">&lt; 0.70</option>
        <option value="gte80">&ge; 0.80</option>
      </select>
      <select id="sortBy">
        <option value="score-asc">Worst first</option>
        <option value="score-desc">Best first</option>
        <option value="latency-desc">Slowest first</option>
        <option value="id-asc">Case ID</option>
      </select>
    </section>
    <section class="table-wrap">
      <table>
        <thead>
          <tr>
            <th class="id">Case</th>
            <th class="status">Status</th>
            <th class="score">Score</th>
            <th class="latency">Latency</th>
            <th class="retrieval">Retrieval</th>
            <th class="question">Question</th>
            <th class="expected">Expected Answer</th>
            <th class="answer">Answer</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
      <div id="empty" class="empty" hidden>No rows match the current filters.</div>
    </section>
  </main>
  <script id="payload" type="application/json">{payload_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById('payload').textContent);
    const rows = payload.rows || [];

    function escapeHtml(value) {{
      return String(value ?? '').replace(/[&<>"']/g, ch => ({{
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }}[ch]));
    }}

    function pct(value) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
      return `${{(Number(value) * 100).toFixed(1)}}%`;
    }}

    function fixed(value, digits = 2) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
      return Number(value).toFixed(digits);
    }}

    function isTriageLike(row) {{
      const answer = row.answer || '';
      return answer.includes('Tôi hiểu bạn đang bị') ||
        answer.includes('Triệu chứng này có nhiều nguyên nhân') ||
        answer.includes('trước hết tôi cần vài thông tin chung');
    }}

    function isRejected(row) {{
      return (row.answer || '').includes('Tôi chỉ hỗ trợ các câu hỏi');
    }}

    function routingLabel(row) {{
      if (isRejected(row)) return '<span class="badge warn">Rejected</span>';
      if (isTriageLike(row)) return '<span class="badge warn">Triage-like</span>';
      if (!row.retrieval) return '<span class="badge warn">No retrieval</span>';
      return '<span class="badge">Normal RAG</span>';
    }}

    function retrievalSummary(row) {{
      if (!row.retrieval) return routingLabel(row);
      const r = row.retrieval || {{}};
      return `${{routingLabel(row)}}<div class="mini">
        <span>MRR ${{fixed(r.mrr)}} · R@5 ${{fixed(r['recall@5'])}}</span>
        <span>Chunk MRR ${{fixed(r.chunk_mrr)}} · C@5 ${{fixed(r['chunk_recall@5'])}}</span>
      </div>`;
    }}

    function renderSummary() {{
      const passed = rows.filter(r => r.passed).length;
      const failed = rows.length - passed;
      const avgScore = rows.reduce((sum, r) => sum + Number(r.score || 0), 0) / Math.max(rows.length, 1);
      const triage = rows.filter(isTriageLike).length;
      const rejected = rows.filter(isRejected).length;
      const noRetrieval = rows.filter(r => !r.retrieval).length;
      const latencies = rows.map(r => Number(r.latency_ms || 0)).filter(Boolean).sort((a, b) => a - b);
      const p95 = latencies.length ? latencies[Math.floor((latencies.length - 1) * 0.95)] : null;
      const overall = payload.summary?.overall ? Object.values(payload.summary.overall)[0] : null;
      const items = [
        ['Rows', rows.length],
        ['Passed', `${{passed}} / ${{rows.length}}`],
        ['Pass rate', overall ? pct(overall.pass_rate) : pct(passed / Math.max(rows.length, 1))],
        ['Avg score', overall ? pct(overall.avg_score) : pct(avgScore)],
        ['Failed', failed],
        ['Triage-like', triage],
        ['Rejected', rejected],
        ['No retrieval', noRetrieval],
        ['P95 latency', p95 ? `${{Math.round(p95)}} ms` : '-'],
        ['Avg retrieval', overall?.avg_retrieval_ms ? `${{Math.round(overall.avg_retrieval_ms)}} ms` : '-'],
        ['Avg generator', overall?.avg_generator_ms ? `${{Math.round(overall.avg_generator_ms)}} ms` : '-'],
        ['Total tokens', overall?.total_tokens ?? '-'],
        ['Error rate', overall ? pct(overall.error_rate) : '-'],
      ];
      document.getElementById('summary').innerHTML = items.map(([label, value]) => `
        <div class="metric"><div class="label">${{escapeHtml(label)}}</div><div class="value">${{escapeHtml(value)}}</div></div>
      `).join('');
    }}

    function compactJson(value) {{
      if (!value) return '-';
      return JSON.stringify(value, null, 2);
    }}

    function flatValue(value) {{
      if (value === null || value === undefined) return '-';
      if (typeof value === 'number') return Number.isInteger(value) ? String(value) : fixed(value, 4);
      if (typeof value === 'boolean') return value ? 'true' : 'false';
      if (Array.isArray(value)) return value.length ? value.join('\\n') : '[]';
      if (typeof value === 'object') return JSON.stringify(value);
      return String(value);
    }}

    function kvBlock(obj) {{
      const entries = Object.entries(obj || {{}});
      if (!entries.length) return '<pre>-</pre>';
      return `<div class="kv">${{entries.map(([key, value]) => `
        <div class="kv-item">
          <div class="kv-key">${{escapeHtml(key)}}</div>
          <div class="kv-value">${{escapeHtml(flatValue(value))}}</div>
        </div>
      `).join('')}}</div>`;
    }}

    function listBlock(items) {{
      if (!items || !items.length) return '<pre>-</pre>';
      return `<div class="source-list">${{items.map(item => `<div>${{escapeHtml(item)}}</div>`).join('')}}</div>`;
    }}

    function rowMatches(row) {{
      const q = document.getElementById('search').value.trim().toLowerCase();
      const status = document.getElementById('statusFilter').value;
      const routing = document.getElementById('routingFilter').value;
      const scoreFilter = document.getElementById('scoreFilter').value;
      const text = [
        row.case_id, row.category, row.priority, row.question, row.expected_answer, row.answer,
        JSON.stringify(row.retrieved_slugs || []),
        JSON.stringify(row.retrieved_chunks || []),
        JSON.stringify(row.judge || {{}})
      ].join(' ').toLowerCase();
      const score = Number(row.score || 0);

      if (q && !text.includes(q)) return false;
      if (status === 'pass' && !row.passed) return false;
      if (status === 'fail' && row.passed) return false;
      if (routing === 'normal' && (!row.retrieval || isTriageLike(row) || isRejected(row))) return false;
      if (routing === 'triage' && !isTriageLike(row)) return false;
      if (routing === 'rejected' && !isRejected(row)) return false;
      if (routing === 'no-retrieval' && row.retrieval) return false;
      if (scoreFilter === 'lt50' && !(score < 0.5)) return false;
      if (scoreFilter === 'lt70' && !(score < 0.7)) return false;
      if (scoreFilter === 'gte80' && !(score >= 0.8)) return false;
      return true;
    }}

    function sortRows(items) {{
      const sortBy = document.getElementById('sortBy').value;
      return [...items].sort((a, b) => {{
        if (sortBy === 'score-desc') return Number(b.score || 0) - Number(a.score || 0);
        if (sortBy === 'latency-desc') return Number(b.latency_ms || 0) - Number(a.latency_ms || 0);
        if (sortBy === 'id-asc') return String(a.case_id || '').localeCompare(String(b.case_id || ''));
        return Number(a.score || 0) - Number(b.score || 0);
      }});
    }}

    function renderRows() {{
      const filtered = sortRows(rows.filter(rowMatches));
      document.getElementById('empty').hidden = filtered.length !== 0;
      document.getElementById('rows').innerHTML = filtered.map(row => {{
        const status = row.passed
          ? '<span class="badge pass">PASS</span>'
          : '<span class="badge fail">FAIL</span>';
        const judge = row.judge || {{}};
        const issueCount = (judge.missing_or_wrong || []).length + (judge.unsupported_claims || []).length;
        const scoreInfo = {{
          score: row.score,
          deterministic_score: row.deterministic_score,
          judge_combined_score: judge.combined_score,
          relevant_score: judge.relevant_score,
          correctness_score: judge.correctness_score,
          faithful_score: judge.faithful_score,
          scoring_mode: row.scoring_mode,
          hard_fail: row.hard_fail,
          forced_direct_answer: row.forced_direct_answer,
          error: row.error,
        }};
        const timingInfo = {{
          latency_ms: row.latency_ms,
          retrieval_ms: row.retrieval_ms,
          generator_ms: row.generator_ms,
          prompt_tokens: row.usage?.prompt_tokens,
          completion_tokens: row.usage?.completion_tokens,
          total_tokens: row.usage?.total_tokens,
          cost_usd: row.usage?.cost_usd,
        }};
        const details = `
          <details>
            <summary>Scoring and status</summary>
            ${{kvBlock(scoreInfo)}}
          </details>
          <details>
            <summary>Judge ${{issueCount ? `(${{issueCount}} issues)` : ''}}</summary>
            ${{kvBlock(row.judge)}}
            <pre>${{escapeHtml(compactJson(row.judge))}}</pre>
          </details>
          <details>
            <summary>Retrieval metrics</summary>
            ${{kvBlock(row.retrieval)}}
          </details>
          <details>
            <summary>Retrieved slugs</summary>
            ${{listBlock(row.retrieved_slugs)}}
          </details>
          <details>
            <summary>Retrieved chunks</summary>
            ${{listBlock(row.retrieved_chunks)}}
          </details>
          <details>
            <summary>Timing and usage</summary>
            ${{kvBlock(timingInfo)}}
          </details>
          <details>
            <summary>Checks</summary>
            <pre>${{escapeHtml(compactJson(row.checks))}}</pre>
          </details>
          <details>
            <summary>Raw row JSON</summary>
            <pre>${{escapeHtml(compactJson(row))}}</pre>
          </details>`;
        return `
          <tr>
            <td class="id"><code>${{escapeHtml(row.case_id)}}</code><div class="muted">${{escapeHtml(row.category || '')}}</div></td>
            <td class="status">${{status}}</td>
            <td class="score"><strong>${{fixed(row.score, 4)}}</strong><div class="mini">
              <span>det ${{fixed(row.deterministic_score, 4)}}</span>
              <span>judge ${{fixed(judge.combined_score, 4)}}</span>
            </div></td>
            <td class="latency">${{Math.round(Number(row.latency_ms || 0))}} ms<div class="mini">
              <span>ret ${{Math.round(Number(row.retrieval_ms || 0))}} ms</span>
              <span>gen ${{Math.round(Number(row.generator_ms || 0))}} ms</span>
            </div></td>
            <td class="retrieval">${{retrievalSummary(row)}}</td>
            <td class="question"><div class="question-text">${{escapeHtml(row.question || '')}}</div></td>
            <td class="expected"><div class="expected-text">${{escapeHtml(row.expected_answer || '-')}}</div></td>
            <td class="answer">
              <div class="answer-text">${{escapeHtml(row.answer || '')}}</div>
              ${{details}}
            </td>
          </tr>`;
      }}).join('');
    }}

    document.getElementById('source').textContent = payload.source;
    for (const id of ['search', 'statusFilter', 'routingFilter', 'scoreFilter', 'sortBy']) {{
      document.getElementById(id).addEventListener('input', renderRows);
    }}
    renderSummary();
    renderRows();
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export an eval JSONL result file to a static HTML viewer.")
    parser.add_argument("result", nargs="?", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    write_html(args.result, args.output)
    print(f"Wrote result viewer: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
