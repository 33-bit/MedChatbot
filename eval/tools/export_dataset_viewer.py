import json
from pathlib import Path

def export_html():
    jsonl_path = Path("eval/datasets/medical_qa_benchmark_v2.jsonl")
    html_path = Path("eval/artifacts/dataset_viewer.html")

    if not jsonl_path.exists():
        print("Dataset not found!")
        return

    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except Exception as e:
                    pass

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical QA Benchmark Dataset</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css" rel="stylesheet">
    <style>
        body {{ background-color: #f8f9fa; padding: 20px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        .card {{ box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-radius: 10px; border: none; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 10px 10px 0 0; margin-bottom: 20px; }}
        .badge-cat {{ font-size: 0.85em; }}
        .badge-priority {{ font-size: 0.8em; }}
        .turns-box {{ background: #f1f2f6; padding: 10px; border-radius: 6px; border-left: 4px solid #3498db; margin-bottom: 5px; }}
        .evidence-box {{ display: flex; flex-direction: column; gap: 8px; font-size: 0.85em; }}
        .evidence-label {{ font-weight: 600; color: #495057; margin-bottom: 3px; }}
        .evidence-item {{ background: #f8f9fa; color: #343a40; padding: 3px 7px; border-radius: 6px; border: 1px solid #dee2e6; margin: 2px 0; }}
        .chunk-id {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.78em; word-break: break-all; }}
        .docs-list {{ font-size: 0.9em; color: #555; }}
        .source-path {{ color: #6c757d; font-size: 0.8em; word-break: break-all; }}
        .meta-box {{ display: flex; flex-wrap: wrap; gap: 5px; }}
        table.dataTable tbody td {{ vertical-align: top; }}
    </style>
</head>
<body>

<div class="container-fluid">
    <div class="card">
        <div class="header d-flex justify-content-between align-items-center">
            <h2 class="m-0">🩺 Medical QA Benchmark Viewer</h2>
            <span class="badge bg-light text-dark fs-6">Total Cases: {len(data)}</span>
        </div>
        <div class="card-body">
            <div class="row g-3 align-items-end mb-3">
                <div class="col-sm-6 col-md-4 col-lg-3">
                    <label for="categoryFilter" class="form-label fw-semibold">Category</label>
                    <select id="categoryFilter" class="form-select">
                        <option value="">All categories</option>
                    </select>
                </div>
            </div>
            <table id="datasetTable" class="table table-hover table-bordered" style="width:100%">
                <thead class="table-light">
                    <tr>
                        <th style="width: 8%">ID</th>
                        <th style="width: 10%">Category</th>
                        <th style="width: 8%">Priority</th>
                        <th style="width: 24%">Question / Context</th>
                        <th style="width: 30%">Reference Answer</th>
                        <th style="width: 15%">Evidence</th>
                        <th style="width: 5%">Meta</th>
                    </tr>
                </thead>
                <tbody>
                </tbody>
            </table>
        </div>
    </div>
</div>

<script src="https://code.jquery.com/jquery-3.7.0.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>

<script>
    const dataset = {json.dumps(data, ensure_ascii=False)};
    const categoryOptions = Array.from(
        new Set(dataset.map(row => row.category).filter(Boolean))
    ).sort();

    function escapeHtml(value) {{
        return String(value ?? '')
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#039;');
    }}

    function basename(path) {{
        return String(path || '').split(/[\\\\/]/).pop();
    }}
    
    function getCategoryBadge(cat) {{
        const colors = {{
            'disease_info': 'bg-primary',
            'drug_info': 'bg-success',
            'emergency': 'bg-danger',
            'diagnostic_flow': 'bg-info text-dark',
            'safety_self_medication': 'bg-warning text-dark',
            'safety_prompt_injection': 'bg-dark',
            'safety_off_topic': 'bg-secondary'
        }};
        const color = colors[cat] || 'bg-secondary';
        return `<span class="badge badge-cat ${{color}}">${{escapeHtml(cat)}}</span>`;
    }}

    function getPriorityBadge(priority) {{
        const color = priority === 'high' ? 'bg-danger' : 'bg-secondary';
        return `<span class="badge badge-priority ${{color}}">${{escapeHtml(priority || '-')}}</span>`;
    }}

    function formatQuestion(row) {{
        if (row.turns && row.turns.length > 0) {{
            let html = '';
            row.turns.forEach((t, i) => {{
                html += `<div class="turns-box"><strong>T${{i+1}}:</strong> ${{escapeHtml(t)}}</div>`;
            }});
            return html;
        }}
        return `<strong>${{escapeHtml(row.question || '')}}</strong>`;
    }}

    function formatEvidence(row) {{
        const parts = [];
        if (row.requires_citation) {{
            parts.push('<span class="badge bg-primary">citation required</span>');
        }}
        if (row.source_docs && row.source_docs.length > 0) {{
            const docs = row.source_docs.map(d => {{
                const title = escapeHtml(d.title || basename(d.path));
                const path = escapeHtml(basename(d.path));
                return `<li>${{title}}${{path ? `<div class="source-path">${{path}}</div>` : ''}}</li>`;
            }}).join('');
            parts.push(`<div><div class="evidence-label">Source docs</div><ul class="ps-3 mb-0 docs-list">${{docs}}</ul></div>`);
        }} else if ((row.category || '').startsWith('safety_')) {{
            parts.push('<div class="text-muted">Global guardrail case</div>');
        }}
        if (row.gold_heading_paths && row.gold_heading_paths.length > 0) {{
            const headings = row.gold_heading_paths
                .map(path => `<div class="evidence-item">${{escapeHtml(path)}}</div>`)
                .join('');
            parts.push(`<div><div class="evidence-label">Heading paths</div>${{headings}}</div>`);
        }}
        if (row.gold_chunks && row.gold_chunks.length > 0) {{
            const chunks = row.gold_chunks
                .map(chunk => `<div class="evidence-item chunk-id">${{escapeHtml(chunk)}}</div>`)
                .join('');
            parts.push(`<div><div class="evidence-label">Gold chunks</div>${{chunks}}</div>`);
        }}
        return `<div class="evidence-box">${{parts.join('') || '<em class="text-muted">No evidence metadata</em>'}}</div>`;
    }}

    function formatMeta(row) {{
        const labels = [];
        labels.push(row.generated ? '<span class="badge bg-info text-dark">generated</span>' : '<span class="badge bg-light text-dark">manual</span>');
        if (row.llm_generated) labels.push('<span class="badge bg-secondary">LLM</span>');
        return `<div class="meta-box">${{labels.join('')}}</div>`;
    }}

    $(document).ready(function() {{
        const table = $('#datasetTable').DataTable({{
            data: dataset,
            pageLength: 25,
            order: [[1, 'asc']], // Order by category
            columns: [
                {{ data: 'id', render: (data, type) => `<code>${{escapeHtml(data)}}</code>` }},
                {{ data: 'category', render: (data, type) => type === 'display' ? getCategoryBadge(data) : data }},
                {{ data: 'priority', render: (data, type) => type === 'display' ? getPriorityBadge(data) : data }},
                {{ data: null, render: (data, type, row) => formatQuestion(row) }},
                {{ data: 'reference_answer', render: (data, type) => type === 'display' ? escapeHtml(data) : data }},
                {{ data: null, render: (data, type, row) => formatEvidence(row), orderable: false }},
                {{ data: null, render: (data, type, row) => formatMeta(row), orderable: false }}
            ],
            language: {{
                search: "Tìm kiếm:",
                lengthMenu: "Hiển thị _MENU_ câu hỏi",
                info: "Đang xem từ _START_ đến _END_ trong tổng số _TOTAL_ câu hỏi",
                paginate: {{
                    first: "Đầu",
                    last: "Cuối",
                    next: "Tiếp",
                    previous: "Trước"
                }}
            }}
        }});

        const categoryFilter = $('#categoryFilter');
        categoryOptions.forEach(cat => {{
            categoryFilter.append($('<option>', {{ value: cat, text: cat }}));
        }});
        categoryFilter.on('change', function() {{
            const selectedCategory = this.value;
            const escapedCategory = $.fn.dataTable.util.escapeRegex(selectedCategory);
            table
                .column(1)
                .search(selectedCategory ? `^${{escapedCategory}}$` : '', true, false)
                .draw();
        }});
    }});
</script>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated viewer at: {html_path.absolute()}")

if __name__ == "__main__":
    export_html()
