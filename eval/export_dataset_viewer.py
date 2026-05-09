import json
import os
from pathlib import Path

def export_html():
    jsonl_path = Path("eval/medical_qa_benchmark.jsonl")
    html_path = Path("eval/dataset_viewer.html")

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
        .keywords-box {{ display: flex; flex-wrap: wrap; gap: 5px; }}
        .keyword {{ background: #e0f7fa; color: #006064; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; border: 1px solid #b2ebf2; }}
        .keyword-not {{ background: #ffebee; color: #c62828; border-color: #ffcdd2; }}
        .docs-list {{ font-size: 0.85em; color: #555; }}
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
            <table id="datasetTable" class="table table-hover table-bordered" style="width:100%">
                <thead class="table-light">
                    <tr>
                        <th style="width: 8%">ID</th>
                        <th style="width: 10%">Category</th>
                        <th style="width: 25%">Question / Context</th>
                        <th style="width: 35%">Reference Answer</th>
                        <th style="width: 12%">Keywords</th>
                        <th style="width: 10%">Source</th>
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
    
    function getCategoryBadge(cat) {{
        const colors = {{
            'disease_info': 'bg-primary',
            'drug_info': 'bg-success',
            'emergency': 'bg-danger',
            'diagnostic_flow': 'bg-info text-dark',
            'safety': 'bg-warning text-dark',
            'direct_answer': 'bg-secondary'
        }};
        const color = colors[cat] || 'bg-secondary';
        return `<span class="badge badge-cat ${{color}}">${{cat}}</span>`;
    }}

    function formatQuestion(row) {{
        if (row.turns && row.turns.length > 0) {{
            let html = '';
            row.turns.forEach((t, i) => {{
                html += `<div class="turns-box"><strong>T${{i+1}}:</strong> ${{t}}</div>`;
            }});
            return html;
        }}
        return `<strong>${{row.question || ''}}</strong>`;
    }}

    function formatKeywords(row) {{
        let html = '<div class="keywords-box">';
        if (row.must_include) {{
            row.must_include.forEach(k => {{
                html += `<span class="keyword">✓ ${{k}}</span>`;
            }});
        }}
        if (row.must_include_any) {{
            row.must_include_any.forEach(group => {{
                if (Array.isArray(group)) {{
                    html += `<span class="keyword">✓ [${{group.join(' OR ')}}]</span>`;
                }}
            }});
        }}
        if (row.must_not_include) {{
            row.must_not_include.forEach(k => {{
                html += `<span class="keyword keyword-not">✗ ${{k}}</span>`;
            }});
        }}
        if (row.requires_emergency_advice) {{
            html += `<span class="badge bg-danger mt-1">Khuyên cấp cứu</span>`;
        }}
        html += '</div>';
        return html;
    }}

    function formatDocs(row) {{
        if (!row.source_docs || row.source_docs.length === 0) return '<em>No doc</em>';
        return `<ul class="ps-3 mb-0 docs-list">` + 
               row.source_docs.map(d => `<li>${{d.title}}</li>`).join('') + 
               `</ul>`;
    }}

    $(document).ready(function() {{
        $('#datasetTable').DataTable({{
            data: dataset,
            pageLength: 25,
            order: [[1, 'asc']], // Order by category
            columns: [
                {{ data: 'id', render: (data, type, row) => `<code>${{data}}</code><br><small class="text-muted">${{row.generated?'Auto':'Manual'}}</small>` }},
                {{ data: 'category', render: (data) => getCategoryBadge(data) }},
                {{ data: null, render: (data, type, row) => formatQuestion(row) }},
                {{ data: 'reference_answer' }},
                {{ data: null, render: (data, type, row) => formatKeywords(row), orderable: false }},
                {{ data: null, render: (data, type, row) => formatDocs(row), orderable: false }}
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
    }});
</script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated viewer at: {html_path.absolute()}")

if __name__ == "__main__":
    export_html()
