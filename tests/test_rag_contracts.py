from __future__ import annotations

import json
import logging

from src.rag import build_qdrant


def test_qdrant_loader_warns_when_chunk_id_missing(tmp_path, caplog):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        json.dumps({"text": "context"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    caplog.set_level(logging.WARNING, logger=build_qdrant.log.name)

    chunks = build_qdrant.load_chunks(chunks_path)

    assert chunks[0]["id"]
    assert "missing chunk_id" in caplog.text
