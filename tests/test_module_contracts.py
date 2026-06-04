from __future__ import annotations

from pathlib import Path

from src.chat import pipeline


def test_removed_dead_runtime_modules_stay_removed():
    assert not Path("src/chat/llm/formatting.py").exists()
    assert not Path("src/chat/persistence.py").exists()


def test_pipeline_does_not_keep_module_scope_retrieval_executor():
    assert not hasattr(pipeline, "_RETRIEVAL_EXECUTOR")
