from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from src.chat.retrieval import preload


def test_preload_temporarily_enables_network_then_restores_offline(monkeypatch):
    calls: list[object] = []
    fake_dense = SimpleNamespace(preload_models=lambda: calls.append("loaded"))
    monkeypatch.setitem(sys.modules, "src.chat.retrieval.dense", fake_dense)
    monkeypatch.setattr(preload, "HF_PRELOAD_RETRIEVAL_MODELS", True)
    monkeypatch.setattr(preload, "HF_OFFLINE_AFTER_PRELOAD", True)
    monkeypatch.setattr(preload, "HF_PRELOAD_REQUIRED", False)
    monkeypatch.setattr(preload, "set_hf_offline", lambda enabled: calls.append(enabled))

    preload.preload_retrieval_models()

    assert calls == [False, "loaded", True]


def test_preload_failure_is_raised_when_required_and_offline_is_restored(monkeypatch):
    calls: list[bool] = []

    def broken_preload() -> None:
        raise RuntimeError("missing model")

    fake_dense = SimpleNamespace(preload_models=broken_preload)
    monkeypatch.setitem(sys.modules, "src.chat.retrieval.dense", fake_dense)
    monkeypatch.setattr(preload, "HF_PRELOAD_RETRIEVAL_MODELS", True)
    monkeypatch.setattr(preload, "HF_OFFLINE_AFTER_PRELOAD", True)
    monkeypatch.setattr(preload, "HF_PRELOAD_REQUIRED", True)
    monkeypatch.setattr(preload, "set_hf_offline", lambda enabled: calls.append(enabled))

    with pytest.raises(RuntimeError, match="missing model"):
        preload.preload_retrieval_models()

    assert calls == [False, True]


def test_preload_disabled_keeps_runtime_offline(monkeypatch):
    calls: list[bool] = []
    monkeypatch.setattr(preload, "HF_PRELOAD_RETRIEVAL_MODELS", False)
    monkeypatch.setattr(preload, "HF_OFFLINE_AFTER_PRELOAD", True)
    monkeypatch.setattr(preload, "set_hf_offline", lambda enabled: calls.append(enabled))

    preload.preload_retrieval_models()

    assert calls == [True]
