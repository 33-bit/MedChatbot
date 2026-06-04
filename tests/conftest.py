from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_service_caches(tmp_path, monkeypatch):
    """Keep tests off the developer's real Redis/SQLite/Neo4j/OpenAI clients."""
    from src.chat import clients

    for factory_name in ("get_sqlite", "get_redis", "get_neo4j", "get_openai"):
        factory = getattr(clients, factory_name, None)
        if hasattr(factory, "cache_clear"):
            factory.cache_clear()

    monkeypatch.setattr(clients, "SQLITE_PATH", str(tmp_path / "chatbot-test.db"))
    yield

    for factory_name in ("get_sqlite", "get_redis", "get_neo4j", "get_openai"):
        factory = getattr(clients, factory_name, None)
        if hasattr(factory, "cache_clear"):
            factory.cache_clear()


@pytest.fixture
def app_client(monkeypatch):
    from fastapi.testclient import TestClient

    from src.server import app as app_module

    async def noop_setup_bot_menu() -> None:
        return None

    monkeypatch.setattr(app_module, "preload_retrieval_models", lambda: None)
    monkeypatch.setattr(app_module, "ensure_fulltext_indexes", lambda: None)
    monkeypatch.setattr(app_module.telegram, "setup_bot_menu", noop_setup_bot_menu)

    with TestClient(app_module.app) as client:
        yield client, app_module
