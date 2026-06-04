from __future__ import annotations

import asyncio
import hashlib

from src.chat.replies import TECHNICAL_ERROR_REPLY


def test_health_endpoint(app_client):
    client, _ = app_client

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_app_uses_lifespan_instead_of_legacy_event_handlers():
    from src.server import app as app_module

    assert app_module.app.router.on_startup == []
    assert app_module.app.router.on_shutdown == []


def test_startup_continues_when_telegram_menu_setup_fails(monkeypatch):
    from src.server import app as app_module

    async def broken_setup_bot_menu() -> None:
        raise RuntimeError("telegram api unavailable")

    monkeypatch.setattr(app_module, "preload_retrieval_models", lambda: None)
    monkeypatch.setattr(app_module, "ensure_fulltext_indexes", lambda: None)
    monkeypatch.setattr(app_module.telegram, "setup_bot_menu", broken_setup_bot_menu)

    asyncio.run(app_module.startup())


def test_startup_initializes_fulltext_indexes(monkeypatch):
    from src.server import app as app_module

    calls: list[str] = []

    async def setup_bot_menu() -> None:
        calls.append("telegram")

    monkeypatch.setattr(app_module, "preload_retrieval_models", lambda: None)
    monkeypatch.setattr(app_module, "ensure_fulltext_indexes", lambda: calls.append("indexes"))
    monkeypatch.setattr(app_module.telegram, "setup_bot_menu", setup_bot_menu)

    asyncio.run(app_module.startup())

    assert calls == ["indexes", "telegram"]


def test_chat_disabled_without_api_key(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "")

    response = client.post("/chat", json={"question": "Tôi bị ho"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Chat API disabled: CHAT_API_KEY not set"


def test_chat_rejects_missing_or_wrong_api_key(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")

    missing = client.post("/chat", json={"question": "Tôi bị ho"})
    wrong = client.post(
        "/chat",
        headers={"X-API-Key": "wrong"},
        json={"question": "Tôi bị ho"},
    )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert wrong.json()["detail"] == "Invalid API key"


def test_chat_rejects_missing_session_id(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")

    def fake_answer(question: str, session_id: str = "default") -> str:
        return "ok"

    monkeypatch.setattr(app_module, "answer", fake_answer)

    response = client.post(
        "/chat",
        headers={"X-API-Key": "secret"},
        json={"question": "Tôi bị ho"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "session_id is required"


def test_chat_scopes_session_to_api_key_and_body_session_id(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")
    seen: list[str] = []

    def fake_answer(question: str, session_id: str = "default") -> str:
        seen.append(session_id)
        return "ok"

    monkeypatch.setattr(app_module, "answer", fake_answer)

    first = client.post(
        "/chat",
        headers={"X-API-Key": "secret"},
        json={"question": "Tôi bị ho", "session_id": "user-a"},
    )
    second = client.post(
        "/chat",
        headers={"X-API-Key": "secret"},
        json={"question": "Tôi bị đau đầu", "session_id": "user-b"},
    )

    expected_first = "api:" + hashlib.sha256(b"secret\x00user-a").hexdigest()[:32]
    expected_second = "api:" + hashlib.sha256(b"secret\x00user-b").hexdigest()[:32]
    assert first.status_code == 200
    assert second.status_code == 200
    assert seen == [expected_first, expected_second]
    assert seen[0] != seen[1]


def test_chat_does_not_pass_raw_body_session_id_to_pipeline(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")
    seen: dict[str, str] = {}

    def fake_answer(question: str, session_id: str = "default") -> str:
        seen["question"] = question
        seen["session_id"] = session_id
        return "ok"

    monkeypatch.setattr(app_module, "answer", fake_answer)

    response = client.post(
        "/chat",
        headers={"X-API-Key": "secret"},
        json={"question": "Tôi bị ho", "session_id": "victim"},
    )

    expected_session = "api:" + hashlib.sha256(b"secret\x00victim").hexdigest()[:32]
    assert response.status_code == 200
    assert response.json() == {"answer": "ok"}
    assert seen == {"question": "Tôi bị ho", "session_id": expected_session}


def test_chat_empty_question_returns_specific_reply(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")

    def fake_answer(question: str, session_id: str = "default") -> str:
        return "Bạn hãy đặt câu hỏi cụ thể nhé." if not question else "unexpected"

    monkeypatch.setattr(app_module, "answer", fake_answer)

    response = client.post(
        "/chat",
        headers={"X-API-Key": "secret"},
        json={"session_id": "user-a"},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "Bạn hãy đặt câu hỏi cụ thể nhé."}


def test_chat_returns_technical_reply_when_pipeline_raises(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")

    def broken_answer(question: str, session_id: str = "default") -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module, "answer", broken_answer)

    response = client.post(
        "/chat",
        headers={"X-API-Key": "secret"},
        json={"question": "Tôi bị đau ngực", "session_id": "user-a"},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": TECHNICAL_ERROR_REPLY}
