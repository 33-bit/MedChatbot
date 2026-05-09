from __future__ import annotations

import hashlib

from src.chat.replies import TECHNICAL_ERROR_REPLY


def test_health_endpoint(app_client):
    client, _ = app_client

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


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


def test_chat_uses_api_key_derived_session_not_body_session_id(app_client, monkeypatch):
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

    expected_session = "api:" + hashlib.sha256(b"secret").hexdigest()[:16]
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
        json={},
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
        json={"question": "Tôi bị đau ngực"},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": TECHNICAL_ERROR_REPLY}
