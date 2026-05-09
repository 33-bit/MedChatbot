from __future__ import annotations

from src.server.channels import messenger, zalo


def test_messenger_verify_accepts_matching_token(app_client, monkeypatch):
    client, _ = app_client
    monkeypatch.setattr(messenger, "MESSENGER_VERIFY_TOKEN", "verify")

    response = client.get(
        "/webhook/messenger",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "verify",
            "hub.challenge": "challenge-text",
        },
    )

    assert response.status_code == 200
    assert response.text == "challenge-text"


def test_messenger_verify_rejects_bad_token(app_client, monkeypatch):
    client, _ = app_client
    monkeypatch.setattr(messenger, "MESSENGER_VERIFY_TOKEN", "verify")

    response = client.get(
        "/webhook/messenger",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong",
            "hub.challenge": "challenge-text",
        },
    )

    assert response.status_code == 403


def test_messenger_webhook_answers_text_message(app_client, monkeypatch):
    client, _ = app_client
    answered: list[tuple[str, str]] = []
    sent: list[tuple[str, str]] = []

    def fake_answer(text: str, session_id: str = "default") -> str:
        answered.append((text, session_id))
        return "reply"

    async def fake_send_text(recipient_id: str, text: str) -> None:
        sent.append((recipient_id, text))

    monkeypatch.setattr(messenger, "answer", fake_answer)
    monkeypatch.setattr(messenger, "send_text", fake_send_text)

    response = client.post(
        "/webhook/messenger",
        json={
            "object": "page",
            "entry": [
                {"messaging": [{"sender": {"id": "u1"}, "message": {"text": "Tôi bị ho"}}]}
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert answered == [("Tôi bị ho", "fb:u1")]
    assert sent == [("u1", "reply")]


def test_zalo_verify_endpoint(app_client):
    client, _ = app_client

    response = client.get("/webhook/zalo")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_zalo_webhook_answers_user_text(app_client, monkeypatch):
    client, _ = app_client
    answered: list[tuple[str, str]] = []
    sent: list[tuple[str, str]] = []

    def fake_answer(text: str, session_id: str = "default") -> str:
        answered.append((text, session_id))
        return "reply"

    async def fake_send_text(user_id: str, text: str) -> None:
        sent.append((user_id, text))

    monkeypatch.setattr(zalo, "answer", fake_answer)
    monkeypatch.setattr(zalo, "send_text", fake_send_text)

    response = client.post(
        "/webhook/zalo",
        json={
            "event_name": "user_send_text",
            "sender": {"id": "z1"},
            "message": {"text": "Tôi bị ho"},
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert answered == [("Tôi bị ho", "zalo:z1")]
    assert sent == [("z1", "reply")]
