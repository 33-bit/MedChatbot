from __future__ import annotations

import asyncio

from src.server.channels import common
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
    calls: list[tuple[str, str, str]] = []

    async def fake_answer_and_send(text, recipient_id, session_id, send_text, **kwargs):
        calls.append((text, recipient_id, session_id))

    monkeypatch.setattr(messenger, "answer_and_send", fake_answer_and_send)

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
    assert calls == [("Tôi bị ho", "u1", "fb:u1")]


def test_zalo_verify_endpoint(app_client):
    client, _ = app_client

    response = client.get("/webhook/zalo")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_zalo_webhook_answers_user_text(app_client, monkeypatch):
    client, _ = app_client
    calls: list[tuple[str, str, str]] = []

    async def fake_answer_and_send(text, recipient_id, session_id, send_text, **kwargs):
        calls.append((text, recipient_id, session_id))

    monkeypatch.setattr(zalo, "ZALO_WEBHOOK_SECRET", "zalo-secret")
    monkeypatch.setattr(zalo, "answer_and_send", fake_answer_and_send)

    response = client.post(
        "/webhook/zalo",
        headers={"X-Bot-Api-Secret-Token": "zalo-secret"},
        json={
            "ok": True,
            "result": {
                "event_name": "message.text.received",
                "message": {
                    "chat": {"id": "zchat1", "chat_type": "PRIVATE"},
                    "text": "Tôi bị ho",
                },
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert calls == [("Tôi bị ho", "zchat1", "zalo:zchat1")]


def test_zalo_webhook_answers_flattened_user_text_payload(app_client, monkeypatch):
    client, _ = app_client
    calls: list[tuple[str, str, str]] = []

    async def fake_answer_and_send(text, recipient_id, session_id, send_text, **kwargs):
        calls.append((text, recipient_id, session_id))

    monkeypatch.setattr(zalo, "ZALO_WEBHOOK_SECRET", "zalo-secret")
    monkeypatch.setattr(zalo, "answer_and_send", fake_answer_and_send)

    response = client.post(
        "/webhook/zalo",
        headers={"X-Bot-Api-Secret-Token": "zalo-secret"},
        json={
            "event_name": "message.text.received",
            "message": {
                "chat": {"id": "zchat1", "chat_type": "PRIVATE"},
                "text": "Tôi bị ho",
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert calls == [("Tôi bị ho", "zchat1", "zalo:zchat1")]


def test_zalo_webhook_rejects_invalid_secret(app_client, monkeypatch):
    client, _ = app_client
    monkeypatch.setattr(zalo, "ZALO_WEBHOOK_SECRET", "zalo-secret")

    response = client.post(
        "/webhook/zalo",
        headers={"X-Bot-Api-Secret-Token": "wrong"},
        json={
            "ok": True,
            "result": {
                "event_name": "message.text.received",
                "message": {
                    "chat": {"id": "zchat1", "chat_type": "PRIVATE"},
                    "text": "Tôi bị ho",
                },
            },
        },
    )

    assert response.status_code == 403


def test_zalo_send_text_uses_bot_platform_api(monkeypatch):
    calls: list[tuple[str, dict, float]] = []

    class Response:
        status_code = 200
        text = '{"ok":true}'

    class Client:
        def __init__(self, timeout: float):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url: str, json: dict):
            calls.append((url, json, self.timeout))
            return Response()

    monkeypatch.setattr(zalo, "ZALO_BOT_TOKEN", "bot-token")
    monkeypatch.setattr(zalo.httpx, "AsyncClient", Client)

    asyncio.run(zalo.send_text("zchat1", "Xin chào"))

    assert calls == [
        (
            "https://bot-api.zaloplatforms.com/botbot-token/sendMessage",
            {"chat_id": "zchat1", "text": "Xin chào"},
            20.0,
        )
    ]


def test_zalo_send_text_adds_best_effort_reply_markup_for_choices(monkeypatch):
    calls: list[dict] = []

    class Response:
        status_code = 200
        text = '{"ok":true}'

    class Client:
        def __init__(self, timeout: float):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url: str, json: dict):
            calls.append(json)
            return Response()

    monkeypatch.setattr(zalo, "ZALO_BOT_TOKEN", "bot-token")
    monkeypatch.setattr(zalo.httpx, "AsyncClient", Client)

    asyncio.run(zalo.send_text("zchat1", "Bạn chọn một ý nhé.", ["Đau nhẹ/vừa", "Đau dữ dội"]))

    assert calls == [
        {
            "chat_id": "zchat1",
            "text": "Bạn chọn một ý nhé.",
            "reply_markup": {
                "keyboard": [
                    [{"text": "Đau nhẹ/vừa"}, {"text": "Đau dữ dội"}],
                ],
                "resize_keyboard": True,
                "one_time_keyboard": True,
            },
        }
    ]


def test_messenger_send_text_uses_quick_replies_for_choices(monkeypatch):
    calls: list[dict] = []

    class Response:
        status_code = 200
        text = '{"ok":true}'

    class Client:
        def __init__(self, timeout: float):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url: str, params: dict, json: dict):
            calls.append(json)
            return Response()

    monkeypatch.setattr(messenger, "MESSENGER_PAGE_TOKEN", "page-token")
    monkeypatch.setattr(messenger.httpx, "AsyncClient", Client)

    asyncio.run(messenger.send_text("u1", "Bạn chọn một ý nhé.", ["Đau nhẹ/vừa", "Đau dữ dội"]))

    assert calls == [
        {
            "recipient": {"id": "u1"},
            "messaging_type": "RESPONSE",
            "message": {
                "text": "Bạn chọn một ý nhé.",
                "quick_replies": [
                    {"content_type": "text", "title": "Đau nhẹ/vừa", "payload": "choice:Đau nhẹ/vừa"},
                    {"content_type": "text", "title": "Đau dữ dội", "payload": "choice:Đau dữ dội"},
                ],
            },
        }
    ]


def test_zalo_send_text_splits_messages_over_platform_limit(monkeypatch):
    calls: list[dict] = []

    class Response:
        status_code = 200
        text = '{"ok":true}'

    class Client:
        def __init__(self, timeout: float):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url: str, json: dict):
            calls.append(json)
            return Response()

    monkeypatch.setattr(zalo, "ZALO_BOT_TOKEN", "bot-token")
    monkeypatch.setattr(zalo.httpx, "AsyncClient", Client)

    asyncio.run(zalo.send_text("zchat1", "a" * 2001))

    assert calls == [
        {"chat_id": "zchat1", "text": "a" * 2000},
        {"chat_id": "zchat1", "text": "a"},
    ]


def test_common_answer_and_send_runs_answer_off_event_loop(monkeypatch):
    calls: list[tuple[str, str]] = []
    sent: list[tuple[str, str]] = []

    def fake_answer_with_choices(text: str, session_id: str = "default"):
        calls.append((text, session_id))
        return common.ChatReply("reply", ())

    async def fake_send_text(recipient_id: str, text: str, choices=None) -> None:
        sent.append((recipient_id, text))

    monkeypatch.setattr(common, "answer_with_choices", fake_answer_with_choices)

    asyncio.run(common.answer_and_send("Tôi bị ho", "u1", "fb:u1", fake_send_text))

    assert calls == [("Tôi bị ho", "fb:u1")]
    assert sent == [("u1", "reply")]


def test_common_answer_and_send_forwards_choices(monkeypatch):
    sent: list[tuple[str, str, list[str]]] = []

    def fake_answer_with_choices(text: str, session_id: str = "default"):
        assert text == "Tôi bị đau bụng"
        assert session_id == "zalo:u1"
        return common.ChatReply("Bạn chọn một ý nhé.", ("Đau nhẹ/vừa", "Đau dữ dội"))

    async def fake_send_text(recipient_id: str, text: str, choices=None) -> None:
        sent.append((recipient_id, text, list(choices or [])))

    monkeypatch.setattr(common, "answer_with_choices", fake_answer_with_choices)

    asyncio.run(common.answer_and_send("Tôi bị đau bụng", "u1", "zalo:u1", fake_send_text))

    assert sent == [("u1", "Bạn chọn một ý nhé.", ["Đau nhẹ/vừa", "Đau dữ dội"])]


def test_common_answer_and_send_sends_technical_reply_on_answer_error(monkeypatch):
    sent: list[tuple[str, str]] = []

    def fail_answer(text: str, session_id: str = "default"):
        raise RuntimeError("llm down")

    async def fake_send_text(recipient_id: str, text: str, choices=None) -> None:
        sent.append((recipient_id, text))

    monkeypatch.setattr(common, "answer_with_choices", fail_answer)

    asyncio.run(common.answer_and_send("Tôi bị ho", "u1", "fb:u1", fake_send_text))

    assert sent == [("u1", common.TECHNICAL_ERROR_REPLY)]
