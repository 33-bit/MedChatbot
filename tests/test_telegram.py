from __future__ import annotations

import asyncio

from src.server.channels import telegram


def test_telegram_rejects_invalid_secret(app_client, monkeypatch):
    client, _ = app_client
    monkeypatch.setattr(telegram, "TELEGRAM_WEBHOOK_SECRET", "expected")

    response = client.post(
        "/webhook/telegram",
        headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"},
        json={"update_id": 1},
    )

    assert response.status_code == 403


def test_telegram_static_commands_are_distinct(monkeypatch):
    sent: list[tuple[int | str, str]] = []

    async def fake_send_text(chat_id: int | str, text: str) -> None:
        sent.append((chat_id, text))

    monkeypatch.setattr(telegram, "send_text", fake_send_text)

    assert asyncio.run(telegram._handle_command(123, "/start"))
    assert asyncio.run(telegram._handle_command(123, "/help"))
    assert asyncio.run(telegram._handle_command(123, "/menu"))

    texts = [text for _, text in sent]
    assert len(texts) == 3
    assert len(set(texts)) == 3
    assert "Xin chào" in texts[0]
    assert "Cách sử dụng" in texts[1]
    assert "Menu lệnh" in texts[2]


def test_telegram_new_command_clears_redis_session(monkeypatch):
    cleared: list[str] = []
    sent: list[str] = []

    async def fake_send_text(chat_id: int | str, text: str) -> None:
        sent.append(text)

    monkeypatch.setattr(telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram, "clear_session", lambda session_id: cleared.append(session_id))

    assert asyncio.run(telegram._handle_command(456, "/new"))

    assert cleared == ["tg:456"]
    assert "xóa ngữ cảnh" in sent[0]


def test_telegram_webhook_ignores_duplicate_update(app_client, monkeypatch):
    client, _ = app_client
    calls: list[tuple[int | str, str]] = []
    monkeypatch.setattr(telegram, "TELEGRAM_WEBHOOK_SECRET", "")
    monkeypatch.setattr(telegram, "reserve_webhook_update", lambda channel, update_id: False)

    async def fake_answer_and_send(chat_id: int | str, text: str) -> None:
        calls.append((chat_id, text))

    monkeypatch.setattr(telegram, "_answer_and_send", fake_answer_and_send)

    response = client.post(
        "/webhook/telegram",
        json={
            "update_id": 10,
            "message": {"chat": {"id": 123}, "text": "Tôi bị ho"},
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert calls == []


def test_telegram_webhook_schedules_answer_for_text_message(app_client, monkeypatch):
    client, _ = app_client
    calls: list[tuple[int | str, str]] = []
    monkeypatch.setattr(telegram, "TELEGRAM_WEBHOOK_SECRET", "")
    monkeypatch.setattr(telegram, "reserve_webhook_update", lambda channel, update_id: True)

    async def fake_answer_and_send(chat_id: int | str, text: str) -> None:
        calls.append((chat_id, text))

    monkeypatch.setattr(telegram, "_answer_and_send", fake_answer_and_send)

    response = client.post(
        "/webhook/telegram",
        json={
            "update_id": 11,
            "message": {"chat": {"id": 123}, "text": "Tôi bị ho"},
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert calls == [(123, "Tôi bị ho")]


def test_telegram_webhook_ignores_non_text_message(app_client, monkeypatch):
    client, _ = app_client
    monkeypatch.setattr(telegram, "TELEGRAM_WEBHOOK_SECRET", "")
    monkeypatch.setattr(telegram, "reserve_webhook_update", lambda channel, update_id: True)

    async def fail_if_called(chat_id: int | str, text: str) -> None:
        raise AssertionError("non-text message should not trigger an answer")

    monkeypatch.setattr(telegram, "_answer_and_send", fail_if_called)

    response = client.post(
        "/webhook/telegram",
        json={"update_id": 12, "message": {"chat": {"id": 123}, "photo": []}},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
