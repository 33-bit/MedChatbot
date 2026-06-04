from __future__ import annotations

import asyncio
import threading

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


def test_telegram_markdown_keeps_code_fence_content_literal():
    html = telegram._md_to_tg_html("```python\n# not a heading\n```\n\n# Heading")

    assert "<pre>python\n# not a heading\n</pre>" in html
    assert "<pre>python\n<b>not a heading</b>\n</pre>" not in html
    assert "<b>Heading</b>" in html


def test_telegram_split_respects_utf8_byte_limit():
    text = "ấ" * 3000

    chunks = telegram._split_for_telegram(text)

    assert len(chunks) > 1
    assert "".join(chunks) == text
    assert all(len(chunk.encode("utf-8")) <= telegram.TG_MAX_LEN for chunk in chunks)


def test_telegram_answer_sends_rating_prompt_after_reply(monkeypatch):
    sent: list[tuple[int | str, str]] = []
    feedback_requests: list[dict] = []
    prompts: list[tuple[int | str, str]] = []

    def fake_answer_with_choices(question: str, session_id: str = "default"):
        assert question == "Tôi bị ho"
        assert session_id == "tg:123"
        return telegram.ChatReply("Bạn nên nghỉ ngơi.", ())

    async def fake_send_text(chat_id: int | str, text: str, choices=None, selection_mode="single") -> None:
        sent.append((chat_id, text))

    def fake_create_feedback_request(
        session_id: str,
        channel: str,
        recipient_id: str,
        question: str,
        answer: str,
    ) -> str:
        feedback_requests.append(
            {
                "session_id": session_id,
                "channel": channel,
                "recipient_id": recipient_id,
                "question": question,
                "answer": answer,
            }
        )
        return "rating-token"

    async def fake_send_rating_prompt(chat_id: int | str, token: str) -> None:
        prompts.append((chat_id, token))

    monkeypatch.setattr(telegram, "answer_with_choices", fake_answer_with_choices, raising=False)
    monkeypatch.setattr(telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram, "create_feedback_request", fake_create_feedback_request, raising=False)
    monkeypatch.setattr(telegram, "_send_rating_prompt", fake_send_rating_prompt, raising=False)

    asyncio.run(telegram._answer_and_send(123, "Tôi bị ho"))

    assert sent == [(123, "Bạn nên nghỉ ngơi.")]
    assert feedback_requests == [
        {
            "session_id": "tg:123",
            "channel": "telegram",
            "recipient_id": "123",
            "question": "Tôi bị ho",
            "answer": "Bạn nên nghỉ ngơi.",
        }
    ]
    assert prompts == [(123, "rating-token")]


def test_telegram_answer_sends_reply_keyboard_for_choices(monkeypatch):
    sent: list[tuple[int | str, str, list[str], str]] = []
    prompts: list[tuple[int | str, str]] = []

    def fake_answer_with_choices(question: str, session_id: str = "default"):
        assert question == "Tôi bị đau bụng"
        assert session_id == "tg:123"
        return telegram.ChatReply("Bạn cho tôi biết thêm nhé.", ("Đau nhẹ/vừa", "Đau dữ dội"))

    async def fake_send_text(chat_id: int | str, text: str, choices=None, selection_mode="single") -> None:
        sent.append((chat_id, text, list(choices or []), selection_mode))

    async def fake_send_rating_prompt(chat_id: int | str, token: str) -> None:
        prompts.append((chat_id, token))

    monkeypatch.setattr(telegram, "answer_with_choices", fake_answer_with_choices, raising=False)
    monkeypatch.setattr(telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram, "create_feedback_request", lambda *args: "rating-token", raising=False)
    monkeypatch.setattr(telegram, "_send_rating_prompt", fake_send_rating_prompt, raising=False)

    asyncio.run(telegram._answer_and_send(123, "Tôi bị đau bụng"))

    assert sent == [(123, "Bạn cho tôi biết thêm nhé.", ["Đau nhẹ/vừa", "Đau dữ dội"], "single")]
    assert prompts == []


def test_telegram_answer_sends_typing_action_while_generating(monkeypatch):
    typing_seen = threading.Event()
    typing_calls: list[int | str] = []

    def fake_answer_with_choices(question: str, session_id: str = "default"):
        typing_seen.wait(timeout=0.2)
        return telegram.ChatReply("Bạn nên nghỉ ngơi.", ())

    async def fake_send_typing_action(chat_id: int | str) -> None:
        typing_calls.append(chat_id)
        typing_seen.set()

    async def fake_send_text(chat_id: int | str, text: str, choices=None, selection_mode="single") -> None:
        return None

    monkeypatch.setattr(telegram, "answer_with_choices", fake_answer_with_choices, raising=False)
    monkeypatch.setattr(telegram, "_send_typing_action", fake_send_typing_action, raising=False)
    monkeypatch.setattr(telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram, "create_feedback_request", lambda *args: "rating-token", raising=False)
    monkeypatch.setattr(telegram, "_send_rating_prompt", fake_send_text, raising=False)

    asyncio.run(telegram._answer_and_send(123, "Tôi bị ho"))

    assert typing_calls == [123]


def test_telegram_rating_keyboard_has_five_callback_scores():
    keyboard = telegram._rating_keyboard("rating-token")

    assert keyboard == {
        "inline_keyboard": [
            [
                {"text": "1", "callback_data": "rate:rating-token:1"},
                {"text": "2", "callback_data": "rate:rating-token:2"},
                {"text": "3", "callback_data": "rate:rating-token:3"},
                {"text": "4", "callback_data": "rate:rating-token:4"},
                {"text": "5", "callback_data": "rate:rating-token:5"},
            ]
        ]
    }


def test_telegram_choice_keyboard_uses_reply_keyboard_buttons():
    keyboard = telegram._choice_keyboard(["Đau nhẹ/vừa", "Đau dữ dội", "Không rõ"])

    assert keyboard == {
        "keyboard": [
            [{"text": "Đau nhẹ/vừa"}, {"text": "Đau dữ dội"}],
            [{"text": "Không rõ"}],
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True,
    }


def test_telegram_multi_choice_keyboard_uses_inline_buttons():
    keyboard = telegram._multi_choice_keyboard(
        "token",
        ("Có nôn", "Có vàng da", "Không"),
        {1},
    )

    assert keyboard == {
        "inline_keyboard": [
            [
                {"text": "Có nôn", "callback_data": "ms:token:0"},
                {"text": "✓ Có vàng da", "callback_data": "ms:token:1"},
            ],
            [{"text": "Không", "callback_data": "ms:token:2"}],
            [{"text": "✅ Xong", "callback_data": "ms:token:done"}],
        ]
    }


def test_telegram_send_text_uses_inline_keyboard_for_multi_choices(monkeypatch):
    calls: list[dict] = []

    class Response:
        status_code = 200
        text = '{"ok":true}'

        def json(self):
            return {"ok": True, "result": {"message_id": 456}}

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

    monkeypatch.setattr(telegram, "TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setattr(telegram.httpx, "AsyncClient", Client)
    monkeypatch.setattr(telegram, "_new_multi_select_token", lambda: "select-token")
    telegram._MULTI_SELECTS.clear()

    asyncio.run(
        telegram.send_text(
            123,
            "Có kèm theo nôn hoặc vàng da không?",
            ("Có nôn", "Có vàng da", "Không"),
            selection_mode="multi",
        )
    )

    assert calls[-1]["reply_markup"] == {
        "inline_keyboard": [
            [
                {"text": "Có nôn", "callback_data": "ms:select-token:0"},
                {"text": "Có vàng da", "callback_data": "ms:select-token:1"},
            ],
            [{"text": "Không", "callback_data": "ms:select-token:2"}],
            [{"text": "✅ Xong", "callback_data": "ms:select-token:done"}],
        ]
    }
    assert telegram._MULTI_SELECTS["select-token"].choices == (
        "Có nôn",
        "Có vàng da",
        "Không",
    )


def test_telegram_send_text_removes_keyboard_when_no_choices(monkeypatch):
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

    monkeypatch.setattr(telegram, "TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setattr(telegram.httpx, "AsyncClient", Client)

    asyncio.run(telegram.send_text(123, "Tôi đã xóa ngữ cảnh hội thoại hiện tại."))

    assert calls[-1]["reply_markup"] == {"remove_keyboard": True}


def test_telegram_multi_select_callback_toggles_and_submits(monkeypatch):
    answers: list[tuple[str, str]] = []
    edits: list[tuple[int | str, int, dict]] = []
    sent: list[tuple[int | str, str]] = []
    background_calls: list[tuple[object, tuple]] = []

    class Background:
        def add_task(self, func, *args):
            background_calls.append((func, args))

    async def fake_answer_callback_query(callback_query_id: str, text: str) -> None:
        answers.append((callback_query_id, text))

    async def fake_edit_message_reply_markup(chat_id: int | str, message_id: int, reply_markup: dict) -> None:
        edits.append((chat_id, message_id, reply_markup))

    async def fake_send_text(chat_id: int | str, text: str, choices=(), selection_mode="single") -> None:
        sent.append((chat_id, text))

    monkeypatch.setattr(telegram, "_answer_callback_query", fake_answer_callback_query, raising=False)
    monkeypatch.setattr(telegram, "_edit_message_reply_markup", fake_edit_message_reply_markup, raising=False)
    monkeypatch.setattr(telegram, "send_text", fake_send_text, raising=False)
    telegram._MULTI_SELECTS.clear()
    telegram._MULTI_SELECTS["select-token"] = telegram.MultiSelectState(
        choices=("Có nôn", "Có vàng da", "Không"),
        selected=set(),
    )

    toggle_update = {
        "id": "cb-1",
        "data": "ms:select-token:1",
        "message": {"message_id": 456, "chat": {"id": 123}},
    }
    done_update = {
        "id": "cb-2",
        "data": "ms:select-token:done",
        "message": {"message_id": 456, "chat": {"id": 123}},
    }
    background = Background()

    assert asyncio.run(telegram._handle_multi_select_callback(toggle_update, background)) is True
    assert telegram._MULTI_SELECTS["select-token"].selected == {1}
    assert edits[-1][2]["inline_keyboard"][0][1]["text"] == "✓ Có vàng da"

    assert asyncio.run(telegram._handle_multi_select_callback(done_update, background)) is True

    assert answers[-1] == ("cb-2", "Đã chọn.")
    assert "select-token" not in telegram._MULTI_SELECTS
    assert sent == [(123, "Người dùng chọn: Có vàng da")]
    assert background_calls == [(telegram._answer_and_send, (123, "Có vàng da"))]


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


def test_telegram_webhook_records_rating_callback(app_client, monkeypatch):
    client, _ = app_client
    events: list[tuple[str, int | str, int | str]] = []
    ratings: list[tuple[str, int]] = []
    callback_answers: list[tuple[str, str]] = []
    deleted_messages: list[tuple[int | str, int]] = []
    monkeypatch.setattr(telegram, "TELEGRAM_WEBHOOK_SECRET", "")
    monkeypatch.setattr(telegram, "reserve_webhook_update", lambda channel, update_id: True)
    monkeypatch.setattr(telegram, "record_feedback_rating", lambda token, rating: ratings.append((token, rating)) or True, raising=False)

    async def fake_answer_callback_query(callback_query_id: str, text: str) -> None:
        events.append(("answer_callback", callback_query_id, text))
        callback_answers.append((callback_query_id, text))

    async def fake_delete_rating_message(chat_id: int | str, message_id: int) -> None:
        events.append(("delete_message", chat_id, message_id))
        deleted_messages.append((chat_id, message_id))

    async def fail_if_called(chat_id: int | str, text: str) -> None:
        raise AssertionError("rating callbacks should not trigger a chatbot answer")

    monkeypatch.setattr(telegram, "_answer_callback_query", fake_answer_callback_query, raising=False)
    monkeypatch.setattr(telegram, "_delete_rating_message", fake_delete_rating_message, raising=False)
    monkeypatch.setattr(telegram, "_answer_and_send", fail_if_called)

    response = client.post(
        "/webhook/telegram",
        json={
            "update_id": 13,
            "callback_query": {
                "id": "callback-1",
                "data": "rate:rating-token:4",
                "message": {"message_id": 456, "chat": {"id": 123}},
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert ratings == [("rating-token", 4)]
    assert callback_answers == [("callback-1", "Cảm ơn bạn đã đánh giá.")]
    assert deleted_messages == [(123, 456)]
    assert events == [
        ("answer_callback", "callback-1", "Cảm ơn bạn đã đánh giá."),
        ("delete_message", 123, 456),
    ]


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
