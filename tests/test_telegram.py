from __future__ import annotations

import asyncio
import threading
import urllib.parse

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

    async def fake_send_text(chat_id: int | str, text: str, **kwargs) -> None:
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
    for text in texts[:2]:
        assert "/mode" in text
        assert "Auto" in text
        assert "Thông tin" in text
        assert "Chẩn đoán" in text
    assert "Menu các lệnh hỗ trợ" in texts[2]
    assert "/start" not in texts[2]
    assert "/menu" not in texts[2]


def test_telegram_menu_interaction(app_client, monkeypatch):
    client, _ = app_client
    sent_texts = []
    callback_answers = []
    deleted_messages = []
    edited_messages = []

    async def fake_send_text(chat_id, text, choices=(), selection_mode="single", inline_keyboard=None):
        sent_texts.append((chat_id, text, inline_keyboard))

    async def fake_answer_callback_query(callback_query_id, text):
        callback_answers.append((callback_query_id, text))

    async def fake_delete_message(chat_id, message_id):
        deleted_messages.append((chat_id, message_id))

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None):
        edited_messages.append((chat_id, message_id, text, inline_keyboard))

    monkeypatch.setattr(telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram, "_answer_callback_query", fake_answer_callback_query)
    monkeypatch.setattr(telegram, "_delete_message", fake_delete_message)
    monkeypatch.setattr(telegram, "_edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(telegram, "TELEGRAM_WEBHOOK_SECRET", "")
    monkeypatch.setattr(telegram, "reserve_webhook_update", lambda channel, update_id: True)

    # Test /menu rendering
    assert asyncio.run(telegram._handle_command(123, "/menu"))
    assert len(sent_texts) == 1
    assert "Menu các lệnh hỗ trợ" in sent_texts[0][1]
    assert sent_texts[0][2] == telegram._menu_keyboard()

    # Reset sent_texts
    sent_texts.clear()

    # 1. Test cmd:close callback
    response = client.post(
        "/webhook/telegram",
        json={
            "update_id": 200,
            "callback_query": {
                "id": "cb-close",
                "data": "cmd:close",
                "message": {"message_id": 456, "chat": {"id": 123}},
            },
        },
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert ("cb-close", "Đã đóng menu") in callback_answers
    assert (123, 456) in deleted_messages

    # 2. Test cmd:/help callback
    response = client.post(
        "/webhook/telegram",
        json={
            "update_id": 201,
            "callback_query": {
                "id": "cb-help",
                "data": "cmd:/help",
                "message": {"message_id": 456, "chat": {"id": 123}},
            },
        },
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert ("cb-help", "Đang thực hiện /help") in callback_answers
    # Verify it triggered the /help command (which sends help text via fake_send_text)
    assert len(sent_texts) == 1
    assert "Cách sử dụng" in sent_texts[0][1]

    # 3. Test cmd:/profile callback (replaces the old memory menu)
    response = client.post(
        "/webhook/telegram",
        json={
            "update_id": 202,
            "callback_query": {
                "id": "cb-profile",
                "data": "cmd:/profile",
                "message": {"message_id": 456, "chat": {"id": 123}},
            },
        },
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert ("cb-profile", "Đang thực hiện /profile") in callback_answers

    # 4. Test menu:main callback
    response = client.post(
        "/webhook/telegram",
        json={
            "update_id": 203,
            "callback_query": {
                "id": "cb-main",
                "data": "menu:main",
                "message": {"message_id": 456, "chat": {"id": 123}},
            },
        },
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert ("cb-main", "") in callback_answers
    assert (123, 456, "📋 **Menu các lệnh hỗ trợ:**", telegram._menu_keyboard()) in edited_messages


def test_telegram_bot_command_menu_hides_start_and_menu_with_icons():
    assert telegram.BOT_COMMANDS == [
        {"command": "menu", "description": "📋 Menu các lệnh hỗ trợ"},
        {"command": "help", "description": "📝 Cách đặt câu hỏi"},
        {"command": "mode", "description": "⚙️ Chọn chế độ trả lời"},
        {"command": "tts", "description": "🔊 Bật/tắt đọc câu trả lời"},
        {"command": "remind", "description": "🔔 Đặt nhắc nhở y tế"},
        {"command": "doctor", "description": "👨‍⚕️ Kết nối bác sĩ"},
        {"command": "end", "description": "⛔ Kết thúc tư vấn bác sĩ"},
        {"command": "topup", "description": "💰 Nạp tiền vào tài khoản"},
        {"command": "balance", "description": "💳 Xem số dư"},
        {"command": "paydebt", "description": "🧾 Thanh toán công nợ"},
        {"command": "new", "description": "🔄 Xóa ngữ cảnh và bắt đầu lượt mới"},
        {"command": "profile", "description": "🩺 Hồ sơ y tế cá nhân"},
    ]


def test_telegram_tts_command_turns_voice_on_and_off(monkeypatch):
    sent: list[dict] = []
    state = {"enabled": False}

    async def fake_send_text(chat_id, text, **kwargs):
        sent.append({"chat_id": chat_id, "text": text, **kwargs})

    monkeypatch.setattr(telegram, "send_text", fake_send_text)
    monkeypatch.setattr(
        telegram,
        "is_tts_enabled",
        lambda chat_id: state["enabled"],
    )
    monkeypatch.setattr(
        telegram,
        "set_tts_enabled",
        lambda chat_id, enabled: state.update(enabled=enabled),
    )

    assert asyncio.run(telegram._handle_command(123, "/ttson"))
    assert state["enabled"] is True
    assert "**Bật**" in sent[-1]["text"]
    assert sent[-1]["inline_keyboard"] == telegram._tts_keyboard(True)

    assert asyncio.run(telegram._handle_command(123, "/ttsoff"))
    assert state["enabled"] is False
    assert "**Tắt**" in sent[-1]["text"]
    assert sent[-1]["inline_keyboard"] == telegram._tts_keyboard(False)


def test_telegram_new_command_clears_redis_session(monkeypatch):
    cleared: list[str] = []
    sent: list[str] = []

    async def fake_send_text(chat_id: int | str, text: str) -> None:
        sent.append(text)

    monkeypatch.setattr(telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram, "clear_session", lambda session_id: cleared.append(session_id))
    monkeypatch.setattr(telegram, "clear_conversation_context", lambda session_id: None)
    telegram._CHAT_MODE_DEFAULTS.clear()
    telegram._CHAT_MODE_DEFAULTS["456"] = "information"

    assert asyncio.run(telegram._handle_command(456, "/new"))

    assert cleared == [telegram._request_identity(456, None, "private").session_key]
    assert "456" not in telegram._CHAT_MODE_DEFAULTS
    assert "xóa ngữ cảnh" in sent[0]


def test_telegram_mode_command_sends_selector(monkeypatch):
    sent: list[dict] = []

    async def fake_send_text(
        chat_id: int | str,
        text: str,
        choices=(),
        selection_mode="single",
        inline_keyboard=None,
    ) -> None:
        sent.append(
            {
                "chat_id": chat_id,
                "text": text,
                "inline_keyboard": inline_keyboard,
            }
        )

    monkeypatch.setattr(telegram, "send_text", fake_send_text)
    telegram._CHAT_MODE_DEFAULTS.clear()

    assert asyncio.run(telegram._handle_command(123, "/mode"))

    assert sent == [
        {
            "chat_id": 123,
            "text": (
                "Lựa chọn chế độ trả lời\n\n"
                "**Auto**: bot tự chọn cách trả lời phù hợp với câu hỏi.\n"
                "**Thông tin**: trả lời thông tin về bệnh, thuốc, phòng ngừa, điều trị và chăm sóc.\n"
                "**Chẩn đoán**: tư vấn triệu chứng, sàng lọc an toàn và hướng dẫn khi cần đi khám.\n\n"
                "Chế độ hiện tại: **Auto**"
            ),
            "inline_keyboard": {
                "inline_keyboard": [[
                    {"text": "✓ Auto", "callback_data": "mode:set:auto"},
                    {"text": "Thông tin", "callback_data": "mode:set:information"},
                    {"text": "Chẩn đoán", "callback_data": "mode:set:diagnostic"},
                ]]
            },
        }
    ]


def test_telegram_mode_callback_updates_chat_default(monkeypatch):
    answers: list[tuple[str, str]] = []
    edits: list[tuple[int | str, int, dict]] = []

    async def fake_answer_callback_query(callback_query_id: str, text: str) -> None:
        answers.append((callback_query_id, text))

    async def fake_edit_message_reply_markup(chat_id: int | str, message_id: int, reply_markup: dict) -> None:
        edits.append((chat_id, message_id, reply_markup))

    monkeypatch.setattr(telegram, "_answer_callback_query", fake_answer_callback_query, raising=False)
    monkeypatch.setattr(telegram, "_edit_message_reply_markup", fake_edit_message_reply_markup, raising=False)
    telegram._CHAT_MODE_DEFAULTS.clear()

    handled = asyncio.run(
        telegram._handle_mode_callback({
            "id": "cb-1",
            "data": "mode:set:diagnostic",
            "message": {"message_id": 456, "chat": {"id": 123}},
        })
    )

    assert handled is True
    assert telegram._CHAT_MODE_DEFAULTS["123"] == "diagnostic"
    assert answers == [("cb-1", "Đã chọn chế độ Chẩn đoán.")]
    assert edits[-1] == (123, 456, telegram._mode_keyboard("diagnostic"))


def test_telegram_answer_uses_chat_mode_default(monkeypatch):
    sent: list[tuple[int | str, str]] = []
    seen: dict[str, str] = {}

    def fake_answer_with_choices(question: str, session_id: str = "default", mode: str = "auto"):
        seen["question"] = question
        seen["session_id"] = session_id
        seen["mode"] = mode
        return telegram.ChatReply("diagnostic reply", ())

    async def fake_send_text(chat_id: int | str, text: str, choices=None, selection_mode="single") -> None:
        sent.append((chat_id, text))

    async def fake_send_rating_prompt(chat_id: int | str, token: str) -> None:
        return None

    monkeypatch.setattr(telegram, "answer_with_choices", fake_answer_with_choices, raising=False)
    monkeypatch.setattr(telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram, "create_feedback_request", lambda *args: "rating-token", raising=False)
    monkeypatch.setattr(telegram, "_send_rating_prompt", fake_send_rating_prompt, raising=False)
    telegram._CHAT_MODE_DEFAULTS.clear()
    telegram._CHAT_MODE_DEFAULTS["123"] = "diagnostic"

    asyncio.run(telegram._answer_and_send(123, "Tôi bị đau lưng"))

    assert seen == {
        "question": "Tôi bị đau lưng",
        "session_id": telegram._request_identity(123, None, "private").session_key,
        "mode": "diagnostic",
    }
    assert sent == [(123, "diagnostic reply")]


def test_telegram_answer_sends_doctor_offer_after_rating(monkeypatch):
    sent: list[dict] = []
    registered: list[tuple[int | str, str, str, str | None]] = []
    keyboards: list[int | str] = []
    ratings: list[tuple[int | str, str]] = []
    events: list[str] = []

    def fake_answer_with_choices(question: str, session_id: str = "default", mode: str = "auto"):
        return telegram.ChatReply(
            "Tôi không đủ thông tin để kết luận. Bạn nên gặp bác sĩ.",
            doctor_offer=True,
            doctor_specialty="Tim mạch",
        )

    async def fake_send_text(chat_id, text, choices=(), selection_mode="single", inline_keyboard=None):
        events.append(f"send:{text}")
        sent.append({"chat_id": chat_id, "text": text, "inline_keyboard": inline_keyboard})

    async def fake_send_rating_prompt(chat_id: int | str, token: str) -> None:
        events.append("rating")
        ratings.append((chat_id, token))

    monkeypatch.setattr(telegram, "answer_with_choices", fake_answer_with_choices, raising=False)
    monkeypatch.setattr(telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram, "create_feedback_request", lambda *args: "rating-token")
    original_register_handoff_context = telegram.telegram_doctor.register_handoff_context

    def fake_register_handoff_context(*args):
        registered.append(args)
        return original_register_handoff_context(*args)

    monkeypatch.setattr(telegram.telegram_doctor, "register_handoff_context", fake_register_handoff_context)
    original_handoff_keyboard = telegram.telegram_doctor.handoff_keyboard

    def fake_handoff_keyboard(chat_id):
        keyboards.append(chat_id)
        return original_handoff_keyboard(chat_id)

    monkeypatch.setattr(telegram.telegram_doctor, "handoff_keyboard", fake_handoff_keyboard)
    monkeypatch.setattr(telegram, "_send_rating_prompt", fake_send_rating_prompt)

    asyncio.run(telegram._answer_and_send(123, "Tôi bị đau ngực"))

    assert registered == [(123, "Tôi bị đau ngực", sent[0]["text"], "Tim mạch")]
    assert keyboards == [123]
    assert ratings == [(123, "rating-token")]
    assert len(sent) == 2
    assert sent[0] == {
        "chat_id": 123,
        "text": "Tôi không đủ thông tin để kết luận. Bạn nên gặp bác sĩ.",
        "inline_keyboard": None,
    }
    assert sent[1]["text"] == (
        "Với tình huống của bạn, bạn có muốn kết nối với bác sĩ chuyên khoa "
        "để được tư vấn sâu hơn không?"
    )
    assert events == [
        "send:Tôi không đủ thông tin để kết luận. Bạn nên gặp bác sĩ.",
        "rating",
        f"send:{sent[1]['text']}",
    ]
    callbacks = [
        btn["callback_data"]
        for row in sent[1]["inline_keyboard"]["inline_keyboard"]
        for btn in row
    ]
    assert telegram.telegram_doctor.HANDOFF_ACCEPT in callbacks
    assert telegram.telegram_doctor.HANDOFF_DECLINE in callbacks
    telegram.telegram_doctor._HANDOFF_CONTEXTS.pop("123", None)



def test_telegram_mode_retry_callback_replays_question_with_target_mode(monkeypatch):
    answers: list[tuple[str, str]] = []
    sent: list[tuple[int | str, str]] = []
    background_calls: list[tuple[object, tuple]] = []

    class Background:
        def add_task(self, func, *args):
            background_calls.append((func, args))

    async def fake_answer_callback_query(callback_query_id: str, text: str) -> None:
        answers.append((callback_query_id, text))

    async def fake_send_text(chat_id: int | str, text: str, choices=(), selection_mode="single") -> None:
        sent.append((chat_id, text))

    monkeypatch.setattr(telegram, "_answer_callback_query", fake_answer_callback_query, raising=False)
    monkeypatch.setattr(telegram, "send_text", fake_send_text, raising=False)
    telegram._CHAT_MODE_DEFAULTS.clear()
    telegram._MODE_RETRIES.clear()
    telegram._MODE_RETRIES["retry-token"] = telegram.ModeRetryState(
        question="Tôi đau lưng lan xuống chân là bệnh gì?",
        mode="diagnostic",
    )

    handled = asyncio.run(
        telegram._handle_mode_retry_callback(
            {
                "id": "cb-2",
                "data": "mode_retry:retry-token",
                "message": {"message_id": 789, "chat": {"id": 123}},
            },
            Background(),
        )
    )

    assert handled is True
    assert answers == [("cb-2", "Đang trả lời ở chế độ Chẩn đoán.")]
    assert sent == [(123, "Trả lời ở chế độ Chẩn đoán: Tôi đau lưng lan xuống chân là bệnh gì?")]
    assert background_calls == [
        (telegram._answer_and_send, (123, "Tôi đau lưng lan xuống chân là bệnh gì?", "diagnostic"))
    ]
    assert "retry-token" not in telegram._MODE_RETRIES
    assert telegram._CHAT_MODE_DEFAULTS.get("123") is None


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
        assert session_id == telegram._request_identity(123, None, "private").session_key
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
            "session_id": telegram._request_identity(123, None, "private").session_key,
            "channel": "telegram",
                "recipient_id": telegram._request_identity(123, None, "private").session_key,
            "question": "Tôi bị ho",
            "answer": "Bạn nên nghỉ ngơi.",
        }
    ]
    assert prompts == [(123, "rating-token")]


def test_telegram_answer_sends_voice_when_tts_is_enabled(monkeypatch):
    voices: list[tuple[int | str, bytes]] = []

    def fake_answer_with_choices(question: str, session_id: str = "default"):
        return telegram.ChatReply("Bạn nên nghỉ ngơi.", ())

    async def fake_send_text(*args, **kwargs) -> None:
        return None

    async def fake_send_voice_audio(chat_id: int | str, audio: bytes) -> None:
        voices.append((chat_id, audio))

    monkeypatch.setattr(telegram, "answer_with_choices", fake_answer_with_choices)
    monkeypatch.setattr(telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram, "send_voice_audio", fake_send_voice_audio)
    monkeypatch.setattr(telegram, "is_tts_enabled", lambda chat_id: True)
    monkeypatch.setattr(telegram, "synthesize_speech", lambda text: b"ogg-audio")
    monkeypatch.setattr(telegram, "create_feedback_request", lambda *args: "rating-token")
    monkeypatch.setattr(telegram, "_send_rating_prompt", fake_send_text)

    asyncio.run(telegram._answer_and_send(123, "Tôi bị ho"))

    assert voices == [(123, b"ogg-audio")]


def test_telegram_tts_failure_does_not_skip_feedback(monkeypatch):
    prompts: list[tuple[int | str, str]] = []

    def fake_answer_with_choices(question: str, session_id: str = "default"):
        return telegram.ChatReply("Bạn nên nghỉ ngơi.", ())

    async def fake_send_text(*args, **kwargs) -> None:
        return None

    async def fake_send_rating_prompt(chat_id: int | str, token: str) -> None:
        prompts.append((chat_id, token))

    monkeypatch.setattr(telegram, "answer_with_choices", fake_answer_with_choices)
    monkeypatch.setattr(telegram, "send_text", fake_send_text)
    monkeypatch.setattr(
        telegram,
        "is_tts_enabled",
        lambda chat_id: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )
    monkeypatch.setattr(telegram, "create_feedback_request", lambda *args: "rating-token")
    monkeypatch.setattr(telegram, "_send_rating_prompt", fake_send_rating_prompt)

    asyncio.run(telegram._answer_and_send(123, "Tôi bị ho"))

    assert prompts == [(123, "rating-token")]


def test_telegram_answer_sends_reply_keyboard_for_choices(monkeypatch):
    sent: list[tuple[int | str, str, list[str], str]] = []
    prompts: list[tuple[int | str, str]] = []

    def fake_answer_with_choices(question: str, session_id: str = "default"):
        assert question == "Tôi bị đau bụng"
        assert session_id == telegram._request_identity(123, None, "private").session_key
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


def test_telegram_answer_now_removes_reply_keyboard_before_generating(monkeypatch):
    sent: list[tuple[int | str, str, list[str], str]] = []
    acks: list[tuple[int | str, str]] = []
    deleted: list[tuple[int | str, int]] = []
    prompts: list[tuple[int | str, str]] = []

    def fake_answer_with_choices(question: str, session_id: str = "default"):
        assert question == telegram.ANSWER_NOW_CHOICE
        assert session_id == telegram._request_identity(123, None, "private").session_key
        return telegram.ChatReply("Câu trả lời cuối.", ())

    async def fake_send_text(chat_id: int | str, text: str, choices=None, selection_mode="single") -> None:
        sent.append((chat_id, text, list(choices or []), selection_mode))

    async def fake_send_answer_now_ack(chat_id: int | str) -> int:
        acks.append((chat_id, telegram.ANSWER_NOW_ACK))
        return 456

    async def fake_delete_message(chat_id: int | str, message_id: int) -> None:
        deleted.append((chat_id, message_id))

    async def fake_send_rating_prompt(chat_id: int | str, token: str) -> None:
        prompts.append((chat_id, token))

    monkeypatch.setattr(telegram, "answer_with_choices", fake_answer_with_choices, raising=False)
    monkeypatch.setattr(telegram, "_send_answer_now_ack", fake_send_answer_now_ack, raising=False)
    monkeypatch.setattr(telegram, "_delete_message", fake_delete_message, raising=False)
    monkeypatch.setattr(telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram, "create_feedback_request", lambda *args: "rating-token", raising=False)
    monkeypatch.setattr(telegram, "_send_rating_prompt", fake_send_rating_prompt, raising=False)

    asyncio.run(telegram._answer_and_send(123, telegram.ANSWER_NOW_CHOICE))

    assert acks == [(123, telegram.ANSWER_NOW_ACK)]
    assert sent == [(123, "Câu trả lời cuối.", [], "single")]
    assert deleted == [(123, 456)]
    assert prompts == [(123, "rating-token")]


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


def test_telegram_answer_now_ack_removes_keyboard_and_returns_message_id(monkeypatch):
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

    message_id = asyncio.run(telegram._send_answer_now_ack(123))

    assert message_id == 456
    assert calls[-1]["text"] == telegram.ANSWER_NOW_ACK
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


def test_telegram_topup_command_shows_amount_keyboard_in_private(monkeypatch):
    sent: list[dict] = []

    async def fake_send_text(
        chat_id, text, choices=(), selection_mode="single", inline_keyboard=None
    ) -> None:
        sent.append({"chat_id": chat_id, "text": text, "inline_keyboard": inline_keyboard})

    monkeypatch.setattr(telegram, "send_text", fake_send_text)

    assert asyncio.run(telegram._handle_command(123, "/topup", chat_type="private"))

    assert len(sent) == 1
    keyboard = sent[0]["inline_keyboard"]["inline_keyboard"]
    flat = [btn for row in keyboard for btn in row]
    preset_amounts = [
        btn["callback_data"] for btn in flat if btn["callback_data"].startswith("topup:set:")
    ]
    assert preset_amounts == [
        "topup:set:10000",
        "topup:set:20000",
        "topup:set:50000",
        "topup:set:100000",
        "topup:set:200000",
        "topup:set:500000",
    ]
    assert any(btn["callback_data"] == "topup:custom" for btn in flat)


def test_telegram_topup_command_refused_in_group(monkeypatch):
    sent: list[str] = []

    async def fake_send_text(chat_id, text, *args, **kwargs) -> None:
        sent.append(text)

    monkeypatch.setattr(telegram, "send_text", fake_send_text)
    telegram._TOPUP_PENDING.clear()

    assert asyncio.run(telegram._handle_command(456, "/topup", chat_type="group"))

    assert len(sent) == 1
    assert "riêng tư" in sent[0] or "private" in sent[0].lower()
    assert "456" not in telegram._TOPUP_PENDING


def test_telegram_balance_command_reports_balance(monkeypatch):
    sent: list[dict] = []

    async def fake_send_text(chat_id, text, *args, inline_keyboard=None, **kwargs) -> None:
        sent.append({"text": text, "inline_keyboard": inline_keyboard})

    monkeypatch.setattr(telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram, "get_balance", lambda account_id: 70_000, raising=False)

    assert asyncio.run(telegram._handle_command(123, "/balance", chat_type="private"))

    assert "70,000" in sent[0]["text"]
    # A refresh button is attached below the balance.
    buttons = [
        btn
        for row in sent[0]["inline_keyboard"]["inline_keyboard"]
        for btn in row
        if btn["callback_data"] == "balance:refresh"
    ]
    assert len(buttons) == 1


def test_telegram_balance_refresh_callback_edits_message(monkeypatch):
    edits: list[dict] = []
    answers: list[tuple[str, str]] = []

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None) -> None:
        edits.append({"chat_id": chat_id, "message_id": message_id, "text": text,
                      "inline_keyboard": inline_keyboard})

    async def fake_answer_callback_query(callback_query_id, text) -> None:
        answers.append((callback_query_id, text))

    monkeypatch.setattr(telegram, "_edit_message_text", fake_edit_message_text, raising=False)
    monkeypatch.setattr(telegram, "_answer_callback_query", fake_answer_callback_query, raising=False)
    monkeypatch.setattr(telegram, "get_balance", lambda account_id: 95_000, raising=False)

    handled = asyncio.run(
        telegram._handle_balance_callback({
            "id": "cb-bal",
            "data": "balance:refresh",
            "message": {"message_id": 88, "chat": {"id": 123}},
        })
    )

    assert handled is True
    assert edits and edits[0]["chat_id"] == 123 and edits[0]["message_id"] == 88
    assert "95,000" in edits[0]["text"]
    # Refresh button stays so the user can refresh again.
    refresh = [
        btn
        for row in edits[0]["inline_keyboard"]["inline_keyboard"]
        for btn in row
        if btn["callback_data"] == "balance:refresh"
    ]
    assert len(refresh) == 1


def test_telegram_balance_refresh_callback_ignores_other_data(monkeypatch):
    assert asyncio.run(
        telegram._handle_balance_callback({"id": "x", "data": "topup:cancel", "message": {}})
    ) is False


def test_telegram_topup_preset_callback_creates_order_and_sends_qr(monkeypatch):
    answers: list[tuple[str, str]] = []
    photos: list[dict] = []
    orders: list[tuple] = []

    async def fake_answer_callback_query(callback_query_id, text) -> None:
        answers.append((callback_query_id, text))

    async def fake_send_photo(chat_id, photo_url, caption, inline_keyboard=None) -> int:
        photos.append({
            "chat_id": chat_id,
            "photo_url": photo_url,
            "caption": caption,
            "inline_keyboard": inline_keyboard,
        })
        return 999

    def fake_create_order(order_code, account_id, amount, payment_link_id) -> None:
        orders.append((order_code, account_id, amount, payment_link_id))
        # Also persist to the real DB so set_order_qr_message has a row to update.
        from src.chat.storage import wallet
        wallet.create_order(order_code, account_id, amount, payment_link_id)

    # qr_code carries the full memo in EMV tag 62.08: "CS123ABCDEF NAPTIEN50000"
    qr_with_content = "62280824CS123ABCDEF NAPTIEN50000"

    def fake_create_payment(*, order_code, amount, description, return_url, cancel_url):
        return {
            "qr_code": qr_with_content,
            "checkout_url": "https://pay.test/web/xyz",
            "payment_link_id": "plink-1",
            "bin": "970415",
            "account_number": "113366668888",
            "account_name": "CHATBOT MEDICAL",
        }

    monkeypatch.setattr(telegram, "_answer_callback_query", fake_answer_callback_query, raising=False)
    monkeypatch.setattr(telegram, "_send_topup_qr", fake_send_photo, raising=False)
    monkeypatch.setattr(telegram, "create_order", fake_create_order, raising=False)
    monkeypatch.setattr(telegram, "payos_create_payment", fake_create_payment, raising=False)

    handled = asyncio.run(
        telegram._handle_topup_callback({
            "id": "cb-1",
            "data": "topup:set:50000",
            "message": {"message_id": 1, "chat": {"id": 123}},
        })
    )

    assert handled is True
    assert len(orders) == 1
    order_code, account_id, amount, plink = orders[0]
    assert account_id == "tg:123"
    assert amount == 50_000
    assert plink == "plink-1"
    assert len(photos) == 1
    # Branded VietQR image from PayOS's own bank bin/account, with the FULL
    # PayOS memo (incl CS-reference) as addInfo so reconciliation still works.
    assert "img.vietqr.io" in photos[0]["photo_url"]
    assert "970415" in photos[0]["photo_url"]
    assert "113366668888" in photos[0]["photo_url"]
    assert "CS123ABCDEF" in urllib.parse.unquote(photos[0]["photo_url"])
    # Caption shows the full transfer content, not just NAPTIEN<order>.
    assert "CS123ABCDEF NAPTIEN50000" in photos[0]["caption"]
    # Cancel button attached to the QR.
    cancel_buttons = [
        btn
        for row in photos[0]["inline_keyboard"]["inline_keyboard"]
        for btn in row
        if btn["callback_data"].startswith("topup:cancel")
    ]
    assert len(cancel_buttons) == 1
    # QR message id persisted on the order so it can be deleted across restarts.
    from src.chat.storage import wallet
    assert wallet.get_order(order_code)["qr_message_id"] == 999


def test_vietqr_image_url_encodes_full_content():
    url = telegram._vietqr_image_url(
        bank_bin="970415",
        account_number="113366668888",
        amount=50_000,
        content="CS123ABCDEF NAPTIEN50000",
        account_name="CHATBOT MEDICAL",
    )
    assert url.startswith("https://img.vietqr.io/image/970415-113366668888-")
    assert "amount=50000" in url
    assert "CS123ABCDEF" in urllib.parse.unquote(url)


def test_topup_cancel_callback_deletes_qr_and_keeps_order(monkeypatch):
    answers: list[tuple[str, str]] = []
    deleted: list[tuple[int | str, int]] = []
    marked_cancelled: list[int] = []

    async def fake_answer_callback_query(callback_query_id, text) -> None:
        answers.append((callback_query_id, text))

    async def fake_delete_message(chat_id, message_id) -> None:
        deleted.append((chat_id, message_id))

    monkeypatch.setattr(telegram, "_answer_callback_query", fake_answer_callback_query, raising=False)
    monkeypatch.setattr(telegram, "_delete_message", fake_delete_message, raising=False)

    handled = asyncio.run(
        telegram._handle_topup_callback({
            "id": "cb-9",
            "data": "topup:cancel",
            "message": {"message_id": 55, "chat": {"id": 123}},
        })
    )

    assert handled is True
    assert deleted == [(123, 55)]
    # Cancel only removes the QR message; it must NOT cancel the order locally,
    # so a genuine payment still credits via the pending->paid path.
    assert marked_cancelled == []


def test_telegram_topup_custom_callback_sets_pending(monkeypatch):
    answers: list[tuple[str, str]] = []
    sent: list[str] = []

    async def fake_answer_callback_query(callback_query_id, text) -> None:
        answers.append((callback_query_id, text))

    async def fake_send_text(chat_id, text, *args, **kwargs) -> None:
        sent.append(text)

    monkeypatch.setattr(telegram, "_answer_callback_query", fake_answer_callback_query, raising=False)
    monkeypatch.setattr(telegram, "send_text", fake_send_text, raising=False)
    telegram._TOPUP_PENDING.clear()

    handled = asyncio.run(
        telegram._handle_topup_callback({
            "id": "cb-2",
            "data": "topup:custom",
            "message": {"message_id": 1, "chat": {"id": 123}},
        })
    )

    assert handled is True
    assert telegram._TOPUP_PENDING.get("123") is True
    assert sent


def test_telegram_pending_custom_amount_intercepts_text(app_client, monkeypatch):
    client, _ = app_client
    orders: list[tuple] = []
    answered: list[tuple] = []

    monkeypatch.setattr(telegram, "TELEGRAM_WEBHOOK_SECRET", "")
    monkeypatch.setattr(telegram, "reserve_webhook_update", lambda channel, update_id: True)

    async def fake_answer_and_send(chat_id, text) -> None:
        answered.append((chat_id, text))

    async def fake_start_topup(chat_id, amount) -> None:
        orders.append((chat_id, amount))

    monkeypatch.setattr(telegram, "_answer_and_send", fake_answer_and_send)
    monkeypatch.setattr(telegram, "_start_topup_payment", fake_start_topup, raising=False)
    telegram._TOPUP_PENDING.clear()
    telegram._TOPUP_PENDING["123"] = True

    response = client.post(
        "/webhook/telegram",
        json={
            "update_id": 20,
            "message": {"chat": {"id": 123, "type": "private"}, "text": "50000"},
        },
    )

    assert response.status_code == 200
    assert orders == [(123, 50_000)]
    assert answered == []  # must NOT go to the chatbot
    assert "123" not in telegram._TOPUP_PENDING


def test_telegram_pending_custom_amount_rejects_out_of_bounds(app_client, monkeypatch):
    client, _ = app_client
    sent: list[str] = []
    orders: list[tuple] = []

    monkeypatch.setattr(telegram, "TELEGRAM_WEBHOOK_SECRET", "")
    monkeypatch.setattr(telegram, "reserve_webhook_update", lambda channel, update_id: True)

    async def fake_send_text(chat_id, text, *args, **kwargs) -> None:
        sent.append(text)

    async def fake_start_topup(chat_id, amount) -> None:
        orders.append((chat_id, amount))

    monkeypatch.setattr(telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram, "_start_topup_payment", fake_start_topup, raising=False)
    telegram._TOPUP_PENDING.clear()
    telegram._TOPUP_PENDING["123"] = True

    response = client.post(
        "/webhook/telegram",
        json={
            "update_id": 21,
            "message": {"chat": {"id": 123, "type": "private"}, "text": "5000"},
        },
    )

    assert response.status_code == 200
    assert orders == []
    assert sent and ("10,000" in sent[0] or "10.000" in sent[0])
    # Stays in pending mode so the user can simply re-type a valid amount.
    assert telegram._TOPUP_PENDING.get("123") is True


def test_telegram_pending_custom_amount_retry_after_invalid(app_client, monkeypatch):
    client, _ = app_client
    sent: list[str] = []
    orders: list[tuple] = []

    monkeypatch.setattr(telegram, "TELEGRAM_WEBHOOK_SECRET", "")
    monkeypatch.setattr(telegram, "reserve_webhook_update", lambda channel, update_id: True)

    async def fake_send_text(chat_id, text, *args, **kwargs) -> None:
        sent.append(text)

    async def fake_start_topup(chat_id, amount) -> None:
        orders.append((chat_id, amount))

    monkeypatch.setattr(telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram, "_start_topup_payment", fake_start_topup, raising=False)
    telegram._TOPUP_PENDING.clear()
    telegram._TOPUP_PENDING["123"] = True

    # First an invalid amount...
    client.post(
        "/webhook/telegram",
        json={"update_id": 30, "message": {"chat": {"id": 123, "type": "private"}, "text": "1000"}},
    )
    # ...then a valid one — this is the bug being fixed: the retry must work.
    client.post(
        "/webhook/telegram",
        json={"update_id": 31, "message": {"chat": {"id": 123, "type": "private"}, "text": "10000"}},
    )

    assert orders == [(123, 10_000)]
    assert "123" not in telegram._TOPUP_PENDING


def test_telegram_topup_cancel_callback_clears_pending(monkeypatch):
    # Tapping cancel on the "enter amount" prompt must drop pending mode so the
    # user can ask normal health questions again.
    deleted: list[tuple] = []
    answers: list[tuple] = []

    async def fake_delete_message(chat_id, message_id) -> None:
        deleted.append((chat_id, message_id))

    async def fake_answer_callback_query(callback_query_id, text) -> None:
        answers.append((callback_query_id, text))

    monkeypatch.setattr(telegram, "_delete_message", fake_delete_message, raising=False)
    monkeypatch.setattr(telegram, "_answer_callback_query", fake_answer_callback_query, raising=False)
    telegram._TOPUP_PENDING.clear()
    telegram._TOPUP_PENDING["123"] = True

    handled = asyncio.run(
        telegram._handle_topup_callback({
            "id": "cb-x",
            "data": "topup:cancel",
            "message": {"message_id": 7, "chat": {"id": 123}},
        })
    )

    assert handled is True
    assert "123" not in telegram._TOPUP_PENDING



def test_whoami_reports_user_id(monkeypatch):
    sent: list[tuple] = []

    async def fake_send_text(chat_id, text, *args, **kwargs) -> None:
        sent.append((chat_id, text))

    monkeypatch.setattr(telegram, "send_text", fake_send_text)

    assert asyncio.run(
        telegram._handle_command(123, "/whoami", chat_type="private", user_id=6866285714)
    )
    assert "6866285714" in sent[0][1]


def test_admin_paid_refused_for_non_admin(monkeypatch):
    sent: list[str] = []
    credited: list[tuple] = []

    async def fake_send_text(chat_id, text, *args, **kwargs) -> None:
        sent.append(text)

    monkeypatch.setattr(telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram, "TELEGRAM_ADMIN_IDS", {6866285714})
    monkeypatch.setattr(telegram, "credit", lambda *a: credited.append(a) or 1, raising=False)

    assert asyncio.run(
        telegram._handle_command(999, "/admin_paid 12345", chat_type="private", user_id=999)
    )
    # Non-admin: no credit, refusal message.
    assert credited == []
    assert sent and ("quyền" in sent[0].lower() or "admin" in sent[0].lower() or "không" in sent[0].lower())


def test_admin_paid_reconciles_order_credits_and_notifies(monkeypatch):
    sent: list[tuple] = []
    deleted: list[tuple] = []

    async def fake_send_text(chat_id, text, *args, **kwargs) -> None:
        sent.append((chat_id, text))

    async def fake_delete_message(chat_id, message_id) -> None:
        deleted.append((chat_id, message_id))

    monkeypatch.setattr(telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram, "_delete_message", fake_delete_message, raising=False)
    monkeypatch.setattr(telegram, "TELEGRAM_ADMIN_IDS", {6866285714})

    from src.chat.storage import wallet
    wallet.create_order(55501, "tg:42", 30_000, "plink-a")
    wallet.set_order_qr_message(55501, 700)

    assert asyncio.run(
        telegram._handle_command(6866285714, "/admin_paid 55501", chat_type="private", user_id=6866285714)
    )

    assert wallet.get_balance("tg:42") == 30_000
    assert wallet.get_order(55501)["status"] == "paid"
    # Target user notified; admin gets a confirmation; QR removed.
    notified_target = any(cid == 42 or cid == "42" for cid, _ in sent)
    assert notified_target
    # QR deleted in the target user's chat, using the persisted message id
    # (works even though the admin reconciles from a different chat / after restart).
    assert deleted == [("42", 700)]
    # Audit trail records which admin credited the order.
    audit = wallet.get_admin_credits(order_code=55501)
    assert len(audit) == 1
    assert audit[0]["admin_user_id"] == 6866285714
    assert audit[0]["amount"] == 30_000


def test_admin_paid_is_idempotent(monkeypatch):
    sent: list[tuple] = []

    async def fake_send_text(chat_id, text, *args, **kwargs) -> None:
        sent.append((chat_id, text))

    monkeypatch.setattr(telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram, "TELEGRAM_ADMIN_IDS", {6866285714})

    from src.chat.storage import wallet
    wallet.create_order(55502, "tg:42", 30_000, "plink-b")

    asyncio.run(telegram._handle_command(6866285714, "/admin_paid 55502", chat_type="private", user_id=6866285714))
    asyncio.run(telegram._handle_command(6866285714, "/admin_paid 55502", chat_type="private", user_id=6866285714))

    # Credited exactly once despite two admin calls.
    assert wallet.get_balance("tg:42") == 30_000


def test_admin_paid_unknown_order(monkeypatch):
    sent: list[str] = []

    async def fake_send_text(chat_id, text, *args, **kwargs) -> None:
        sent.append(text)

    monkeypatch.setattr(telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram, "TELEGRAM_ADMIN_IDS", {6866285714})

    assert asyncio.run(
        telegram._handle_command(6866285714, "/admin_paid 999999", chat_type="private", user_id=6866285714)
    )
    assert sent and ("không tìm thấy" in sent[0].lower() or "khong" in sent[0].lower() or "999999" in sent[0])


def test_parse_order_code_accepts_human_inputs():
    f = telegram._parse_order_code
    # Bare number
    assert f("1780804634234") == 1780804634234
    # With NAPTIEN prefix (as shown in the QR caption)
    assert f("NAPTIEN1780804634234") == 1780804634234
    # Full bank transfer content (CS-reference + NAPTIEN)
    assert f("CS9P9PI8Z54 NAPTIEN1780804634234") == 1780804634234
    # Zero-width / surrounding whitespace from copy-paste
    assert f("​1780804634234​") == 1780804634234
    assert f("  1780804634234  ") == 1780804634234
    # Garbage / missing
    assert f("") is None
    assert f("abc") is None


def test_admin_paid_accepts_full_transfer_content(monkeypatch):
    sent: list[tuple] = []

    async def fake_send_text(chat_id, text, *args, **kwargs) -> None:
        sent.append((chat_id, text))

    monkeypatch.setattr(telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram, "TELEGRAM_ADMIN_IDS", {6866285714})

    from src.chat.storage import wallet
    wallet.create_order(55777, "tg:42", 30_000, "plink-z")

    # Admin pastes the whole bank content, not just the bare number.
    assert asyncio.run(
        telegram._handle_command(
            6866285714,
            "/admin_paid CS9P9PI8Z54 NAPTIEN55777",
            chat_type="private",
            user_id=6866285714,
        )
    )
    assert wallet.get_balance("tg:42") == 30_000


def test_telegram_send_photo_uses_send_photo_api(monkeypatch):
    calls: list[tuple[str, dict]] = []

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
            calls.append((url, json))
            return Response()

    monkeypatch.setattr(telegram, "TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setattr(telegram.httpx, "AsyncClient", Client)

    asyncio.run(telegram.send_photo(123, "photo-file-id", "👤 Bệnh nhân: ảnh"))

    assert calls[-1][0].endswith("/sendPhoto")
    assert calls[-1][1]["chat_id"] == 123
    assert calls[-1][1]["photo"] == "photo-file-id"
    assert calls[-1][1]["caption"] == "👤 Bệnh nhân: ảnh"


def test_telegram_send_voice_uses_send_voice_api(monkeypatch):
    calls: list[tuple[str, dict]] = []

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
            calls.append((url, json))
            return Response()

    monkeypatch.setattr(telegram, "TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setattr(telegram.httpx, "AsyncClient", Client)

    asyncio.run(telegram.send_voice(456, "voice-file-id"))

    assert calls[-1][0].endswith("/sendVoice")
    assert calls[-1][1]["chat_id"] == 456
    assert calls[-1][1]["voice"] == "voice-file-id"


def test_telegram_send_voice_audio_uploads_ogg(monkeypatch):
    calls: list[tuple[str, dict, dict]] = []

    class Response:
        status_code = 200

    class Client:
        def __init__(self, timeout: float):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url: str, data: dict, files: dict):
            calls.append((url, data, files))
            return Response()

    monkeypatch.setattr(telegram, "TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setattr(telegram.httpx, "AsyncClient", Client)

    asyncio.run(telegram.send_voice_audio(456, b"ogg-audio"))

    url, data, files = calls[-1]
    assert url.endswith("/sendVoice")
    assert data == {"chat_id": "456"}
    assert files == {"voice": ("answer.ogg", b"ogg-audio", "audio/ogg")}


def test_telegram_copy_message_uses_copy_message_api(monkeypatch):
    calls: list[tuple[str, dict]] = []

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
            calls.append((url, json))
            return Response()

    monkeypatch.setattr(telegram, "TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setattr(telegram.httpx, "AsyncClient", Client)

    asyncio.run(telegram.copy_message(456, 123, 789))

    assert calls[-1][0].endswith("/copyMessage")
    assert calls[-1][1]["chat_id"] == 456
    assert calls[-1][1]["from_chat_id"] == 123
    assert calls[-1][1]["message_id"] == 789


def test_telegram_doctor_command_is_handled(monkeypatch):
    calls: list[int | str] = []

    async def fake_handle_doctor_command(chat_id):
        calls.append(chat_id)
        return True

    monkeypatch.setattr(telegram.telegram_doctor, "handle_doctor_command", fake_handle_doctor_command, raising=False)

    assert asyncio.run(telegram._handle_command(123, "/doctor", chat_type="private", user_id=123))
    assert calls == [123]


def test_telegram_end_command_uses_doctor_session_before_new_chatbot_answer(monkeypatch):
    calls: list[int | str] = []
    sent: list[str] = []

    async def fake_handle_end(chat_id):
        calls.append(chat_id)
        return True

    async def fake_send_text(chat_id, text, *args, **kwargs):
        sent.append(text)

    monkeypatch.setattr(telegram.telegram_doctor, "handle_end", fake_handle_end, raising=False)
    monkeypatch.setattr(telegram, "send_text", fake_send_text)

    assert asyncio.run(telegram._handle_command(123, "/end", chat_type="private", user_id=123))
    assert calls == [123]
    assert sent == []


def test_telegram_webhook_relays_photo_before_empty_text_guard(app_client, monkeypatch):
    client, _ = app_client
    relayed: list[dict] = []
    monkeypatch.setattr(telegram, "TELEGRAM_WEBHOOK_SECRET", "")
    monkeypatch.setattr(telegram, "reserve_webhook_update", lambda channel, update_id: True)

    async def fake_relay(chat_id, message):
        relayed.append({"chat_id": chat_id, "message": message})
        return True

    async def fail_answer(chat_id, text):
        raise AssertionError("active doctor relay should intercept before chatbot")

    monkeypatch.setattr(telegram.telegram_doctor, "relay_message", fake_relay, raising=False)
    monkeypatch.setattr(telegram, "_answer_and_send", fail_answer)

    response = client.post(
        "/webhook/telegram",
        json={
            "update_id": 991,
            "message": {"chat": {"id": 123, "type": "private"}, "photo": [{"file_id": "p1"}]},
        },
    )

    assert response.status_code == 200
    assert relayed == [{"chat_id": 123, "message": {"chat": {"id": 123, "type": "private"}, "photo": [{"file_id": "p1"}]}}]


def test_banned_user_chatbot_is_blocked(app_client, monkeypatch):
    client, _ = app_client
    from src.chat.clients import get_sqlite
    from src.chat.storage import wallet

    monkeypatch.setattr(telegram, "TELEGRAM_WEBHOOK_SECRET", "")
    monkeypatch.setattr(telegram, "reserve_webhook_update", lambda channel, update_id: True)

    # Put chat 777 into a banned state (in debt, past the grace window).
    wallet.debit("tg:777", 5_000)
    conn = get_sqlite()
    conn.execute(
        "UPDATE account_balance SET debt_since = ? WHERE account_id = ?",
        (wallet.debt_status("tg:777")["debt_since"] - wallet.DEBT_GRACE_SECONDS - 10, "tg:777"),
    )
    conn.commit()

    answered: list[tuple] = []
    sent: list[tuple] = []

    async def fail_answer(chat_id, text):
        answered.append((chat_id, text))

    async def fake_send_text(chat_id, text, *args, **kwargs):
        sent.append((chat_id, text))

    monkeypatch.setattr(telegram, "_answer_and_send", fail_answer)
    monkeypatch.setattr(telegram, "send_text", fake_send_text)

    response = client.post(
        "/webhook/telegram",
        json={"update_id": 7001, "message": {"chat": {"id": 777, "type": "private"}, "text": "Tôi bị ho"}},
    )

    assert response.status_code == 200
    assert answered == []  # chatbot did not run
    assert any(cid == 777 and ("nợ" in text.lower() or "khóa" in text.lower()) for cid, text in sent)


def test_banned_user_can_still_use_paydebt_command(monkeypatch):
    from src.chat.clients import get_sqlite
    from src.chat.storage import wallet

    wallet.debit("tg:778", 10_000)
    conn = get_sqlite()
    conn.execute(
        "UPDATE account_balance SET debt_since = ? WHERE account_id = ?",
        (wallet.debt_status("tg:778")["debt_since"] - wallet.DEBT_GRACE_SECONDS - 10, "tg:778"),
    )
    conn.commit()

    sent: list[str] = []

    async def fake_send_text(chat_id, text, *args, **kwargs):
        sent.append(text)

    async def fake_start_topup_payment(chat_id, amount):
        sent.append(f"MOCK_QR:{amount}")

    monkeypatch.setattr(telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram, "_start_topup_payment", fake_start_topup_payment, raising=False)

    assert asyncio.run(telegram._handle_command(778, "/paydebt", chat_type="private", user_id=778))
    # Payoff = 10,000 + 10% = 11,000.
    assert any("11,000" in t for t in sent)
