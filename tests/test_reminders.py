import datetime
from zoneinfo import ZoneInfo
import time
import json
from unittest.mock import patch, AsyncMock, MagicMock
import pytest
import httpx
from src.chat.storage.recurrence import next_occurrence, TZ
from src.chat.storage.reminders import (
    init_reminders_db, create_reminder_draft, confirm_reminder_draft,
    list_active_reminders, delete_reminder, claim_due_reminders,
    complete_delivery, release_claim, count_active_reminders,
    check_duplicate_active_or_pending, delete_reminder_draft
)
from src.chat.storage.reminder_parser import (
    check_reminder_prefilter,
    direct_reminder_fallback,
    is_explicit_reminder_request,
    parse_multi_turn_reminder,
    parse_reminder_natural_language,
)
from src.server.channels.telegram import run_reminder_tick, send_checked_reminder_message

def test_recurrence_one_time():
    sched = {"type": "one_time", "datetime": "2026-06-22 08:30"}
    after = datetime.datetime(2026, 6, 21, 12, 0, tzinfo=TZ)
    res = next_occurrence(sched, after)
    assert res == datetime.datetime(2026, 6, 22, 8, 30, tzinfo=TZ)
    
    # Already passed
    after_passed = datetime.datetime(2026, 6, 22, 9, 0, tzinfo=TZ)
    assert next_occurrence(sched, after_passed) is None

def test_recurrence_daily():
    sched = {"type": "daily", "times": ["08:00", "20:00"]}
    # Today before 08:00
    after = datetime.datetime(2026, 6, 21, 5, 0, tzinfo=TZ)
    assert next_occurrence(sched, after) == datetime.datetime(2026, 6, 21, 8, 0, tzinfo=TZ)
    
    # Today between 08:00 and 20:00
    after2 = datetime.datetime(2026, 6, 21, 10, 0, tzinfo=TZ)
    assert next_occurrence(sched, after2) == datetime.datetime(2026, 6, 21, 20, 0, tzinfo=TZ)
    
    # Today after 20:00 -> tomorrow 08:00
    after3 = datetime.datetime(2026, 6, 21, 21, 0, tzinfo=TZ)
    assert next_occurrence(sched, after3) == datetime.datetime(2026, 6, 22, 8, 0, tzinfo=TZ)

def test_recurrence_weekdays():
    # Monday and Wednesday (0 and 2)
    sched = {"type": "weekdays", "days": [0, 2], "times": ["09:00"]}
    # Sunday, June 21, 2026 -> next is Monday, June 22
    after = datetime.datetime(2026, 6, 21, 12, 0, tzinfo=TZ)
    assert next_occurrence(sched, after) == datetime.datetime(2026, 6, 22, 9, 0, tzinfo=TZ)
    
    # Monday after 09:00 -> next is Wednesday, June 24
    after2 = datetime.datetime(2026, 6, 22, 10, 0, tzinfo=TZ)
    assert next_occurrence(sched, after2) == datetime.datetime(2026, 6, 24, 9, 0, tzinfo=TZ)

def test_recurrence_interval_and_recovery():
    # Every 8 hours starting 2026-06-21 08:00
    sched = {"type": "interval", "unit": "hours", "value": 8, "start_datetime": "2026-06-21 08:00"}
    after = datetime.datetime(2026, 6, 21, 10, 0, tzinfo=TZ)
    # Next should be 16:00
    assert next_occurrence(sched, after) == datetime.datetime(2026, 6, 21, 16, 0, tzinfo=TZ)
    
    # Downtime recovery: server down for 24 hours (now is 2026-06-22 10:00)
    after_downtime = datetime.datetime(2026, 6, 22, 10, 0, tzinfo=TZ)
    # Next should be 16:00 on June 22
    assert next_occurrence(sched, after_downtime) == datetime.datetime(2026, 6, 22, 16, 0, tzinfo=TZ)

def test_recurrence_end_date():
    sched = {"type": "daily", "times": ["08:00"], "end_date": "2026-06-22"}
    after = datetime.datetime(2026, 6, 21, 12, 0, tzinfo=TZ)
    # Next is June 22 08:00 (allowed)
    assert next_occurrence(sched, after) == datetime.datetime(2026, 6, 22, 8, 0, tzinfo=TZ)
    
    after2 = datetime.datetime(2026, 6, 22, 9, 0, tzinfo=TZ)
    # Next would be June 23, but end_date is June 22 (inclusive). So None.
    assert next_occurrence(sched, after2) is None

def test_sqlite_reminders_lifecycle():
    init_reminders_db()
    
    chat_id = 12345
    user_id = 67890
    
    # 1. Create draft
    sched = {"type": "daily", "times": ["08:00"]}
    draft_id = create_reminder_draft(
        chat_id=chat_id,
        user_id=user_id,
        medical_type="medication",
        reminder_text="Uống Panadol",
        schedule=sched,
        next_fire_at=int(time.time()) + 100,
        end_date=None,
        source="direct"
    )
    assert draft_id > 0
    
    assert count_active_reminders(chat_id, user_id) == 0
    
    # 2. Confirm draft
    reminder = confirm_reminder_draft(chat_id, user_id, draft_id)
    assert reminder is not None
    assert reminder["status"] == "active"
    
    assert count_active_reminders(chat_id, user_id) == 1
    
    # 3. Check duplicate active
    assert check_duplicate_active_or_pending(chat_id, user_id, "medication", "Uống Panadol", '{"type": "daily", "times": ["08:00"]}') is True
    
    # 4. List active
    active = list_active_reminders(chat_id, user_id)
    assert len(active) == 1
    assert active[0]["reminder_text"] == "Uống Panadol"
    
    # 5. Claim due reminders
    now = int(time.time())
    due = claim_due_reminders(now, lease_duration=60)
    assert len(due) == 0
    
    due_future = claim_due_reminders(now + 200, lease_duration=60)
    assert len(due_future) == 1
    assert due_future[0]["id"] == reminder["id"]
    
    assert len(claim_due_reminders(now + 200, lease_duration=60)) == 0
    
    # 6. Complete delivery
    complete_delivery(reminder["id"], next_fire_at=now + 500)
    
    active_after = list_active_reminders(chat_id, user_id)
    assert active_after[0]["next_fire_at"] == now + 500
    
    # 7. Permanent delete
    deleted = delete_reminder(chat_id, user_id, reminder["id"])
    assert deleted is True
    assert count_active_reminders(chat_id, user_id) == 0

def test_max_active_reminders_limit():
    init_reminders_db()
    chat_id = 999
    user_id = 888
    
    for i in range(20):
        draft_id = create_reminder_draft(
            chat_id=chat_id,
            user_id=user_id,
            medical_type="medication",
            reminder_text=f"Thuốc {i}",
            schedule={"type": "one_time", "datetime": f"2026-06-22 08:{i:02d}"},
            next_fire_at=int(time.time()) + 1000,
            end_date=None,
            source="direct"
        )
        confirm_reminder_draft(chat_id, user_id, draft_id)
        
    assert count_active_reminders(chat_id, user_id) == 20
    
    draft_id_21 = create_reminder_draft(
        chat_id=chat_id,
        user_id=user_id,
        medical_type="medication",
        reminder_text="Thuốc 21",
        schedule={"type": "one_time", "datetime": "2026-06-22 09:00"},
        next_fire_at=int(time.time()) + 1000,
        end_date=None,
        source="direct"
    )
    assert confirm_reminder_draft(chat_id, user_id, draft_id_21) is None
    assert count_active_reminders(chat_id, user_id) == 20

def test_prefilter():
    assert check_reminder_prefilter("nhắc tôi uống thuốc") is True
    assert check_reminder_prefilter("Lịch hẹn khám lúc 9h") is True
    assert check_reminder_prefilter("Tôi bị đau bụng nhiều ngày") is False

@patch("src.chat.storage.reminder_parser.call_mini")
def test_parse_reminder_nl(mock_call):
    tz = ZoneInfo("Asia/Ho_Chi_Minh")
    now = datetime.datetime(2026, 6, 21, 10, 0, tzinfo=tz)
    
    mock_call.return_value = {
        "is_direct_request": True,
        "is_ordinary_mention": False,
        "medical_type": "medication",
        "reminder_text": "Uống thuốc Panadol",
        "schedule": {"type": "daily", "times": ["08:00", "20:00"]},
        "end_date": None,
        "is_ambiguous": False,
        "is_past": False
    }
    res = parse_reminder_natural_language("nhắc tôi uống thuốc Panadol lúc 8h sáng và 8h tối", now)
    assert res is not None
    assert res["is_direct_request"] is True
    assert res["medical_type"] == "medication"
    assert res["schedule"]["times"] == ["08:00", "20:00"]

@pytest.mark.anyio
@patch("src.server.channels.telegram.send_checked_reminder_message", new_callable=AsyncMock)
async def test_dispatcher_success_flow(mock_send):
    init_reminders_db()
    mock_send.return_value = None
    
    draft_id = create_reminder_draft(
        chat_id=111,
        user_id=222,
        medical_type="medication",
        reminder_text="Thuốc A",
        schedule={"type": "one_time", "datetime": "2026-06-21 08:00"},
        next_fire_at=int(time.time()) - 10,
        end_date=None,
        source="direct"
    )
    reminder = confirm_reminder_draft(111, 222, draft_id)
    assert reminder is not None
    
    await run_reminder_tick()
    
    assert len(list_active_reminders(111, 222)) == 0
    mock_send.assert_called_once()


from src.chat.storage.reminders import (
    get_pending_conversation, upsert_pending_conversation, delete_pending_conversation
)

def test_pending_conversation_lifecycle():
    init_reminders_db()
    chat_id = 11111
    user_id = 22222
    
    # 1. Initially None
    assert get_pending_conversation(chat_id, user_id) is None
    
    # 2. Upsert
    upsert_pending_conversation(
        chat_id=chat_id,
        user_id=user_id,
        original_request="Đặt lịch uống thuốc",
        partial_fields={"medical_type": "medication"},
        turns=["Đặt lịch uống thuốc"],
        missing_fields=["schedule"]
    )
    
    pending = get_pending_conversation(chat_id, user_id)
    assert pending is not None
    assert pending["original_request"] == "Đặt lịch uống thuốc"
    assert pending["partial_fields"] == {"medical_type": "medication"}
    assert pending["turns"] == ["Đặt lịch uống thuốc"]
    assert pending["missing_fields"] == ["schedule"]
    assert pending["source"] == "direct"
    
    # 3. Expiration check (simulate expired)
    upsert_pending_conversation(
        chat_id=chat_id,
        user_id=user_id,
        original_request="Đặt lịch uống thuốc",
        partial_fields={"medical_type": "medication"},
        turns=["Đặt lịch uống thuốc"],
        missing_fields=["schedule"],
        expires_at=int(time.time()) - 10  # 10s in the past
    )
    assert get_pending_conversation(chat_id, user_id) is None
    
    # 4. Delete
    upsert_pending_conversation(
        chat_id=chat_id,
        user_id=user_id,
        original_request="Đặt lịch uống thuốc",
        partial_fields={"medical_type": "medication"},
        turns=["Đặt lịch uống thuốc"],
        missing_fields=["schedule"]
    )
    delete_pending_conversation(chat_id, user_id)
    assert get_pending_conversation(chat_id, user_id) is None


@patch("src.chat.storage.reminder_parser.call_mini")
def test_parse_multi_turn_reminder(mock_call):
    tz = ZoneInfo("Asia/Ho_Chi_Minh")
    now = datetime.datetime(2026, 6, 21, 10, 0, tzinfo=tz)
    
    # Turn 1: Incomplete request
    mock_call.return_value = {
        "is_relevant_followup": True,
        "is_canceled": False,
        "is_direct_request": True,
        "is_ordinary_mention": False,
        "merged_fields": {"medical_type": "medication", "reminder_text": "Uống Panadol", "schedule": None, "end_date": None},
        "missing_fields": ["schedule"],
        "is_complete": False,
        "is_ambiguous": False,
        "is_past": False,
        "clarification_prompt": "Bạn muốn uống thuốc vào lúc mấy giờ?"
    }
    
    res1 = parse_multi_turn_reminder("Đặt lịch uống Panadol cho tôi", now, None)
    assert res1["is_complete"] is False
    assert "schedule" in res1["missing_fields"]
    
    # Turn 2: Follow-up completing it
    prior = {
        "original_request": "Đặt lịch uống Panadol cho tôi",
        "partial_fields": {"medical_type": "medication", "reminder_text": "Uống Panadol", "schedule": None, "end_date": None},
        "turns": ["Đặt lịch uống Panadol cho tôi", "Bạn muốn uống thuốc vào lúc mấy giờ?"],
        "missing_fields": ["schedule"]
    }
    
    mock_call.return_value = {
        "is_relevant_followup": True,
        "is_canceled": False,
        "merged_fields": {
            "medical_type": "medication",
            "reminder_text": "Uống Panadol",
            "schedule": {"type": "daily", "times": ["11:00"]},
            "end_date": None
        },
        "missing_fields": [],
        "is_complete": True,
        "is_ambiguous": False,
        "is_past": False,
        "clarification_prompt": None
    }
    
    res2 = parse_multi_turn_reminder("11h trưa hàng ngày", now, prior)
    assert res2["is_complete"] is True
    assert res2["merged_fields"]["schedule"]["times"] == ["11:00"]


@pytest.mark.anyio
async def test_telegram_multi_turn_reminders(monkeypatch):
    from src.chat.storage.reminders import init_reminders_db, get_pending_conversation, delete_pending_conversation
    from src.server.channels import telegram
    
    init_reminders_db()
    chat_id = 99999
    user_id = 88888
    
    # Clean any pending
    delete_pending_conversation(chat_id, user_id)
    
    # Mock send_text and call_mini
    sent_messages = []
    async def mock_send_text(c_id, text, choices=None, selection_mode=None, inline_keyboard=None):
        sent_messages.append({"text": text, "keyboard": inline_keyboard})
        return 123
        
    monkeypatch.setattr(telegram, "send_text", mock_send_text)
    
    # Turn 1: Incomplete direct request
    # Mock LLM for turn 1
    mock_responses = [
        # Response 1
        {
            "is_relevant_followup": True,
            "is_canceled": False,
            "is_direct_request": True,
            "is_ordinary_mention": False,
            "merged_fields": {"medical_type": "medication", "reminder_text": "uống thuốc", "schedule": None, "end_date": None},
            "missing_fields": ["schedule"],
            "is_complete": False,
            "is_ambiguous": False,
            "is_past": False,
            "clarification_prompt": "Bạn muốn uống thuốc vào lúc mấy giờ?"
        },
        # Response 2
        {
            "is_relevant_followup": True,
            "is_canceled": False,
            "merged_fields": {
                "medical_type": "medication",
                "reminder_text": "uống thuốc",
                "schedule": {"type": "one_time", "datetime": "2026-06-22 11:00"},
                "end_date": None
            },
            "missing_fields": [],
            "is_complete": True,
            "is_ambiguous": False,
            "is_past": False,
            "clarification_prompt": None
        }
    ]
    
    def mock_call_mini(prompt, text, stage):
        return mock_responses.pop(0)
        
    monkeypatch.setattr("src.chat.storage.reminder_parser.call_mini", mock_call_mini)
    
    # Process turn 1
    res1 = await telegram._process_reminder_input_flow(chat_id, user_id, "Đặt lịch uống thuốc cho tôi", is_direct=True)
    assert res1 is True
    assert len(sent_messages) == 1
    assert "Bạn muốn uống thuốc vào lúc mấy giờ?" in sent_messages[-1]["text"]
    assert sent_messages[-1]["keyboard"] == {"inline_keyboard": [[{"text": "❌ Hủy", "callback_data": "remind:cancel_conv"}]]}
    
    # Pending conversation should exist
    pending = get_pending_conversation(chat_id, user_id)
    assert pending is not None
    assert pending["original_request"] == "Đặt lịch uống thuốc cho tôi"
    
    # Turn 2: Follow-up completing the request
    # Since there is a pending conversation, we intercept in _answer_and_send
    await telegram._answer_and_send(chat_id, "Sáng mai 11h", user_id=user_id, chat_type="private")
    
    # Conversation should be deleted
    assert get_pending_conversation(chat_id, user_id) is None
    
    # Confirmation draft should have been proposed
    assert len(sent_messages) == 2
    assert "Đề xuất tạo nhắc nhở y tế" in sent_messages[-1]["text"]
    assert "Một lần vào 2026-06-22 11:00" in sent_messages[-1]["text"]


@pytest.mark.anyio
async def test_pending_conv_unrelated_question(monkeypatch):
    from src.chat.storage.reminders import init_reminders_db, get_pending_conversation, upsert_pending_conversation
    from src.server.channels import telegram
    
    init_reminders_db()
    chat_id = 77777
    user_id = 66666
    
    upsert_pending_conversation(
        chat_id=chat_id,
        user_id=user_id,
        original_request="Đặt lịch uống thuốc",
        partial_fields={"medical_type": "medication"},
        turns=["Đặt lịch uống thuốc"],
        missing_fields=["schedule"]
    )
    
    mock_parser_response = {
        "is_relevant_followup": False,
        "is_canceled": False,
        "is_direct_request": False,
        "is_ordinary_mention": False,
        "merged_fields": {},
        "missing_fields": [],
        "is_complete": False,
        "is_ambiguous": False,
        "is_past": False,
        "clarification_prompt": None
    }
    monkeypatch.setattr("src.chat.storage.reminder_parser.call_mini", lambda *args, **kwargs: mock_parser_response)
    
    from src.server.channels.telegram import ChatReply
    monkeypatch.setattr("src.server.channels.telegram.answer_with_choices", lambda *args, **kwargs: ChatReply("Bác sĩ Chatbot xin chào!"))
    
    sent_messages = []
    async def mock_send_text(c_id, text, choices=None, selection_mode=None, inline_keyboard=None):
        sent_messages.append(text)
        return 123
    monkeypatch.setattr(telegram, "send_text", mock_send_text)
    
    await telegram._answer_and_send(chat_id, "Tôi bị ho và đau họng", user_id=user_id, chat_type="private")
    
    assert any("Bác sĩ Chatbot" in msg for msg in sent_messages)
    
    pending = get_pending_conversation(chat_id, user_id)
    assert pending is not None
    assert pending["original_request"] == "Đặt lịch uống thuốc"


@pytest.mark.anyio
async def test_pending_conv_cancel(monkeypatch):
    from src.chat.storage.reminders import init_reminders_db, get_pending_conversation, upsert_pending_conversation
    from src.server.channels import telegram
    
    init_reminders_db()
    monkeypatch.setattr(telegram, "TELEGRAM_WEBHOOK_SECRET", None)
    chat_id = 55555
    user_id = 44444
    
    upsert_pending_conversation(
        chat_id=chat_id,
        user_id=user_id,
        original_request="Đặt lịch uống thuốc",
        partial_fields={"medical_type": "medication"},
        turns=["Đặt lịch uống thuốc"],
        missing_fields=["schedule"]
    )
    
    monkeypatch.setattr(telegram, "send_text", AsyncMock(return_value=123))
    
    res = await telegram._handle_command(chat_id, "/cancel", chat_type="private", user_id=user_id)
    assert res is True
    
    assert get_pending_conversation(chat_id, user_id) is None
    
    upsert_pending_conversation(
        chat_id=chat_id,
        user_id=user_id,
        original_request="Đặt lịch uống thuốc",
        partial_fields={"medical_type": "medication"},
        turns=["Đặt lịch uống thuốc"],
        missing_fields=["schedule"]
    )
    
    payload = {
        "callback_query": {
            "id": "cb_cancel",
            "data": "remind:cancel_conv",
            "from": {"id": user_id},
            "message": {
                "chat": {"id": chat_id},
                "message_id": 999
            }
        }
    }
    
    request = MagicMock()
    request.json = AsyncMock(return_value=payload)
    
    mock_answer = AsyncMock()
    mock_edit = AsyncMock()
    monkeypatch.setattr(telegram, "_answer_callback_query", mock_answer)
    monkeypatch.setattr(telegram, "_edit_message_text", mock_edit)
    monkeypatch.setattr(telegram, "reserve_webhook_update", lambda *args: True)
    
    from fastapi import BackgroundTasks
    bg_tasks = BackgroundTasks()
    
    await telegram.telegram_webhook(request, bg_tasks)
    
    assert get_pending_conversation(chat_id, user_id) is None
    mock_answer.assert_called_once_with("cb_cancel", "Đã hủy thiết lập.")
    mock_edit.assert_called_once_with(chat_id, 999, "❌ Đã hủy thiết lập nhắc nhở.")


def test_expiration_cleanup():
    from src.chat.storage.reminders import (
        init_reminders_db, get_pending_conversation, upsert_pending_conversation, cleanup_expired_drafts
    )
    init_reminders_db()
    chat_id = 33333
    user_id = 22222
    
    upsert_pending_conversation(
        chat_id=chat_id,
        user_id=user_id,
        original_request="Đặt lịch uống thuốc",
        partial_fields={"medical_type": "medication"},
        turns=["Đặt lịch uống thuốc"],
        missing_fields=["schedule"],
        expires_at=int(time.time()) - 100
    )
    
    cleanup_expired_drafts()
    
    assert get_pending_conversation(chat_id, user_id) is None


def test_explicit_reminder_detection_and_safe_fallback():
    assert is_explicit_reminder_request("đặt lịch uống thuốc cho tôi") is True
    assert is_explicit_reminder_request("Nhắc tôi đi khám") is True
    assert is_explicit_reminder_request("I have a medicine question") is False

    fallback = direct_reminder_fallback("đặt lịch uống thuốc cho tôi")
    assert fallback["is_direct_request"] is True
    assert fallback["merged_fields"]["medical_type"] == "medication"
    assert fallback["merged_fields"]["reminder_text"] == "Uống thuốc"
    assert fallback["missing_fields"] == ["schedule"]


@pytest.mark.anyio
@pytest.mark.parametrize("parser_result", [
    None,
    {
        "is_relevant_followup": True,
        "is_canceled": False,
        "merged_fields": {
            "medical_type": "medication",
            "reminder_text": "uống thuốc",
            "schedule": None,
            "end_date": None,
        },
        "missing_fields": ["schedule"],
        "is_complete": False,
        "is_ambiguous": True,
        "is_past": False,
        "clarification_prompt": "Bạn muốn uống thuốc vào lúc nào trong ngày?",
    },
])
async def test_explicit_reminder_never_falls_through_to_rag(monkeypatch, parser_result):
    from src.chat.storage.reminders import delete_pending_conversation, init_reminders_db
    from src.server.channels import telegram

    chat_id = 99101
    user_id = 99102
    init_reminders_db()
    delete_pending_conversation(chat_id, user_id)

    monkeypatch.setattr(
        "src.chat.storage.reminder_parser.parse_multi_turn_reminder",
        lambda *args, **kwargs: parser_result,
    )
    process_reminder = AsyncMock(return_value=True)
    monkeypatch.setattr(telegram, "_process_reminder_input_flow", process_reminder)
    monkeypatch.setattr(
        telegram,
        "answer_with_choices",
        MagicMock(side_effect=AssertionError("explicit reminder reached medical RAG")),
    )

    async def wait_for_stop(_chat_id, stop):
        await stop.wait()

    monkeypatch.setattr(telegram, "_keep_typing", wait_for_stop)

    await telegram._answer_and_send(
        chat_id,
        "đặt lịch uống thuốc cho tôi",
        user_id=user_id,
        chat_type="private",
    )

    process_reminder.assert_awaited_once()
    assert process_reminder.await_args.kwargs["is_direct"] is True
    assert process_reminder.await_args.kwargs["parsed_reminder"] is not None


@pytest.mark.anyio
async def test_ambiguous_direct_reminder_is_persisted_before_clarification(monkeypatch):
    from src.chat.storage.reminders import (
        delete_pending_conversation,
        get_pending_conversation,
        init_reminders_db,
    )
    from src.server.channels import telegram

    chat_id = 99103
    user_id = 99104
    init_reminders_db()
    delete_pending_conversation(chat_id, user_id)
    monkeypatch.setattr(telegram, "send_text", AsyncMock(return_value=123))

    parsed = {
        "is_relevant_followup": True,
        "is_canceled": False,
        "merged_fields": {
            "medical_type": "medication",
            "reminder_text": "uống thuốc",
            "schedule": None,
            "end_date": None,
        },
        "missing_fields": ["schedule"],
        "is_complete": False,
        "is_ambiguous": True,
        "is_past": False,
        "clarification_prompt": "Bạn muốn uống thuốc vào lúc nào trong ngày?",
    }

    handled = await telegram._process_reminder_input_flow(
        chat_id,
        user_id,
        "đặt lịch uống thuốc cho tôi",
        is_direct=True,
        parsed_reminder=parsed,
    )

    assert handled is True
    pending = get_pending_conversation(chat_id, user_id)
    assert pending is not None
    assert pending["partial_fields"]["medical_type"] == "medication"
    assert pending["missing_fields"] == ["schedule"]


@pytest.mark.anyio
async def test_confirmation_draft_accepts_free_text_correction(monkeypatch):
    from src.chat.storage.reminders import (
        create_reminder_draft,
        delete_reminder_drafts_for_user,
        get_latest_reminder_draft,
        init_reminders_db,
    )
    from src.server.channels import telegram

    chat_id = 99105
    user_id = 99106
    init_reminders_db()
    delete_reminder_drafts_for_user(chat_id, user_id)
    old_draft_id = create_reminder_draft(
        chat_id=chat_id,
        user_id=user_id,
        medical_type="medication",
        reminder_text="uống thuốc",
        schedule={"type": "one_time", "datetime": "2026-06-22 12:00"},
        next_fire_at=int(datetime.datetime(2026, 6, 22, 12, 0, tzinfo=TZ).timestamp()),
        end_date=None,
        source="direct",
    )

    corrected = {
        "is_relevant_followup": True,
        "is_canceled": False,
        "corrected": True,
        "merged_fields": {
            "medical_type": "medication",
            "reminder_text": "Vitamin A",
            "schedule": {"type": "one_time", "datetime": "2026-06-22 12:00"},
            "end_date": None,
        },
        "missing_fields": [],
        "is_complete": True,
        "is_ambiguous": False,
        "is_past": False,
        "clarification_prompt": None,
    }
    monkeypatch.setattr(
        "src.chat.storage.reminder_parser.parse_multi_turn_reminder",
        lambda *args, **kwargs: corrected,
    )
    monkeypatch.setattr(telegram, "send_text", AsyncMock(return_value=123))
    monkeypatch.setattr(
        telegram,
        "answer_with_choices",
        MagicMock(side_effect=AssertionError("draft correction reached medical RAG")),
    )

    async def wait_for_stop(_chat_id, stop):
        await stop.wait()

    monkeypatch.setattr(telegram, "_keep_typing", wait_for_stop)

    await telegram._answer_and_send(
        chat_id,
        "đổi tên nội dung thành vitamin A",
        user_id=user_id,
        chat_type="private",
    )

    latest = get_latest_reminder_draft(chat_id, user_id)
    assert latest is not None
    assert latest["id"] != old_draft_id
    assert latest["reminder_text"] == "Vitamin A"


@pytest.mark.anyio
async def test_cancel_command_deletes_confirmation_draft(monkeypatch):
    from src.chat.storage.reminders import (
        create_reminder_draft,
        delete_reminder_drafts_for_user,
        get_latest_reminder_draft,
        init_reminders_db,
    )
    from src.server.channels import telegram

    chat_id = 99107
    user_id = 99108
    init_reminders_db()
    delete_reminder_drafts_for_user(chat_id, user_id)
    create_reminder_draft(
        chat_id=chat_id,
        user_id=user_id,
        medical_type="medication",
        reminder_text="Vitamin A",
        schedule={"type": "one_time", "datetime": "2026-06-22 12:00"},
        next_fire_at=int(datetime.datetime(2026, 6, 22, 12, 0, tzinfo=TZ).timestamp()),
        end_date=None,
        source="direct",
    )
    send_text = AsyncMock(return_value=123)
    monkeypatch.setattr(telegram, "send_text", send_text)

    handled = await telegram._handle_command(
        chat_id,
        "/cancel",
        chat_type="private",
        user_id=user_id,
    )

    assert handled is True
    assert get_latest_reminder_draft(chat_id, user_id) is None
    send_text.assert_awaited_once_with(chat_id, "❌ Đã hủy thiết lập nhắc nhở.")


@pytest.mark.anyio
async def test_reminder_list_renders_plain_text_with_edit_and_delete(monkeypatch):
    from src.chat.storage.reminders import (
        confirm_reminder_draft,
        create_reminder_draft,
        init_reminders_db,
        list_active_reminders,
    )
    from src.server.channels import telegram

    chat_id = 99109
    user_id = 99110
    init_reminders_db()
    for reminder in list_active_reminders(chat_id, user_id):
        delete_reminder(chat_id, user_id, reminder["id"])
    draft_id = create_reminder_draft(
        chat_id=chat_id,
        user_id=user_id,
        medical_type="medication",
        reminder_text="vitamin A",
        schedule={"type": "one_time", "datetime": "2026-06-22 12:00"},
        next_fire_at=int(datetime.datetime(2026, 6, 22, 12, 0, tzinfo=TZ).timestamp()),
        end_date=None,
        source="direct",
    )
    reminder = confirm_reminder_draft(chat_id, user_id, draft_id)
    edit_message = AsyncMock()
    monkeypatch.setattr(telegram, "_edit_message_text", edit_message)

    await telegram._show_reminders_list(chat_id, user_id, message_id=123)

    rendered = edit_message.await_args.args[2]
    keyboard = edit_message.await_args.kwargs["inline_keyboard"]["inline_keyboard"]
    assert "*" not in rendered
    assert "Nội dung: vitamin A" in rendered
    assert "Lịch nhắc: Một lần vào 2026-06-22 12:00" in rendered
    assert keyboard[0][0]["callback_data"] == f"remind:edit:{reminder['id']}"
    assert keyboard[0][1]["callback_data"] == f"remind:remove:{reminder['id']}"


def test_confirm_edit_draft_updates_existing_reminder():
    from src.chat.storage.reminders import (
        confirm_reminder_draft,
        create_reminder_draft,
        init_reminders_db,
        list_active_reminders,
    )

    chat_id = 99111
    user_id = 99112
    init_reminders_db()
    original_draft = create_reminder_draft(
        chat_id=chat_id,
        user_id=user_id,
        medical_type="medication",
        reminder_text="uống thuốc",
        schedule={"type": "one_time", "datetime": "2026-06-22 12:00"},
        next_fire_at=int(datetime.datetime(2026, 6, 22, 12, 0, tzinfo=TZ).timestamp()),
        end_date=None,
        source="direct",
    )
    original = confirm_reminder_draft(chat_id, user_id, original_draft)
    edit_draft = create_reminder_draft(
        chat_id=chat_id,
        user_id=user_id,
        medical_type="medication",
        reminder_text="vitamin A",
        schedule={"type": "daily", "times": ["13:00"]},
        next_fire_at=int(datetime.datetime(2026, 6, 22, 13, 0, tzinfo=TZ).timestamp()),
        end_date=None,
        source=f"edit:{original['id']}",
    )

    updated = confirm_reminder_draft(chat_id, user_id, edit_draft)

    active = list_active_reminders(chat_id, user_id)
    assert updated["id"] == original["id"]
    assert updated["reminder_text"] == "vitamin A"
    assert len(active) == 1
    assert active[0]["schedule"] == {"type": "daily", "times": ["13:00"]}


@pytest.mark.anyio
async def test_multi_turn_edit_preserves_original_reminder_id(monkeypatch):
    from src.chat.storage.reminders import (
        confirm_reminder_draft,
        create_reminder_draft,
        delete_pending_conversation,
        delete_reminder_drafts_for_user,
        get_latest_reminder_draft,
        get_pending_conversation,
        init_reminders_db,
        list_active_reminders,
    )
    from src.server.channels import telegram

    chat_id = 99113
    user_id = 99114
    init_reminders_db()
    delete_pending_conversation(chat_id, user_id)
    delete_reminder_drafts_for_user(chat_id, user_id)
    for reminder in list_active_reminders(chat_id, user_id):
        delete_reminder(chat_id, user_id, reminder["id"])

    original_draft = create_reminder_draft(
        chat_id=chat_id,
        user_id=user_id,
        medical_type="medication",
        reminder_text="vitamin A",
        schedule={"type": "one_time", "datetime": "2026-06-22 12:00"},
        next_fire_at=int(datetime.datetime(2026, 6, 22, 12, 0, tzinfo=TZ).timestamp()),
        end_date=None,
        source="direct",
    )
    original = confirm_reminder_draft(chat_id, user_id, original_draft)
    create_reminder_draft(
        chat_id=chat_id,
        user_id=user_id,
        medical_type="medication",
        reminder_text="vitamin A",
        schedule={"type": "one_time", "datetime": "2026-06-22 12:00"},
        next_fire_at=int(datetime.datetime(2026, 6, 22, 12, 0, tzinfo=TZ).timestamp()),
        end_date=None,
        source=f"edit:{original['id']}",
    )

    parser_results = [
        {
            "is_relevant_followup": True,
            "is_canceled": False,
            "corrected": True,
            "merged_fields": {
                "medical_type": "medication",
                "reminder_text": "vitamin A",
                "schedule": None,
                "end_date": None,
            },
            "missing_fields": ["schedule"],
            "is_complete": False,
            "is_ambiguous": True,
            "is_past": False,
            "clarification_prompt": "Bạn muốn nhắc vào mấy giờ mỗi ngày?",
        },
        {
            "is_relevant_followup": True,
            "is_canceled": False,
            "corrected": False,
            "merged_fields": {
                "medical_type": "medication",
                "reminder_text": "vitamin A",
                "schedule": {"type": "daily", "times": ["12:00"]},
                "end_date": None,
            },
            "missing_fields": [],
            "is_complete": True,
            "is_ambiguous": False,
            "is_past": False,
            "clarification_prompt": None,
        },
    ]
    monkeypatch.setattr(
        "src.chat.storage.reminder_parser.parse_multi_turn_reminder",
        lambda *args, **kwargs: parser_results.pop(0),
    )
    monkeypatch.setattr(telegram, "send_text", AsyncMock(return_value=123))
    monkeypatch.setattr(
        telegram,
        "answer_with_choices",
        MagicMock(side_effect=AssertionError("multi-turn edit reached medical RAG")),
    )

    async def wait_for_stop(_chat_id, stop):
        await stop.wait()

    monkeypatch.setattr(telegram, "_keep_typing", wait_for_stop)

    await telegram._answer_and_send(
        chat_id,
        "đổi thành nhắc hàng ngày",
        user_id=user_id,
        chat_type="private",
    )
    pending = get_pending_conversation(chat_id, user_id)
    assert pending is not None
    assert pending["source"] == f"edit:{original['id']}"

    await telegram._answer_and_send(
        chat_id,
        "12h",
        user_id=user_id,
        chat_type="private",
    )
    edited_draft = get_latest_reminder_draft(chat_id, user_id)
    assert edited_draft is not None
    assert edited_draft["source"] == f"edit:{original['id']}"

    updated = confirm_reminder_draft(chat_id, user_id, edited_draft["id"])
    active = list_active_reminders(chat_id, user_id)
    assert updated["id"] == original["id"]
    assert len(active) == 1
    assert active[0]["schedule"] == {"type": "daily", "times": ["12:00"]}
