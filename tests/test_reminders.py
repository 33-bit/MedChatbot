import datetime
from zoneinfo import ZoneInfo
import time
import json
from unittest.mock import patch, AsyncMock
import pytest
import httpx
from src.chat.storage.recurrence import next_occurrence, TZ
from src.chat.storage.reminders import (
    init_reminders_db, create_reminder_draft, confirm_reminder_draft,
    list_active_reminders, delete_reminder, claim_due_reminders,
    complete_delivery, release_claim, count_active_reminders,
    check_duplicate_active_or_pending, delete_reminder_draft
)
from src.chat.storage.reminder_parser import parse_reminder_natural_language, check_reminder_prefilter, parse_multi_turn_reminder
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


