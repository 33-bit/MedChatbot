import datetime
import json
import logging
import re
from zoneinfo import ZoneInfo
from src.chat.llm.mini import call_mini

log = logging.getLogger(__name__)
TZ = ZoneInfo("Asia/Ho_Chi_Minh")

_DIRECT_REMINDER_PATTERNS = (
    r"^\s*/remind(?:@\w+)?\s+add\b",
    r"\b(?:đặt|tạo)\s+(?:lịch|nhắc(?:\s*nhở)?)\b",
    r"\bnhắc(?:\s*nhở)?\s+(?:cho\s+)?(?:tôi|mình|em|anh|chị)\b",
    r"\bremind\s+me\b",
    r"\b(?:set|create|add)\s+(?:a\s+)?reminder\b",
)


def is_explicit_reminder_request(text: str) -> bool:
    """Recognize direct reminder commands without relying on LLM output."""
    return any(
        re.search(pattern, text or "", flags=re.IGNORECASE)
        for pattern in _DIRECT_REMINDER_PATTERNS
    )


def direct_reminder_fallback(text: str) -> dict:
    """Build a safe partial request when reminder parsing fails."""
    normalized = (text or "").lower()
    medication = bool(re.search(r"\b(?:uống|thuốc|medicine|medication|pill)\b", normalized))
    clinic = bool(re.search(r"\b(?:khám|bác\s*sĩ|clinic|doctor|appointment)\b", normalized))

    if medication and not clinic:
        medical_type = "medication"
        reminder_text = "Uống thuốc"
        missing_fields = ["schedule"]
        prompt = "Bạn muốn được nhắc uống thuốc vào ngày và giờ nào?"
    elif clinic and not medication:
        medical_type = "clinic"
        reminder_text = "Đi khám"
        missing_fields = ["schedule"]
        prompt = "Bạn muốn được nhắc đi khám vào ngày và giờ nào?"
    else:
        medical_type = None
        reminder_text = None
        missing_fields = ["medical_type", "reminder_text", "schedule"]
        prompt = "Bạn muốn được nhắc việc gì, vào ngày và giờ nào?"

    return {
        "is_relevant_followup": True,
        "is_canceled": False,
        "is_direct_request": True,
        "is_ordinary_mention": False,
        "merged_fields": {
            "medical_type": medical_type,
            "reminder_text": reminder_text,
            "schedule": None,
            "end_date": None,
        },
        "missing_fields": missing_fields,
        "is_complete": False,
        "is_ambiguous": False,
        "is_past": False,
        "clarification_prompt": prompt,
    }

def check_reminder_prefilter(text: str) -> bool:
    text_lower = (text or "").lower()
    keywords = [
        "nhắc", "remind", "hẹn", "khám", "uống", "thuốc", "lịch", "sáng", 
        "trưa", "chiều", "tối", "mai", "giờ", "phút", "tiếng", "hàng ngày", 
        "mỗi", "clinic", "pill", "medicine", "doctor", "schedule", "alarm", "báo thức"
    ]
    if any(kw in text_lower for kw in keywords):
        return True
    if re.search(r'\d+', text_lower):
        return True
    return False

def parse_reminder_natural_language(text: str, current_time_context: datetime.datetime) -> dict | None:
    current_time_str = current_time_context.astimezone(TZ).strftime("%Y-%m-%d %H:%M")
    weekday_str = current_time_context.astimezone(TZ).strftime("%A")
    
    system_prompt = f"""You are an expert medical assistant parsing Telegram messages to extract medical reminders (medication or clinic visits).
Current time context: {current_time_str} (Asia/Ho_Chi_Minh timezone, weekday: {weekday_str}).

Your task is to parse the user's input and return a JSON object with the following fields:
1. "is_direct_request": bool (true if the user explicitly asks to set a reminder/alarm, e.g. "nhắc tôi...", "set reminder...", "/remind add...").
2. "is_ordinary_mention": bool (true if the user is not directly asking for a reminder, but mentions a concrete schedule/appointment they have, e.g. "Tôi có lịch khám lúc 9h sáng mai", "Bác sĩ dặn uống thuốc này lúc 8h tối").
3. "medical_type": "medication" | "clinic" | null (use "medication" for taking medicine/pills, "clinic" for doctor appointments/checkups).
4. "reminder_text": str | null (the description of the reminder, e.g. "Uống thuốc Panadol", "Khám răng tại nha khoa").
5. "schedule": dict | null (the schedule specification, or null if missing/ambiguous).
6. "end_date": "YYYY-MM-DD" | null (the end date of the reminder if specified, inclusive).
7. "is_ambiguous": bool (true if a schedule is mentioned but lacks specific time details, e.g. "sáng mai", "uống thuốc hàng ngày" without time, "tuần sau").
8. "is_past": bool (true if the scheduled time is in the past relative to current_time).

Rules for Medication Schedule extraction:
- Never infer dosage or timing if they are not explicitly stated by the user.
- The user must explicitly state the action and concrete schedule.

Rules for Clinic Schedule extraction:
- The user must explicitly state their schedule (e.g. "Tôi có lịch hẹn khám lúc 9h sáng mai"). Do not extract if the user is only asking for advice (e.g. "tôi nên đi khám khi nào?").

Rules for Schedule formats in JSON:
- One-time: {{"type": "one_time", "datetime": "YYYY-MM-DD HH:MM"}}
- Daily: {{"type": "daily", "times": ["HH:MM", "HH:MM", ...]}}
- Weekdays: {{"type": "weekdays", "days": [0, 1, ...], "times": ["HH:MM", ...]}} (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun)
- Interval: {{"type": "interval", "unit": "hours" | "days", "value": N, "start_datetime": "YYYY-MM-DD HH:MM"}}

Resolve relative times (like "in 30 minutes", "sáng mai" resolved to 08:00 tomorrow if ambiguous but if the user specified a time like "9h sáng mai", resolve to 09:00 tomorrow) based on current_time: {current_time_str}.
Return ONLY a valid JSON object.
"""

    try:
        res = call_mini(system_prompt, text, stage="reminder_parser")
        if not isinstance(res, dict):
            return None
        return res
    except Exception as e:
        log.warning("Failed parsing natural language reminder: %s", e)
        return None


def parse_multi_turn_reminder(
    text: str,
    current_time_context: datetime.datetime,
    prior_context: dict | None = None
) -> dict | None:
    current_time_str = current_time_context.astimezone(TZ).strftime("%Y-%m-%d %H:%M")
    weekday_str = current_time_context.astimezone(TZ).strftime("%A")
    
    prior_context_str = ""
    if prior_context:
        prior_context_str = f"""
Here is the context of the active pending reminder conversation so far:
- Original Request: {prior_context.get('original_request')}
- Extracted Fields So Far: {json.dumps(prior_context.get('partial_fields'))}
- Conversation Turns History: {json.dumps(prior_context.get('turns'))}
- Missing Fields from last turn: {json.dumps(prior_context.get('missing_fields'))}
"""
    
    system_prompt = f"""You are an expert medical assistant parsing Telegram messages to extract or complete medical reminders (medication or clinic visits).
Current time context: {current_time_str} (Asia/Ho_Chi_Minh timezone, weekday: {weekday_str}).
{prior_context_str}

Your task is to parse the user's latest input: "{text}" and return a JSON object.

If prior context is provided, determine if this new message is a relevant follow-up to complete or correct the pending reminder:
1. "is_relevant_followup": bool. True if the user is answering a question about the reminder, providing time, schedule, drug name, or correcting it. If the user asks a completely unrelated medical/health question (e.g. "Tôi bị ho", "Tôi bị đau đầu"), says general greetings, or changes the topic, set this to false.
2. "is_canceled": bool. True if the user explicitly asks to cancel (e.g., "thôi", "hủy đi", "cancel", "không cần nữa").
3. "corrected": bool. True if the user is correcting previously provided information (e.g., "không, lúc 9h tối", "sửa lại là thứ 2").
4. "merged_fields": dict containing the merged reminder data:
   - "medical_type": "medication" | "clinic" | null
   - "reminder_text": str | null
   - "schedule": dict | null
   - "end_date": "YYYY-MM-DD" | null
   Merges the new details with any fields already extracted in prior context. If the user corrects a field, overwrite it.
5. "missing_fields": list of strings of fields still needed. We need:
   - "medical_type" (if missing)
   - "reminder_text" (if missing. NOTE: Generic text like "uống thuốc" is valid without a specific drug name, but a schedule remains mandatory).
   - "schedule" (if missing, incomplete, or ambiguous).
6. "is_complete": bool. True if we now have a valid "medical_type", "reminder_text", and a valid, non-ambiguous, non-past "schedule" (i.e. "missing_fields" is empty).
7. "is_ambiguous": bool. True if a schedule is mentioned but lacks specific time details (e.g. "uống thuốc hàng ngày" without time).
8. "is_past": bool. True if the scheduled time is in the past relative to current_time.
9. "clarification_prompt": str | null. If is_complete is false and is_relevant_followup is true, provide a friendly, natural Vietnamese or English question (matching the user's language) to ask for the missing fields. Do not infer dosage or timing.

If NO prior context is provided:
1. "is_relevant_followup": bool (always true).
2. "is_canceled": bool. True if the user starts with a cancel request.
3. "is_direct_request": bool. True if the user explicitly asks to set a reminder/alarm (e.g. "nhắc tôi...", "set reminder...", "/remind add...").
4. "is_ordinary_mention": bool. True if the user is not directly asking for a reminder, but mentions a concrete schedule/appointment (e.g. "Tôi có lịch khám lúc 9h sáng mai").
5. "merged_fields": dict of extracted reminder data (medical_type, reminder_text, schedule, end_date).
6. "missing_fields": list of strings of fields still needed.
7. "is_complete": bool (same as above).
8. "is_ambiguous": bool (same as above).
9. "is_past": bool (same as above).
10. "clarification_prompt": str | null (same as above).

Rules:
- Never infer medication dosage or timing that the user has not provided.
- The user must explicitly state the schedule.
- Standard schedule formats in JSON:
  - One-time: {{"type": "one_time", "datetime": "YYYY-MM-DD HH:MM"}}
  - Daily: {{"type": "daily", "times": ["HH:MM", "HH:MM", ...]}}
  - Weekdays: {{"type": "weekdays", "days": [0, 1, ...], "times": ["HH:MM", ...]}} (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun)
  - Interval: {{"type": "interval", "unit": "hours" | "days", "value": N, "start_datetime": "YYYY-MM-DD HH:MM"}}

Return ONLY a valid JSON object.
"""
    try:
        res = call_mini(system_prompt, text, stage="reminder_parser")
        if not isinstance(res, dict):
            return None
        return res
    except Exception as e:
        log.warning("Failed parsing multi turn natural language reminder: %s", e)
        return None
