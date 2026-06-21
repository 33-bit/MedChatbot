import datetime
import json
import logging
import re
from zoneinfo import ZoneInfo
from src.chat.llm.mini import call_mini

log = logging.getLogger(__name__)
TZ = ZoneInfo("Asia/Ho_Chi_Minh")

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
