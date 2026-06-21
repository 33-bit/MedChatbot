import datetime
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Ho_Chi_Minh")

def next_occurrence(schedule: dict, after: datetime.datetime) -> datetime.datetime | None:
    # Ensure after is zone-aware in Asia/Ho_Chi_Minh timezone
    after_tz = after.astimezone(TZ)
    
    schedule_type = schedule.get("type")
    end_date_str = schedule.get("end_date")
    end_dt = None
    if end_date_str:
        # Inclusive of the end day -> until end of that day (23:59:59)
        try:
            end_dt = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").replace(tzinfo=TZ) + datetime.timedelta(days=1)
        except ValueError:
            pass
        
    next_dt = None
    
    if schedule_type == "one_time":
        dt_str = schedule.get("datetime")
        try:
            dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
            if dt > after_tz:
                next_dt = dt
        except ValueError:
            pass
            
    elif schedule_type == "daily":
        times = schedule.get("times", [])
        candidates = []
        for time_str in times:
            try:
                t = datetime.datetime.strptime(time_str, "%H:%M").time()
                dt_today = datetime.datetime.combine(after_tz.date(), t).replace(tzinfo=TZ)
                if dt_today > after_tz:
                    candidates.append(dt_today)
                dt_tomorrow = datetime.datetime.combine(after_tz.date() + datetime.timedelta(days=1), t).replace(tzinfo=TZ)
                candidates.append(dt_tomorrow)
            except ValueError:
                pass
        if candidates:
            next_dt = min(candidates)
            
    elif schedule_type == "weekdays":
        days = schedule.get("days", [])
        times = schedule.get("times", [])
        candidates = []
        for i in range(8):
            d = after_tz.date() + datetime.timedelta(days=i)
            if d.weekday() in days:
                for time_str in times:
                    try:
                        t = datetime.datetime.strptime(time_str, "%H:%M").time()
                        dt = datetime.datetime.combine(d, t).replace(tzinfo=TZ)
                        if dt > after_tz:
                            candidates.append(dt)
                    except ValueError:
                        pass
        if candidates:
            next_dt = min(candidates)
            
    elif schedule_type == "interval":
        unit = schedule.get("unit")
        value = schedule.get("value")
        start_str = schedule.get("start_datetime")
        try:
            start_dt = datetime.datetime.strptime(start_str, "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
            if after_tz < start_dt:
                next_dt = start_dt
            else:
                if unit == "hours":
                    delta = datetime.timedelta(hours=value)
                elif unit == "days":
                    delta = datetime.timedelta(days=value)
                else:
                    return None
                
                diff = after_tz - start_dt
                num_intervals = int(diff.total_seconds() // delta.total_seconds()) + 1
                next_dt = start_dt + num_intervals * delta
        except (ValueError, ZeroDivisionError):
            pass

    if next_dt and end_dt and next_dt >= end_dt:
        return None
    return next_dt

def format_schedule_vietnamese(schedule: dict) -> str:
    stype = schedule.get("type")
    if stype == "one_time":
        return f"Một lần vào {schedule.get('datetime')}"
    elif stype == "daily":
        times = ", ".join(schedule.get("times", []))
        return f"Hàng ngày lúc {times}"
    elif stype == "weekdays":
        days_map = {0: "T2", 1: "T3", 2: "T4", 3: "T5", 4: "T6", 5: "T7", 6: "CN"}
        days = ", ".join(days_map[d] for d in schedule.get("days", []) if d in days_map)
        times = ", ".join(schedule.get("times", []))
        return f"Thứ {days} lúc {times}"
    elif stype == "interval":
        unit = "giờ" if schedule.get("unit") == "hours" else "ngày"
        val = schedule.get("value")
        start = schedule.get("start_datetime")
        return f"Mỗi {val} {unit} bắt đầu từ {start}"
    return "Không xác định"
