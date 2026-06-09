from __future__ import annotations

import sqlite3

import pytest
from redis.exceptions import RedisError

from src.chat.guards import quota
from src.chat.storage import feedback
from src.chat.storage import rate_limit
from src.chat.storage import redis_session
from src.chat.storage import sqlite_profile
from src.chat.storage import webhook_dedupe
from src.chat.storage import session as session_store
from src.chat.storage.session import PatientSession


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.counts: dict[str, int] = {}
        self.deleted: list[str] = []

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.data[key] = value

    def delete(self, key: str) -> None:
        self.deleted.append(key)
        self.data.pop(key, None)

    def incr(self, key: str) -> int:
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    def expire(self, key: str, ttl: int) -> None:
        return None


class BrokenSqlite:
    def __init__(self) -> None:
        self.rolled_back = False

    def execute(self, *args, **kwargs):
        raise sqlite3.Error("database is locked")

    def rollback(self) -> None:
        self.rolled_back = True


def test_session_load_and_save_fail_open_when_redis_missing(monkeypatch):
    monkeypatch.setattr(redis_session, "get_redis", lambda: (_ for _ in ()).throw(RuntimeError("missing")))

    loaded = session_store.load_session("s")
    session_store.save_session(PatientSession(session_id="s"))
    session_store.clear_session("s")

    assert loaded == PatientSession(session_id="s")


def test_clear_session_only_deletes_redis_not_sqlite_consultation(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(redis_session, "get_redis", lambda: fake_redis)

    session_store.log_consultation("tg:1", "q", "a")
    session_store.clear_session("tg:1")

    consultations = session_store.get_past_consultations("tg:1")
    assert fake_redis.deleted == ["session:tg:1"]
    assert len(consultations) == 1
    assert consultations[0]["question"] == "q"
    assert consultations[0]["answer"] == "a"


def test_rate_limit_blocks_after_configured_threshold(monkeypatch):
    monkeypatch.setattr(rate_limit, "RATE_LIMIT_PER_MINUTE", 2)

    assert session_store.check_rate_limit("s") is True
    assert session_store.check_rate_limit("s") is True
    assert session_store.check_rate_limit("s") is False


def test_quota_fails_open_when_redis_unconfigured(monkeypatch):
    monkeypatch.setattr(quota, "get_redis", lambda: (_ for _ in ()).throw(RuntimeError("missing")))

    assert quota.check_both("s") == (True, "")


def test_quota_fails_open_when_redis_errors(monkeypatch):
    monkeypatch.setattr(quota, "get_redis", lambda: (_ for _ in ()).throw(RedisError("down")))

    assert quota.check_both("s") == (True, "")


def test_quota_blocks_session_before_global(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(quota, "get_redis", lambda: fake_redis)
    monkeypatch.setattr(quota, "SESSION_LLM_QUOTA_PER_DAY", 1)

    assert quota.check_session_quota("s") is True
    assert quota.check_both("s")[0] is False


def test_quota_blocks_global_after_configured_threshold(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(quota, "get_redis", lambda: fake_redis)
    monkeypatch.setattr(quota, "SESSION_LLM_QUOTA_PER_DAY", 100)
    monkeypatch.setattr(quota, "GLOBAL_LLM_QUOTA_PER_MINUTE", 1)

    assert quota.check_both("s") == (True, "")
    allowed, message = quota.check_both("other")

    assert allowed is False
    assert message == "Hệ thống đang quá tải. Vui lòng thử lại sau 1 phút."


def test_reserve_webhook_update_accepts_only_first_delivery():
    assert session_store.reserve_webhook_update("telegram", 123) is True
    assert session_store.reserve_webhook_update("telegram", 123) is False


def test_reserve_webhook_update_uses_memory_fallback_when_sqlite_errors(monkeypatch):
    broken = BrokenSqlite()
    webhook_dedupe._MEMORY_UPDATES.clear()
    monkeypatch.setattr(webhook_dedupe, "get_sqlite", lambda: broken)

    assert session_store.reserve_webhook_update("telegram", 223) is True
    assert session_store.reserve_webhook_update("telegram", 223) is False
    assert session_store.reserve_webhook_update("telegram", 224) is True
    assert broken.rolled_back is True


def test_feedback_request_can_record_rating():
    token = feedback.create_feedback_request(
        "tg:1",
        "telegram",
        "1",
        "Tôi bị ho",
        "Bạn nên nghỉ ngơi.",
    )

    assert feedback.record_feedback_rating(token, 5) is True
    assert feedback.record_feedback_rating(token, 2) is False

    row = feedback.get_feedback(token)
    assert row == {
        "token": token,
        "session_id": "tg:1",
        "channel": "telegram",
        "recipient_id": "1",
        "question": "Tôi bị ho",
        "answer": "Bạn nên nghỉ ngơi.",
        "rating": 5,
    }


def test_standalone_webhook_dedupe_uses_memory_fallback_when_sqlite_errors(monkeypatch):
    broken = BrokenSqlite()
    webhook_dedupe._MEMORY_UPDATES.clear()
    monkeypatch.setattr(webhook_dedupe, "get_sqlite", lambda: broken)

    assert webhook_dedupe.reserve_webhook_update("telegram", 323) is True
    assert webhook_dedupe.reserve_webhook_update("telegram", 323) is False
    assert webhook_dedupe.reserve_webhook_update("telegram", 324) is True
    assert broken.rolled_back is True


def test_session_module_is_storage_facade():
    assert session_store.load_session is redis_session.load_session
    assert session_store.save_session is redis_session.save_session
    assert session_store.clear_session is redis_session.clear_session
    assert session_store.log_consultation is sqlite_profile.log_consultation
    assert session_store.get_past_consultations is sqlite_profile.get_past_consultations
    assert session_store.save_profile is sqlite_profile.save_profile
    assert session_store.check_rate_limit is rate_limit.check_rate_limit
    assert session_store.reserve_webhook_update is webhook_dedupe.reserve_webhook_update


def test_chat_trace_storage_round_trips_meta():
    from src.chat.storage import traces

    saved = traces.save_chat_trace(
        trace_id="trace-1",
        session_id="debug-user",
        internal_session_id="api:scoped",
        mode="auto",
        question="Tôi bị ho",
        answer="Bạn nên nghỉ ngơi.",
        meta={
            "trace_id": "trace-1",
            "latency_ms_total": 12.5,
            "outcome": "informational",
            "timings": [{"stage": "total", "ms": 12.5, "fields": {"outcome": "informational"}}],
        },
    )

    loaded = traces.get_chat_trace("trace-1", internal_session_id="api:scoped")

    assert saved == loaded
    assert loaded == {
        "trace_id": "trace-1",
        "session_id": "debug-user",
        "internal_session_id": "api:scoped",
        "mode": "auto",
        "question": "Tôi bị ho",
        "answer": "Bạn nên nghỉ ngơi.",
        "created_at": saved["created_at"],
        "meta": {
            "trace_id": "trace-1",
            "latency_ms_total": 12.5,
            "outcome": "informational",
            "timings": [{"stage": "total", "ms": 12.5, "fields": {"outcome": "informational"}}],
        },
    }


def test_chat_trace_list_filters_and_limits():
    from src.chat.storage import traces

    traces.save_chat_trace(
        trace_id="trace-old",
        session_id="debug-user",
        internal_session_id="api:debug-user",
        mode="auto",
        question="old question",
        answer="old answer",
        meta={"trace_id": "trace-old", "latency_ms_total": 30, "outcome": "diagnostic"},
        created_at=100.0,
    )
    traces.save_chat_trace(
        trace_id="trace-new",
        session_id="debug-user",
        internal_session_id="api:debug-user",
        mode="information",
        question="new question",
        answer="new answer",
        meta={"trace_id": "trace-new", "latency_ms_total": 10, "route_label": "informational"},
        created_at=200.0,
    )
    traces.save_chat_trace(
        trace_id="trace-other",
        session_id="other-user",
        internal_session_id="api:other",
        mode="auto",
        question="other question",
        answer="other answer",
        meta={"trace_id": "trace-other", "latency_ms_total": 5},
        created_at=300.0,
    )

    summaries = traces.list_chat_traces(
        internal_session_id="api:debug-user",
        trace_id=None,
        limit=1,
    )
    exact = traces.list_chat_traces(
        internal_session_id="api:debug-user",
        trace_id="trace-old",
        limit=20,
    )

    assert [row["trace_id"] for row in summaries] == ["trace-new"]
    assert summaries[0]["answer_preview"] == "new answer"
    assert summaries[0]["latency_ms_total"] == 10
    assert summaries[0]["route"] == "informational"
    assert [row["trace_id"] for row in exact] == ["trace-old"]


def test_chat_trace_reads_are_scoped_to_internal_session_id():
    from src.chat.storage import traces

    traces.save_chat_trace(
        trace_id="trace-private",
        session_id="debug-user",
        internal_session_id="api:owner",
        mode="auto",
        question="private question",
        answer="private answer",
        meta={"trace_id": "trace-private"},
    )

    assert traces.get_chat_trace("trace-private", internal_session_id="api:other") is None
    assert traces.list_chat_traces(internal_session_id="api:other") == []


def test_chat_trace_duplicate_id_raises_integrity_error():
    from src.chat.storage import traces

    payload = {
        "trace_id": "trace-duplicate",
        "session_id": "debug-user",
        "internal_session_id": "api:owner",
        "mode": "auto",
        "question": "first question",
        "answer": "first answer",
        "meta": {"trace_id": "trace-duplicate"},
    }
    traces.save_chat_trace(**payload)

    with pytest.raises(sqlite3.IntegrityError):
        traces.save_chat_trace(
            **{
                **payload,
                "question": "second question",
                "answer": "second answer",
            }
        )

    loaded = traces.get_chat_trace("trace-duplicate", internal_session_id="api:owner")
    assert loaded is not None
    assert loaded["question"] == "first question"
    assert loaded["answer"] == "first answer"
