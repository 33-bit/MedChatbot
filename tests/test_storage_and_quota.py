from __future__ import annotations

from redis.exceptions import RedisError

from src.chat.guards import quota
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


def test_session_load_and_save_fail_open_when_redis_missing(monkeypatch):
    monkeypatch.setattr(session_store, "get_redis", lambda: (_ for _ in ()).throw(RuntimeError("missing")))

    loaded = session_store.load_session("s")
    session_store.save_session(PatientSession(session_id="s"))
    session_store.clear_session("s")

    assert loaded == PatientSession(session_id="s")


def test_clear_session_only_deletes_redis_not_sqlite_consultation(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(session_store, "get_redis", lambda: fake_redis)

    session_store.log_consultation("tg:1", "q", "a")
    session_store.clear_session("tg:1")

    consultations = session_store.get_past_consultations("tg:1")
    assert fake_redis.deleted == ["session:tg:1"]
    assert len(consultations) == 1
    assert consultations[0]["question"] == "q"
    assert consultations[0]["answer"] == "a"


def test_rate_limit_blocks_after_configured_threshold(monkeypatch):
    monkeypatch.setattr(session_store, "RATE_LIMIT_PER_MINUTE", 2)

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
