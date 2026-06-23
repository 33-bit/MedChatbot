from __future__ import annotations

from src.chat.storage import cleanup


class FakeCursor:
    rowcount = 1


class FakeSqlite:
    def __init__(self) -> None:
        self.statements: list[str] = []
        self.commits = 0

    def execute(self, statement: str, params=()):
        self.statements.append(statement)
        return FakeCursor()

    def commit(self) -> None:
        self.commits += 1


def test_cleanup_does_not_vacuum_by_default(monkeypatch):
    db = FakeSqlite()
    monkeypatch.setattr(cleanup, "get_sqlite", lambda: db)

    stats = cleanup.run(verbose=False)

    assert stats == {
        "consultations_deleted": 1,
        "profiles_deleted": 1,
        "traces_deleted": 1,
        "feedback_deleted": 1,
        "expired_profile_facts_deleted": 1,
        "superseded_profile_facts_deleted": 1,
        "orphaned_profile_subjects_deleted": 1,
        "rate_limit_purged": 1,
        "doctor_consultations_ended": 0,
    }
    assert "VACUUM" not in db.statements
    assert db.commits == 1


def test_cleanup_vacuums_only_when_requested(monkeypatch):
    db = FakeSqlite()
    monkeypatch.setattr(cleanup, "get_sqlite", lambda: db)

    cleanup.run(verbose=False, vacuum=True)

    assert db.statements[-1] == "VACUUM"
