from __future__ import annotations

import subprocess
import sys

import pytest

import run_vps


class FakeProcess:
    def __init__(self, running: bool = True):
        self.running = running
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self):
        return None if self.running else 0

    def terminate(self):
        self.terminated = True
        self.running = False

    def wait(self, timeout=None):
        self.wait_calls += 1
        return 0

    def kill(self):
        self.killed = True
        self.running = False


def test_kg_needs_build_when_graph_is_empty(monkeypatch):
    monkeypatch.setattr(run_vps, "_kg_node_count", lambda: 0)

    assert run_vps.kg_needs_build() is True


def test_kg_does_not_need_build_when_graph_has_nodes(monkeypatch):
    monkeypatch.setattr(run_vps, "_kg_node_count", lambda: 3)

    assert run_vps.kg_needs_build() is False


def test_qdrant_build_modes_only_include_missing_or_empty_collections(monkeypatch):
    monkeypatch.setattr(
        run_vps,
        "_qdrant_collection_points",
        lambda name: {
            run_vps.DISEASES_COLLECTION: 0,
            run_vps.DRUGS_COLLECTION: 10,
        }[name],
    )

    assert run_vps.qdrant_build_modes() == ["--diseases"]


def test_ensure_bootstrap_data_runs_only_needed_builders(monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.setattr(run_vps, "kg_needs_build", lambda: True)
    monkeypatch.setattr(run_vps, "qdrant_build_modes", lambda: ["--drugs"])
    monkeypatch.setattr(run_vps, "_run_module", lambda *args: commands.append(list(args)))
    monkeypatch.setattr(run_vps, "ensure_fulltext_indexes", lambda: None)

    run_vps.ensure_bootstrap_data()

    assert commands == [
        ["src.rag.kg_builder", "--clear"],
        ["src.rag.build_qdrant", "--drugs"],
    ]


def test_ensure_service_tracks_only_processes_it_starts():
    owned: list[FakeProcess] = []
    ready_checks = iter([False, True])
    process = FakeProcess()

    run_vps.ensure_service(
        "redis",
        is_ready=lambda: next(ready_checks),
        start_process=lambda: process,
        owned_processes=owned,
    )

    assert owned == [process]


def test_ensure_service_leaves_already_running_service_unowned():
    owned: list[FakeProcess] = []

    run_vps.ensure_service(
        "redis",
        is_ready=lambda: True,
        start_process=lambda: FakeProcess(),
        owned_processes=owned,
    )

    assert owned == []


def test_stop_owned_processes_terminates_live_processes_in_reverse_order():
    first = FakeProcess()
    second = FakeProcess()

    run_vps.stop_owned_processes([first, second])

    assert second.terminated is True
    assert first.terminated is True
    assert second.wait_calls == 1
    assert first.wait_calls == 1


def test_stop_owned_processes_kills_process_after_timeout():
    process = FakeProcess()
    calls = 0

    def timeout(*, timeout=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.TimeoutExpired(cmd="demo", timeout=timeout)
        return 0

    process.wait = timeout  # type: ignore[method-assign]

    run_vps.stop_owned_processes([process])

    assert process.terminated is True
    assert process.killed is True


def test_parse_args_accepts_reload():
    args = run_vps.parse_args(["--reload"])

    assert args.reload is True


def test_start_fastapi_omits_reload_by_default(monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.setattr(
        run_vps,
        "_start_process",
        lambda command: commands.append(command) or FakeProcess(),
    )

    run_vps.start_fastapi()

    assert commands == [
        [
            sys.executable,
            "run.py",
            "--host",
            run_vps.FASTAPI_HOST,
            "--port",
            str(run_vps.FASTAPI_PORT),
        ]
    ]


def test_start_fastapi_includes_reload_when_requested(monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.setattr(
        run_vps,
        "_start_process",
        lambda command: commands.append(command) or FakeProcess(),
    )

    run_vps.start_fastapi(reload=True)

    assert commands == [
        [
            sys.executable,
            "run.py",
            "--host",
            run_vps.FASTAPI_HOST,
            "--port",
            str(run_vps.FASTAPI_PORT),
            "--reload",
        ]
    ]


def test_telegram_webhook_url_strips_trailing_slash():
    assert run_vps.telegram_webhook_url("https://demo.ngrok-free.app/") == (
        "https://demo.ngrok-free.app/webhook/telegram"
    )


def test_zalo_webhook_url_strips_trailing_slash():
    assert run_vps.zalo_webhook_url("https://demo.ngrok-free.app/") == (
        "https://demo.ngrok-free.app/webhook/zalo"
    )


def test_register_zalo_webhook_calls_bot_platform_set_webhook(monkeypatch):
    calls: list[tuple[str, dict, int]] = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True, "result": {"url": "https://demo.ngrok-free.app/webhook/zalo"}}

    def fake_post(url: str, data: dict, timeout: int):
        calls.append((url, data, timeout))
        return Response()

    monkeypatch.setattr(run_vps, "ZALO_BOT_TOKEN", "bot-token")
    monkeypatch.setattr(run_vps, "ZALO_WEBHOOK_SECRET", "zalo-secret")
    monkeypatch.setattr(run_vps.httpx, "post", fake_post)

    run_vps.register_zalo_webhook("https://demo.ngrok-free.app/")

    assert calls == [
        (
            "https://bot-api.zaloplatforms.com/botbot-token/setWebhook",
            {
                "url": "https://demo.ngrok-free.app/webhook/zalo",
                "secret_token": "zalo-secret",
            },
            20,
        )
    ]


def test_validate_config_allows_zalo_without_telegram(monkeypatch):
    monkeypatch.setattr(run_vps, "REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setattr(run_vps, "NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setattr(run_vps, "NEO4J_PASSWORD", "secret")
    monkeypatch.setattr(run_vps, "QDRANT_URL", "http://localhost:6333")
    monkeypatch.setattr(run_vps, "NGROK_AUTHTOKEN", "ngrok")
    monkeypatch.setattr(run_vps, "TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr(run_vps, "TELEGRAM_WEBHOOK_SECRET", "")
    monkeypatch.setattr(run_vps, "ZALO_BOT_TOKEN", "zalo-token")
    monkeypatch.setattr(run_vps, "ZALO_WEBHOOK_SECRET", "zalo-secret")

    run_vps.validate_config()


def test_validate_config_requires_a_messaging_channel(monkeypatch):
    monkeypatch.setattr(run_vps, "REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setattr(run_vps, "NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setattr(run_vps, "NEO4J_PASSWORD", "secret")
    monkeypatch.setattr(run_vps, "QDRANT_URL", "http://localhost:6333")
    monkeypatch.setattr(run_vps, "NGROK_AUTHTOKEN", "ngrok")
    monkeypatch.setattr(run_vps, "TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr(run_vps, "TELEGRAM_WEBHOOK_SECRET", "")
    monkeypatch.setattr(run_vps, "ZALO_BOT_TOKEN", "")
    monkeypatch.setattr(run_vps, "ZALO_WEBHOOK_SECRET", "")

    with pytest.raises(RuntimeError, match="At least one messaging channel"):
        run_vps.validate_config()


def test_start_owned_process_tracks_process():
    process = FakeProcess()
    owned: list[FakeProcess] = []

    run_vps.start_owned_process(lambda: process, owned)

    assert owned == [process]
