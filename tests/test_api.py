from __future__ import annotations

import asyncio
import uuid

from src.chat.replies import TECHNICAL_ERROR_REPLY


def test_health_endpoint(app_client):
    client, _ = app_client

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_bachmai_source_pdf_endpoint_serves_split_pdf(
    app_client,
    monkeypatch,
    tmp_path,
):
    client, app_module = app_client
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4\ntest")
    monkeypatch.setattr(
        app_module,
        "resolve_bachmai_source_pdf",
        lambda slug: pdf if slug == "may_day" else None,
    )

    response = client.get("/sources/bachmai/may_day.pdf")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"].startswith("inline;")
    assert response.content == b"%PDF-1.4\ntest"


def test_bachmai_source_pdf_endpoint_returns_404_for_unknown_slug(
    app_client,
    monkeypatch,
):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "resolve_bachmai_source_pdf", lambda slug: None)

    response = client.get("/sources/bachmai/not_found.pdf")

    assert response.status_code == 404


def test_app_uses_lifespan_instead_of_legacy_event_handlers():
    from src.server import app as app_module

    assert app_module.app.router.on_startup == []
    assert app_module.app.router.on_shutdown == []


def test_startup_continues_when_telegram_menu_setup_fails(monkeypatch):
    from src.server import app as app_module

    async def broken_setup_bot_menu() -> None:
        raise RuntimeError("telegram api unavailable")

    monkeypatch.setattr(app_module, "preload_retrieval_models", lambda: None)
    monkeypatch.setattr(app_module, "ensure_fulltext_indexes", lambda: None)
    monkeypatch.setattr(app_module.telegram, "setup_bot_menu", broken_setup_bot_menu)

    asyncio.run(app_module.startup())


def test_startup_initializes_fulltext_indexes(monkeypatch):
    from src.server import app as app_module

    calls: list[str] = []

    async def setup_bot_menu() -> None:
        calls.append("telegram")

    monkeypatch.setattr(app_module, "preload_retrieval_models", lambda: None)
    monkeypatch.setattr(app_module, "ensure_fulltext_indexes", lambda: calls.append("indexes"))
    monkeypatch.setattr(app_module.telegram, "setup_bot_menu", setup_bot_menu)

    asyncio.run(app_module.startup())

    assert calls == ["indexes", "telegram"]


def test_chat_disabled_without_api_key(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "")

    response = client.post("/chat", json={"question": "Tôi bị ho"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Chat API disabled: CHAT_API_KEY not set"


def test_chat_rejects_missing_or_wrong_api_key(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")

    missing = client.post("/chat", json={"question": "Tôi bị ho"})
    wrong = client.post(
        "/chat",
        headers={"X-API-Key": "wrong"},
        json={"question": "Tôi bị ho"},
    )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert wrong.json()["detail"] == "Invalid API key"


def test_chat_rejects_missing_session_id(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")

    def fake_answer(question: str, session_id: str = "default") -> str:
        return "ok"

    monkeypatch.setattr(app_module, "answer", fake_answer)

    response = client.post(
        "/chat",
        headers={"X-API-Key": "secret"},
        json={"question": "Tôi bị ho"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "session_id is required"


def test_chat_scopes_session_to_api_key_and_body_session_id(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")
    seen: list[str] = []

    def fake_answer(question: str, session_id: str = "default") -> str:
        seen.append(session_id)
        return "ok"

    monkeypatch.setattr(app_module, "answer", fake_answer)

    first = client.post(
        "/chat",
        headers={"X-API-Key": "secret"},
        json={"question": "Tôi bị ho", "session_id": "user-a"},
    )
    second = client.post(
        "/chat",
        headers={"X-API-Key": "secret"},
        json={"question": "Tôi bị đau đầu", "session_id": "user-b"},
    )

    expected_first = app_module._scoped_api_session_id("secret", "user-a")
    expected_second = app_module._scoped_api_session_id("secret", "user-b")
    assert first.status_code == 200
    assert second.status_code == 200
    assert seen == [expected_first, expected_second]
    assert seen[0] != seen[1]


def test_chat_does_not_pass_raw_body_session_id_to_pipeline(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")
    seen: dict[str, str] = {}

    def fake_answer(question: str, session_id: str = "default") -> str:
        seen["question"] = question
        seen["session_id"] = session_id
        return "ok"

    monkeypatch.setattr(app_module, "answer", fake_answer)

    response = client.post(
        "/chat",
        headers={"X-API-Key": "secret"},
        json={"question": "Tôi bị ho", "session_id": "victim"},
    )

    expected_session = app_module._scoped_api_session_id("secret", "victim")
    assert response.status_code == 200
    assert response.json() == {"answer": "ok"}
    assert seen == {"question": "Tôi bị ho", "session_id": expected_session}


def test_chat_empty_question_returns_specific_reply(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")

    def fake_answer(question: str, session_id: str = "default") -> str:
        return "Bạn hãy đặt câu hỏi cụ thể nhé." if not question else "unexpected"

    monkeypatch.setattr(app_module, "answer", fake_answer)

    response = client.post(
        "/chat",
        headers={"X-API-Key": "secret"},
        json={"session_id": "user-a"},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "Bạn hãy đặt câu hỏi cụ thể nhé."}


def test_chat_returns_technical_reply_when_pipeline_raises(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")

    def broken_answer(question: str, session_id: str = "default") -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module, "answer", broken_answer)

    response = client.post(
        "/chat",
        headers={"X-API-Key": "secret"},
        json={"question": "Tôi bị đau ngực", "session_id": "user-a"},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": TECHNICAL_ERROR_REPLY}


def test_debug_chat_route_page_is_served(app_client):
    client, _ = app_client
    response = client.get("/debug/chat-route")
    assert response.status_code == 200
    assert "Chat Route Debug Console" in response.text
    assert 'id="workflow-graph"' in response.text
    assert 'id="node-inspector"' in response.text
    assert "app.js" in response.text  # script is now external


def test_debug_console_app_js_is_served(app_client):
    client, _ = app_client
    response = client.get("/debug/chat-route/static/app.js")
    assert response.status_code == 200
    assert "renderWorkflowGraph" in response.text


def test_debug_console_static_asset_rejects_unknown(app_client):
    client, _ = app_client
    assert client.get("/debug/chat-route/static/secrets.txt").status_code == 404


def test_debug_chat_route_run_requires_api_key(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")

    missing = client.post(
        "/debug/chat-route/run",
        json={"question": "Tôi bị ho", "session_id": "debug-user"},
    )
    wrong = client.post(
        "/debug/chat-route/run",
        headers={"X-API-Key": "wrong"},
        json={"question": "Tôi bị ho", "session_id": "debug-user"},
    )

    assert missing.status_code == 401
    assert wrong.status_code == 401


def test_debug_chat_route_run_scopes_saves_and_returns_trace(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")
    seen: dict[str, str] = {}

    def fake_answer_with_meta(question: str, session_id: str = "default", mode: str = "auto"):
        seen["question"] = question
        seen["session_id"] = session_id
        seen["mode"] = mode
        return "debug answer", {
            "trace_id": "trace-api",
            "latency_ms_total": 42.0,
            "route_label": "informational",
            "timings": [{"stage": "total", "ms": 42.0, "fields": {"outcome": "informational"}}],
        }

    monkeypatch.setattr(app_module, "answer_with_meta", fake_answer_with_meta)

    response = client.post(
        "/debug/chat-route/run",
        headers={"X-API-Key": "secret"},
        json={"question": "Tôi bị ho", "session_id": "debug-user", "mode": "information"},
    )

    expected_internal = app_module._scoped_api_session_id("secret", "debug-user")
    data = response.json()
    assert response.status_code == 200
    assert seen == {
        "question": "Tôi bị ho",
        "session_id": expected_internal,
        "mode": "information",
    }
    persisted_trace_id = data["trace"]["trace_id"]
    assert str(uuid.UUID(persisted_trace_id)) == persisted_trace_id
    assert persisted_trace_id != "trace-api"
    assert data["trace"]["session_id"] == expected_internal
    assert data["trace"]["internal_session_id"] == expected_internal
    assert data["trace"]["answer"] == "debug answer"
    assert data["trace"]["created_at"] > 0
    assert data["trace"]["meta"]["trace_id"] == persisted_trace_id
    assert data["trace"]["meta"]["pipeline_trace_id"] == "trace-api"


def test_debug_chat_route_lists_and_loads_saved_trace(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")

    def fake_answer_with_meta(question: str, session_id: str = "default", mode: str = "auto"):
        return "answer", {
            "trace_id": "trace-list",
            "latency_ms_total": 7.0,
            "outcome": "diagnostic",
            "timings": [{"stage": "total", "ms": 7.0, "fields": {"outcome": "diagnostic"}}],
        }

    monkeypatch.setattr(app_module, "answer_with_meta", fake_answer_with_meta)
    run = client.post(
        "/debug/chat-route/run",
        headers={"X-API-Key": "secret"},
        json={"question": "Tôi đau đầu", "session_id": "debug-user"},
    )
    trace_id = run.json()["trace"]["trace_id"]

    listed = client.get(
        "/debug/chat-route/traces?session_id=debug-user",
        headers={"X-API-Key": "secret"},
    )
    detail = client.get(
        f"/debug/chat-route/traces/{trace_id}?session_id=debug-user",
        headers={"X-API-Key": "secret"},
    )
    missing = client.get(
        "/debug/chat-route/traces/missing?session_id=debug-user",
        headers={"X-API-Key": "secret"},
    )
    wrong_session_list = client.get(
        "/debug/chat-route/traces?session_id=other-user",
        headers={"X-API-Key": "secret"},
    )
    wrong_session_detail = client.get(
        f"/debug/chat-route/traces/{trace_id}?session_id=other-user",
        headers={"X-API-Key": "secret"},
    )

    assert listed.status_code == 200
    assert listed.json()["traces"][0]["trace_id"] == trace_id
    assert detail.status_code == 200
    assert detail.json()["trace"]["question"] == "Tôi đau đầu"
    assert missing.status_code == 404
    assert wrong_session_list.status_code == 200
    assert wrong_session_list.json() == {"traces": []}
    assert wrong_session_detail.status_code == 404


def test_debug_chat_route_list_and_detail_require_api_key(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")

    missing_list = client.get("/debug/chat-route/traces?session_id=debug-user")
    wrong_list = client.get(
        "/debug/chat-route/traces?session_id=debug-user",
        headers={"X-API-Key": "wrong"},
    )
    missing_detail = client.get(
        "/debug/chat-route/traces/trace-list?session_id=debug-user"
    )
    wrong_detail = client.get(
        "/debug/chat-route/traces/trace-list?session_id=debug-user",
        headers={"X-API-Key": "wrong"},
    )

    assert missing_list.status_code == 401
    assert wrong_list.status_code == 401
    assert missing_detail.status_code == 401
    assert wrong_detail.status_code == 401


def test_debug_chat_route_list_and_detail_require_session_id(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")

    missing_list = client.get(
        "/debug/chat-route/traces",
        headers={"X-API-Key": "secret"},
    )
    missing_detail = client.get(
        "/debug/chat-route/traces/trace-list",
        headers={"X-API-Key": "secret"},
    )

    assert missing_list.status_code == 400
    assert missing_detail.status_code == 400


def test_debug_chat_route_run_uses_fallback_trace_id_and_created_at_when_pipeline_raises(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")

    def broken_answer_with_meta(question: str, session_id: str = "default", mode: str = "auto"):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module, "answer_with_meta", broken_answer_with_meta)

    response = client.post(
        "/debug/chat-route/run",
        headers={"X-API-Key": "secret"},
        json={"question": "Tôi bị ho", "session_id": "debug-user"},
    )

    data = response.json()
    trace_id = data["trace"]["trace_id"]
    assert response.status_code == 200
    assert trace_id
    assert data["trace"]["meta"]["trace_id"] == trace_id
    assert data["trace"]["created_at"] > 0
    assert data["trace"]["answer"] == TECHNICAL_ERROR_REPLY
    assert "warning" not in data

    replay = client.get(
        f"/debug/chat-route/traces/{trace_id}?session_id=debug-user",
        headers={"X-API-Key": "secret"},
    )

    assert replay.status_code == 200
    assert replay.json()["trace"]["trace_id"] == trace_id
    assert replay.json()["trace"]["answer"] == TECHNICAL_ERROR_REPLY


def test_debug_chat_route_run_warns_when_trace_save_fails(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")

    def fake_answer_with_meta(question: str, session_id: str = "default", mode: str = "auto"):
        return "debug answer", {"latency_ms_total": 1.0}

    def broken_save_chat_trace(**kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(app_module, "answer_with_meta", fake_answer_with_meta)
    monkeypatch.setattr(app_module, "save_chat_trace", broken_save_chat_trace)

    response = client.post(
        "/debug/chat-route/run",
        headers={"X-API-Key": "secret"},
        json={"question": "Tôi bị ho", "session_id": "debug-user"},
    )

    data = response.json()
    assert response.status_code == 200
    assert data["warning"] == "trace_persistence_failed"
    assert data["trace"]["trace_id"]
    assert data["trace"]["created_at"] > 0
    assert data["trace"]["meta"]["trace_id"] == data["trace"]["trace_id"]


def test_debug_chat_route_runs_with_same_pipeline_trace_id_get_unique_persisted_ids(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")

    def fake_answer_with_meta(question: str, session_id: str = "default", mode: str = "auto"):
        return "answer", {"trace_id": "short-pipeline-id"}

    monkeypatch.setattr(app_module, "answer_with_meta", fake_answer_with_meta)

    first = client.post(
        "/debug/chat-route/run",
        headers={"X-API-Key": "secret"},
        json={"question": "first", "session_id": "debug-user"},
    ).json()["trace"]
    second = client.post(
        "/debug/chat-route/run",
        headers={"X-API-Key": "secret"},
        json={"question": "second", "session_id": "debug-user"},
    ).json()["trace"]

    assert first["trace_id"] != second["trace_id"]
    assert first["meta"]["pipeline_trace_id"] == "short-pipeline-id"
    assert second["meta"]["pipeline_trace_id"] == "short-pipeline-id"
    assert first["meta"]["trace_id"] == first["trace_id"]
    assert second["meta"]["trace_id"] == second["trace_id"]

    listed = client.get(
        "/debug/chat-route/traces?session_id=debug-user",
        headers={"X-API-Key": "secret"},
    )
    assert listed.status_code == 200
    assert {row["trace_id"] for row in listed.json()["traces"]} == {
        first["trace_id"],
        second["trace_id"],
    }


def test_debug_chat_route_page_shows_created_at_fields(app_client):
    client, _ = app_client

    response = client.get("/debug/chat-route/static/app.js")

    assert response.status_code == 200
    assert "created_at" in response.text
    assert 'params.set("session_id"' in response.text
    assert "loadTrace(trace.trace_id)" in response.text


def test_debug_chat_route_stream_requires_api_key(app_client, monkeypatch):
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")
    missing = client.post("/debug/chat-route/stream",
                          json={"question": "Tôi bị ho", "session_id": "debug-user"})
    wrong = client.post("/debug/chat-route/stream",
                        headers={"X-API-Key": "wrong"},
                        json={"question": "Tôi bị ho", "session_id": "debug-user"})
    assert missing.status_code == 401
    assert wrong.status_code == 401


def test_debug_chat_route_stream_emits_nodes_then_done_and_persists(app_client, monkeypatch):
    import json as _json
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")

    def fake_answer_with_meta(question, session_id="default", mode="auto"):
        # Drive node events through the REAL sink installed by the endpoint.
        from src.chat import pipeline
        import time
        pipeline._log_timing("t", "route", time.perf_counter(), label="informational")
        pipeline._log_timing("t", "generate", time.perf_counter(), chars=5)
        return "stream answer", {
            "trace_id": "trace-stream",
            "latency_ms_total": 9.0,
            "timings": [{"stage": "total", "ms": 9.0, "fields": {}}],
            "graph_nodes": [],
        }

    monkeypatch.setattr(app_module, "answer_with_meta", fake_answer_with_meta)

    with client.stream("POST", "/debug/chat-route/stream",
                       headers={"X-API-Key": "secret"},
                       json={"question": "Tôi bị ho", "session_id": "debug-user",
                             "mode": "information"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = []
        for line in resp.iter_lines():
            if line.startswith("data:"):
                events.append(_json.loads(line[len("data:"):].strip()))

    node_ids = [e["id"] for e in events if e.get("type") == "node"]
    done = [e for e in events if e.get("type") == "done"]
    assert "route" in node_ids and "generate" in node_ids
    assert len(done) == 1
    saved_trace = done[0]["trace"]
    assert saved_trace["answer"] == "stream answer"
    assert saved_trace["trace_id"]

    replay = client.get(
        f"/debug/chat-route/traces/{saved_trace['trace_id']}?session_id=debug-user",
        headers={"X-API-Key": "secret"},
    )
    assert replay.status_code == 200


def test_debug_chat_route_stream_done_on_worker_exception(app_client, monkeypatch):
    import json as _json
    from src.chat.replies import TECHNICAL_ERROR_REPLY
    client, app_module = app_client
    monkeypatch.setattr(app_module, "CHAT_API_KEY", "secret")

    def broken(question, session_id="default", mode="auto"):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module, "answer_with_meta", broken)

    with client.stream("POST", "/debug/chat-route/stream",
                       headers={"X-API-Key": "secret"},
                       json={"question": "x", "session_id": "debug-user"}) as resp:
        events = []
        for line in resp.iter_lines():
            if line.startswith("data:"):
                events.append(_json.loads(line[len("data:"):].strip()))

    done = [e for e in events if e.get("type") == "done"]
    assert len(done) == 1
    assert done[0]["trace"]["answer"] == TECHNICAL_ERROR_REPLY
