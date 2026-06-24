"""Focused tests for API server session-control endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


@pytest.fixture
def adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    return adapter


@pytest.fixture
def auth_adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-test"}))
    adapter._session_db = session_db
    return adapter


def _create_session_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_get("/api/sessions", adapter._handle_list_sessions)
    app.router.add_post("/api/sessions", adapter._handle_create_session)
    app.router.add_get("/api/sessions/{session_id}", adapter._handle_get_session)
    app.router.add_patch("/api/sessions/{session_id}", adapter._handle_patch_session)
    app.router.add_delete("/api/sessions/{session_id}", adapter._handle_delete_session)
    app.router.add_get("/api/sessions/{session_id}/messages", adapter._handle_session_messages)
    app.router.add_put("/api/sessions/{session_id}/messages", adapter._handle_replace_session_messages)
    app.router.add_delete("/api/sessions/{session_id}/messages", adapter._handle_delete_session_messages)
    app.router.add_post("/api/sessions/{session_id}/fork", adapter._handle_fork_session)
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    app.router.add_post("/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)
    return app


@pytest.mark.asyncio
async def test_capabilities_advertises_session_control_surface(adapter):
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/v1/capabilities")
        assert resp.status == 200
        data = await resp.json()

    features = data["features"]
    assert features["session_resources"] is True
    assert features["session_chat"] is True
    assert features["session_chat_streaming"] is True
    assert features["session_fork"] is True
    assert features["admin_config_rw"] is False
    assert features["memory_write_api"] is False
    assert features["skills_api"] is True
    assert features["realtime_voice"] is False
    assert data["endpoints"]["sessions"] == {"method": "GET", "path": "/api/sessions"}
    assert data["endpoints"]["session_chat_stream"] == {
        "method": "POST",
        "path": "/api/sessions/{session_id}/chat/stream",
    }


@pytest.mark.asyncio
async def test_run_agent_binds_api_session_context_for_tool_env(adapter, monkeypatch):
    """API-server request sessions should reach tools and terminal subprocess env."""
    monkeypatch.setenv("HERMES_SESSION_ID", "stale-session")
    observed = {}

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self, session_id: str):
            self.session_id = session_id

        def run_conversation(self, user_message, conversation_history, task_id):
            from gateway.session_context import get_session_env
            from tools.environments.local import _make_run_env

            observed["task_id"] = task_id
            observed["context_session_id"] = get_session_env("HERMES_SESSION_ID")
            observed["context_platform"] = get_session_env("HERMES_SESSION_PLATFORM")
            observed["context_session_key"] = get_session_env("HERMES_SESSION_KEY")
            observed["child_session_id"] = _make_run_env({}).get("HERMES_SESSION_ID")
            return {"final_response": "ok"}

    def fake_create_agent(**kwargs):
        return FakeAgent(kwargs["session_id"])

    monkeypatch.setattr(adapter, "_create_agent", fake_create_agent)

    result, usage = await adapter._run_agent(
        user_message="hello",
        conversation_history=[],
        session_id="request-session",
        gateway_session_key="request-key",
    )

    assert result["session_id"] == "request-session"
    assert usage == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    assert observed == {
        "task_id": "request-session",
        "context_session_id": "request-session",
        "context_platform": "api_server",
        "context_session_key": "request-key",
        "child_session_id": "request-session",
    }


@pytest.mark.asyncio
async def test_session_crud_and_message_history(adapter, session_db):
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        create_resp = await cli.post("/api/sessions", json={"title": "Mobile chat", "model": "test-model"})
        assert create_resp.status == 201
        created = await create_resp.json()
        session_id = created["session"]["id"]
        assert created["object"] == "hermes.session"
        assert created["session"]["title"] == "Mobile chat"

        session_db.append_message(session_id, "user", "hello from phone")
        session_db.append_message(session_id, "assistant", "hello from hermes")

        list_resp = await cli.get("/api/sessions?limit=10&offset=0")
        assert list_resp.status == 200
        listed = await list_resp.json()
        assert listed["object"] == "list"
        assert [s["id"] for s in listed["data"]] == [session_id]
        assert listed["data"][0]["message_count"] == 2

        get_resp = await cli.get(f"/api/sessions/{session_id}")
        assert get_resp.status == 200
        got = await get_resp.json()
        assert got["session"]["id"] == session_id
        assert got["session"]["message_count"] == 2

        messages_resp = await cli.get(f"/api/sessions/{session_id}/messages")
        assert messages_resp.status == 200
        messages = await messages_resp.json()
        assert messages["object"] == "list"
        assert [m["role"] for m in messages["data"]] == ["user", "assistant"]
        assert messages["data"][0]["content"] == "hello from phone"

        patch_resp = await cli.patch(f"/api/sessions/{session_id}", json={"title": "Renamed"})
        assert patch_resp.status == 200
        patched = await patch_resp.json()
        assert patched["session"]["title"] == "Renamed"

        delete_resp = await cli.delete(f"/api/sessions/{session_id}")
        assert delete_resp.status == 200
        deleted = await delete_resp.json()
        assert deleted == {"object": "hermes.session.deleted", "id": session_id, "deleted": True}
        assert session_db.get_session(session_id) is None


@pytest.mark.asyncio
async def test_session_messages_follow_compression_tip(adapter, session_db):
    source_id = session_db.create_session("source-session", "api_server")
    session_db.append_message(source_id, "user", "before compression")
    session_db.end_session(source_id, "compression")
    session_db.create_session("tip-session", "api_server", parent_session_id=source_id)
    session_db.replace_messages(source_id, [])
    session_db.append_message("tip-session", "user", "after compression")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        messages_resp = await cli.get(f"/api/sessions/{source_id}/messages")
        assert messages_resp.status == 200
        messages = await messages_resp.json()

    assert messages["object"] == "list"
    assert messages["session_id"] == "tip-session"
    assert [m["content"] for m in messages["data"]] == ["after compression"]


@pytest.mark.asyncio
async def test_session_fork_uses_current_sessiondb_branch_primitives(adapter, session_db):
    source_id = session_db.create_session("source-session", "api_server", model="test-model")
    session_db.set_session_title(source_id, "Original")
    session_db.append_message(source_id, "user", "first path")
    session_db.append_message(source_id, "assistant", "answer")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(f"/api/sessions/{source_id}/fork", json={"title": "Alternative"})
        assert resp.status == 201
        payload = await resp.json()

    fork = payload["session"]
    assert payload["object"] == "hermes.session"
    assert fork["id"] != source_id
    assert fork["parent_session_id"] == source_id
    assert fork["title"] == "Alternative"
    assert [m["content"] for m in session_db.get_messages(fork["id"])] == ["first path", "answer"]
    assert session_db.get_session(source_id)["end_reason"] == "branched"


@pytest.mark.asyncio
async def test_session_chat_loads_history_and_preserves_session_headers(auth_adapter, session_db):
    session_id = session_db.create_session("chat-session", "api_server")
    session_db.set_session_title(session_id, "Chat")
    session_db.append_message(session_id, "user", "earlier")
    session_db.append_message(session_id, "assistant", "prior answer")

    mock_run = AsyncMock(return_value=({"final_response": "fresh answer", "session_id": session_id}, {"total_tokens": 3}))
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "next", "system_message": "stay focused"},
                headers={"Authorization": "Bearer sk-test", "X-Hermes-Session-Key": "client-42"},
            )
            assert resp.status == 200
            payload = await resp.json()

    assert resp.headers["X-Hermes-Session-Id"] == session_id
    assert resp.headers["X-Hermes-Session-Key"] == "client-42"
    assert payload["object"] == "hermes.session.chat.completion"
    assert payload["session_id"] == session_id
    assert payload["message"]["role"] == "assistant"
    assert payload["message"]["content"] == "fresh answer"
    mock_run.assert_awaited_once()
    _, kwargs = mock_run.call_args
    assert kwargs["session_id"] == session_id
    assert kwargs["gateway_session_key"] == "client-42"
    assert kwargs["ephemeral_system_prompt"] == "stay focused"
    history = kwargs["conversation_history"]
    assert len(history) == 2
    assert isinstance(history[0].pop("timestamp"), (int, float))
    assert isinstance(history[1].pop("timestamp"), (int, float))
    assert history == [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "prior answer"},
    ]


@pytest.mark.asyncio
async def test_session_chat_accepts_multimodal_message(auth_adapter, session_db):
    session_id = session_db.create_session("image-session", "api_server")
    image_payload = [
        {"type": "input_text", "text": "What's in this image?"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
    ]
    expected_user_message = [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]

    mock_run = AsyncMock(return_value=({"final_response": "A cat.", "session_id": session_id}, {"total_tokens": 4}))
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": image_payload},
                headers={"Authorization": "Bearer sk-test"},
            )
            assert resp.status == 200, await resp.text()

    _, kwargs = mock_run.call_args
    assert kwargs["user_message"] == expected_user_message


@pytest.mark.asyncio
async def test_session_chat_stream_accepts_multimodal_message(adapter, session_db):
    session_id = session_db.create_session("image-stream-session", "api_server")
    image_payload = [
        {"type": "input_text", "text": "What's in this image?"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
    ]
    expected_user_message = [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    captured_kwargs = {}

    async def fake_run(**kwargs):
        captured_kwargs.update(kwargs)
        kwargs["stream_delta_callback"]("A cat.")
        return {"final_response": "A cat.", "session_id": session_id}, {"total_tokens": 4}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={"message": image_payload},
            )
            assert resp.status == 200, await resp.text()
            assert resp.headers["Content-Type"].startswith("text/event-stream")
            body = await resp.text()

    assert "event: assistant.completed" in body
    assert captured_kwargs["user_message"] == expected_user_message


@pytest.mark.asyncio
async def test_session_chat_stream_emits_lifecycle_events_and_keepalive_safe_shape(adapter, session_db):
    session_id = session_db.create_session("stream-session", "api_server")
    session_db.set_session_title(session_id, "Stream")

    async def fake_run(**kwargs):
        kwargs["stream_delta_callback"]("Hello")
        kwargs["stream_delta_callback"](" world")
        kwargs["tool_progress_callback"]("reasoning.available", tool_name="_thinking", preview="thinking")
        return {"final_response": "Hello world", "session_id": session_id}, {"total_tokens": 2}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(f"/api/sessions/{session_id}/chat/stream", json={"message": "stream please"})
            assert resp.status == 200
            assert resp.headers["Content-Type"].startswith("text/event-stream")
            body = await resp.text()

    assert "event: run.started" in body
    assert "event: message.started" in body
    assert "event: assistant.delta" in body
    assert "Hello world" in body
    assert "event: tool.progress" in body
    assert "event: assistant.completed" in body
    assert "event: run.completed" in body
    assert "event: done" in body


@pytest.mark.asyncio
async def test_session_chat_stream_run_completed_carries_turn_transcript(adapter, session_db):
    """run.completed must include the full interleaved turn transcript so a
    client that lost intermediate (pre-tool-call) assistant text from the live
    delta stream can reconcile without a separate /messages fetch. Refs #34703.
    """
    import json as _json

    session_id = session_db.create_session("transcript-session", "api_server")

    async def fake_run(**kwargs):
        # Stream the intermediate planning text the way a real turn would.
        kwargs["stream_delta_callback"]("Let me search for that:")
        kwargs["stream_delta_callback"]("Here is the summary.")
        result = {
            "final_response": "Here is the summary.",
            "session_id": session_id,
            "messages": [
                {"role": "user", "content": "search then summarize"},
                {
                    "role": "assistant",
                    "content": "Let me search for that:",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "content": "results", "tool_call_id": "call_1", "tool_name": "web_search"},
                {"role": "assistant", "content": "Here is the summary."},
            ],
        }
        return result, {"total_tokens": 6}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={"message": "search then summarize"},
            )
            assert resp.status == 200
            body = await resp.text()

    # Pull the run.completed event payload out of the SSE body.
    run_completed_payload = None
    for block in body.split("\n\n"):
        if "event: run.completed" in block:
            for line in block.splitlines():
                if line.startswith("data: "):
                    run_completed_payload = _json.loads(line[len("data: "):])
            break
    assert run_completed_payload is not None, body
    messages = run_completed_payload.get("messages")
    assert isinstance(messages, list) and messages, run_completed_payload

    # The colon-ended intermediate text that preceded the tool call must be present.
    contents = [m.get("content") for m in messages]
    assert "Let me search for that:" in contents
    assert "Here is the summary." in contents
    # No prior-turn user message should leak into the per-turn slice.
    assert all(m.get("role") in ("assistant", "tool") for m in messages)
    # The tool call is preserved alongside the intermediate text.
    assert any(m.get("tool_calls") for m in messages)



@pytest.mark.asyncio
async def test_session_endpoints_require_auth_when_key_configured(auth_adapter):
    app = _create_session_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/api/sessions")
        assert resp.status == 401
        body = await resp.json()
        assert body["error"]["code"] == "invalid_api_key"

        ok = await cli.get("/api/sessions", headers={"Authorization": "Bearer sk-test"})
        assert ok.status == 200
        data = await ok.json()
        assert data["object"] == "list"
        assert data["data"] == []


@pytest.mark.asyncio
async def test_session_header_rejected_without_api_key(adapter, session_db):
    session_id = session_db.create_session("unsafe-session", "api_server")
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            f"/api/sessions/{session_id}/chat",
            json={"message": "hello"},
            headers={"X-Hermes-Session-Key": "client-42"},
        )
        assert resp.status == 403
        data = await resp.json()
        assert "X-Hermes-Session-Key requires API key" in data["error"]["message"]


@pytest.mark.asyncio
async def test_put_messages_seeds_a_missing_session(adapter, session_db):
    """PUT creates the session row when absent (the cold-seed path after a fresh
    state.db) and stores the supplied transcript."""
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.put(
            "/api/sessions/cold-seed/messages",
            json={"messages": [
                {"role": "user", "content": "restored question"},
                {"role": "assistant", "content": "restored answer"},
            ]},
        )
        assert resp.status == 200, await resp.text()
        assert await resp.json() == {
            "object": "hermes.session.messages",
            "session_id": "cold-seed",
            "count": 2,
        }

    assert session_db.get_session("cold-seed") is not None
    assert [m["content"] for m in session_db.get_messages("cold-seed")] == [
        "restored question",
        "restored answer",
    ]


@pytest.mark.asyncio
async def test_put_messages_preserves_reasoning_and_tool_calls_round_trip(adapter, session_db):
    """A transcript PUT back carries assistant reasoning + tool calls verbatim —
    the full-fidelity property the cold-seed relies on."""
    session_id = session_db.create_session("fidelity", "api_server")
    transcript = [
        {"role": "user", "content": "search please"},
        {
            "role": "assistant",
            "content": "calling search",
            "reasoning_content": "the user wants a search",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "content": "results", "tool_call_id": "call_1", "tool_name": "web_search"},
        {"role": "assistant", "content": "here you go"},
    ]
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.put(f"/api/sessions/{session_id}/messages", json={"messages": transcript})
        assert resp.status == 200, await resp.text()

    restored = session_db.get_messages_as_conversation(session_id)
    assert [m["role"] for m in restored] == ["user", "assistant", "tool", "assistant"]
    assert restored[1]["reasoning_content"] == "the user wants a search"
    assert restored[1]["tool_calls"][0]["function"]["name"] == "web_search"
    assert restored[2]["tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_put_messages_replaces_the_existing_transcript(adapter, session_db):
    """PUT is idempotent set — the prior transcript is wiped, not appended to."""
    session_id = session_db.create_session("replace-me", "api_server")
    session_db.append_message(session_id, "user", "old one")
    session_db.append_message(session_id, "assistant", "old two")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.put(
            f"/api/sessions/{session_id}/messages",
            json={"messages": [{"role": "user", "content": "only this"}]},
        )
        assert resp.status == 200

    assert [m["content"] for m in session_db.get_messages(session_id)] == ["only this"]


@pytest.mark.asyncio
async def test_put_messages_rejects_a_non_list_body(adapter, session_db):
    session_id = session_db.create_session("bad-body", "api_server")
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.put(f"/api/sessions/{session_id}/messages", json={"messages": "nope"})
        assert resp.status == 400
        assert (await resp.json())["error"]["code"] == "invalid_messages"


@pytest.mark.asyncio
async def test_delete_messages_truncates_from_id_keeping_the_prefix(adapter, session_db):
    """DELETE ?from=<id> removes that message and everything after it, leaving
    the earlier rows untouched and the counters recomputed."""
    session_id = session_db.create_session("truncate", "api_server")
    for i in range(4):
        session_db.append_message(session_id, "user" if i % 2 == 0 else "assistant", f"msg {i}")
    boundary_id = session_db.get_messages(session_id)[2]["id"]

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.delete(f"/api/sessions/{session_id}/messages?from={boundary_id}")
        assert resp.status == 200, await resp.text()
        assert await resp.json() == {
            "object": "hermes.session.messages.truncated",
            "session_id": session_id,
            "deleted": 2,
        }

    assert [m["content"] for m in session_db.get_messages(session_id)] == ["msg 0", "msg 1"]
    assert session_db.get_session(session_id)["message_count"] == 2


@pytest.mark.asyncio
async def test_delete_messages_requires_the_from_param(adapter, session_db):
    session_id = session_db.create_session("no-from", "api_server")
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.delete(f"/api/sessions/{session_id}/messages")
        assert resp.status == 400
        assert (await resp.json())["error"]["code"] == "invalid_from"


@pytest.mark.asyncio
async def test_delete_messages_rejects_a_non_integer_from(adapter, session_db):
    session_id = session_db.create_session("bad-from", "api_server")
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.delete(f"/api/sessions/{session_id}/messages?from=abc")
        assert resp.status == 400
        assert (await resp.json())["error"]["code"] == "invalid_from"


@pytest.mark.asyncio
async def test_delete_messages_404_when_session_missing(adapter):
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.delete("/api/sessions/ghost/messages?from=1")
        assert resp.status == 404
        assert (await resp.json())["error"]["code"] == "session_not_found"


@pytest.mark.asyncio
async def test_message_writes_require_auth_when_key_configured(auth_adapter, session_db):
    session_id = session_db.create_session("guarded", "api_server")
    app = _create_session_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        put_resp = await cli.put(f"/api/sessions/{session_id}/messages", json={"messages": []})
        assert put_resp.status == 401
        del_resp = await cli.delete(f"/api/sessions/{session_id}/messages?from=1")
        assert del_resp.status == 401


@pytest.mark.asyncio
async def test_delete_messages_rejects_a_non_positive_from(adapter, session_db):
    """from < 1 would match every row (ids are positive autoincrements) and
    silently wipe the transcript — it must be rejected, not run."""
    session_id = session_db.create_session("floor", "api_server")
    session_db.append_message(session_id, "user", "keep me")
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        for bad in ("0", "-1"):
            resp = await cli.delete(f"/api/sessions/{session_id}/messages?from={bad}")
            assert resp.status == 400, await resp.text()
            assert (await resp.json())["error"]["code"] == "invalid_from"
    # The transcript is untouched by a rejected request.
    assert [m["content"] for m in session_db.get_messages(session_id)] == ["keep me"]


@pytest.mark.asyncio
async def test_delete_messages_from_beyond_range_is_an_idempotent_noop(adapter, session_db):
    """A from_id past the last message deletes nothing and returns 0 — the
    idempotent-retry contract the edit-resend flow relies on."""
    session_id = session_db.create_session("idempotent", "api_server")
    session_db.append_message(session_id, "user", "one")
    session_db.append_message(session_id, "assistant", "two")
    beyond = session_db.get_messages(session_id)[-1]["id"] + 1000

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.delete(f"/api/sessions/{session_id}/messages?from={beyond}")
        assert resp.status == 200, await resp.text()
        assert (await resp.json())["deleted"] == 0

    assert [m["content"] for m in session_db.get_messages(session_id)] == ["one", "two"]
