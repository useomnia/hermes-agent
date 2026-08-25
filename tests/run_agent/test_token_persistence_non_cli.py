from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch
import json
import sys

from run_agent import AIAgent


def _mock_response(*, usage: dict, content: str = "done", model: str | None = "test/model"):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model=model,
        usage=SimpleNamespace(**usage),
    )


def _make_agent(session_db, *, platform: str, response_model: str | None = "test/model"):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="openrouter/auto",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=session_db,
            session_id=f"{platform}-session",
            platform=platform,
        )
    agent.client = MagicMock()
    agent.client.chat.completions.create.return_value = _mock_response(
        model=response_model,
        usage={
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
        }
    )
    return agent


def test_run_conversation_persists_tokens_for_telegram_sessions():
    session_db = MagicMock()
    agent = _make_agent(session_db, platform="telegram")

    result = agent.run_conversation("hello")

    assert result["final_response"] == "done"
    session_db.update_token_counts.assert_called_once()
    assert session_db.update_token_counts.call_args.args[0] == "telegram-session"


def test_run_conversation_persists_tokens_for_cron_sessions():
    session_db = MagicMock()
    agent = _make_agent(session_db, platform="cron")

    result = agent.run_conversation("hello")

    assert result["final_response"] == "done"
    session_db.update_token_counts.assert_called_once()
    assert session_db.update_token_counts.call_args.args[0] == "cron-session"


def test_run_conversation_records_observed_response_model_without_changing_requested_model():
    session_db = MagicMock()
    agent = _make_agent(
        session_db,
        platform="telegram",
        response_model="anthropic/claude-sonnet-4",
    )

    result = agent.run_conversation("hello")

    assert result["final_response"] == "done"
    kwargs = session_db.update_token_counts.call_args.kwargs
    assert kwargs["model"] == "openrouter/auto"
    assert kwargs["usage_model"] == "anthropic/claude-sonnet-4"


def test_run_conversation_falls_back_to_requested_model_when_response_model_missing():
    session_db = MagicMock()
    agent = _make_agent(session_db, platform="telegram", response_model=None)

    result = agent.run_conversation("hello")

    assert result["final_response"] == "done"
    kwargs = session_db.update_token_counts.call_args.kwargs
    assert kwargs["model"] == "openrouter/auto"
    assert kwargs["usage_model"] == "openrouter/auto"


def test_session_search_lazily_opens_db_when_entrypoint_did_not_pass_one(monkeypatch):
    sentinel_db = object()
    captured = {}

    class FakeSessionDB:
        def __new__(cls):
            return sentinel_db

    hermes_state = ModuleType("hermes_state")
    hermes_state.SessionDB = FakeSessionDB
    monkeypatch.setitem(sys.modules, "hermes_state", hermes_state)

    def fake_session_search(**kwargs):
        captured.update(kwargs)
        return json.dumps({"success": True, "results": []})

    # The agent dispatches through the registry, whose built-in handler closes
    # over the already-imported module. Patch that handler target directly.
    monkeypatch.setattr("tools.session_search_tool.session_search", fake_session_search)

    agent = _make_agent(None, platform="acp")
    result = json.loads(agent._invoke_tool("session_search", {"query": "Hermes"}, "task-id"))

    assert result["success"] is True
    assert captured["db"] is sentinel_db
    assert captured["query"] == "Hermes"
    assert agent._session_db is sentinel_db
