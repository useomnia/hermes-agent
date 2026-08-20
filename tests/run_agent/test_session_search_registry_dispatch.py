"""Regression coverage for registry-backed session_search dispatch.

The built-in tool needs the owning agent's SessionDB, while plugins may
intentionally override its registry entry.  Both agent execution paths must
keep the former context and honor the latter override.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from tools.registry import registry


def _make_agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id="session-current",
        )


def _tool_call():
    return SimpleNamespace(
        id="session-search-1",
        function=SimpleNamespace(
            name="session_search",
            arguments=json.dumps({"query": "retention policy"}),
        ),
    )


def _run(agent, path: str):
    if path == "sequential":
        messages = []
        assistant_message = SimpleNamespace(content="", tool_calls=[_tool_call()])
        agent._execute_tool_calls_sequential(assistant_message, messages, "task-1")
        return json.loads(messages[-1]["content"])
    if path == "runtime_helper":
        return json.loads(
            agent._invoke_tool("session_search", {"query": "retention policy"}, "task-1")
        )
    raise AssertionError(f"unknown execution path: {path}")


def _restore_entry(entry) -> None:
    registry.register(
        name=entry.name,
        toolset=entry.toolset,
        schema=entry.schema,
        handler=entry.handler,
        check_fn=entry.check_fn,
        requires_env=entry.requires_env,
        is_async=entry.is_async,
        description=entry.description,
        emoji=entry.emoji,
        max_result_size_chars=entry.max_result_size_chars,
        dynamic_schema_overrides=entry.dynamic_schema_overrides,
        override=True,
    )


@pytest.mark.parametrize("path", ["sequential", "runtime_helper"])
@pytest.mark.parametrize("has_session_db", [True, False])
def test_registered_session_search_override_wins_in_every_agent_path(path, has_session_db):
    agent = _make_agent()
    session_db = object() if has_session_db else None
    agent._get_session_db_for_recall = MagicMock(return_value=session_db)
    original = registry.get_entry("session_search")
    assert original is not None
    captured = {}

    def plugin_handler(args, **kwargs):
        captured.update({"args": args, **kwargs})
        return json.dumps({"implementation": "plugin"})

    try:
        registry.register(
            name="session_search",
            toolset="plugin_session_search",
            schema={"name": "session_search", "parameters": {"type": "object"}},
            handler=plugin_handler,
            override=True,
        )

        assert _run(agent, path) == {"implementation": "plugin"}
        assert captured["args"] == {"query": "retention policy"}
        assert captured["db"] is session_db
        assert captured["current_session_id"] == "session-current"
    finally:
        _restore_entry(original)


@pytest.mark.parametrize("path", ["sequential", "runtime_helper"])
def test_builtin_session_search_handler_keeps_agent_context(monkeypatch, path):
    agent = _make_agent()
    session_db = object()
    agent._get_session_db_for_recall = MagicMock(return_value=session_db)
    captured = {}

    def builtin_session_search(**kwargs):
        captured.update(kwargs)
        return json.dumps({"implementation": "builtin"})

    monkeypatch.setattr("tools.session_search_tool.session_search", builtin_session_search)

    assert _run(agent, path) == {"implementation": "builtin"}
    assert captured["query"] == "retention policy"
    assert captured["db"] is session_db
    assert captured["current_session_id"] == "session-current"


@pytest.mark.parametrize("path", ["sequential", "runtime_helper"])
def test_builtin_session_search_without_a_session_db_keeps_unavailable_error(monkeypatch, path):
    agent = _make_agent()
    agent._get_session_db_for_recall = MagicMock(return_value=None)
    builtin_session_search = MagicMock(side_effect=AssertionError("must not dispatch"))

    monkeypatch.setattr("hermes_state.format_session_db_unavailable", lambda: "Session DB unavailable")
    monkeypatch.setattr("tools.session_search_tool.session_search", builtin_session_search)

    assert _run(agent, path) == {
        "success": False,
        "error": "Session DB unavailable",
    }
    builtin_session_search.assert_not_called()
