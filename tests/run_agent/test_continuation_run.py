"""A no-user continuation resumes saved history and never re-runs an unapproved call."""

import json
from types import SimpleNamespace

import pytest


class _FakeChatCompletions:
    def __init__(self):
        self.requests: list[list[dict]] = []

    def create(self, **kwargs):
        self.requests.append([dict(message) for message in kwargs["messages"]])
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="continued", reasoning=None, tool_calls=[]),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )


def _call(call_id: str, name: str) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


@pytest.fixture
def run(monkeypatch):
    from run_agent import AIAgent

    completions = _FakeChatCompletions()
    dispatched: list[str] = []
    monkeypatch.setattr(
        "run_agent.OpenAI",
        lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    monkeypatch.setattr(
        "run_agent.get_tool_definitions",
        lambda *args, **kwargs: [
            {"function": {"name": name}} for name in ("mcp_write", "terminal", "web_search")
        ],
    )

    def _dispatch(name, args, task_id=None, **kwargs):
        dispatched.append(name)
        return json.dumps({"ok": True})

    monkeypatch.setattr("run_agent.handle_function_call", _dispatch)

    def _run(history):
        agent = AIAgent(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:8080/v1",
            platform="cli",
            max_iterations=3,
            quiet_mode=True,
            skip_memory=True,
        )
        agent._disable_streaming = True
        result = agent.run_conversation("", conversation_history=history, continuation=True)
        return result, completions.requests, dispatched

    return _run


def test_continues_from_a_tool_result_without_a_user_message(run):
    history = [
        {"role": "user", "content": "research this"},
        {"role": "assistant", "content": "", "tool_calls": [_call("c1", "web_search")]},
        {"role": "tool", "tool_call_id": "c1", "content": "results"},
    ]
    result, requests, dispatched = run(history)

    assert result["final_response"].startswith("continued")
    roles = [m["role"] for m in requests[0] if m["role"] != "system"]
    assert roles == ["user", "assistant", "tool"]
    assert [m["content"] for m in requests[0] if m["role"] == "user"] == ["research this"]
    assert dispatched == []


def test_only_the_approved_call_is_re_dispatched(run):
    history = [
        {"role": "user", "content": "write it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("w1", "mcp_write"), _call("t2", "terminal")],
            "display_metadata": {
                "_omnio_resolved_approvals": {
                    "w1": {"scope": "once", "tool_name": "mcp_write", "arguments": "{}"}
                }
            },
        },
    ]
    _result, requests, dispatched = run(history)

    assert dispatched == ["mcp_write"]
    tool_ids = [m.get("tool_call_id") for m in requests[0] if m["role"] == "tool"]
    assert "w1" in tool_ids


def test_a_mismatched_grant_never_dispatches(run):
    history = [
        {"role": "user", "content": "write it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("w1", "mcp_write")],
            "display_metadata": {
                "_omnio_resolved_approvals": {
                    "w1": {"scope": "once", "tool_name": "other_tool", "arguments": "{}"}
                }
            },
        },
    ]
    _result, _requests, dispatched = run(history)

    assert dispatched == []
