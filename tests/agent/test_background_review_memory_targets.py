"""Structured memory-store reporting for background review results."""

import json

from agent.background_review import ReviewSummary, changed_memory_targets


def _call(call_id, tool, **args):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": call_id, "function": {"name": tool, "arguments": json.dumps(args)}}
        ],
    }


def _result(call_id, **data):
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(data)}


def test_reports_each_changed_store_once():
    msgs = [
        _call("a", "memory", action="add", target="user", content="x"),
        _result("a", success=True, target="user", message="Entry added."),
        _call("b", "memory", action="replace", target="memory", old_text="o", content="n"),
        _result("b", success=True, target="memory", message="Entry replaced."),
        _call("c", "memory", action="remove", target="user", old_text="x"),
        _result("c", success=True, target="user", message="Entry removed."),
    ]
    assert sorted(changed_memory_targets(msgs, [])) == ["memory", "user"]


def test_default_target_is_memory_and_batch_counts():
    msgs = [
        _call("a", "memory", operations=[{"action": "add", "content": "x"}]),
        _result("a", success=True, message="Applied 1 operation(s)."),
    ]
    assert changed_memory_targets(msgs, []) == ["memory"]


def test_skill_only_failed_and_noop_results_report_nothing():
    msgs = [
        _call("s", "skill_manage", action="create", name="n"),
        _result("s", success=True, message="Skill created."),
        _call("f", "memory", action="add", target="user", content="x"),
        _result("f", success=False, error="over limit"),
        _call("d", "memory", action="add", target="user", content="x"),
        _result("d", success=True, target="user", message="Entry already exists (no duplicate added)."),
    ]
    assert changed_memory_targets(msgs, []) == []


def test_results_inherited_from_prior_history_are_ignored():
    prior = [_result("a", success=True, target="user", message="Entry added.")]
    msgs = [
        _call("a", "memory", action="add", target="user", content="x"),
        _result("a", success=True, target="user", message="Entry added."),
    ]
    assert changed_memory_targets(msgs, prior) == []


def test_review_summary_is_a_plain_string_with_targets():
    summary = ReviewSummary("text", ["user"])
    assert summary == "text" and isinstance(summary, str)
    assert summary.memory_targets == ("user",)
    assert ReviewSummary("text").memory_targets == ()
