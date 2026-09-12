"""Tests for _repair_tool_call_arguments — malformed JSON repair pipeline."""

import json

from run_agent import _repair_tool_call_arguments
from agent.message_sanitization import (
    _close_truncated_json,
    _repair_tool_call_arguments_detailed,
)


class TestRepairToolCallArguments:
    """Verify each repair stage in the pipeline."""

    # -- Stage 1: empty / whitespace-only --

    def test_empty_string_returns_empty_object(self):
        assert _repair_tool_call_arguments("", "t") == "{}"

    def test_whitespace_only_returns_empty_object(self):
        assert _repair_tool_call_arguments("   \n\t  ", "t") == "{}"

    def test_none_type_returns_empty_object(self):
        """Non-string input (e.g. None from a broken model response)."""
        assert _repair_tool_call_arguments(None, "t") == "{}"

    # -- Stage 2: Python None literal --

    def test_python_none_literal(self):
        assert _repair_tool_call_arguments("None", "t") == "{}"

    def test_python_none_with_whitespace(self):
        assert _repair_tool_call_arguments("  None  ", "t") == "{}"

    # -- Stage 3: trailing comma repair --

    def test_trailing_comma_in_object(self):
        result = _repair_tool_call_arguments('{"key": "value",}', "t")
        assert json.loads(result) == {"key": "value"}

    def test_trailing_comma_in_array(self):
        result = _repair_tool_call_arguments('{"a": [1, 2,]}', "t")
        parsed = json.loads(result)
        assert parsed == {"a": [1, 2]}

    def test_multiple_trailing_commas(self):
        result = _repair_tool_call_arguments('{"a": 1, "b": 2,}', "t")
        parsed = json.loads(result)
        assert parsed["a"] == 1
        assert parsed["b"] == 2

    # -- Stage 4: unclosed brackets --

    def test_unclosed_brace(self):
        result = _repair_tool_call_arguments('{"key": "value"', "t")
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_unclosed_bracket_and_brace(self):
        result = _repair_tool_call_arguments('{"a": [1, 2', "t")
        # Bracket counting adds ']' then '}', producing {"a": [1, 2]}
        # which is valid JSON.  But the naive count can't always recover
        # complex nesting — verify we at least get valid JSON.
        json.loads(result)

    # -- Stage 5: excess closing delimiters --

    def test_extra_closing_brace(self):
        result = _repair_tool_call_arguments('{"key": "value"}}', "t")
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_extra_closing_bracket(self):
        result = _repair_tool_call_arguments('{"a": [1]]}', "t")
        # Should produce valid JSON
        json.loads(result)

    # -- Stage 6: last resort --

    def test_unrepairable_garbage_returns_empty_object(self):
        assert _repair_tool_call_arguments("totally not json", "t") == "{}"

    def test_unrepairable_partial_returns_empty_object(self):
        # Truncated in the middle of a string key — bracket closing won't help
        assert _repair_tool_call_arguments('{"truncated": "val', "t") == "{}"

    # -- Valid JSON passthrough (this path is via except, but still works) --

    def test_already_valid_json_passes_through(self):
        """When json.loads fails for a non-JSON reason (shouldn't normally
        happen), but the repair pipeline still produces valid output."""
        raw = '{"path": "/tmp/foo", "content": "hello"}'
        result = _repair_tool_call_arguments(raw, "t")
        parsed = json.loads(result)
        assert parsed["path"] == "/tmp/foo"

    # -- Combined repairs --

    def test_trailing_comma_plus_unclosed_brace(self):
        result = _repair_tool_call_arguments('{"a": 1, "b": 2,', "t")
        # Trailing comma stripped first, then closing brace added.
        # May or may not fully recover — verify valid JSON at minimum.
        json.loads(result)

    def test_real_world_glm_truncation(self):
        """Simulates GLM-5.1 truncating mid-argument."""
        raw = '{"command": "ls -la /tmp", "timeout": 30, "background":'
        result = _repair_tool_call_arguments(raw, "terminal")
        # Should at least be valid JSON, even if background is lost
        json.loads(result)

    # -- Stage 0: strict=False (literal control chars in strings) --
    # llama.cpp backends sometimes emit literal tabs/newlines inside JSON
    # string values. strict=False accepts these; we re-serialise to the
    # canonical wire form (#12068).

    def test_literal_newline_inside_string_value(self):
        raw = '{"summary": "line one\nline two"}'
        result = _repair_tool_call_arguments(raw, "t")
        parsed = json.loads(result)
        assert parsed == {"summary": "line one\nline two"}

    def test_literal_tab_inside_string_value(self):
        raw = '{"summary": "col1\tcol2"}'
        result = _repair_tool_call_arguments(raw, "t")
        parsed = json.loads(result)
        assert parsed == {"summary": "col1\tcol2"}

    def test_literal_control_char_reserialised_to_wire_form(self):
        """After repair, the output must parse under strict=True."""
        raw = '{"msg": "has\tliteral\ttabs"}'
        result = _repair_tool_call_arguments(raw, "t")
        # strict=True must now accept this
        parsed = json.loads(result)
        assert parsed["msg"] == "has\tliteral\ttabs"

    # -- Stage 4: control-char escape fallback --

    def test_control_chars_with_trailing_comma(self):
        """strict=False fails due to trailing comma, but brace-count pass
        + control-char escape rescues it."""
        raw = '{"msg": "line\none",}'
        result = _repair_tool_call_arguments(raw, "t")
        parsed = json.loads(result)
        assert "line" in parsed["msg"]



class TestTruncatedStringRecovery:
    """A fragment cut mid-string is recoverable, but only as a *lossy* repair.

    Closing the quote makes the JSON parse again while silently dropping
    whatever bytes the stream lost, so the two entry points diverge on
    purpose: the detailed form hands back the recovered text for transcript
    use, the string form reports ``"{}"`` so an executing caller refuses the
    call instead of acting on incomplete arguments.
    """

    def test_detailed_recovers_truncated_string_as_lossy(self):
        raw = '{"query":"Notion replace entire page content markdown update existing document m'
        result = _repair_tool_call_arguments_detailed(raw, "tool_search")
        assert result.ok is True
        assert result.lossy is True
        assert json.loads(result.arguments) == {
            "query": "Notion replace entire page content markdown update existing document m"
        }

    def test_string_form_refuses_a_lossy_recovery(self):
        """The execution path must not run a tool on truncated arguments."""
        raw = '{"path":"/tmp/a.md","content":"the first half of the fi'
        assert _repair_tool_call_arguments(raw, "write_file") == "{}"

    def test_detailed_marks_intact_repairs_non_lossy(self):
        result = _repair_tool_call_arguments_detailed('{"key": "value",}', "t")
        assert result.ok is True
        assert result.lossy is False
        assert json.loads(result.arguments) == {"key": "value"}

    def test_detailed_reports_unrepairable(self):
        result = _repair_tool_call_arguments_detailed("totally not json", "t")
        assert (result.arguments, result.ok, result.lossy) == ("{}", False, False)


class TestStructuralClosing:
    """Containers close innermost-first, and delimiters inside strings are text."""

    def test_nested_containers_close_in_lifo_order(self):
        """Counting openers would emit ``}]`` and discard the whole call."""
        result = _repair_tool_call_arguments('{"a": [1, 2', "t")
        assert json.loads(result) == {"a": [1, 2]}

    def test_object_inside_array_closes_in_order(self):
        result = _repair_tool_call_arguments('{"a": [1, {"b": 2', "t")
        assert json.loads(result) == {"a": [1, {"b": 2}]}

    def test_bracket_inside_string_value_is_not_structure(self):
        result = _repair_tool_call_arguments('{"a":"[","b":[1,2', "t")
        assert json.loads(result) == {"a": "[", "b": [1, 2]}

    def test_brace_inside_string_value_is_not_structure(self):
        result = _repair_tool_call_arguments('{"a":"{","b":[1,2', "t")
        assert json.loads(result) == {"a": "{", "b": [1, 2]}

    def test_escaped_quote_does_not_close_the_string(self):
        raw = '{"a":"say \\"hi\\"","b":[1'
        result = _repair_tool_call_arguments(raw, "t")
        assert json.loads(result) == {"a": 'say "hi"', "b": [1]}

    def test_balanced_json_is_untouched(self):
        raw = '{"a": [1, 2], "b": {"c": 3}}'
        assert json.loads(_repair_tool_call_arguments(raw, "t")) == {
            "a": [1, 2], "b": {"c": 3},
        }

    def test_close_truncated_json_reports_lossiness(self):
        assert _close_truncated_json('{"a": [1, 2') == ('{"a": [1, 2]}', False)
        assert _close_truncated_json('{"a": "va') == ('{"a": "va"}', True)
