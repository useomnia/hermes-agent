#!/usr/bin/env python3
"""Tests for calling MCP tools from an execute_code script.

The core sandbox tools have hand-written stubs; MCP tools cannot, because a
session's servers — and, for a multi-tenant deployment, its per-tenant tool
allow-list — are only known at runtime. They are admitted by PREDICATE over the
session's own resolved tool list and get stubs generated from their registered
schemas.

The security property under test is that the predicate cannot widen the
sandbox's reach beyond the session's: a tool absent from the session list must be
absent from the sandbox, whatever the registry knows about it.
"""

import json
import os
import unittest
from unittest.mock import patch

import pytest

os.environ["TERMINAL_ENV"] = "local"


@pytest.fixture(autouse=True)
def _force_local_terminal(monkeypatch):
    """Mirror the sibling execute_code suites — pin the local backend under xdist."""
    monkeypatch.setenv("TERMINAL_ENV", "local")


from tools.code_execution_tool import (
    SANDBOX_ALLOWED_TOOLS,
    _is_mcp_tool,
    _mcp_stub_source,
    _resolve_sandbox_tools,
    build_execute_code_schema,
    execute_code,
    generate_hermes_tools_module,
)
from tools.registry import registry

MCP_TOOL = "mcp__omnia__get_brands_prompts"
MCP_TOOL_SCHEMA = {
    "name": MCP_TOOL,
    "description": "List the prompts monitored for a brand.",
    "parameters": {
        "type": "object",
        "properties": {
            "brandId": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["brandId"],
    },
}


def _register_mcp_tool(name=MCP_TOOL, schema=MCP_TOOL_SCHEMA, toolset="mcp-omnia"):
    registry.register(
        name=name,
        toolset=toolset,
        schema=schema,
        handler=lambda args, **kw: json.dumps({"prompts": [{"id": "p1"}]}),
        description=str(schema.get("description") or ""),
        emoji="🔌",
    )


class _WithMcpTool(unittest.TestCase):
    """Registers a fake MCP tool for the duration of each test."""

    def setUp(self):
        _register_mcp_tool()

    def tearDown(self):
        registry.deregister(MCP_TOOL)


# ---------------------------------------------------------------------------
# Which tools a script may call
# ---------------------------------------------------------------------------

class TestResolveSandboxTools(_WithMcpTool):
    def test_mcp_tool_in_the_session_is_callable(self):
        resolved = _resolve_sandbox_tools(["terminal", MCP_TOOL])
        self.assertIn(MCP_TOOL, resolved)
        self.assertIn("terminal", resolved)

    def test_mcp_tool_absent_from_the_session_is_not_callable(self):
        """The registry knows this tool; the session was not granted it. The
        sandbox must not be a way around that."""
        resolved = _resolve_sandbox_tools(["terminal"])
        self.assertNotIn(MCP_TOOL, resolved)

    def test_non_mcp_non_core_tools_stay_out(self):
        """Only core + MCP are admitted — an unrelated tool the session has (a
        plugin tool, an agent-loop tool) is not made callable by this change."""
        resolved = _resolve_sandbox_tools(["terminal", "todo", "delegate_task"])
        self.assertEqual(resolved, frozenset({"terminal"}))

    def test_empty_session_list_falls_back_to_core_tools(self):
        """No list means the caller could not tell us, not "grant nothing" — the
        execution path still has to produce a working sandbox."""
        self.assertEqual(_resolve_sandbox_tools([]), SANDBOX_ALLOWED_TOOLS)
        self.assertEqual(_resolve_sandbox_tools(None), SANDBOX_ALLOWED_TOOLS)

    def test_session_with_only_unusable_tools_falls_back_to_core(self):
        self.assertEqual(_resolve_sandbox_tools(["todo"]), SANDBOX_ALLOWED_TOOLS)

    def test_authoritative_caller_gets_no_fallback(self):
        """A caller whose list IS the session's surface must not be handed the
        core seven when the session has none of them."""
        self.assertEqual(
            _resolve_sandbox_tools(["todo"], fallback_to_core=False), frozenset()
        )
        self.assertEqual(
            _resolve_sandbox_tools([MCP_TOOL], fallback_to_core=False),
            frozenset({MCP_TOOL}),
        )

    def test_is_mcp_tool_ignores_core_and_unknown_names(self):
        self.assertTrue(_is_mcp_tool(MCP_TOOL))
        self.assertFalse(_is_mcp_tool("terminal"))
        self.assertFalse(_is_mcp_tool("not_a_registered_tool_at_all"))


# ---------------------------------------------------------------------------
# Generated stubs
# ---------------------------------------------------------------------------

class TestMcpStubSource(_WithMcpTool):
    def test_stub_dispatches_the_registered_tool_name(self):
        source = _mcp_stub_source(MCP_TOOL)
        self.assertIsNotNone(source)
        self.assertIn(f"def {MCP_TOOL}(**kwargs):", source)
        self.assertIn(f"_call('{MCP_TOOL}', kwargs)", source)

    def test_stub_documents_required_and_optional_parameters(self):
        source = _mcp_stub_source(MCP_TOOL)
        self.assertIn("List the prompts monitored for a brand.", source)
        self.assertIn("Required: brandId", source)
        self.assertIn("Optional: limit", source)

    def test_stub_for_a_no_argument_tool_says_so(self):
        name = "mcp__omnia__who_am_i"
        _register_mcp_tool(
            name=name,
            schema={"name": name, "description": "Who am I.", "parameters": {}},
        )
        try:
            source = _mcp_stub_source(name)
            self.assertIn("Takes no arguments.", source)
        finally:
            registry.deregister(name)

    def test_stub_is_skipped_for_a_name_that_is_not_an_identifier(self):
        """A generated module must always import; a name we can't express as a
        function is dropped rather than emitted as broken source."""
        name = "mcp__omnia__has-a-dash"
        _register_mcp_tool(name=name, schema={"name": name, "parameters": {}})
        try:
            self.assertIsNone(_mcp_stub_source(name))
        finally:
            registry.deregister(name)

    def test_generated_module_is_valid_python_and_exports_the_tool(self):
        source = generate_hermes_tools_module([MCP_TOOL, "terminal"])
        compile(source, "hermes_tools.py", "exec")  # must import cleanly
        self.assertIn(f"def {MCP_TOOL}(**kwargs):", source)
        self.assertIn("def terminal(", source)

    def test_generated_module_omits_an_undeclared_tool(self):
        source = generate_hermes_tools_module(["terminal"])
        self.assertNotIn(MCP_TOOL, source)


# ---------------------------------------------------------------------------
# What the model is told
# ---------------------------------------------------------------------------

class TestSchemaDescription(_WithMcpTool):
    def test_description_mentions_mcp_tools_without_listing_them_all(self):
        many = [f"mcp__omnia__tool_{i}" for i in range(40)]
        for name in many:
            _register_mcp_tool(name=name, schema={"name": name, "parameters": {}})
        try:
            schema = build_execute_code_schema({"terminal", *many}, mode="project")
            description = schema["description"]
            self.assertIn(f"all {len(many)} MCP tools", description)
            # One example name is fine; forty would be a per-turn context tax.
            listed = sum(1 for name in many if name in description)
            self.assertEqual(listed, 1)
        finally:
            for name in many:
                registry.deregister(name)

    def test_description_says_nothing_about_mcp_when_none_are_callable(self):
        schema = build_execute_code_schema({"terminal"}, mode="project")
        self.assertNotIn("MCP", schema["description"])

    def test_import_example_prefers_a_core_tool(self):
        """The example must stay copy-pasteable — a 40-char MCP name as the
        headline import teaches the wrong default."""
        schema = build_execute_code_schema({"terminal", MCP_TOOL}, mode="project")
        code_doc = schema["parameters"]["properties"]["code"]["description"]
        self.assertIn("from hermes_tools import terminal", code_doc)

    def test_limits_reflect_the_configured_caps(self):
        """A deployment that raises the call cap must not still advertise the
        default, or the model plans around a ceiling that isn't there."""
        with patch(
            "tools.code_execution_tool._load_config",
            return_value={"timeout": 600, "max_tool_calls": 150},
        ):
            schema = build_execute_code_schema({"terminal"}, mode="project")
        self.assertIn("10-minute timeout", schema["description"])
        self.assertIn("max 150 tool calls per script", schema["description"])

    def test_limits_render_odd_timeouts_in_seconds(self):
        with patch(
            "tools.code_execution_tool._load_config",
            return_value={"timeout": 90, "max_tool_calls": 50},
        ):
            schema = build_execute_code_schema({"terminal"}, mode="project")
        self.assertIn("90s timeout", schema["description"])


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

class TestScriptCallsMcpTool(_WithMcpTool):
    """A script calls the MCP tool and only its own stdout comes back."""

    def _run(self, code, enabled_tools):
        dispatched = []

        def _dispatch(function_name, function_args, task_id=None, **kwargs):
            dispatched.append((function_name, function_args))
            if function_name == MCP_TOOL:
                return json.dumps({"prompts": [{"id": "p1"}, {"id": "p2"}]})
            return json.dumps({"error": f"Unknown tool: {function_name}"})

        with patch("model_tools.handle_function_call", side_effect=_dispatch):
            raw = execute_code(code=code, task_id="test-mcp-ptc", enabled_tools=enabled_tools)
        return json.loads(raw), dispatched

    def test_script_can_call_an_mcp_tool_and_reduce_the_result(self):
        result, dispatched = self._run(
            f"from hermes_tools import {MCP_TOOL}\n"
            f"data = {MCP_TOOL}(brandId='b1')\n"
            "print(len(data['prompts']))\n",
            enabled_tools=["terminal", MCP_TOOL],
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["output"].strip(), "2")
        self.assertEqual(dispatched, [(MCP_TOOL, {"brandId": "b1"})])
        # The per-call payload stayed in the script: only the reduction is here.
        self.assertNotIn("p1", result["output"])

    def test_mcp_calls_are_accounted_like_any_other_inner_call(self):
        result, _ = self._run(
            f"from hermes_tools import {MCP_TOOL}\n"
            f"for b in ('b1', 'b2', 'b3'):\n"
            f"    {MCP_TOOL}(brandId=b)\n"
            "print('done')\n",
            enabled_tools=[MCP_TOOL],
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tool_calls_made"], 3)
        self.assertEqual(result["inner_tool_calls"], {MCP_TOOL: 3})

    def test_rpc_rejects_an_mcp_tool_the_session_lacks(self):
        """Server-side enforcement, not just an absent stub: a script that names
        the tool directly over the RPC must still be refused."""
        result, dispatched = self._run(
            "import hermes_tools\n"
            f"print(hermes_tools._call({MCP_TOOL!r}, {{'brandId': 'b1'}}))\n",
            enabled_tools=["terminal"],
        )
        self.assertEqual(result["status"], "success")  # the script itself ran
        self.assertIn("not available in execute_code", result["output"])
        self.assertEqual(dispatched, [])
