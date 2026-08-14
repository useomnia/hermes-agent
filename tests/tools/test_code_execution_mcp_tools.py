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


from model_tools import _uncollapse_bridge_tool_names
from tools.code_execution_tool import (
    DEFAULT_SANDBOX_MODULE,
    SANDBOX_ALLOWED_TOOLS,
    _sandbox_module_name,
    _is_mcp_tool,
    _mcp_stub_source,
    _resolve_remote_script_cwd,
    _ship_file_to_remote,
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

    def _run(self, code, enabled_tools, mcp_result=None):
        dispatched = []

        def _dispatch(function_name, function_args, task_id=None, **kwargs):
            dispatched.append((function_name, function_args))
            if function_name == MCP_TOOL:
                if mcp_result is not None:
                    return mcp_result
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

    def test_a_real_mcp_envelope_arrives_decoded(self):
        """The shape mcp_tool.py actually returns, end to end.

        This fixture is the one that matters: the sibling tests hand back a bare
        payload, and that is why the double-encoded envelope reached production
        unnoticed. A script must be able to walk into the payload directly.
        """
        envelope = json.dumps(
            {"result": json.dumps({"data": {"prompts": [{"id": "p1"}, {"id": "p2"}]}})}
        )
        result, _ = self._run(
            f"from hermes_tools import {MCP_TOOL}\n"
            f"r = {MCP_TOOL}(brandId='b1')\n"
            "print(len(r['result']['data']['prompts']))\n",
            enabled_tools=[MCP_TOOL],
            mcp_result=envelope,
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["output"].strip(), "2")

    def test_a_prose_envelope_stays_a_string_end_to_end(self):
        envelope = json.dumps({"result": "No prompts found for that brand."})
        result, _ = self._run(
            f"from hermes_tools import {MCP_TOOL}\n"
            f"r = {MCP_TOOL}(brandId='b1')\n"
            "print(type(r['result']).__name__)\n",
            enabled_tools=[MCP_TOOL],
            mcp_result=envelope,
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["output"].strip(), "str")

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


# ---------------------------------------------------------------------------
# Where a remote script runs
# ---------------------------------------------------------------------------

class TestRemoteScriptCwd(unittest.TestCase):
    """A script on a remote backend must run where terminal() runs.

    The staging dir is deleted after the run, so a script that wrote a file
    relative to it would lose that file — the one place a script can silently
    destroy its own output.
    """

    STAGING = "/tmp/.hermes-session/hermes_exec_abc123"

    def test_project_mode_defers_to_the_backend_default(self):
        """No recorded cwd means "wherever a bare terminal command lands", which
        the backend already knows — so don't cd at all."""
        self.assertIsNone(
            _resolve_remote_script_cwd("project", self.STAGING, "task-with-no-record")
        )

    def test_project_mode_follows_the_session_cwd_record(self):
        with patch("tools.terminal_tool.get_session_cwd", return_value="/home/brand"):
            self.assertEqual(
                _resolve_remote_script_cwd("project", self.STAGING, "task-1"),
                "/home/brand",
            )

    def test_project_mode_falls_back_to_a_registered_override(self):
        with patch("tools.terminal_tool.get_session_cwd", return_value=None):
            with patch(
                "tools.file_tools._registered_task_cwd_override",
                return_value="/home/work",
            ):
                self.assertEqual(
                    _resolve_remote_script_cwd("project", self.STAGING, "task-1"),
                    "/home/work",
                )

    def test_strict_mode_keeps_the_staging_dir(self):
        self.assertEqual(
            _resolve_remote_script_cwd("strict", self.STAGING, "task-1"), self.STAGING
        )

    def test_a_remote_path_is_never_stat_checked(self):
        """These paths live on the remote machine. An isdir() check here asks the
        wrong filesystem and would reject every valid answer."""
        with patch("tools.terminal_tool.get_session_cwd", return_value="/home/brand"):
            with patch("os.path.isdir", return_value=False):
                self.assertEqual(
                    _resolve_remote_script_cwd("project", self.STAGING, "task-1"),
                    "/home/brand",
                )


class TestBackendAwareCwdNote(unittest.TestCase):
    """The description must not promise a local venv to a remote session."""

    def _describe(self, backend, mode="project"):
        with patch(
            "tools.terminal_tool._get_env_config", return_value={"env_type": backend}
        ):
            return build_execute_code_schema({"terminal"}, mode=mode)["description"]

    def test_remote_backend_describes_the_terminal_machine(self):
        description = self._describe("sprites")
        self.assertIn("same machine as terminal()", description)
        self.assertIn("that machine's python3", description)
        self.assertNotIn("active venv's python", description)

    def test_local_backend_keeps_the_venv_wording(self):
        self.assertIn("active venv's python", self._describe("local"))

    def test_strict_mode_warns_the_temp_dir_is_deleted(self):
        self.assertIn("deleted afterwards", self._describe("sprites", mode="strict"))


class TestModuleIntrospection(_WithMcpTool):
    """A script must be able to discover the exact names it can call.

    Guessing at an MCP prefix and failing the import is the expensive failure
    mode — it burns the turn before any work happens.
    """

    def test_list_tools_reports_every_generated_name(self):
        source = generate_hermes_tools_module([MCP_TOOL, "terminal"])
        namespace: dict = {}
        exec(compile(source, "hermes_tools.py", "exec"), namespace)
        self.assertEqual(sorted(namespace["list_tools"]()), sorted([MCP_TOOL, "terminal"]))

    def test_list_tools_excludes_tools_that_were_not_generated(self):
        source = generate_hermes_tools_module(["terminal"])
        namespace: dict = {}
        exec(compile(source, "hermes_tools.py", "exec"), namespace)
        self.assertEqual(namespace["list_tools"](), ["terminal"])

    def test_list_tools_is_a_copy(self):
        """A script mutating the returned list must not corrupt the module."""
        source = generate_hermes_tools_module(["terminal"])
        namespace: dict = {}
        exec(compile(source, "hermes_tools.py", "exec"), namespace)
        namespace["list_tools"]().append("web_search")
        self.assertEqual(namespace["list_tools"](), ["terminal"])

    def test_description_points_at_list_tools_when_mcp_is_available(self):
        schema = build_execute_code_schema({"terminal", MCP_TOOL}, mode="project")
        self.assertIn("list_tools()", schema["description"])


class TestSandboxModuleName(unittest.TestCase):
    """The stub module's name is agent-visible, so it is configurable.

    It lands in the tool description AND in every script's import line, so a
    deployment whose agent must not identify its harness has to be able to change
    it — and the value has to be usable as both a filename and an import.
    """

    def _with_module(self, value):
        return patch(
            "tools.code_execution_tool._load_config", return_value={"module_name": value}
        )

    def test_default_is_unchanged_when_unset(self):
        with patch("tools.code_execution_tool._load_config", return_value={}):
            self.assertEqual(_sandbox_module_name(), DEFAULT_SANDBOX_MODULE)

    def test_configured_name_is_used(self):
        with self._with_module("omnio_tools"):
            self.assertEqual(_sandbox_module_name(), "omnio_tools")

    def test_invalid_names_fall_back_rather_than_breaking_the_import(self):
        for bad in ("123bad", "has-a-dash", "two words", "class", ""):
            with self.subTest(bad=bad), self._with_module(bad):
                self.assertEqual(_sandbox_module_name(), DEFAULT_SANDBOX_MODULE)

    def test_description_uses_the_configured_name_everywhere(self):
        with self._with_module("omnio_tools"):
            schema = build_execute_code_schema({"terminal"}, mode="project")
        blob = schema["description"] + schema["parameters"]["properties"]["code"]["description"]
        self.assertIn("from omnio_tools import", blob)
        self.assertNotIn("hermes", blob.lower())


class TestBridgeCollapsedSessionList(unittest.TestCase):
    """A script must see the session's real tools, not the collapsed ones.

    The agent loop passes the MODEL-FACING names. With the tool-search bridge
    active those have MCP and plugin tools replaced by tool_search/tool_describe/
    tool_call, so a sandbox reading them concludes the session has no MCP tools —
    while the tool description, built before collapsing, says it does.
    """

    def test_no_bridge_tools_means_the_list_is_returned_unchanged(self):
        original = ["terminal", "read_file"]
        self.assertEqual(_uncollapse_bridge_tool_names(original, None, None), original)

    def test_empty_input_is_returned_unchanged(self):
        self.assertEqual(_uncollapse_bridge_tool_names([], None, None), [])
        self.assertIsNone(_uncollapse_bridge_tool_names(None, None, None))

    def test_a_collapsed_list_regains_the_deferred_tools(self):
        collapsed = ["terminal", "tool_search", "tool_describe", "tool_call"]
        real_defs = [
            {"function": {"name": "terminal"}},
            {"function": {"name": MCP_TOOL}},
        ]
        with patch("model_tools.get_tool_definitions", return_value=real_defs):
            resolved = _uncollapse_bridge_tool_names(collapsed, ["hermes-cli"], None)
        self.assertIn(MCP_TOOL, resolved)
        self.assertIn("terminal", resolved)

    def test_rebuild_is_scoped_to_the_sessions_own_toolsets(self):
        """Scoping is what stops this widening the surface: the rebuild must be
        asked for the session's toolsets, not the whole registry."""
        captured = {}

        def _fake_defs(**kwargs):
            captured.update(kwargs)
            return []

        with patch("model_tools.get_tool_definitions", side_effect=_fake_defs):
            _uncollapse_bridge_tool_names(["tool_call"], ["hermes-cli"], ["cronjob"])
        self.assertEqual(captured.get("enabled_toolsets"), ["hermes-cli"])
        self.assertEqual(captured.get("disabled_toolsets"), ["cronjob"])
        self.assertTrue(captured.get("skip_tool_search_assembly"))

    def test_a_failed_rebuild_leaves_the_list_alone(self):
        with patch("model_tools.get_tool_definitions", side_effect=RuntimeError("boom")):
            self.assertEqual(
                _uncollapse_bridge_tool_names(["terminal", "tool_call"], None, None),
                ["terminal", "tool_call"],
            )


class TestShipFileToRemote(unittest.TestCase):
    """Shipping a file must not depend on the file fitting in a command.

    The generated tools module is as large as the session's tool surface — one stub
    per MCP tool — and the shell fallback puts its bytes inside a command string,
    which backends length-limit. A file write has no such limit.
    """

    class _EnvWithWriter:
        def __init__(self, ok=True, raises=False):
            self.ok, self.raises = ok, raises
            self.written, self.commands = [], []

        def write_file_content(self, path, content):
            if self.raises:
                raise RuntimeError("api rejected it")
            self.written.append((path, content))
            return self.ok

        def execute(self, command, cwd=None, timeout=None):
            self.commands.append(command)
            return {"output": "", "returncode": 0}

    class _EnvShellOnly:
        def __init__(self):
            self.commands = []

        def execute(self, command, cwd=None, timeout=None):
            self.commands.append(command)
            return {"output": "", "returncode": 0}

    def test_prefers_the_backends_own_file_write(self):
        env = self._EnvWithWriter()
        _ship_file_to_remote(env, "/tmp/x/omnio_tools.py", "print('hi')")
        self.assertEqual(env.written, [("/tmp/x/omnio_tools.py", "print('hi')")])
        self.assertEqual(env.commands, [])

    def test_falls_back_to_the_shell_when_the_writer_declines(self):
        env = self._EnvWithWriter(ok=False)
        _ship_file_to_remote(env, "/tmp/x/mod.py", "print('hi')")
        self.assertTrue(env.commands)

    def test_falls_back_to_the_shell_when_the_writer_raises(self):
        env = self._EnvWithWriter(raises=True)
        _ship_file_to_remote(env, "/tmp/x/mod.py", "print('hi')")
        self.assertTrue(env.commands)

    def test_shell_fallback_chunks_a_large_file(self):
        """A module the size of a real MCP catalog must not become one command."""
        env = self._EnvShellOnly()
        _ship_file_to_remote(env, "/tmp/x/mod.py", "x = 1\n" * 20_000)
        self.assertGreater(len(env.commands), 2)
        for command in env.commands:
            self.assertLess(len(command), 32_000)
        self.assertTrue(env.commands[0].startswith("printf %s "))
        self.assertIn(">>", env.commands[1])
        self.assertIn("base64 -d", env.commands[-1])
        self.assertIn("rm -f", env.commands[-1])

    def test_shell_fallback_writes_a_small_file_in_one_chunk(self):
        env = self._EnvShellOnly()
        _ship_file_to_remote(env, "/tmp/x/mod.py", "print('hi')")
        self.assertEqual(len(env.commands), 2)  # one write + the decode
        self.assertIn(">", env.commands[0])
        self.assertNotIn(">>", env.commands[0])


# ---------------------------------------------------------------------------
# Decoding what an MCP server sends back
# ---------------------------------------------------------------------------

class TestResultDecoding(unittest.TestCase):
    """An MCP tool answers with {"result": <text>}, and a server whose payload is
    JSON therefore sends it double-encoded.

    Without decoding, every caller needs a second json.loads on every call — and
    the model has to discover that by running a probe script first, which is the
    round trip this exists to remove. The decode is deliberately narrow: it must
    never reinterpret an answer the server meant as text.
    """

    def _decoder(self, transport="uds"):
        source = generate_hermes_tools_module(["terminal"], transport=transport)
        namespace: dict = {}
        exec(compile(source, "hermes_tools.py", "exec"), namespace)
        return namespace["_decode_result"]

    def test_both_transports_ship_the_decoder(self):
        for transport in ("uds", "file"):
            with self.subTest(transport=transport):
                self.assertTrue(callable(self._decoder(transport)))

    def test_json_object_payload_is_parsed(self):
        decode = self._decoder()
        got = decode({"result": '{"data": {"prompts": [{"id": "p1"}]}}'})
        self.assertEqual(got, {"result": {"data": {"prompts": [{"id": "p1"}]}}})

    def test_json_array_payload_is_parsed(self):
        decode = self._decoder()
        self.assertEqual(decode({"result": "[1, 2, 3]"}), {"result": [1, 2, 3]})

    def test_structured_content_is_preserved_alongside(self):
        decode = self._decoder()
        got = decode({"result": '{"a": 1}', "structuredContent": {"b": 2}})
        self.assertEqual(got, {"result": {"a": 1}, "structuredContent": {"b": 2}})

    def test_prose_text_is_left_alone(self):
        decode = self._decoder()
        payload = {"result": "No prompts found for that brand."}
        self.assertEqual(decode(payload), payload)

    def test_a_scalar_that_looks_like_json_stays_text(self):
        """"42" is a valid JSON document. Turning it into an int would silently
        change a server's text answer into a number."""
        decode = self._decoder()
        for text in ("42", "true", "null", '"quoted"'):
            with self.subTest(text=text):
                self.assertEqual(decode({"result": text}), {"result": text})

    def test_an_already_structured_result_is_untouched(self):
        decode = self._decoder()
        payload = {"result": {"data": [1]}}
        self.assertEqual(decode(payload), payload)

    def test_a_dict_with_other_keys_is_not_the_envelope(self):
        """Our own tools may use a "result" key; only the MCP envelope shape
        is eligible."""
        decode = self._decoder()
        payload = {"result": '{"a": 1}', "exit_code": 0}
        self.assertEqual(decode(payload), payload)

    def test_non_envelope_values_pass_through(self):
        decode = self._decoder()
        for value in ({"content": "x"}, [1, 2], "text", 7, None):
            with self.subTest(value=value):
                self.assertEqual(decode(value), value)

    def test_malformed_json_is_returned_as_the_server_sent_it(self):
        decode = self._decoder()
        payload = {"result": '{"a": 1'}
        self.assertEqual(decode(payload), payload)

    def test_the_stub_docstring_states_the_envelope(self):
        """The decode is invisible to the model, so the stub has to say it."""
        _register_mcp_tool()
        try:
            source = _mcp_stub_source(MCP_TOOL)
        finally:
            registry.deregister(MCP_TOOL)
        self.assertIn('Returns {"result": payload}', source)
        self.assertIn("already decoded", source)
