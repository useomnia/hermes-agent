"""Session path identity across API run scopes and the Toolbox HTTP adapter."""

import json
import shlex
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway import session_context
from run_agent import AIAgent
from tools import approval, file_tools, terminal_tool
from tools.environments.base import BaseEnvironment
from tools.environments.file_sync import FileSyncManager
from tools.environments.sprites import SpritesEnvironment, SpritesFileOperations


@pytest.fixture
def rpc_pair(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    requests = []

    class Toolbox(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def reply(self, value, code=200):
            self.send_response(code)
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((
                self.path,
                self.headers.get("X-Hermes-Session-Id"),
                payload,
            ))
            assert self.headers["Authorization"] == "Bearer test-pair"
            assert self.headers["X-Omnio-Brand"] == "brand-a"
            if self.path == "/exec":
                # Run the adapter's actual marker wrapper, without a local-env substitute.
                result = subprocess.run(
                    ["bash", "-c", payload["command"]],
                    cwd=payload["cwd"],
                    input=payload.get("stdin"),
                    text=True,
                    capture_output=True,
                    timeout=5,
                )
                self.reply({
                    "output": result.stdout + result.stderr,
                    "returncode": result.returncode,
                })
                return
            paths = {
                key: Path(payload[key])
                for key in ("path", "src", "dst")
                if key in payload
            }
            if not all(
                path.is_absolute() and path.resolve().is_relative_to(root)
                for path in paths.values()
            ):
                self.reply(
                    {
                        "detail": "path must be absolute and inside the fixture workspace"
                    },
                    400,
                )
                return
            operation = payload["operation"]
            if operation == "move":
                paths["src"].rename(paths["dst"])
                self.reply({})
                return
            path = paths.get("path")
            if path is None:
                self.reply({"detail": "path is required"}, 400)
                return
            if operation == "write":
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(payload["content"], encoding="utf-8")
                self.reply({"bytesWritten": len(payload["content"].encode())})
            elif operation == "delete":
                path.unlink()
                self.reply({})
            elif operation in {"read", "readRaw"}:
                if not path.is_file():
                    self.reply({"error": f"File not found: {path}"})
                else:
                    content = path.read_text(encoding="utf-8")
                    self.reply({
                        "content": content,
                        "totalLines": len(content.splitlines()),
                        "fileSize": path.stat().st_size,
                    })
            else:
                self.reply({"detail": "unsupported operation"}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Toolbox)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = SpritesEnvironment.__new__(SpritesEnvironment)
    BaseEnvironment.__init__(env, cwd=str(root), timeout=5)
    env._prefer_nonlogin = True
    env.toolbox_url = f"http://127.0.0.1:{server.server_port}"
    env.bearer_token = "test-pair"
    env.brand = "brand-a"
    env._cache_sync_lock = threading.Lock()
    env._sync_manager = FileSyncManager(
        lambda: [], env._sprites_upload, env._sprites_delete
    )
    env._cache_sync_manager = FileSyncManager(
        lambda: [], env._upload_cache_file, env._delete_cache_files
    )
    monkeypatch.setenv("TERMINAL_ENV", "sprites")
    monkeypatch.setenv("TERMINAL_CWD", str(root))
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": env})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": "sprites",
            "cwd": str(root),
            "timeout": 5,
            "lifetime_seconds": 3600,
        },
    )
    monkeypatch.setattr(
        file_tools, "_file_ops_cache", {"default": SpritesFileOperations(env)}
    )
    monkeypatch.setattr(file_tools, "_read_tracker", {})
    try:
        yield root, env, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def dispatch():
    definitions = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in ("terminal", "patch", "write_file", "read_file")
    ]
    with (
        patch("run_agent.get_tool_definitions", return_value=definitions),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    def run(task, run_id, calls):
        approval_token = approval.set_current_session_key(run_id)
        session_token = session_context._SESSION_ID.set(task)
        setattr(agent, "session_id", task)
        setattr(agent, "_turn_failed_file_mutations", {})
        setattr(agent, "_turn_file_mutation_paths", set())
        messages = []
        tool_calls = [
            SimpleNamespace(
                id=f"{run_id}-{index}",
                type="function",
                function=SimpleNamespace(name=name, arguments=json.dumps(args)),
            )
            for index, (name, args) in enumerate(calls)
        ]
        try:
            agent._execute_tool_calls(
                SimpleNamespace(tool_calls=tool_calls), messages, task
            )
            return [json.loads(message["content"]) for message in messages]
        finally:
            session_context._SESSION_ID.reset(session_token)
            approval.reset_current_session_key(approval_token)

    return run


def test_api_runs_share_their_conversations_cwd_across_toolbox_rpc(rpc_pair, dispatch):
    root, env, requests = rpc_pair
    first, second = root / "first", root / "second"
    first.mkdir()
    second.mkdir()
    for task, run_id, directory, content in (
        ("conversation-one", "approval-run-one", first, "first"),
        ("conversation-two", "approval-run-two", second, "second"),
    ):
        (directory / "update.txt").write_text("before\n", encoding="utf-8")
        (directory / "delete.txt").write_text("delete\n", encoding="utf-8")
        (directory / "move.txt").write_text("move\n", encoding="utf-8")
        results = dispatch(
            task,
            run_id,
            [
                ("terminal", {"command": f"cd {shlex.quote(str(directory))}"}),
                (
                    "patch",
                    {
                        "mode": "patch",
                        "patch": (
                            f"*** Begin Patch\n*** Add File: draft.txt\n+{content}\n"
                            "*** Update File: update.txt\n@@\n-before\n+after\n"
                            "*** Delete File: delete.txt\n"
                            "*** Move File: move.txt -> moved.txt\n*** End Patch"
                        ),
                    },
                ),
                ("read_file", {"path": "draft.txt"}),
            ],
        )
        assert results[0]["exit_code"] == 0
        assert results[1]["success"] is True, results
        assert set(results[1]["files_targeted"]) == {
            str(directory / f"{name}.txt")
            for name in ("draft", "update", "delete", "move", "moved")
        }
        assert (directory / "update.txt").read_text(encoding="utf-8") == "after\n"
        assert (directory / "moved.txt").read_text(encoding="utf-8") == "move\n"
        assert not (directory / "delete.txt").exists()
        assert not (directory / "move.txt").exists()
        assert content in results[2]["content"], results
        assert (directory / "draft.txt").read_text(encoding="utf-8") == content
        assert terminal_tool.get_session_cwd(task) == str(directory)
        assert terminal_tool.get_session_cwd(run_id) is None
    assert env.cwd == str(second)
    later = dispatch(
        "conversation-one",
        "approval-run-later",
        [
            ("terminal", {"command": "pwd"}),
            ("write_file", {"path": "later.txt", "content": "later turn"}),
            ("read_file", {"path": "later.txt"}),
        ],
    )
    assert later[0]["output"] == str(first)
    assert later[1]["resolved_path"] == str(first / "later.txt")
    assert "later turn" in later[2]["content"]
    assert (first / "later.txt").read_text(encoding="utf-8") == "later turn"
    assert not (root / "draft.txt").exists()
    assert not (second / "later.txt").exists()
    assert all(
        session in {"conversation-one", "conversation-two"}
        for _, session, _ in requests
    )
    assert all(
        Path(payload[key]).is_absolute()
        for route, _, payload in requests
        if route == "/files"
        for key in ("path", "src", "dst")
        if key in payload
    )
    assert [payload["cwd"] for route, _, payload in requests if route == "/exec"] == [
        str(root),
        str(root),
        str(first),
    ]


def test_relative_reads_send_the_task_resolved_target_to_toolbox(rpc_pair, dispatch):
    root, env, requests = rpc_pair
    own = root / "own"
    other = root / "other"
    own.mkdir()
    other.mkdir()
    (own / "note.txt").write_text("owned bytes", encoding="utf-8")
    (other / "note.txt").write_text("other session decoy", encoding="utf-8")
    env.cwd = str(other)
    terminal_tool.record_session_cwd("conversation-one", str(own))

    result = dispatch(
        "conversation-one", "approval-read-run", [("read_file", {"path": "note.txt"})]
    )

    assert "owned bytes" in result[0]["content"], result
    assert requests[-1] == (
        "/files",
        "conversation-one",
        {
            "operation": "read",
            "path": str(own / "note.txt"),
            "offset": 1,
            "limit": 500,
        },
    )


def test_untasked_terminal_cwd_uses_the_file_tools_default_key(rpc_pair):
    root, _env, _requests = rpc_pair
    directory = root / "default-task"
    directory.mkdir()
    token = approval.set_current_session_key("approval-only-run")
    try:
        result = json.loads(
            terminal_tool.terminal_tool(command=f"cd {shlex.quote(str(directory))}")
        )
        written = json.loads(
            file_tools.write_file_tool(path="default.txt", content="default")
        )
    finally:
        approval.reset_current_session_key(token)

    assert result["exit_code"] == 0
    assert terminal_tool.get_session_cwd("default") == str(directory)
    assert terminal_tool.get_session_cwd("approval-only-run") is None
    assert written["resolved_path"] == str(directory / "default.txt")
    assert (directory / "default.txt").read_text(encoding="utf-8") == "default"


def test_background_cwd_uses_task_identity_and_process_ownership_uses_approval(
    rpc_pair, monkeypatch
):
    from tools import process_registry as process_module

    root, env, _requests = rpc_pair
    directory = root / "background-task"
    directory.mkdir()
    calls = []

    class Registry:
        pending_watchers = []

        def spawn_via_env(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(id="proc-test", pid=1234)

        def _write_checkpoint(self):
            pass

    monkeypatch.setattr(process_module, "process_registry", Registry())
    terminal_tool.record_session_cwd("conversation-one", str(directory))
    token = approval.set_current_session_key("approval-background-run")
    try:
        result = json.loads(
            terminal_tool.terminal_tool(
                command="pwd", task_id="conversation-one", background=True
            )
        )
    finally:
        approval.reset_current_session_key(token)

    assert result["exit_code"] == 0
    assert calls == [
        {
            "env": env,
            "command": "pwd",
            "cwd": str(directory),
            "task_id": "default",
            "session_key": "approval-background-run",
            "origin_session_id": "",
            "origin_turn_id": "",
        }
    ]
