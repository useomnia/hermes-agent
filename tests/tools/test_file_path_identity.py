"""File paths agree at the real parser, tool, adapter and verifier boundaries."""

import json
from contextlib import contextmanager

import pytest

from agent.tool_dispatch_helpers import _extract_parallel_scope_paths
from run_agent import AIAgent
from tools import file_state, file_tools, terminal_tool
from tools.environments.sprites import SpritesEnvironment, SpritesFileOperations


class CaptureFiles:
    cwd = "/home/other-session"

    def __init__(self):
        self.files = {}
        self.requests = []

    def file_request(self, payload):
        self.requests.append(payload)
        for key in ("path", "src", "dst"):
            if key in payload:
                assert payload[key].startswith("/home/"), payload
        path = payload.get("path")
        operation = payload["operation"]
        if operation == "readRaw":
            return (
                {"content": self.files[path]}
                if path in self.files
                else {"error": "File not found"}
            )
        if operation == "write":
            self.files[path] = payload["content"]
            return {"bytesWritten": len(payload["content"])}
        if operation == "delete":
            del self.files[path]
            return {}
        if operation == "move":
            self.files[payload["dst"]] = self.files.pop(payload["src"])
            return {}
        raise AssertionError(payload)


@pytest.fixture
def sandbox(monkeypatch):
    env = CaptureFiles()
    ops = SpritesFileOperations(env)
    monkeypatch.setenv("TERMINAL_ENV", "sprites")
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda task_id: ops)
    terminal_tool.record_session_cwd("one", "/home/work-one")
    terminal_tool.record_session_cwd("two", "/home/work-two")
    return env


def body(operation):
    return f"*** Begin Patch\n{operation}\n*** End Patch"


@pytest.mark.parametrize("spelling", ["relative", "absolute", "tilde"])
def test_every_operation_uses_the_same_targets_for_scopes_locks_and_io(
    sandbox, monkeypatch, spelling
):
    root = "/home" if spelling == "tilde" else "/home/work-one"
    prefix = {"relative": "", "absolute": root + "/", "tilde": "~/"}[spelling]
    sandbox.files.update({
        f"{root}/update.txt": "before",
        f"{root}/delete.txt": "delete",
        f"{root}/source.txt": "move",
    })
    patch = body(
        f"***Add File: {prefix}add.txt\n+literal *** Add File: untouched.txt\n"
        f"***Update File: {prefix}update.txt\n@@\n-before\n+after\n"
        f"***Delete File: {prefix}delete.txt\n"
        f"***Move File: {prefix}source.txt -> {prefix}dest.txt"
    )
    args = {"mode": "patch", "patch": patch}
    expected = {
        f"{root}/{name}.txt" for name in ("add", "update", "delete", "source", "dest")
    }
    locks = []
    actual_lock = file_state.lock_path

    @contextmanager
    def observed_lock(path):
        locks.append(path)
        with actual_lock(path):
            yield

    monkeypatch.setattr(file_state, "lock_path", observed_lock)
    scopes = _extract_parallel_scope_paths("patch", args, task_id="one")
    result = json.loads(file_tools.patch_tool(**args, task_id="one"))

    assert result["success"] is True, result
    assert {str(path) for path in scopes} == expected
    assert locks == sorted(expected)
    assert set(result["files_targeted"]) == expected
    assert set(result["files_modified"]) == expected
    assert sandbox.files == {
        f"{root}/add.txt": "literal *** Add File: untouched.txt",
        f"{root}/update.txt": "after",
        f"{root}/dest.txt": "move",
    }
    requested = {
        value
        for request in sandbox.requests
        for key, value in request.items()
        if key in {"path", "src", "dst"}
    }
    assert requested == expected


def test_shared_file_backend_preserves_each_sessions_directory(sandbox):
    for session in ("one", "two", "one"):
        result = json.loads(
            file_tools.patch_tool(
                mode="patch",
                patch=body("*** Add File: note.txt\n+" + session),
                task_id=session,
            )
        )
        assert result["success"] is True
    assert sandbox.files == {
        "/home/work-one/note.txt": "one",
        "/home/work-two/note.txt": "two",
    }


def test_existing_mount_aliases_share_scheduling_and_recovery_identity(sandbox):
    terminal_tool.record_session_cwd("one", "/brand")
    expected = "/home/brand/note.txt"
    for path in (
        "note.txt",
        "/brand/note.txt",
        "/home/brand/note.txt",
        "~/brand/note.txt",
    ):
        scopes = _extract_parallel_scope_paths("patch", {"path": path}, task_id="one")
        assert [str(scope) for scope in scopes] == [expected]
        assert str(file_tools._resolve_path_for_task(path, "one")) == expected
    result = json.loads(
        file_tools.patch_tool(
            mode="patch", patch=body("***Add File: note.txt\n+brand"), task_id="one"
        )
    )
    assert result["files_targeted"] == [expected]
    assert sandbox.files == {expected: "brand"}


@pytest.mark.parametrize(
    "path", ["/skills/helper.py", "/home/skills/helper.py", "~/skills/helper.py"]
)
def test_canonical_skill_alias_still_refreshes_the_existing_projection(path):
    env = object.__new__(SpritesEnvironment)
    refreshed = []
    env.sync_skill_files = lambda: refreshed.append("skills")
    env.sync_projected_path(path)
    assert refreshed == ["skills"]


def test_sensitive_raw_spelling_is_still_guarded_after_resolution(sandbox, monkeypatch):
    from pathlib import PurePosixPath

    monkeypatch.setattr(
        file_tools,
        "_resolve_path_for_task",
        lambda *_: PurePosixPath("/home/harmless.txt"),
    )
    result = json.loads(
        file_tools.patch_tool(
            mode="patch", patch=body("***Add File: /etc/hosts\n+bad"), task_id="one"
        )
    )
    assert "sensitive" in result["error"]
    assert sandbox.requests == []


@pytest.mark.parametrize(
    "header",
    [
        "***Add File: ../escape.txt\n+bad",
        "***Update File: /etc/hosts\n@@\n-old\n+bad",
        "***Delete File: /etc/hosts",
        "***Move File: safe.txt -> ../escape.txt",
        "***Move File: /etc/hosts -> safe.txt",
        "***Move File: safe.txt -> /etc/hosts",
    ],
)
def test_raw_traversal_and_protected_endpoints_are_rejected_before_io(sandbox, header):
    result = json.loads(
        file_tools.patch_tool(mode="patch", patch=body(header), task_id="one")
    )
    assert result.get("error"), result
    assert sandbox.requests == []


def test_equivalent_recovery_clears_only_the_same_file_and_keeps_first_error(sandbox):
    agent = object.__new__(AIAgent)
    agent._turn_failed_file_mutations = {}
    agent._turn_file_mutation_paths = set()

    def record(path, error=None):
        result = json.dumps({"error": error} if error else {"success": True})
        agent._record_file_mutation_result(
            "patch",
            {"mode": "patch", "patch": body(f"***Add File: {path}\n+text")},
            result,
            bool(error),
            task_id="one",
        )

    record("note.txt", "first failure")
    terminal_tool.record_session_cwd("one", "/home/work-two")
    record("note.txt")
    assert set(agent._turn_failed_file_mutations) == {"/home/work-one/note.txt"}
    record("/home/work-one/note.txt", "second failure")
    assert (
        agent._turn_failed_file_mutations["/home/work-one/note.txt"]["error_preview"]
        == "first failure"
    )
    record("/home/work-one/note.txt")
    assert not agent._turn_failed_file_mutations
    record("~/home-note.txt", "home failure")
    record("/home/home-note.txt")
    assert not agent._turn_failed_file_mutations


def test_failure_receipt_keeps_captured_identity_after_directory_change(sandbox):
    result = file_tools.patch_tool(
        mode="patch",
        patch=body("***Update File: missing.txt\n@@\n-before\n+after"),
        task_id="one",
    )
    terminal_tool.record_session_cwd("one", "/home/work-two")
    agent = object.__new__(AIAgent)
    agent._turn_failed_file_mutations = {}
    agent._turn_file_mutation_paths = set()
    agent._record_file_mutation_result(
        "patch",
        {
            "mode": "patch",
            "patch": body("***Update File: missing.txt\n@@\n-before\n+after"),
        },
        result,
        True,
        task_id="one",
    )
    assert set(agent._turn_failed_file_mutations) == {"/home/work-one/missing.txt"}


def test_move_recovery_clears_each_endpoint_only_when_it_lands(sandbox):
    agent = object.__new__(AIAgent)
    agent._turn_failed_file_mutations = {}
    agent._turn_file_mutation_paths = set()
    args = {"mode": "patch", "patch": body("***Move File: old.txt -> new.txt")}
    agent._record_file_mutation_result(
        "patch", args, json.dumps({"error": "move failed"}), True, task_id="one"
    )
    assert set(agent._turn_failed_file_mutations) == {
        "/home/work-one/old.txt",
        "/home/work-one/new.txt",
    }
    agent._record_file_mutation_result(
        "patch",
        {"path": "/home/work-one/old.txt"},
        json.dumps({"success": True}),
        False,
        task_id="one",
    )
    assert set(agent._turn_failed_file_mutations) == {"/home/work-one/new.txt"}
    agent._record_file_mutation_result(
        "patch", {"path": "new.txt"}, "{}", False, task_id="one"
    )
    assert set(agent._turn_failed_file_mutations) == {"/home/work-one/new.txt"}
