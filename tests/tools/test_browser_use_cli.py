"""Focused contracts for the managed Browser Use CLI surface."""

from __future__ import annotations

import json
import os
import stat
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import browser_use_cli as bu


def test_same_explicit_session_isolated_by_conversation(monkeypatch):
    from gateway import session_context

    values = {"HERMES_SESSION_ID": "conversation-a"}
    monkeypatch.setattr(session_context, "get_session_env", lambda key: values.get(key))
    first = bu._derive_bu_name("child-a", "shared-name")

    values["HERMES_SESSION_ID"] = "conversation-b"
    second = bu._derive_bu_name("child-b", "shared-name")

    assert first != second
    assert first.startswith("bu-") and second.startswith("bu-")
    assert "shared-name" not in first
    assert "default" not in first


def test_task_aware_omnio_cdp_fails_closed(monkeypatch):
    from tools import browser_tool

    calls = []
    monkeypatch.setenv("BROWSER_CDP_URL_TEMPLATE", "http://relay/sessions/{session_id}")
    monkeypatch.setattr(
        browser_tool,
        "_get_task_cdp_override",
        lambda task_id: calls.append(task_id) or "",
    )
    monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)

    error = bu._resolve_backend_cdp(
        {}, "task-17", session_name="same", bu_name="bu-hash"
    )

    assert error and "refusing to attach" in error
    assert calls == ["task-17"]


def test_omnio_template_outranks_process_cdp_override(monkeypatch):
    """A raw process endpoint cannot escape conversation-scoped Toolbox CDP."""
    from tools import browser_tool

    monkeypatch.setenv("BROWSER_CDP_URL_TEMPLATE", "http://relay/sessions/{session_id}")
    monkeypatch.setenv("BU_CDP_WS", "ws://unscoped-process-endpoint")
    monkeypatch.setattr(
        browser_tool,
        "_get_task_cdp_override",
        lambda task_id: f"http://relay/sessions/{task_id}",
    )
    env = {}

    error = bu._resolve_backend_cdp(env, "conversation-17")

    assert error is None
    assert env["BU_CDP_URL"] == "http://relay/sessions/conversation-17"
    assert "BU_CDP_WS" not in env


def test_subprocess_environment_is_scrubbed_and_floored(monkeypatch):
    from tools import browser_tool

    monkeypatch.setattr(
        browser_tool,
        "_build_browser_env",
        lambda: {
            "PATH": "/only/version-manager",
            "PYTHONPATH": "/wrong/hermes/site-packages",
            "PYTHONHOME": "/wrong/python",
            "ANONYMIZED_TELEMETRY": "true",
            "HERMES_SESSION_ID": "raw-conversation",
            "BU_AUTOSPAWN": "1",
            "BROWSER_CDP_URL_TEMPLATE": "http://relay/{session_id}",
        },
    )
    env = bu._base_subprocess_env()

    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert "HERMES_SESSION_ID" not in env
    assert "BU_AUTOSPAWN" not in env
    assert "BROWSER_CDP_URL_TEMPLATE" not in env
    assert env["ANONYMIZED_TELEMETRY"] == "false"
    assert env["BROWSER_USE_TELEMETRY"] == "0"
    assert "/usr/bin" in env["PATH"] or os.name == "nt"


def test_stale_managed_cli_is_not_resolved(tmp_path, monkeypatch):
    executable = tmp_path / "browser-use"
    executable.write_text("#!/bin/sh\nprintf '0.1.8\\n'\n")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(bu, "_managed_bin_dir", lambda: str(tmp_path))

    assert bu._find_cli() is None


def test_managed_browser_use_cli_is_shared_from_root_for_profile_gateway(
    tmp_path, monkeypatch
):
    profile_home = tmp_path / "profiles" / "brand-a"
    managed_bin = tmp_path / "bin"
    managed_bin.mkdir()
    executable = managed_bin / "browser-use"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setattr(
        bu, "_managed_cli_is_current", lambda path: path == str(executable)
    )

    assert bu._find_cli() == [str(executable)]


def test_missing_managed_cli_explains_omnio_agent_side_reprovision(monkeypatch):
    monkeypatch.setattr(bu, "_find_cli", lambda: None)
    monkeypatch.setattr(bu, "_omnio_template_cdp_configured", lambda: True)

    result = json.loads(bu.browser_exec("print(page_info())"))

    assert "Reprovision this Omnio sandbox" in result["error"]
    assert "Toolbox terminal" in result["error"]
    assert "hermes tools" not in result["error"]


def test_missing_managed_cli_keeps_standalone_install_guidance(monkeypatch):
    monkeypatch.setattr(bu, "_find_cli", lambda: None)
    monkeypatch.setattr(bu, "_omnio_template_cdp_configured", lambda: False)

    result = json.loads(bu.browser_exec("print(page_info())"))

    assert "hermes tools" in result["error"]
    assert "Reprovision this Omnio sandbox" not in result["error"]


def test_install_cli_forces_exact_package_pin(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(bu, "_managed_bin_dir", lambda: str(tmp_path))
    monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda: None)
    monkeypatch.setattr(
        bu, "_find_cli", lambda: [str(tmp_path / "browser-use")] if calls else None
    )
    monkeypatch.setattr(
        bu.shutil, "which", lambda name: "/managed/uv" if name == "uv" else None
    )

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bu.subprocess, "run", fake_run)
    ok, _message = bu.install_cli()

    assert ok
    assert calls[0][0] == [
        "/managed/uv",
        "tool",
        "install",
        "--force",
        "browser-use==0.13.8",
    ]
    assert calls[0][1]["env"]["UV_NO_CONFIG"] == "1"
    assert calls[0][1]["env"]["UV_TOOL_BIN_DIR"] == str(tmp_path)


def test_timeout_stops_exact_named_harness(monkeypatch):
    calls = []

    monkeypatch.setattr(bu, "_find_cli", lambda: ["/managed/browser-use"])
    monkeypatch.setattr(bu, "_base_subprocess_env", lambda: {"PATH": "/usr/bin"})
    monkeypatch.setattr(bu, "_blocked_url_in_code", lambda code: None)
    monkeypatch.setattr(bu, "_resolve_backend_cdp", lambda *args, **kwargs: None)
    monkeypatch.setattr(bu, "_workspace_dir", lambda *args: None)

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[-1] != "--reload":
            raise __import__("subprocess").TimeoutExpired(command, 5)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bu.subprocess, "run", fake_run)
    result = json.loads(
        bu.browser_exec(
            "print('work')", session="shared", task_id="conv-1", timeout_s=5
        )
    )

    assert "timed out" in result["error"]
    assert calls[-1][0] == ["/managed/browser-use", "--reload"]
    assert calls[-1][1]["env"]["BU_NAME"].startswith("bu-")
    assert bu._ACTIVE_HARNESSES == {}


def test_omnio_cdp_uses_isolated_default_harness_instance(monkeypatch, tmp_path):
    """Omnio's Toolbox tab is reused while IPC/temp state stays per session."""
    calls = []
    monkeypatch.setenv("BROWSER_CDP_URL_TEMPLATE", "http://relay/{session_id}")
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: str(tmp_path / "home")
    )
    monkeypatch.setattr(bu, "_find_cli", lambda: ["/managed/browser-use"])
    monkeypatch.setattr(
        bu,
        "_base_subprocess_env",
        lambda: {
            "PATH": "/usr/bin",
            "BROWSER_USE_API_KEY": "must-not-reach-local-runtime",
            "BU_AUTOSPAWN": "1",
        },
    )
    monkeypatch.setattr(bu, "_blocked_url_in_code", lambda code: None)
    monkeypatch.setattr(bu, "_resolve_backend_cdp", lambda *args, **kwargs: None)
    monkeypatch.setattr(bu, "_workspace_dir", lambda *args: str(tmp_path / "workspace"))

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bu.subprocess, "run", fake_run)
    from gateway import session_context

    nonce = f"{os.getpid()}-{time.time_ns()}"
    values = {"HERMES_SESSION_ID": f"conversation-omnio-a-{nonce}"}
    monkeypatch.setattr(session_context, "get_session_env", lambda key: values.get(key))
    monkeypatch.setattr(bu, "_owner_pid_is_alive", lambda pid: True)
    bu.browser_exec("print('a')", session="", task_id="child-a")
    first_env = calls[-1][1]["env"]

    values["HERMES_SESSION_ID"] = f"conversation-omnio-b-{nonce}"
    bu.browser_exec("print('b')", session="", task_id="child-b")
    second_env = calls[-1][1]["env"]

    assert first_env["BU_NAME"] == second_env["BU_NAME"] == "default"
    assert "BROWSER_USE_API_KEY" not in first_env
    assert first_env["BU_AUTOSPAWN"] == "0"
    assert first_env["BH_RUNTIME_DIR"] != second_env["BH_RUNTIME_DIR"]
    assert first_env["BH_TMP_DIR"] != second_env["BH_TMP_DIR"]
    assert "conversation-omnio-a" not in first_env["BH_RUNTIME_DIR"]
    assert "conversation-omnio-a" not in first_env["BH_TMP_DIR"]
    runtime_path = Path(first_env["BH_RUNTIME_DIR"])
    tmp_path_for_harness = Path(first_env["BH_TMP_DIR"])
    assert runtime_path.parent == Path("/tmp")
    for private_path in (runtime_path, tmp_path_for_harness):
        metadata = private_path.lstat()
        assert stat.S_ISDIR(metadata.st_mode)
        assert not stat.S_ISLNK(metadata.st_mode)
        assert stat.S_IMODE(metadata.st_mode) == 0o700
        if hasattr(os, "getuid"):
            assert metadata.st_uid == os.getuid()
    owner_marker = runtime_path / "gateway.owner_pid"
    assert owner_marker.read_text(encoding="ascii").strip() == str(os.getpid())
    assert stat.S_IMODE(owner_marker.stat().st_mode) == 0o600
    bu.cleanup_all_browser_use()
    reloads = [item for item in calls if item[0][-1] == "--reload"]
    assert len(reloads) == 2
    assert all(item[1]["env"]["BU_NAME"] == "default" for item in reloads)
    with bu._HARNESS_LOCK:
        bu._ACTIVE_HARNESSES.clear()


def test_omnio_named_sessions_keep_hashed_dedicated_daemons(monkeypatch, tmp_path):
    """Explicit names keep dedicated tabs and private runtime state."""
    calls = []
    monkeypatch.setenv("BROWSER_CDP_URL_TEMPLATE", "http://relay/{session_id}")
    monkeypatch.setattr(bu, "_find_cli", lambda: ["/managed/browser-use"])
    monkeypatch.setattr(bu, "_base_subprocess_env", lambda: {"PATH": "/usr/bin"})
    monkeypatch.setattr(bu, "_blocked_url_in_code", lambda code: None)
    monkeypatch.setattr(bu, "_resolve_backend_cdp", lambda *args, **kwargs: None)
    monkeypatch.setattr(bu, "_workspace_dir", lambda *args: str(tmp_path / "workspace"))
    monkeypatch.setattr(
        bu.subprocess,
        "run",
        lambda command, **kwargs: (
            calls.append((command, kwargs))
            or SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )
    from gateway import session_context

    named_conversation = f"same-conversation-{os.getpid()}-{time.time_ns()}"
    monkeypatch.setattr(
        session_context,
        "get_session_env",
        lambda key: named_conversation if key == "HERMES_SESSION_ID" else None,
    )
    monkeypatch.setattr(bu, "_owner_pid_is_alive", lambda pid: True)
    bu.browser_exec("print('one')", session="one", task_id="child-one")
    bu.browser_exec("print('two')", session="two", task_id="child-two")
    first_env = calls[-2][1]["env"]
    second_env = calls[-1][1]["env"]

    assert first_env["BU_NAME"].startswith("bu-")
    assert second_env["BU_NAME"].startswith("bu-")
    assert first_env["BU_NAME"] != second_env["BU_NAME"]
    assert first_env["BU_NAME"] != "default"
    assert first_env["BH_RUNTIME_DIR"] != second_env["BH_RUNTIME_DIR"]
    assert first_env["BH_TMP_DIR"] != second_env["BH_TMP_DIR"]
    for env in (first_env, second_env):
        runtime = Path(env["BH_RUNTIME_DIR"])
        assert runtime.parent == Path("/tmp")
        assert stat.S_IMODE(runtime.stat().st_mode) == 0o700
        marker = runtime / "gateway.owner_pid"
        assert marker.read_text(encoding="ascii").strip() == str(os.getpid())
        assert stat.S_IMODE(marker.stat().st_mode) == 0o600
    with bu._HARNESS_LOCK:
        bu._ACTIVE_HARNESSES.clear()


def test_omnio_stale_owner_reloads_exact_harness(monkeypatch, tmp_path):
    """A dead gateway owner is reaped without touching another daemon."""
    calls = []
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: str(tmp_path / "home")
    )
    monkeypatch.setattr(bu, "_owner_pid_is_alive", lambda pid: True)
    env = {}
    logical = f"stale-owner-{os.getpid()}-{time.time_ns()}"
    assert (
        bu._configure_omnio_harness_dirs(
            env, logical, harness_name="bu-stale-owner", cmd=["/managed/browser-use"]
        )
        is None
    )
    runtime = Path(env["BH_RUNTIME_DIR"])
    marker = runtime / "gateway.owner_pid"
    marker.write_text("999999\n", encoding="ascii")
    marker.chmod(0o600)

    monkeypatch.setattr(bu, "_owner_pid_is_alive", lambda pid: False)
    monkeypatch.setattr(
        bu,
        "_stop_harness_daemon",
        lambda *args: calls.append(args) or True,
    )
    assert (
        bu._configure_omnio_harness_dirs(
            env, logical, harness_name="bu-stale-owner", cmd=["/managed/browser-use"]
        )
        is None
    )
    assert calls
    assert calls[0][0] == "bu-stale-owner"
    assert calls[0][1]["BU_NAME"] == "bu-stale-owner"


def test_omnio_live_owner_is_not_taken_over(monkeypatch, tmp_path):
    """A live gateway owner blocks a second process from sharing the daemon."""
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: str(tmp_path / "home")
    )
    monkeypatch.setattr(bu, "_owner_pid_is_alive", lambda pid: True)
    env = {}
    logical = f"live-owner-{os.getpid()}-{time.time_ns()}"
    assert (
        bu._configure_omnio_harness_dirs(
            env, logical, harness_name="bu-live-owner", cmd=["/managed/browser-use"]
        )
        is None
    )
    marker = Path(env["BH_RUNTIME_DIR"]) / "gateway.owner_pid"
    marker.write_text("424242\n", encoding="ascii")
    marker.chmod(0o600)

    error = bu._configure_omnio_harness_dirs(
        {}, logical, harness_name="bu-live-owner", cmd=["/managed/browser-use"]
    )

    assert error and "live gateway process" in error
    assert marker.read_text(encoding="ascii").strip() == "424242"


def test_omnio_dead_owner_requires_successful_reclaim(monkeypatch, tmp_path):
    """A failed exact reload never overwrites the stale ownership marker."""
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: str(tmp_path / "home")
    )
    monkeypatch.setattr(bu, "_owner_pid_is_alive", lambda pid: False)
    monkeypatch.setattr(bu, "_stop_harness_daemon", lambda *args: False)
    env = {}
    logical = f"failed-reclaim-{os.getpid()}-{time.time_ns()}"
    # Seed the deterministic runtime and then replace its marker with a dead
    # owner PID, just as a crashed gateway would leave it.
    assert (
        bu._configure_omnio_harness_dirs(
            env, logical, harness_name="bu-failed-reclaim", cmd=["/managed/browser-use"]
        )
        is None
    )
    marker = Path(env["BH_RUNTIME_DIR"]) / "gateway.owner_pid"
    marker.write_text("999999\n", encoding="ascii")
    marker.chmod(0o600)

    error = bu._configure_omnio_harness_dirs(
        {}, logical, harness_name="bu-failed-reclaim", cmd=["/managed/browser-use"]
    )

    assert error and "could not be reloaded safely" in error
    assert marker.read_text(encoding="ascii").strip() == "999999"


def test_owner_marker_rejects_symlink(tmp_path):
    """The gateway marker cannot be redirected through a symlink."""
    if not hasattr(os, "O_NOFOLLOW"):
        return
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    target = tmp_path / "outside"
    target.write_text("keep", encoding="ascii")
    (runtime / "gateway.owner_pid").symlink_to(target)

    error = bu._write_omnio_owner_pid(runtime)

    assert error and "owner marker" in error
    assert target.read_text(encoding="ascii") == "keep"


def test_owner_pid_probe_uses_cross_platform_fallback(monkeypatch):
    """PID fallback must not use the Windows-unsafe os.kill probe."""
    from gateway import status

    def unavailable_probe(_pid):
        raise OSError("simulated helper failure")

    monkeypatch.setattr(status, "_pid_exists", unavailable_probe)
    monkeypatch.setattr(
        bu.os,
        "kill",
        lambda *_args: pytest.fail("os.kill must not be used"),
    )
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(pid_exists=lambda _pid: True),
    )

    assert bu._owner_pid_is_alive(12345) is True


def test_owner_pid_probe_keeps_uncertainty_alive(monkeypatch):
    """A failure of both probes must not permit unsafe owner takeover."""
    from gateway import status

    def unknown_probe(_pid):
        raise OSError("unknown")

    monkeypatch.setattr(status, "_pid_exists", unknown_probe)

    def unknown_psutil_probe(_pid):
        raise OSError("unknown")

    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(pid_exists=unknown_psutil_probe),
    )

    assert bu._owner_pid_is_alive(12345) is True


def test_inactivity_cleanup_only_stops_stale_named_daemon(monkeypatch):
    stopped = []
    monkeypatch.setattr(bu, "_harness_inactivity_timeout", lambda: 30)
    monkeypatch.setattr(
        bu,
        "_stop_harness_daemon",
        lambda name, env, cmd: stopped.append(name) or True,
    )
    with bu._HARNESS_LOCK:
        bu._ACTIVE_HARNESSES.clear()
        bu._ACTIVE_HARNESSES.update({
            "bu-stale": {
                "last_activity": 0,
                "env": {},
                "cmd": ["bu"],
                "in_flight": 0,
            },
            "bu-fresh": {
                "last_activity": time.time(),
                "env": {},
                "cmd": ["bu"],
                "in_flight": 0,
            },
            # A long-running browser_exec must retain its lease even when
            # its last activity predates the inactivity TTL.
            "bu-active": {
                "last_activity": 0,
                "env": {},
                "cmd": ["bu"],
                "in_flight": 1,
            },
        })

    bu._cleanup_inactive_harnesses()

    assert stopped == ["bu-stale"]
    with bu._HARNESS_LOCK:
        assert "bu-stale" not in bu._ACTIVE_HARNESSES
        assert "bu-active" in bu._ACTIVE_HARNESSES
        bu._ACTIVE_HARNESSES.clear()


def test_workspace_follows_canonical_conversation_and_session(monkeypatch, tmp_path):
    from gateway import session_context

    monkeypatch.delenv("BH_AGENT_WORKSPACE", raising=False)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: str(tmp_path))
    values = {"HERMES_SESSION_ID": "conversation-a"}
    monkeypatch.setattr(session_context, "get_session_env", lambda key: values.get(key))

    parent = bu._workspace_dir("parent-task", "same-label")
    delegated = bu._workspace_dir("child-task", "same-label")
    other_label = bu._workspace_dir("child-task", "different-label")

    assert parent == delegated
    assert parent != other_label
    assert parent and "conversation-a" not in parent
    assert "same-label" not in parent


def test_workspace_ignores_unkeyed_inherited_path(monkeypatch, tmp_path):
    from gateway import session_context

    monkeypatch.setenv("BH_AGENT_WORKSPACE", str(tmp_path / "raw-session-name"))
    monkeypatch.setattr(
        session_context,
        "get_session_env",
        lambda key: "conversation-safe" if key == "HERMES_SESSION_ID" else None,
    )
    workspace = bu._workspace_dir("child", "label")

    assert workspace != str(tmp_path / "raw-session-name")
    assert workspace and Path(workspace).name.startswith("bu-")


class _RemoteResponse:
    def __init__(self, status_code=200, payload=None, *, body=b"", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.content = body
        self.headers = headers or {}
        self.closed = False

    def json(self):
        if isinstance(self._payload, BaseException):
            raise self._payload
        return self._payload

    def iter_content(self, chunk_size=0):  # noqa: ARG002
        yield self.content

    def close(self):
        self.closed = True


def _configure_remote_omnio(monkeypatch):
    monkeypatch.setenv(
        "OMNIO_BROWSER_EXEC_URL",
        "http://127.0.0.1:8642/internal/toolbox/browser/exec",
    )
    monkeypatch.setenv("OMNIO_TOOLBOX_URL", "http://127.0.0.1:8642/internal/toolbox")
    monkeypatch.setenv("OMNIO_TOOLBOX_BEARER", "pair-secret")
    monkeypatch.setenv("OMNIO_TOOLBOX_BRAND", "brand-a")
    monkeypatch.setattr(
        bu, "_canonical_browser_session_id", lambda task_id: "hermes-session"
    )
    monkeypatch.setattr(bu, "_native_vision_enabled", lambda: False)


def test_remote_browser_exec_preserves_result_contract_and_pair_headers(monkeypatch):
    _configure_remote_omnio(monkeypatch)
    calls = []

    def fail_local(*_args, **_kwargs):
        raise AssertionError("Omnio remote execution must not resolve the local CLI")

    monkeypatch.setattr(bu, "_find_cli", fail_local)
    monkeypatch.setattr(bu, "_base_subprocess_env", fail_local)

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return _RemoteResponse(
            payload={
                "output": "done",
                "returncode": 0,
                "workspace": "/home/brand/workspace",
                "session": "named",
                "stderr": " warning ",
                "screenshotPath": "",
                "downloads": [],
            }
        )

    monkeypatch.setattr(bu.requests, "post", fake_post)
    result = json.loads(
        bu.browser_exec("print('done')", session="named", timeout_s=2, task_id="child")
    )

    assert result == {
        "success": True,
        "exit_code": 0,
        "output": "done",
        "workspace": "/home/brand/workspace",
        "downloads": [],
        "session": "named",
        "stderr": "warning",
    }
    url, kwargs = calls[0]
    assert url.endswith("/browser/exec")
    assert kwargs["headers"] == {
        "Authorization": "Bearer pair-secret",
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",
        "X-Omnio-Brand": "brand-a",
        "X-Hermes-Session-Id": "hermes-session",
    }
    assert kwargs["json"]["code"] == "print('done')"
    assert kwargs["json"]["session"] == "named"
    assert kwargs["json"]["timeoutSeconds"] == 5
    assert kwargs["json"]["executionId"]
    assert kwargs["timeout"] == 10


def test_remote_browser_exec_never_falls_back_when_capability_is_present(monkeypatch):
    _configure_remote_omnio(monkeypatch)
    local_calls = []
    monkeypatch.setattr(bu, "_find_cli", lambda: local_calls.append(True))
    monkeypatch.setattr(
        bu.requests,
        "post",
        lambda *_args, **_kwargs: _RemoteResponse(
            payload={
                "output": "ok",
                "returncode": 0,
                "workspace": "/home/brand",
                "downloads": [],
            }
        ),
    )

    result = json.loads(bu.browser_exec("print('ok')", task_id="child"))

    assert result["success"] is True
    assert local_calls == []


def test_remote_browser_exec_missing_pair_auth_fails_closed(monkeypatch):
    _configure_remote_omnio(monkeypatch)
    monkeypatch.delenv("OMNIO_TOOLBOX_BEARER")
    monkeypatch.setattr(
        bu,
        "_find_cli",
        lambda: pytest.fail("remote capability must not fall back to local CLI"),
    )
    result = json.loads(bu.browser_exec("print('auth')", task_id="child"))

    assert "pair credentials" in result["error"]
    assert "Reprovision this Omnio sandbox" in result["error"]


def test_remote_browser_exec_runs_url_safety_before_forwarding(monkeypatch):
    _configure_remote_omnio(monkeypatch)
    calls = []
    monkeypatch.setattr(bu, "_blocked_url_in_code", lambda _code: "Blocked unsafe URL")
    monkeypatch.setattr(
        bu.requests,
        "post",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = json.loads(bu.browser_exec("new_tab('https://unsafe.test')"))

    assert result == {"error": "Blocked unsafe URL"}
    assert calls == []


@pytest.mark.parametrize("status_code", [404, 405, 422, 501])
def test_remote_browser_exec_old_toolbox_requires_reprovision(monkeypatch, status_code):
    _configure_remote_omnio(monkeypatch)
    calls = []
    monkeypatch.setattr(
        bu.requests,
        "post",
        lambda url, **kwargs: (
            calls.append((url, kwargs)) or _RemoteResponse(status_code=status_code)
        ),
    )

    result = json.loads(bu.browser_exec("print('old')", task_id="child"))

    assert "Reprovision this Omnio sandbox" in result["error"]
    assert len(calls) == 1


@pytest.mark.parametrize("status_code", [500, 503])
def test_remote_browser_exec_server_error_is_actionable_without_replay(
    monkeypatch, status_code
):
    _configure_remote_omnio(monkeypatch)
    calls = []
    monkeypatch.setattr(
        bu.requests,
        "post",
        lambda url, **kwargs: (
            calls.append((url, kwargs)) or _RemoteResponse(status_code=status_code)
        ),
    )

    result = json.loads(bu.browser_exec("print('server')", task_id="child"))

    assert "browser exec failed with HTTP" in result["error"]
    assert "not replayed" in result["error"]
    assert len(calls) == 1


def test_remote_browser_exec_timeout_cancels_once_without_replay(monkeypatch):
    _configure_remote_omnio(monkeypatch)
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith("/cancel"):
            return _RemoteResponse(payload={"cancelled": True})
        raise bu.requests.Timeout("request exceeded transport budget")

    monkeypatch.setattr(bu.requests, "post", fake_post)
    result = json.loads(
        bu.browser_exec("print('maybe ran')", timeout_s=5, task_id="child")
    )

    assert "timed out after 5s" in result["error"]
    assert "do not replay" in result["error"]
    assert [url.rsplit("/", 1)[-1] for url, _kwargs in calls] == ["exec", "cancel"]
    assert calls[1][1]["json"]["executionId"] == calls[0][1]["json"]["executionId"]


def test_remote_browser_exec_invalid_response_encoding_is_not_replayed(monkeypatch):
    _configure_remote_omnio(monkeypatch)
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        raise bu.requests.exceptions.ContentDecodingError(
            "response body was not valid gzip"
        )

    monkeypatch.setattr(bu.requests, "post", fake_post)
    result = json.loads(bu.browser_exec("print('maybe ran')", task_id="child"))

    assert "invalid remote response encoding" in result["error"]
    assert "may already have executed" in result["error"]
    assert "do not replay" in result["error"]
    assert [url.rsplit("/", 1)[-1] for url, _kwargs in calls] == ["exec"]


def test_remote_browser_exec_rejects_unsafe_screenshot_without_fetch(monkeypatch):
    _configure_remote_omnio(monkeypatch)
    posts = []
    gets = []
    monkeypatch.setattr(
        bu.requests,
        "post",
        lambda url, **kwargs: (
            posts.append((url, kwargs))
            or _RemoteResponse(
                payload={
                    "output": "captured",
                    "returncode": 0,
                    "workspace": "/home/brand",
                    "downloads": [],
                    "screenshotPath": "/etc/shadow.png",
                }
            )
        ),
    )
    monkeypatch.setattr(bu.requests, "get", lambda *args, **kwargs: gets.append(args))

    result = json.loads(bu.browser_exec("capture_screenshot()", task_id="child"))

    assert "unsafe screenshot path" in result["error"]
    assert len(posts) == 1
    assert gets == []


def test_remote_browser_exec_nonvision_keeps_virtual_screenshot_path(monkeypatch):
    _configure_remote_omnio(monkeypatch)
    posts = []
    gets = []
    monkeypatch.setattr(
        bu.requests,
        "post",
        lambda url, **kwargs: (
            posts.append((url, kwargs))
            or _RemoteResponse(
                payload={
                    "output": "captured",
                    "returncode": 0,
                    "workspace": "/home/brand",
                    "downloads": [],
                    "screenshotPath": "/tmp/screenshots/shot.png",
                }
            )
        ),
    )
    monkeypatch.setattr(bu.requests, "get", lambda *args, **kwargs: gets.append(args))

    result = json.loads(bu.browser_exec("capture_screenshot()", task_id="child"))

    assert result["screenshot_path"] == "/tmp/screenshots/shot.png"
    assert gets == []


def test_remote_browser_exec_native_screenshot_fetches_and_cleans_temp(monkeypatch):
    _configure_remote_omnio(monkeypatch)
    monkeypatch.setattr(bu, "_native_vision_enabled", lambda: True)
    post_response = _RemoteResponse(
        payload={
            "output": "captured",
            "returncode": 0,
            "workspace": "/home/brand",
            "downloads": [],
            "screenshotPath": "/tmp/screenshots/shot.png",
        }
    )
    file_response = _RemoteResponse(body=b"PNG bytes", headers={"Content-Length": "9"})
    observed = []

    monkeypatch.setattr(bu.requests, "post", lambda *_args, **_kwargs: post_response)
    monkeypatch.setattr(
        bu.requests,
        "get",
        lambda url, **kwargs: observed.append((url, kwargs)) or file_response,
    )

    def fake_native(result, path, *, display_path=None):
        observed.append((path, display_path, Path(path).exists()))
        assert result["screenshot_path"] == display_path
        return {
            "_multimodal": True,
            "text_summary": json.dumps({**result, "screenshot_path": display_path}),
            "meta": {"screenshot_path": display_path},
        }

    monkeypatch.setattr(bu, "_native_screenshot_result", fake_native)
    result = bu.browser_exec("capture_screenshot()", task_id="child")

    assert result["meta"]["screenshot_path"] == "/tmp/screenshots/shot.png"
    assert observed[0][0].endswith("/files")
    assert observed[0][1]["params"] == {"path": "/tmp/screenshots/shot.png"}
    assert observed[0][1]["headers"]["Authorization"] == "Bearer pair-secret"
    assert observed[1][2] is True
    assert not Path(observed[1][0]).exists()


def test_remote_browser_exec_skips_oversized_native_attachment_without_replay(
    monkeypatch,
):
    _configure_remote_omnio(monkeypatch)
    monkeypatch.setattr(bu, "_native_vision_enabled", lambda: True)
    monkeypatch.setattr(
        bu.requests,
        "post",
        lambda *_args, **_kwargs: _RemoteResponse(
            payload={
                "output": "captured",
                "returncode": 0,
                "workspace": "/home/brand",
                "downloads": [],
                "screenshotPath": "/tmp/screenshots/huge.png",
            }
        ),
    )
    gets = []
    monkeypatch.setattr(
        bu.requests,
        "get",
        lambda *args, **kwargs: (
            gets.append((args, kwargs))
            or _RemoteResponse(
                body=b"not-read",
                headers={"Content-Length": str(bu._REMOTE_SCREENSHOT_MAX_BYTES + 1)},
            )
        ),
    )

    result = json.loads(bu.browser_exec("capture_screenshot()", task_id="child"))

    assert result["screenshot_path"] == "/tmp/screenshots/huge.png"
    assert len(gets) == 1


def test_remote_browser_exec_native_fetch_failure_does_not_invite_replay(monkeypatch):
    _configure_remote_omnio(monkeypatch)
    monkeypatch.setattr(bu, "_native_vision_enabled", lambda: True)
    monkeypatch.setattr(
        bu.requests,
        "post",
        lambda *_args, **_kwargs: _RemoteResponse(
            payload={
                "output": "captured once",
                "returncode": 0,
                "workspace": "/home/brand",
                "downloads": [],
                "screenshotPath": "/tmp/screenshots/shot.png",
            }
        ),
    )
    monkeypatch.setattr(
        bu.requests,
        "get",
        lambda *_args, **_kwargs: _RemoteResponse(
            body=b"", headers={"Content-Length": "0"}
        ),
    )

    result = json.loads(bu.browser_exec("capture_screenshot()", task_id="child"))

    assert result["success"] is True
    assert result["output"] == "captured once"
    assert result["screenshot_path"] == "/tmp/screenshots/shot.png"
