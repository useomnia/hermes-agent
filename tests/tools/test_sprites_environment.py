import json
from typing import cast


def _run_terminal_with_existing_env(monkeypatch, env):
    import tools.terminal_tool as terminal_tool

    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": "sprites",
            "timeout": 30,
            "cwd": "/brand",
        },
    )
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda *_args, **_kwargs: {"approved": True},
    )
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": env})
    monkeypatch.setattr(terminal_tool, "_last_activity", {"default": 0.0})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})

    return json.loads(terminal_tool.terminal_tool("echo hello"))


def test_get_env_config_should_default_sprites_to_brand_workspace(monkeypatch):
    from tools.terminal_tool import _get_env_config

    monkeypatch.setenv("TERMINAL_ENV", "sprites")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setenv("OMNIO_TOOLBOX_URL", "https://toolbox.example")
    monkeypatch.setenv("OMNIO_TOOLBOX_BEARER", "pair-secret")
    monkeypatch.setenv("OMNIO_TOOLBOX_BRAND", "brand-123")

    config = _get_env_config()

    assert config["env_type"] == "sprites"
    assert config["cwd"] == "/brand"
    assert config["sprites_url"] == "https://toolbox.example"
    assert config["sprites_bearer"] == "pair-secret"
    assert config["sprites_brand"] == "brand-123"


def test_create_environment_should_construct_sprites_environment(monkeypatch):
    from tools import terminal_tool
    import tools.environments.sprites as sprites_module

    captured = {}

    class FakeSpritesEnvironment:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(sprites_module, "SpritesEnvironment", FakeSpritesEnvironment)

    env = terminal_tool._create_environment(
        env_type="sprites",
        image="",
        cwd="/brand",
        timeout=30,
        container_config={
            "sprites_url": "https://toolbox.example",
            "sprites_bearer": "pair-secret",
            "sprites_brand": "brand-123",
        },
    )

    assert isinstance(env, FakeSpritesEnvironment)
    assert captured == {
        "toolbox_url": "https://toolbox.example",
        "bearer_token": "pair-secret",
        "brand": "brand-123",
        "cwd": "/brand",
        "timeout": 30,
    }


def test_sprites_environment_should_send_exec_to_toolbox():
    from tools.environments.sprites import SpritesEnvironment

    env = SpritesEnvironment.__new__(SpritesEnvironment)
    env.cwd = "/brand"
    env.timeout = 60
    env.toolbox_url = "https://toolbox.example"
    env.bearer_token = "pair-secret"
    env.brand = "brand-123"

    calls = []

    def fake_request(path, payload=None, *, timeout=None, method="POST"):
        calls.append({"path": path, "payload": payload, "timeout": timeout, "method": method})
        return {"output": "ok\n", "returncode": 0}

    env._request_json = fake_request
    handle = env._run_bash("echo ok", timeout=9, stdin_data="payload")

    assert handle.wait(timeout=2) == 0
    assert handle.stdout.read() == "ok\n"
    assert calls == [
        {
            "path": "/exec",
            "payload": {
                "command": "echo ok",
                "cwd": "/brand",
                "login": False,
                "stdin": "payload",
                "timeoutSeconds": 9,
            },
            "timeout": 14,
            "method": "POST",
        }
    ]


def test_sprites_environment_should_use_toolbox_temp_session_dir():
    from tools.environments.sprites import SpritesEnvironment

    env = SpritesEnvironment.__new__(SpritesEnvironment)

    assert env.get_temp_dir() == "/tmp/.hermes-session"


def test_sprites_environment_should_upload_skills_with_dedicated_operation(tmp_path):
    from tools.environments.sprites import SpritesEnvironment

    skill = tmp_path / "SKILL.md"
    skill.write_text("name: demo\n", encoding="utf-8")
    env = SpritesEnvironment.__new__(SpritesEnvironment)
    requests = []
    env.file_request = requests.append

    env._sprites_upload(str(skill), "/skills/demo/SKILL.md")

    assert requests == [
        {
            "operation": "writeSkills",
            "path": "/skills/demo/SKILL.md",
            "contentBase64": "bmFtZTogZGVtbwo=",
            "encoding": "base64",
        }
    ]


def test_sprites_environment_should_bulk_upload_skills_with_dedicated_operation(tmp_path):
    from tools.environments.sprites import SpritesEnvironment

    skill = tmp_path / "SKILL.md"
    skill.write_text("name: demo\n", encoding="utf-8")
    env = SpritesEnvironment.__new__(SpritesEnvironment)
    requests = []
    env.file_request = requests.append

    env._sprites_bulk_upload([(str(skill), "/skills/demo/SKILL.md")])

    assert requests == [
        {
            "operation": "writeSkills",
            "files": [
                {
                    "path": "/skills/demo/SKILL.md",
                    "contentBase64": "bmFtZTogZGVtbwo=",
                    "encoding": "base64",
                }
            ],
        }
    ]


def test_sprites_environment_should_bound_and_split_skill_batches(tmp_path, monkeypatch):
    import tools.environments.sprites as sprites_module
    from tools.environments.sprites import SpritesEnvironment

    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    env = SpritesEnvironment.__new__(SpritesEnvironment)
    requests = []
    env.file_request = requests.append
    monkeypatch.setattr(sprites_module, "_MAX_SKILL_BATCH_FILES", 1)

    env._sprites_bulk_upload(
        [
            (str(first), "/skills/demo/first.txt"),
            (str(second), "/skills/demo/second.txt"),
        ]
    )

    assert len(requests) == 2
    assert [request["files"][0]["path"] for request in requests] == [
        "/skills/demo/first.txt",
        "/skills/demo/second.txt",
    ]


def test_sprites_environment_should_refuse_symlinked_skill_source(tmp_path):
    import pytest

    from tools.environments.sprites import SpritesEnvironment, SpritesToolboxError

    target = tmp_path / "target.txt"
    target.write_text("secret", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    env = SpritesEnvironment.__new__(SpritesEnvironment)

    with pytest.raises(SpritesToolboxError, match="refused non-file"):
        env._sprites_upload(str(link), "/skills/demo/link.txt")


def test_sprites_environment_should_delete_skills_with_dedicated_operation():
    from tools.environments.sprites import SpritesEnvironment

    env = SpritesEnvironment.__new__(SpritesEnvironment)
    requests = []
    env.file_request = requests.append

    env._sprites_delete(["/skills/demo/SKILL.md"])

    assert requests == [
        {"operation": "deleteSkills", "path": "/skills/demo/SKILL.md", "missingOk": True}
    ]


def test_sprites_request_should_send_bearer_and_brand_headers(monkeypatch):
    import tools.environments.sprites as sprites_module
    from tools.environments.sprites import SpritesEnvironment

    env = SpritesEnvironment.__new__(SpritesEnvironment)
    env.toolbox_url = "https://toolbox.example"
    env.bearer_token = "pair-secret"
    env.brand = "brand-123"
    env.timeout = 60

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _limit):
            return json.dumps({"ok": True}).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["body"] = request.data.decode()
        return FakeResponse()

    monkeypatch.setattr(sprites_module._URL_OPENER, "open", fake_urlopen)

    response = env._request_json("/health", {"ping": True}, timeout=5)

    assert response == {"ok": True}
    assert captured["url"] == "https://toolbox.example/health"
    assert captured["timeout"] == 5
    assert captured["headers"]["Authorization"] == "Bearer pair-secret"
    assert captured["headers"]["X-omnio-brand"] == "brand-123"
    assert json.loads(captured["body"]) == {"ping": True}


def test_sprites_request_should_reject_oversized_response(monkeypatch):
    import pytest

    import tools.environments.sprites as sprites_module
    from tools.environments.sprites import SpritesEnvironment, SpritesToolboxError

    env = SpritesEnvironment.__new__(SpritesEnvironment)
    env.toolbox_url = "https://toolbox.example"
    env.bearer_token = "pair-secret"
    env.brand = "brand-123"
    env.timeout = 60

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, limit):
            return b"x" * limit

    monkeypatch.setattr(
        sprites_module._URL_OPENER,
        "open",
        lambda request, timeout: FakeResponse(),
    )

    with pytest.raises(SpritesToolboxError, match="response exceeded"):
        env._request_json("/health", timeout=5)


def test_sprites_request_should_parse_structured_toolbox_error(monkeypatch):
    import io
    import urllib.error
    from email.message import Message

    import pytest

    import tools.environments.sprites as sprites_module
    from tools.environments.sprites import SpritesEnvironment, SpritesToolboxError

    env = SpritesEnvironment.__new__(SpritesEnvironment)
    env.toolbox_url = "https://toolbox.example"
    env.bearer_token = "pair-secret"
    env.brand = "brand-123"
    env.timeout = 60
    body = json.dumps({
        "error": {
            "code": "EXEC_START_FAILED",
            "phase": "start",
            "retryable": True,
            "commandStarted": False,
            "requestId": "req-123",
            "message": "executor unavailable",
        }
    }).encode()
    http_error = urllib.error.HTTPError(
        "https://toolbox.example/exec",
        503,
        "Service Unavailable",
        Message(),
        io.BytesIO(body),
    )

    def raise_http_error(request, timeout):
        raise http_error

    monkeypatch.setattr(
        sprites_module._URL_OPENER,
        "open",
        raise_http_error,
    )

    with pytest.raises(SpritesToolboxError) as exc_info:
        env._request_json("/exec", timeout=5)

    error = exc_info.value
    assert error.code == "EXEC_START_FAILED"
    assert error.phase == "start"
    assert error.retryable is True
    assert error.command_started is False
    assert error.request_id == "req-123"
    assert error.http_status == 503
    assert "executor unavailable" in str(error)
    assert "code=EXEC_START_FAILED" in str(error)
    assert "phase=start" in str(error)


def test_sprites_request_should_treat_v4_error_as_nonretryable(monkeypatch):
    import io
    import urllib.error
    from email.message import Message

    import pytest

    import tools.environments.sprites as sprites_module
    from tools.environments.sprites import SpritesEnvironment, SpritesToolboxError

    env = SpritesEnvironment.__new__(SpritesEnvironment)
    env.toolbox_url = "https://toolbox.example"
    env.bearer_token = "pair-secret"
    env.brand = "brand-123"
    env.timeout = 60
    http_error = urllib.error.HTTPError(
        "https://toolbox.example/exec",
        503,
        "Service Unavailable",
        Message(),
        io.BytesIO(b'{"error":"toolbox is still resuming"}'),
    )

    def raise_http_error(request, timeout):
        raise http_error

    monkeypatch.setattr(
        sprites_module._URL_OPENER,
        "open",
        raise_http_error,
    )

    with pytest.raises(SpritesToolboxError) as exc_info:
        env._request_json("/exec", timeout=5)

    error = exc_info.value
    assert error.retryable is False
    assert error.command_started is None
    assert error.code is None
    assert error.phase is None
    assert "toolbox is still resuming" in str(error)


def test_terminal_should_retry_confirmed_not_started_error_once(monkeypatch):
    import tools.terminal_tool as terminal_tool
    from tools.environments.sprites import SpritesToolboxError

    error = SpritesToolboxError(
        "Toolbox API /exec failed with HTTP 503: executor unavailable",
        code="EXEC_START_FAILED",
        phase="start",
        retryable=True,
        command_started=False,
        request_id="req-123",
        http_status=503,
    )

    class FakeEnv:
        cwd = "/brand"

        def __init__(self):
            self.calls = 0

        def execute(self, command, **kwargs):
            self.calls += 1
            raise error

    env = FakeEnv()
    sleeps = []
    monkeypatch.setattr(terminal_tool.time, "sleep", sleeps.append)

    result = _run_terminal_with_existing_env(monkeypatch, env)

    assert env.calls == 2
    assert sleeps == [terminal_tool._CONFIRMED_NOT_STARTED_RETRY_DELAY_SECONDS]
    assert result["exit_code"] == -1
    assert "executor unavailable" in result["error"]
    assert "code=EXEC_START_FAILED" in result["error"]
    assert "phase=start" in result["error"]


def test_terminal_should_not_retry_ambiguous_timeout(monkeypatch):
    import tools.terminal_tool as terminal_tool

    class FakeEnv:
        cwd = "/brand"

        def __init__(self):
            self.calls = 0

        def execute(self, command, **kwargs):
            self.calls += 1
            raise TimeoutError("read timed out after request was sent")

    env = FakeEnv()
    sleeps = []
    monkeypatch.setattr(terminal_tool.time, "sleep", sleeps.append)

    result = _run_terminal_with_existing_env(monkeypatch, env)

    assert env.calls == 1
    assert sleeps == []
    assert result["exit_code"] == -1
    assert "read timed out after request was sent" in result["error"]


def test_terminal_should_not_retry_unstructured_v4_error(monkeypatch):
    import tools.terminal_tool as terminal_tool
    from tools.environments.sprites import SpritesToolboxError

    error = SpritesToolboxError(
        "Toolbox API /exec failed with HTTP 503: toolbox is still resuming",
        http_status=503,
    )

    class FakeEnv:
        cwd = "/brand"

        def __init__(self):
            self.calls = 0

        def execute(self, command, **kwargs):
            self.calls += 1
            raise error

    env = FakeEnv()
    sleeps = []
    monkeypatch.setattr(terminal_tool.time, "sleep", sleeps.append)

    result = _run_terminal_with_existing_env(monkeypatch, env)

    assert env.calls == 1
    assert sleeps == []
    assert "toolbox is still resuming" in result["error"]


def test_terminal_should_not_retry_user_command_exit_one(monkeypatch):
    class FakeEnv:
        cwd = "/brand"

        def __init__(self):
            self.calls = 0

        def execute(self, command, **kwargs):
            self.calls += 1
            return {"output": "", "returncode": 1}

    env = FakeEnv()

    result = _run_terminal_with_existing_env(monkeypatch, env)

    assert env.calls == 1
    assert result == {
        "output": "",
        "exit_code": 1,
        "error": None,
    }


def test_toolbox_url_should_require_a_safe_origin():
    import pytest

    from tools.environments.sprites import _normalize_toolbox_url

    assert _normalize_toolbox_url("https://toolbox.example/") == "https://toolbox.example"
    assert _normalize_toolbox_url("http://127.0.0.1:8643") == "http://127.0.0.1:8643"
    with pytest.raises(ValueError, match="HTTPS outside loopback"):
        _normalize_toolbox_url("http://toolbox.example")
    with pytest.raises(ValueError, match="must not contain a path"):
        _normalize_toolbox_url("https://toolbox.example/redirect")


def test_sprites_file_operations_should_use_files_endpoint_for_write():
    from tools.environments.sprites import SpritesEnvironment, SpritesFileOperations

    class FakeEnv:
        cwd = "/brand"
        config = None

        def __init__(self):
            self.requests = []

        def file_request(self, payload):
            self.requests.append(payload)
            if payload["operation"] == "readRaw":
                return {"error": "File not found"}
            if payload["operation"] == "write":
                return {"bytesWritten": len(payload["content"].encode("utf-8"))}
            raise AssertionError(f"unexpected operation: {payload['operation']}")

        def execute(self, command, cwd=None, **kwargs):
            return {"output": "", "returncode": 0}

    env = FakeEnv()
    ops = SpritesFileOperations(cast(SpritesEnvironment, env))

    result = ops.write_file("/brand/example.txt", "hello")

    assert result.error is None
    assert result.bytes_written == 5
    assert env.requests == [
        {"operation": "readRaw", "path": "/brand/example.txt"},
        {
            "operation": "write",
            "path": "/brand/example.txt",
            "content": "hello",
            "encoding": "utf-8",
        },
    ]


def test_execute_code_guard_should_approve_sprites_backend():
    from tools.approval import check_execute_code_guard

    result = check_execute_code_guard("import os", "sprites")

    assert result["approved"] is True
