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


def test_get_env_config_should_default_sprites_to_home_workspace(monkeypatch):
    from tools.terminal_tool import _get_env_config

    monkeypatch.setenv("TERMINAL_ENV", "sprites")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setenv("OMNIO_TOOLBOX_URL", "https://toolbox.example")
    monkeypatch.setenv("OMNIO_TOOLBOX_BEARER", "pair-secret")
    monkeypatch.setenv("OMNIO_TOOLBOX_BRAND", "brand-123")

    config = _get_env_config()

    assert config["env_type"] == "sprites"
    assert config["cwd"] == "/home"
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

    def fake_request(
        path,
        payload=None,
        *,
        timeout=None,
        method="POST",
        retry_exec_predispatch=False,
        retry_deadline_seconds=None,
        cancel_event=None,
    ):
        calls.append(
            {
                "path": path,
                "payload": payload,
                "timeout": timeout,
                "method": method,
                "retry_exec_predispatch": retry_exec_predispatch,
                "retry_deadline_seconds": retry_deadline_seconds,
                "cancel_event": cancel_event is not None,
            }
        )
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
            "retry_exec_predispatch": True,
            "retry_deadline_seconds": 9,
            "cancel_event": True,
        }
    ]


def test_sprites_environment_should_stream_raw_file_bytes(monkeypatch):
    import tools.environments.sprites as sprites_module
    from tools.environments.sprites import SpritesEnvironment

    env = SpritesEnvironment.__new__(SpritesEnvironment)
    env.toolbox_url = "https://toolbox.example/internal/toolbox"
    env.bearer_token = "pair-secret"
    env.brand = "brand-123"
    env.timeout = 60
    observed = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, limit):
            observed["limit"] = limit
            return b"raw-image-bytes"[:limit]

    def fake_open(request, timeout):
        observed["url"] = request.full_url
        observed["authorization"] = request.get_header("Authorization")
        observed["brand"] = request.get_header("X-omnio-brand")
        observed["method"] = request.get_method()
        observed["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(sprites_module._URL_OPENER, "open", fake_open)

    result = env.read_file_bytes("/home/image with spaces.png", max_bytes=7)

    assert result == b"raw-ima"
    assert observed == {
        "url": (
            "https://toolbox.example/internal/toolbox/files?"
            "path=%2Fhome%2Fimage+with+spaces.png"
        ),
        "authorization": "Bearer pair-secret",
        "brand": "brand-123",
        "method": "GET",
        "timeout": 60,
        "limit": 7,
    }


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
        env._request_json(
            "/exec",
            {"command": "pwd", "cwd": "/"},
            timeout=5,
        )

    error = exc_info.value
    assert error.code == "EXEC_START_FAILED"
    assert error.phase == "start"
    assert error.retryable is True
    assert error.command_started is False
    assert error.request_id == "req-123"
    assert error.http_status == 503
    assert error.request_cwd == "/"
    assert "executor unavailable" in str(error)
    assert '"retryable": true' in str(error)
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


def test_sprites_exec_should_retry_twice_then_succeed(monkeypatch, caplog):
    import io
    import urllib.error
    from email.message import Message

    import tools.environments.sprites as sprites_module
    from tools.environments.sprites import SpritesEnvironment

    env = SpritesEnvironment.__new__(SpritesEnvironment)
    env.toolbox_url = "https://toolbox.example"
    env.bearer_token = "pair-secret"
    env.brand = "brand-123"
    env.timeout = 60
    attempts = 0

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _limit):
            return b'{"output":"ok\\n","returncode":0}'

    def fake_open(request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            body = json.dumps(
                {
                    "error": {
                        "code": "pre_dispatch_failed",
                        "phase": "pre_dispatch",
                        "retryable": True,
                        "commandStarted": False,
                        "requestId": f"req-{attempts}",
                    }
                }
            ).encode()
            raise urllib.error.HTTPError(
                request.full_url,
                503,
                "Service Unavailable",
                Message(),
                io.BytesIO(body),
            )
        return FakeResponse()

    sleeps = []
    monkeypatch.setattr(sprites_module._URL_OPENER, "open", fake_open)
    monkeypatch.setattr(sprites_module.time, "sleep", sleeps.append)

    response = env._request_json(
        "/exec",
        {"command": "echo ok"},
        retry_exec_predispatch=True,
        retry_deadline_seconds=30,
    )

    assert response == {"output": "ok\n", "returncode": 0}
    assert attempts == 3
    assert sleeps == list(sprites_module._EXEC_PREDISPATCH_RETRY_DELAYS_SECONDS)
    assert caplog.text.count("Retrying Toolbox API /exec") == 2
    assert "code=pre_dispatch_failed" in caplog.text
    assert "phase=pre_dispatch" in caplog.text
    assert "requestId=req-1" in caplog.text
    assert "requestId=req-2" in caplog.text


def test_sprites_exec_should_stop_after_two_retries(monkeypatch):
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
    attempts = 0

    def fake_open(request, timeout):
        nonlocal attempts
        attempts += 1
        body = json.dumps(
            {
                "error": {
                    "code": "pre_dispatch_failed",
                    "phase": "pre_dispatch",
                    "retryable": True,
                    "commandStarted": False,
                    "requestId": f"req-{attempts}",
                }
            }
        ).encode()
        raise urllib.error.HTTPError(
            request.full_url,
            503,
            "Service Unavailable",
            Message(),
            io.BytesIO(body),
        )

    sleeps = []
    monkeypatch.setattr(sprites_module._URL_OPENER, "open", fake_open)
    monkeypatch.setattr(sprites_module.time, "sleep", sleeps.append)

    with pytest.raises(SpritesToolboxError) as exc_info:
        env._request_json(
            "/exec",
            {"command": "echo ok"},
            retry_exec_predispatch=True,
        )

    assert attempts == 3
    assert sleeps == list(sprites_module._EXEC_PREDISPATCH_RETRY_DELAYS_SECONDS)
    assert exc_info.value.request_id == "req-3"


def test_sprites_exec_should_not_retry_past_command_deadline(monkeypatch):
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
    attempts = 0
    clock = 0.0

    def fake_open(request, timeout):
        nonlocal attempts
        attempts += 1
        body = json.dumps(
            {
                "error": {
                    "code": "pre_dispatch_failed",
                    "phase": "pre_dispatch",
                    "retryable": True,
                    "commandStarted": False,
                    "requestId": f"req-{attempts}",
                }
            }
        ).encode()
        raise urllib.error.HTTPError(
            request.full_url,
            503,
            "Service Unavailable",
            Message(),
            io.BytesIO(body),
        )

    def fake_sleep(delay):
        nonlocal clock
        clock += delay

    monkeypatch.setattr(sprites_module._URL_OPENER, "open", fake_open)
    monkeypatch.setattr(sprites_module.time, "monotonic", lambda: clock)
    monkeypatch.setattr(sprites_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(
        sprites_module,
        "_EXEC_PREDISPATCH_RETRY_DELAYS_SECONDS",
        (0.0, 2.0),
    )

    with pytest.raises(SpritesToolboxError) as exc_info:
        env._request_json(
            "/exec",
            {"command": "echo ok"},
            retry_exec_predispatch=True,
            retry_deadline_seconds=1,
        )

    assert attempts == 2
    assert clock == 0.0
    assert exc_info.value.request_id == "req-2"


def test_sprites_run_bash_should_apply_short_command_deadline(monkeypatch):
    import io
    import urllib.error
    from email.message import Message

    import tools.environments.sprites as sprites_module
    from tools.environments.sprites import SpritesEnvironment

    env = SpritesEnvironment.__new__(SpritesEnvironment)
    env.cwd = "/brand"
    env.toolbox_url = "https://toolbox.example"
    env.bearer_token = "pair-secret"
    env.brand = "brand-123"
    env.timeout = 60
    attempts = 0

    def fake_open(request, timeout):
        nonlocal attempts
        attempts += 1
        body = json.dumps(
            {
                "error": {
                    "code": "pre_dispatch_failed",
                    "phase": "pre_dispatch",
                    "retryable": True,
                    "commandStarted": False,
                    "requestId": f"req-{attempts}",
                }
            }
        ).encode()
        raise urllib.error.HTTPError(
            request.full_url,
            503,
            "Service Unavailable",
            Message(),
            io.BytesIO(body),
        )

    monkeypatch.setattr(sprites_module._URL_OPENER, "open", fake_open)

    handle = env._run_bash("echo ok", timeout=1)

    assert handle.wait(timeout=1) == 1
    assert attempts == 1


def test_sprites_exec_should_stop_retrying_when_handle_is_killed(monkeypatch):
    import io
    import threading
    import urllib.error
    from email.message import Message

    import tools.environments.sprites as sprites_module
    from tools.environments.sprites import SpritesEnvironment

    env = SpritesEnvironment.__new__(SpritesEnvironment)
    env.cwd = "/brand"
    env.toolbox_url = "https://toolbox.example"
    env.bearer_token = "pair-secret"
    env.brand = "brand-123"
    env.timeout = 60
    attempts = 0
    first_attempt = threading.Event()

    def fake_open(request, timeout):
        nonlocal attempts
        attempts += 1
        first_attempt.set()
        body = json.dumps(
            {
                "error": {
                    "code": "pre_dispatch_failed",
                    "phase": "pre_dispatch",
                    "retryable": True,
                    "commandStarted": False,
                    "requestId": f"req-{attempts}",
                }
            }
        ).encode()
        raise urllib.error.HTTPError(
            request.full_url,
            503,
            "Service Unavailable",
            Message(),
            io.BytesIO(body),
        )

    monkeypatch.setattr(sprites_module._URL_OPENER, "open", fake_open)

    handle = env._run_bash("echo ok", timeout=30)
    assert first_attempt.wait(timeout=1)
    handle.kill()

    assert handle.wait(timeout=1) == 1
    assert attempts == 1


def test_sprites_exec_should_not_retry_unsafe_http_errors(monkeypatch):
    import io
    import urllib.error
    from email.message import Message

    import pytest

    import tools.environments.sprites as sprites_module
    from tools.environments.sprites import SpritesEnvironment, SpritesToolboxError

    cases = [
        (
            503,
            {
                "code": "already_started",
                "retryable": True,
                "commandStarted": True,
            },
        ),
        (
            503,
            {
                "code": "permanent_failure",
                "retryable": False,
                "commandStarted": False,
            },
        ),
        (
            503,
            {
                "code": "ambiguous_failure",
                "retryable": True,
            },
        ),
        (
            500,
            {
                "code": "wrong_status",
                "retryable": True,
                "commandStarted": False,
            },
        ),
    ]

    for status, error_payload in cases:
        env = SpritesEnvironment.__new__(SpritesEnvironment)
        env.toolbox_url = "https://toolbox.example"
        env.bearer_token = "pair-secret"
        env.brand = "brand-123"
        env.timeout = 60
        attempts = 0

        def fake_open(request, timeout):
            nonlocal attempts
            attempts += 1
            body = json.dumps({"error": error_payload}).encode()
            raise urllib.error.HTTPError(
                request.full_url,
                status,
                "HTTP error",
                Message(),
                io.BytesIO(body),
            )

        sleeps = []
        monkeypatch.setattr(sprites_module._URL_OPENER, "open", fake_open)
        monkeypatch.setattr(sprites_module.time, "sleep", sleeps.append)

        with pytest.raises(SpritesToolboxError):
            env._request_json(
                "/exec",
                {"command": "echo ok"},
                retry_exec_predispatch=True,
            )

        assert attempts == 1
        assert sleeps == []


def test_sprites_exec_should_not_retry_unparseable_503(monkeypatch):
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
    attempts = 0

    def fake_open(request, timeout):
        nonlocal attempts
        attempts += 1
        raise urllib.error.HTTPError(
            request.full_url,
            503,
            "Service Unavailable",
            Message(),
            io.BytesIO(b"not json"),
        )

    sleeps = []
    monkeypatch.setattr(sprites_module._URL_OPENER, "open", fake_open)
    monkeypatch.setattr(sprites_module.time, "sleep", sleeps.append)

    with pytest.raises(SpritesToolboxError):
        env._request_json(
            "/exec",
            {"command": "echo ok"},
            retry_exec_predispatch=True,
        )

    assert attempts == 1
    assert sleeps == []


def test_terminal_should_render_exhausted_sprites_infra_error(monkeypatch, caplog):
    from tools.environments.sprites import SpritesToolboxError

    error = SpritesToolboxError(
        "Toolbox API /exec failed with HTTP 503: executor unavailable",
        detail="executor unavailable",
        code="pre_dispatch_failed",
        phase="pre_dispatch",
        retryable=True,
        command_started=False,
        request_id="req-final",
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

    result = _run_terminal_with_existing_env(monkeypatch, env)

    assert env.calls == 1
    assert result["exit_code"] == -1
    assert result["error"] == (
        "terminal temporarily unavailable "
        "(infrastructure issue, code=pre_dispatch_failed); retry shortly"
    )
    assert "HTTP 503" not in result["error"]
    assert "Toolbox API /exec failed with HTTP 503" in caplog.text


def test_terminal_should_render_sprites_client_error_with_actual_cwd(monkeypatch, caplog):
    from tools.environments.sprites import SpritesToolboxError

    server_detail = (
        "path must be under an available toolbox workspace"
    )
    error = SpritesToolboxError(
        'Toolbox API /exec failed with HTTP 400: {"detail":"workspace rejected"}',
        detail=server_detail,
        http_status=400,
        request_cwd="/",
    )

    class FakeEnv:
        cwd = "/brand"

        def execute(self, command, **kwargs):
            raise error

    result = _run_terminal_with_existing_env(monkeypatch, FakeEnv())

    assert result["error"] == (
        "command not run: cwd '/' - "
        "path must be under an available toolbox workspace"
    )
    assert "HTTP 400" not in result["error"]
    assert '{"detail"' not in result["error"]
    assert 'HTTP 400: {"detail":"workspace rejected"}' in caplog.text


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


def test_terminal_should_render_unstructured_infra_error(monkeypatch):
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
    assert result["error"] == (
        "terminal temporarily unavailable (infrastructure issue); retry shortly"
    )
    assert "toolbox is still resuming" not in result["error"]


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


def test_toolbox_url_should_require_a_safe_origin_and_preserve_base_path():
    import pytest

    from tools.environments.sprites import _normalize_toolbox_url

    # A bare origin is preserved (gateway paired with a direct-forwarding proxy).
    assert _normalize_toolbox_url("https://toolbox.example/") == "https://toolbox.example"
    assert _normalize_toolbox_url("http://127.0.0.1:8643") == "http://127.0.0.1:8643"
    # A base path is preserved so the gateway can target the proxy's loopback
    # Toolbox forwarder; the trailing slash is stripped so appended endpoint
    # paths (e.g. /exec) never produce a doubled separator.
    assert (
        _normalize_toolbox_url("http://127.0.0.1:8642/internal/toolbox")
        == "http://127.0.0.1:8642/internal/toolbox"
    )
    assert (
        _normalize_toolbox_url("http://127.0.0.1:8642/internal/toolbox/")
        == "http://127.0.0.1:8642/internal/toolbox"
    )
    # Unsafe shapes are still rejected.
    with pytest.raises(ValueError, match="HTTPS outside loopback"):
        _normalize_toolbox_url("http://toolbox.example")
    with pytest.raises(ValueError, match="credentials, a query, or a fragment"):
        _normalize_toolbox_url("https://toolbox.example/base?foo=bar")
    with pytest.raises(ValueError, match="credentials, a query, or a fragment"):
        _normalize_toolbox_url("https://user:pass@toolbox.example/")


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


def test_sprites_file_operations_should_render_toolbox_errors(caplog):
    from tools.environments.sprites import (
        SpritesEnvironment,
        SpritesFileOperations,
        SpritesToolboxError,
    )

    class FakeEnv:
        cwd = "/brand"
        config = None

        def file_request(self, payload):
            raise SpritesToolboxError(
                'Toolbox API /files failed with HTTP 400: {"detail":"bad path"}',
                detail="path is outside the available workspace",
                http_status=400,
            )

    ops = SpritesFileOperations(cast(SpritesEnvironment, FakeEnv()))

    result = ops.read_file("/outside/example.txt")

    assert result.error == (
        "file operation not run: path '/outside/example.txt' - "
        "path is outside the available workspace"
    )
    assert "HTTP 400" not in result.error
    assert 'HTTP 400: {"detail":"bad path"}' in caplog.text


def test_sprites_file_operations_expand_tilde_with_sprite_home():
    from tools.environments.sprites import SpritesEnvironment, SpritesFileOperations

    class FakeEnv:
        cwd = "/brand"
        config = None

        def __init__(self):
            self.requests = []

        def execute(self, command, cwd=None, **kwargs):
            assert command == "echo $HOME"
            return {"output": "/home/oai/share\n", "returncode": 0}

        def file_request(self, payload):
            self.requests.append(payload)
            return {"content": "brief", "totalLines": 1, "fileSize": 5}

    env = FakeEnv()
    ops = SpritesFileOperations(cast(SpritesEnvironment, env))

    result = ops.read_file("~/brand/brief.md")

    assert result.error is None
    assert env.requests == [
        {
            "operation": "read",
            "path": "/home/oai/share/brand/brief.md",
            "offset": 1,
            "limit": 500,
        }
    ]


def test_sprites_environment_should_write_content_through_files_endpoint():
    from tools.environments.sprites import SpritesEnvironment

    env = SpritesEnvironment.__new__(SpritesEnvironment)
    requests = []

    def file_request(payload):
        requests.append(payload)
        return {"bytesWritten": len(payload["content"].encode("utf-8")), "dirsCreated": True}

    env.file_request = file_request

    assert env.write_file_content("/tmp/hermes-results/large.txt", "full result") is True
    assert requests == [
        {
            "operation": "write",
            "path": "/tmp/hermes-results/large.txt",
            "content": "full result",
            "encoding": "utf-8",
        }
    ]


def test_sprites_environment_should_reject_file_content_over_two_mib():
    import pytest

    import tools.environments.sprites as sprites_module
    from tools.environments.sprites import SpritesEnvironment, SpritesToolboxError

    env = SpritesEnvironment.__new__(SpritesEnvironment)
    requests = []
    env.file_request = requests.append
    content = "x" * (sprites_module._MAX_FILE_CONTENT_BYTES + 1)

    with pytest.raises(SpritesToolboxError, match="write content exceeded"):
        env.write_file_content("/tmp/hermes-results/too-large.txt", content)

    assert requests == []


def test_execute_code_guard_should_approve_sprites_backend():
    from tools.approval import check_execute_code_guard

    result = check_execute_code_guard("import os", "sprites")

    assert result["approved"] is True
