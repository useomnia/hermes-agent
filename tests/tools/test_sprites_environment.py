import json
from typing import cast


def test_get_env_config_should_default_sprites_to_brand_workspace(monkeypatch):
    from tools.terminal_tool import _get_env_config

    monkeypatch.setenv("TERMINAL_ENV", "sprites")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setenv("TERMINAL_SPRITES_URL", "https://runtime.example")
    monkeypatch.setenv("TERMINAL_SPRITES_BEARER", "pair-secret")
    monkeypatch.setenv("TERMINAL_SPRITES_BRAND", "brand-123")

    config = _get_env_config()

    assert config["env_type"] == "sprites"
    assert config["cwd"] == "/brand"
    assert config["sprites_url"] == "https://runtime.example"
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
            "sprites_url": "https://runtime.example",
            "sprites_bearer": "pair-secret",
            "sprites_brand": "brand-123",
        },
    )

    assert isinstance(env, FakeSpritesEnvironment)
    assert captured == {
        "runtime_url": "https://runtime.example",
        "bearer_token": "pair-secret",
        "brand": "brand-123",
        "cwd": "/brand",
        "timeout": 30,
    }


def test_sprites_environment_should_send_exec_to_runtime():
    from tools.environments.sprites import SpritesEnvironment

    env = SpritesEnvironment.__new__(SpritesEnvironment)
    env.cwd = "/brand"
    env.timeout = 60
    env.runtime_url = "https://runtime.example"
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


def test_sprites_environment_should_use_durable_runtime_session_dir():
    from tools.environments.sprites import SpritesEnvironment

    env = SpritesEnvironment.__new__(SpritesEnvironment)

    assert env.get_temp_dir() == "/scratch/.hermes-session"


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
    from tools.environments.sprites import SpritesEnvironment

    env = SpritesEnvironment.__new__(SpritesEnvironment)
    env.runtime_url = "https://runtime.example"
    env.bearer_token = "pair-secret"
    env.brand = "brand-123"
    env.timeout = 60

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"ok": True}).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["body"] = request.data.decode()
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    response = env._request_json("/health", {"ping": True}, timeout=5)

    assert response == {"ok": True}
    assert captured["url"] == "https://runtime.example/health"
    assert captured["timeout"] == 5
    assert captured["headers"]["Authorization"] == "Bearer pair-secret"
    assert captured["headers"]["X-omnio-brand"] == "brand-123"
    assert json.loads(captured["body"]) == {"ping": True}


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
