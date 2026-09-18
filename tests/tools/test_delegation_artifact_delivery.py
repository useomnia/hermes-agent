"""Worker artifacts cross the real HTTP client into a separate filesystem."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from tools.credential_files import to_agent_visible_cache_path
from tools.delegate_tool import _spill_summary_to_file
from tools.delegation_live_log import create_live_transcripts
from tools.environments.file_sync import (
    FileSyncManager, SPRITES_DELEGATION_ROOT, iter_sprites_delegation_files,
)
from tools.environments.sprites import SpritesEnvironment, SpritesFileOperations, SpritesToolboxError


@pytest.fixture
def pair(tmp_path, monkeypatch):
    home = tmp_path / "harness" / "brand-a"
    home.mkdir(parents=True)
    toolbox = tmp_path / "toolbox"
    toolbox.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_ENV", "sprites")
    requests = []

    class Files(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            path = toolbox / payload["path"].lstrip("/")
            operation = payload["operation"]
            if operation == "write":
                assert len(payload["content"].encode()) <= 2 * 1024 * 1024
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(payload["content"])
                result = {"bytesWritten": len(payload["content"].encode())}
            elif operation == "delete":
                path.unlink(missing_ok=True)
                result = {"ok": True}
            else:
                try:
                    content = path.read_text()
                    result = {"content": content, "totalLines": len(content.splitlines())}
                except FileNotFoundError:
                    result = {"error": "file not found"}
            self.reply(result)

        def do_PUT(self):
            query = parse_qs(urlsplit(self.path).query)
            assert query["overwrite"] == ["true"]
            data = self.rfile.read(int(self.headers["Content-Length"]))
            path = toolbox / query["path"][0].lstrip("/")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            requests.append({"operation": "raw-write", "path": query["path"][0]})
            self.reply({"bytesWritten": len(data)})

        def reply(self, result):
            assert self.headers["X-Omnio-Brand"] == "brand-a"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Files)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = SpritesEnvironment.__new__(SpritesEnvironment)
    env.toolbox_url = f"http://127.0.0.1:{server.server_port}"
    env.bearer_token = "test-token"
    env.brand = "brand-a"
    env.timeout = 5
    env.cwd = "/brand"
    env._delegation_sync_lock = threading.Lock()
    env._delegation_sync_manager = FileSyncManager(
        iter_sprites_delegation_files, env._upload_delegation_artifact,
        env._delete_delegation_artifacts,
    )
    yield home, toolbox, env, requests
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_summary_path_can_be_read_completely_from_toolbox(pair):
    home, toolbox, env, requests = pair
    expected = "worker instruction\n" * 3000 + "FINAL RESULT"
    path = _spill_summary_to_file(0, expected)
    assert path.startswith("/tmp/.omnio-session/cache/delegation/")
    result = SpritesFileOperations(env).read_file_raw(path)
    assert result.content == expected
    assert (toolbox / path.lstrip("/")).read_text() == expected


def test_live_log_refreshes_before_each_parent_read(pair):
    home, toolbox, env, requests = pair
    _, writers, paths = create_live_transcripts([{"goal": "write"}])
    assert paths[0].startswith("/tmp/.omnio-session/cache/delegation/")
    writers[0].assistant_text("FIRST OBSERVATION")
    assert "FIRST OBSERVATION" in SpritesFileOperations(env).read_file_raw(paths[0]).content
    writers[0].assistant_text("FINAL OBSERVATION")
    assert "FINAL OBSERVATION" in SpritesFileOperations(env).read_file_raw(paths[0]).content


def test_profile_credentials_other_caches_and_symlinks_are_not_transferred(pair):
    home, toolbox, env, requests = pair
    path = _spill_summary_to_file(0, "legitimate")
    secret = home / "auth.json"
    secret.write_text('{"credential":"never copy"}')
    cache = home / "cache" / "delegation"
    (cache / "secret.json").symlink_to(secret)
    (cache / "outside").symlink_to(home, target_is_directory=True)
    (home / "cache" / "other.json").write_text("private")
    env.sync_delegation_artifacts()
    assert [item["path"] for item in requests] == [path]
    assert to_agent_visible_cache_path(str(secret)) == str(secret)
    assert to_agent_visible_cache_path(str(cache / ".." / "other.json")) == str(cache / ".." / "other.json")


def test_large_artifact_uses_raw_upload_without_truncation(pair):
    home, toolbox, env, requests = pair
    expected = "result line\n" * 210_000 + "FINAL RECORD"
    path = _spill_summary_to_file(0, expected)
    env.sync_delegation_artifacts()
    assert (toolbox / path.lstrip("/")).read_text() == expected
    assert requests[0]["operation"] == "raw-write"


def test_invalid_artifact_destination_is_rejected(pair):
    home, toolbox, env, requests = pair
    with pytest.raises(SpritesToolboxError, match="outside delegation"):
        env._upload_delegation_artifact(str(home / "auth.json"), SPRITES_DELEGATION_ROOT + "/../../auth.json")
    assert not requests


def test_failed_transfer_is_reported_and_retried(pair, monkeypatch):
    home, toolbox, env, requests = pair
    path = _spill_summary_to_file(0, "complete")
    original = env.write_file_content
    monkeypatch.setattr(env, "write_file_content", lambda *args: False)
    with pytest.raises(SpritesToolboxError, match="transfer failed"):
        env.sync_delegation_artifacts()
    monkeypatch.setattr(env, "write_file_content", original)
    env.sync_delegation_artifacts()
    assert (toolbox / path.lstrip("/")).read_text() == "complete"
