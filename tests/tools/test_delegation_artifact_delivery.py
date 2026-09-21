"""Harness cache artifacts (worker output, stdout recovery, stored pages,
screenshots) cross the HTTP client into a separate Toolbox filesystem."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from tools.credential_files import (
    _CACHE_DIRS, from_agent_visible_cache_path, to_agent_visible_cache_path,
)
from tools.delegate_tool import _spill_summary_to_file
from tools.delegation_live_log import create_live_transcripts
from tools.environments import file_sync
from tools.environments.file_sync import (
    FileSyncManager, SPRITES_CACHE_ROOT, SPRITES_DELEGATION_ROOT, iter_sprites_cache_files,
)
from tools.environments.sprites import SpritesEnvironment, SpritesFileOperations, SpritesToolboxError
from tools.code_execution_tool import _truncate_stdout_text


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
                    if len(content.encode()) > 2 * 1024 * 1024:
                        result = {"error": f"File too large: {payload['path']}"}
                    else:
                        result = {"content": content, "totalLines": len(content.splitlines())}
                except FileNotFoundError:
                    result = {"error": "file not found"}
            self.reply(result)

        def do_GET(self):
            query = parse_qs(urlsplit(self.path).query)
            assert self.headers["X-Omnio-Brand"] == "brand-a"
            assert self.headers["Authorization"] == "Bearer test-token"
            path = toolbox / query["path"][0].lstrip("/")
            requests.append({"operation": "raw-read", "path": query["path"][0]})
            if not path.is_file():
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(path.read_bytes())

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
    env._cache_sync_lock = threading.Lock()
    env._cache_sync_manager = FileSyncManager(
        iter_sprites_cache_files, env._upload_cache_file, env._delete_cache_files,
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
    # Files at the cache ROOT (model metadata, encrypted secret caches) are
    # not part of the projected set: only the listed subdirectories cross.
    (home / "cache" / "other.json").write_text("private")
    (home / "cache" / "bws_cache.enc.json").write_text("encrypted")
    env.sync_cache_files()
    assert [item["path"] for item in requests] == [path]
    assert to_agent_visible_cache_path(str(secret)) == str(secret)
    assert to_agent_visible_cache_path(str(cache / ".." / "other.json")) == str(cache / ".." / "other.json")


def test_every_cache_subdirectory_maps_to_the_toolbox_and_back(pair):
    home, toolbox, env, requests = pair
    for subpath, _old in _CACHE_DIRS:
        host = home / subpath / "nested" / "artifact.bin"
        host.parent.mkdir(parents=True, exist_ok=True)
        host.write_bytes(b"\x00\x01binary\xff")
        agent = to_agent_visible_cache_path(str(host))
        assert agent == f"{SPRITES_CACHE_ROOT}/{subpath.removeprefix('cache/')}/nested/artifact.bin"
        assert from_agent_visible_cache_path(agent) == str(host)
    env.sync_cache_files()
    landed = sorted(item["path"] for item in requests)
    assert landed == sorted(
        f"{SPRITES_CACHE_ROOT}/{subpath.removeprefix('cache/')}/nested/artifact.bin"
        for subpath, _old in _CACHE_DIRS
    )
    for subpath, _old in _CACHE_DIRS:
        copy = toolbox / f"{SPRITES_CACHE_ROOT}/{subpath.removeprefix('cache/')}/nested/artifact.bin".lstrip("/")
        assert copy.read_bytes() == b"\x00\x01binary\xff"


def test_symlinked_profile_home_still_maps(tmp_path, monkeypatch):
    real = tmp_path / "real-home"
    (real / "cache" / "web").mkdir(parents=True)
    link = tmp_path / "linked-home"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(link))
    monkeypatch.setenv("TERMINAL_ENV", "sprites")
    page = link / "cache" / "web" / "page.md"
    page.write_text("stored")
    assert to_agent_visible_cache_path(str(page)) == f"{SPRITES_CACHE_ROOT}/web/page.md"
    assert to_agent_visible_cache_path(str(page.resolve())) == f"{SPRITES_CACHE_ROOT}/web/page.md"


def test_paths_are_not_translated_on_other_backends(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "cache" / "web").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    page = home / "cache" / "web" / "page.md"
    page.write_text("stored")
    for backend in ("local", "ssh", "modal"):
        monkeypatch.setenv("TERMINAL_ENV", backend)
        assert to_agent_visible_cache_path(str(page)) == str(page)
        assert from_agent_visible_cache_path(f"{SPRITES_CACHE_ROOT}/web/page.md") == f"{SPRITES_CACHE_ROOT}/web/page.md"


def test_oversized_cache_files_stay_on_the_harness(pair, monkeypatch):
    home, toolbox, env, requests = pair
    monkeypatch.setattr(file_sync, "SPRITES_CACHE_FILE_MAX_BYTES", 1024)
    videos = home / "cache" / "videos"
    videos.mkdir(parents=True)
    (videos / "small.mp4").write_bytes(b"v" * 512)
    (videos / "huge.mp4").write_bytes(b"v" * 4096)
    assert [remote for _host, remote in iter_sprites_cache_files()] == [f"{SPRITES_CACHE_ROOT}/videos/small.mp4"]
    env.sync_cache_files()
    assert [item["path"] for item in requests] == [f"{SPRITES_CACHE_ROOT}/videos/small.mp4"]


def test_text_artifacts_are_redacted_and_media_is_byte_exact(pair):
    home, toolbox, env, requests = pair
    secret = "ghp_" + "b" * 36
    web = home / "cache" / "web"
    web.mkdir(parents=True)
    (web / "page.md").write_text(f"title\ntoken {secret}\n")
    shots = home / "cache" / "screenshots"
    shots.mkdir(parents=True)
    png = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) + secret.encode()
    (shots / "shot.png").write_bytes(png)
    env.sync_cache_files()
    stored_page = (toolbox / f"{SPRITES_CACHE_ROOT}/web/page.md".lstrip("/")).read_text()
    assert secret not in stored_page and "title" in stored_page
    assert (toolbox / f"{SPRITES_CACHE_ROOT}/screenshots/shot.png".lstrip("/")).read_bytes() == png


def test_raw_reads_under_the_cache_refresh_the_projection_first(pair):
    home, toolbox, env, requests = pair
    shots = home / "cache" / "screenshots"
    shots.mkdir(parents=True)
    (shots / "shot.png").write_bytes(b"\x89PNG first")
    agent_path = to_agent_visible_cache_path(str(shots / "shot.png"))
    assert env.read_file_bytes(agent_path, max_bytes=64) == b"\x89PNG first"
    (shots / "shot.png").write_bytes(b"\x89PNG second, longer")
    assert env.read_file_bytes(agent_path, max_bytes=64) == b"\x89PNG second, longer"
    # A read outside the projected cache does not trigger a sync round.
    before = len(requests)
    (toolbox / "brand").mkdir()
    (toolbox / "brand" / "notes.txt").write_text("brand")
    assert env.read_file_bytes("/brand/notes.txt", max_bytes=64) == b"brand"
    assert [item["operation"] for item in requests[before:]] == ["raw-read"]


def test_web_and_browser_footers_name_the_toolbox_path(pair):
    home, toolbox, env, requests = pair
    from tools.web_tools import _truncate_with_footer
    from tools.browser_tool import _truncate_snapshot

    page, truncated = _truncate_with_footer("line\n" * 5000, "https://example.com/doc", 2000)
    assert truncated
    assert f"{SPRITES_CACHE_ROOT}/web/" in page
    assert str(home) not in page
    snapshot = _truncate_snapshot("- element\n" * 5000, max_chars=2000)
    assert f"{SPRITES_CACHE_ROOT}/web/browser-snapshot-" in snapshot
    assert str(home) not in snapshot


def test_large_artifact_uses_raw_upload_without_truncation(pair):
    home, toolbox, env, requests = pair
    expected = "result line\n" * 210_000 + "FINAL RECORD"
    path = _spill_summary_to_file(0, expected)
    env.sync_cache_files()
    assert (toolbox / path.lstrip("/")).read_text() == expected
    assert requests[0]["operation"] == "raw-write"
    assert SpritesFileOperations(env).read_file_raw(path).content == expected


def test_stdout_recovery_is_readable_through_toolbox_file_tools(pair):
    home, toolbox, env, requests = pair
    secret = "ghp_" + "a" * 36
    expected = "before\n" * 200_000 + "MIDDLE_RECORD\n" + secret + "\nafter\n" * 200_000
    output, metadata = _truncate_stdout_text(expected)
    assert "MIDDLE_RECORD" not in output
    path = metadata["stdout_spill_path"]
    assert path.startswith("/tmp/.omnio-session/cache/exec/")
    # Host-side canonical copy, projected to the Toolbox before the first read.
    assert len(list(home.glob("cache/exec/*"))) == 1
    assert not requests
    result = SpritesFileOperations(env).read_file_raw(path)
    assert "MIDDLE_RECORD" in result.content
    assert secret not in result.content
    assert result.content.startswith("before\n") and result.content.endswith("after\n")
    assert (toolbox / path.lstrip("/")).read_text() == result.content
    assert requests[0]["operation"] == "raw-write"
    assert metadata["stdout_spill_truncated"] is False
    page = SpritesFileOperations(env).read_file(path, offset=200_001, limit=1)
    assert page.error is None
    assert "MIDDLE_RECORD" in page.content
    assert "before" not in page.content and "after" not in page.content
    assert page.truncated is True
    assert "offset=200002" in page.hint


def test_stdout_artifact_read_is_bounded(pair):
    home, toolbox, env, requests = pair
    path = "/tmp/.omnio-session/cache/exec/too-large.txt"
    local = toolbox / path.lstrip("/")
    local.parent.mkdir(parents=True)
    local.write_bytes(b"x" * (5 * 1024 * 1024 + 1))
    result = SpritesFileOperations(env).read_file_raw(path)
    assert result.error and "exceeds" in result.error and "offset/limit" in result.error
    assert not result.content
    paged = SpritesFileOperations(env).read_file(path, offset=1, limit=10)
    assert paged.error and "exceeds" in paged.error


def test_failed_stdout_transfer_surfaces_as_a_read_error_not_a_host_path(pair, monkeypatch):
    home, toolbox, env, requests = pair

    def failed(*args):
        raise SpritesToolboxError("Artifact transfer failed")

    monkeypatch.setattr(env, "_write_raw_artifact", failed)
    output, metadata = _truncate_stdout_text("before\n" * 15000 + "END")
    assert output.endswith("END")
    path = metadata["stdout_spill_path"]
    assert path.startswith("/tmp/.omnio-session/cache/exec/")
    assert str(home) not in path
    result = SpritesFileOperations(env).read_file_raw(path)
    assert result.error and "temporarily unavailable" in result.error
    assert not requests


def test_invalid_artifact_destination_is_rejected(pair):
    home, toolbox, env, requests = pair
    with pytest.raises(SpritesToolboxError, match="outside the Toolbox cache"):
        env._upload_cache_file(str(home / "auth.json"), SPRITES_DELEGATION_ROOT + "/../../../auth.json")
    assert not requests


def test_failed_transfer_is_reported_and_retried(pair, monkeypatch):
    home, toolbox, env, requests = pair
    path = _spill_summary_to_file(0, "complete")
    original = env._write_raw_artifact

    def failed(*args):
        raise SpritesToolboxError("Artifact transfer failed")

    monkeypatch.setattr(env, "_write_raw_artifact", failed)
    with pytest.raises(SpritesToolboxError, match="transfer failed"):
        env.sync_cache_files()
    monkeypatch.setattr(env, "_write_raw_artifact", original)
    env.sync_cache_files()
    assert (toolbox / path.lstrip("/")).read_text() == "complete"
