import base64
import os
import threading
from pathlib import Path

import pytest


@pytest.fixture
def skill_tree(monkeypatch, tmp_path):
    home = tmp_path / "active-profile"
    skills = home / "skills"
    skills.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return skills


def _projection_environment():
    """Use the real projection, manager and bulk encoder with in-memory I/O."""
    from tools.environments.file_sync import FileSyncManager, iter_sprites_sync_files
    from tools.environments.sprites import SpritesEnvironment

    env = SpritesEnvironment.__new__(SpritesEnvironment)
    env._cache_sync_lock = threading.Lock()
    remote = {}

    def file_request(payload):
        if payload["operation"] == "writeSkills":
            for entry in payload["files"]:
                remote[entry["path"]] = base64.b64decode(entry["contentBase64"])
        else:
            assert payload["operation"] == "deleteSkills"
            remote.pop(payload["path"], None)
        return {"ok": True}

    env.file_request = file_request
    env._sync_manager = FileSyncManager(
        get_files_fn=iter_sprites_sync_files,
        upload_fn=env._sprites_upload,
        bulk_upload_fn=env._sprites_bulk_upload,
        delete_fn=env._sprites_delete,
    )
    return env, remote


def test_iter_sprites_sync_files_should_include_only_skills(monkeypatch, tmp_path):
    from tools.environments.file_sync import iter_sprites_sync_files
    import tools.credential_files as credential_files

    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("name: demo\n", encoding="utf-8")

    monkeypatch.setattr(credential_files, "get_credential_file_mounts", lambda: [])
    monkeypatch.setattr(
        credential_files,
        "iter_skills_files",
        lambda container_base: [
            {
                "host_path": str(skill_file),
                "container_path": f"{container_base}/demo/SKILL.md",
            }
        ],
    )

    assert iter_sprites_sync_files("/skills") == [
        (str(skill_file), "/skills/demo/SKILL.md")
    ]


def test_iter_sprites_sync_files_should_refuse_credentials(monkeypatch, tmp_path):
    from tools.environments.file_sync import iter_sprites_sync_files
    import tools.credential_files as credential_files

    credential = tmp_path / "secret.json"
    credential.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        credential_files,
        "get_credential_file_mounts",
        lambda: [
            {
                "host_path": str(credential),
                "container_path": "/root/.hermes/credentials/secret.json",
            }
        ],
    )
    monkeypatch.setattr(credential_files, "iter_skills_files", lambda container_base: [])

    with pytest.raises(RuntimeError, match="refuses to sync credential files"):
        iter_sprites_sync_files("/skills")


def test_projection_omits_only_active_profile_root_usage_files(monkeypatch, skill_tree, tmp_path):
    from agent import skill_utils
    from tools.environments.file_sync import iter_sprites_sync_files

    internal = [".usage.json", ".usage.json.lock", ".usage_active.tmp"]
    helpers = [
        "SKILL.md", "root.py", ".hidden", ".usage_notes.md", ".usage_tmp.json",
        "demo/SKILL.md", "demo/.usage.json", "demo/.usage.json.lock",
        "demo/.usage_active.tmp", ".hidden-dir/helper.py", "demo/.hidden.py",
    ]
    for name in internal + helpers:
        path = skill_tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")

    external = tmp_path / "external-skills"
    external.mkdir()
    for name in internal:
        (external / name).write_text(name, encoding="utf-8")
    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", lambda: [external])

    projected = dict(iter_sprites_sync_files("/skills"))

    assert projected == {
        **{str(skill_tree / name): f"/skills/{name}" for name in helpers},
        **{str(external / name): f"/skills/external_skills/0/{name}" for name in internal},
    }
    assert all((skill_tree / name).is_file() for name in internal)


def test_projection_keeps_symlink_exclusion(skill_tree, tmp_path):
    from tools.environments.file_sync import iter_sprites_sync_files

    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = skill_tree / "escape.txt"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    assert iter_sprites_sync_files() == []


def test_usage_counters_and_lock_remain_on_agent_after_projection(skill_tree):
    from tools.skill_usage import bump_view, load_usage

    helper = skill_tree / "demo.py"
    helper.write_text("print('helper')", encoding="utf-8")
    bump_view("demo")
    env, remote = _projection_environment()

    env.sync_skill_files()

    assert load_usage()["demo"]["view_count"] == 1
    assert (skill_tree / ".usage.json").is_file()
    assert (skill_tree / ".usage.json.lock").is_file()
    assert remote == {"/skills/demo.py": b"print('helper')"}


@pytest.mark.parametrize("rename_at", ["before_lstat", "before_read", "after_upload"])
def test_usage_rename_never_breaks_command_preflight(monkeypatch, skill_tree, rename_at):
    from tools import skill_usage
    from tools.environments import file_sync
    from tools.environments.base import BaseEnvironment

    helper = skill_tree / "helper.py"
    helper.write_text("print('helper')", encoding="utf-8")
    env, remote = _projection_environment()
    renamed = []
    replace = os.replace
    read_bytes = Path.read_bytes
    mtime_key = file_sync._file_mtime_key
    file_request = env.file_request

    def before_replace(source, target):
        pending = Path(source)
        assert pending.is_file()

        def commit():
            if not renamed:
                replace(source, target)
                renamed.append(True)

        # The helper fallback makes the interleaving happen on both versions:
        # the fixed projection omits the temporary path before selection.
        get_files = env._sync_manager._get_files_fn
        trigger = helper

        def files_for_sync():
            nonlocal trigger
            files = get_files()
            if any(Path(source) == pending for source, _ in files):
                trigger = pending
            return files

        def after_key(path):
            result = mtime_key(path)
            if rename_at == "before_lstat" and Path(path) == trigger:
                commit()
            return result

        def before_read(path):
            if rename_at == "before_read" and path == trigger:
                commit()
            return read_bytes(path)

        def after_upload(payload):
            result = file_request(payload)
            if rename_at == "after_upload":
                commit()
            return result

        with monkeypatch.context() as scheduling:
            scheduling.setattr(file_sync, "_file_mtime_key", after_key)
            scheduling.setattr(Path, "read_bytes", before_read)
            scheduling.setattr(env, "file_request", after_upload)
            scheduling.setattr(env._sync_manager, "_get_files_fn", files_for_sync)
            try:
                BaseEnvironment.execute(env, "echo unrelated")
            except Prepared:
                pass
        commit()

    # Prove execute reaches command preparation only after its real preflight.
    dispatched = []

    class Prepared(Exception):
        pass

    def prepare(command):
        dispatched.append(command)
        raise Prepared

    env._prepare_command = prepare
    env.sync_cache_files = lambda **kwargs: None
    monkeypatch.setattr(skill_usage.os, "replace", before_replace)

    # save_usage catches a regression's sync error; observable assertions below
    # distinguish that failure from reaching the command-preparation sentinel.
    skill_usage.save_usage({"demo": {"view_count": 1}})

    assert renamed
    assert dispatched == ["echo unrelated"]
    assert remote == {"/skills/helper.py": b"print('helper')"}
    assert skill_usage.load_usage()["demo"]["view_count"] == 1


@pytest.mark.parametrize("failure", [FileNotFoundError("missing helper"), OSError("transfer failed")])
def test_genuine_helper_failures_still_block_command_preflight(monkeypatch, skill_tree, failure):
    from tools.environments.base import BaseEnvironment

    helper = skill_tree / "helper.py"
    helper.write_text("print('helper')", encoding="utf-8")
    env, remote = _projection_environment()
    dispatched = []
    env._prepare_command = lambda command: dispatched.append(command)

    if isinstance(failure, FileNotFoundError):
        read_bytes = Path.read_bytes

        def read_helper(path):
            if path == helper:
                raise failure
            return read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", read_helper)
    else:
        def fail_transfer(payload):
            raise failure

        env.file_request = fail_transfer

    with pytest.raises(type(failure), match=str(failure)):
        BaseEnvironment.execute(env, "echo unrelated")

    assert dispatched == []
    assert remote == {}
    assert env._sync_manager._synced_files == {}
