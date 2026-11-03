"""Shared public identities and complete user-directed package copies."""

import json
import os
from pathlib import Path

import pytest

from agent import prompt_builder, skill_commands, skill_utils
from tools import skill_manager_tool as manager, skill_package, skill_usage, skills_tool


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    root = tmp_path / "profile" / "skills"
    root.mkdir(parents=True)
    (root / ".bundled_manifest").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root.parent))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", root)
    monkeypatch.setattr(manager, "SKILLS_DIR", root)
    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", lambda: [])
    prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=True)
    skills_tool._SKILLS_CACHE.clear()
    yield root
    prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=True)
    skill_commands._skill_commands = {}
    skills_tool._SKILLS_CACHE.clear()


def main(name="helper", body="Use this procedure.\n", extra=""):
    return f"---\nname: {name}\ndescription: Follow this procedure.\n{extra}---\n# Instructions\n{body}"


def package(root, relative, name="helper", body="Use this procedure.\n", extra=""):
    directory = root / relative
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(main(name, body, extra), encoding="utf-8")
    return directory


def view(name, **kwargs):
    return json.loads(skills_tool.skill_view(name, preprocess=False, **kwargs))


def offered():
    return json.loads(skills_tool.skills_list())["skills"]


def test_all_offers_load_exact_entries_and_writes_share_identity(catalog):
    one = package(catalog, "ops/first", body="FIRST")
    two = package(catalog, "custom/second", body="SECOND")
    assert not view("helper")["success"]
    names = {entry["name"] for entry in offered()}
    assert names == {"ops/first", "custom/second"}
    for name, expected in (("ops/first", one), ("custom/second", two)):
        loaded = view(name)
        assert loaded["success"] and loaded["content"] == (expected / "SKILL.md").read_text(encoding="utf-8")
        target = manager.skill_management_target(name)
        assert target["success"] and target["editable"]
        assert target["skill_dir"] == str(expected)
    assert manager.skill_management_target("helper")["code"] == "ambiguous"
    assert not json.loads(manager.skill_manage("edit", "helper", content=main(body="WRONG")))["success"]
    assert "FIRST" in (one / "SKILL.md").read_text(encoding="utf-8")
    assert "SECOND" in (two / "SKILL.md").read_text(encoding="utf-8")
    index = prompt_builder.build_skills_system_prompt()
    assert "ops/first" in index and "custom/second" in index
    commands = skill_commands.scan_skill_commands()
    assert {entry["name"] for entry in commands.values()} == names
    assert json.loads(manager.skill_manage("edit", "custom/second", content=main("changed", "EDIT")))["success"]
    assert "EDIT" in (two / "SKILL.md").read_text(encoding="utf-8")


def test_declared_name_reads_and_writes_and_creation_alias_collision(catalog):
    directory = package(catalog, "ops/directory", name="declared")
    assert manager._find_skill("declared")["path"] == directory
    assert json.loads(manager.skill_manage("patch", "declared", old_string="Use this procedure.", new_string="UPDATED"))["success"]
    assert "UPDATED" in view("declared")["content"]
    assert not json.loads(manager.skill_manage("create", "declared", content=main("declared")))["success"]
    assert not (catalog / "declared").exists()


@pytest.mark.parametrize("lookup", ["reference", "owner/reference", "owner:reference"])
def test_package_owned_markdown_never_becomes_a_skill(catalog, lookup):
    directory = package(catalog, "owner", name="owner")
    (directory / "reference.md").write_text("Owned support", encoding="utf-8")
    assert not view(lookup)["success"]
    assert view("owner", file_path="reference.md")["content"] == "Owned support"
    assert {entry["name"] for entry in offered()} == {"owner"}


def test_root_doc_does_not_collide_with_real_skill_and_legacy_still_reads(catalog):
    owner = package(catalog, "owner", name="owner")
    (owner / "helper.md").write_text("Owned support", encoding="utf-8")
    real = package(catalog, "ops/real")
    assert view("helper")["content"] == (real / "SKILL.md").read_text(encoding="utf-8")
    (catalog / "legacy.md").write_text("Legacy body", encoding="utf-8")
    assert view("legacy")["content"] == "Legacy body"


def test_cross_root_same_path_is_omitted_not_shadowed(catalog, tmp_path, monkeypatch):
    external = tmp_path / "external"
    package(catalog, "same")
    package(external, "same", body="EXTERNAL")
    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", lambda: [external])
    assert offered() == []
    assert not view("helper")["success"]
    assert not view("same")["success"]
    assert prompt_builder.build_skills_system_prompt() == ""
    assert skill_commands.scan_skill_commands() == {}
    assert manager.skill_management_target("same")["code"] == "ambiguous"


def test_cross_root_distinct_paths_get_loadable_names(catalog, tmp_path, monkeypatch):
    external = tmp_path / "external"
    package(catalog, "local/one", body="LOCAL")
    package(external, "external/two", body="EXTERNAL")
    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", lambda: [external])
    assert {entry["name"] for entry in offered()} == {"local/one", "external/two"}
    assert "LOCAL" in view("local/one")["content"]
    assert "EXTERNAL" in view("external/two")["content"]
    assert not manager.skill_management_target("external/two")["editable"]


def test_cold_snapshot_warm_and_compact_keep_exact_names(catalog):
    package(catalog, "ops/one")
    package(catalog, "other/two", body="SECOND")
    first = prompt_builder.build_skills_system_prompt(compact_categories={"ops"})
    assert "ops/one" in first and "other/two" in first
    prompt_builder.clear_skills_system_prompt_cache()
    second = prompt_builder.build_skills_system_prompt(compact_categories={"ops"})
    third = prompt_builder.build_skills_system_prompt(compact_categories={"ops"})
    assert first == second == third


def test_nonowned_and_unknown_provenance_fail_closed(catalog):
    directory = package(catalog, "ops/short", name="declared")
    (catalog / ".bundled_manifest").write_text("short:hash\n", encoding="utf-8")
    assert not manager.skill_management_target("declared")["editable"]
    (catalog / ".bundled_manifest").unlink()
    assert manager.skill_management_target("declared")["provenance"] == "unknown"
    assert not manager.skill_management_target("declared")["editable"]
    (catalog / ".bundled_manifest").write_text("", encoding="utf-8")
    assert manager.skill_management_target("declared")["editable"]
    hub = catalog / ".hub"
    hub.mkdir()
    (hub / "lock.json").write_text(json.dumps({"installed": {"short": {"install_path": "ops/short"}}}), encoding="utf-8")
    assert not manager.skill_management_target("declared")["editable"]
    assert directory.exists()


def test_local_copy_preserves_binary_body_author_and_new_identity(catalog):
    source = package(catalog, "shipped/helper", body="Exact body.\r\n", extra="metadata:\n  id: old-id\n  author: Original\n")
    binary = b"\x00\xff\x80\r\n"
    (source / "image.bin").write_bytes(binary)
    (source / ".hidden").write_bytes(b"hidden")
    before = {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    result = manager.copy_skill_package("helper", target_name="custom-copy")
    assert result["success"], result
    assert result["file_count"] == 3
    destination = catalog / "custom-copy"
    assert (destination / "image.bin").read_bytes() == binary
    assert (destination / ".hidden").read_bytes() == b"hidden"
    metadata, body = skill_utils.parse_frontmatter((destination / "SKILL.md").read_bytes().decode())
    _, original_body = skill_utils.parse_frontmatter(before["SKILL.md"].decode())
    assert body == original_body
    assert metadata["name"] == "custom-copy"
    assert metadata["metadata"] == {"author": "Original"}
    assert {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob("*") if p.is_file()} == before
    assert view("custom-copy")["success"]
    assert manager.skill_management_target("custom-copy")["editable"]
    assert not skill_usage.is_curator_managed("custom-copy")
    assert not list((catalog / ".hub").glob("copy-*"))


def test_remote_complete_bundle_uses_same_custom_path_and_requested_content(catalog):
    files = {"SKILL.md": main(extra="author: Source\nmetadata:\n  id: repo-id\n").encode(), "scripts/run.bin": b"\xff\x00"}
    result = manager.copy_skill_package(target_name="repo-copy", files=files, content=main(body="USER EDIT"))
    assert result["success"], result
    metadata, body = skill_utils.parse_frontmatter((catalog / "repo-copy/SKILL.md").read_text(encoding="utf-8"))
    assert metadata["author"] == "Source" and metadata["name"] == "repo-copy"
    assert "USER EDIT" in body
    assert (catalog / "repo-copy/scripts/run.bin").read_bytes() == b"\xff\x00"


@pytest.mark.parametrize("bad", ["../escape", "/escape", "a/../escape", "a//b", "a\\b", "C:/escape", "x\x00y"])
def test_remote_package_rejects_unsafe_paths_before_publication(catalog, bad):
    result = manager.copy_skill_package(target_name="new-copy", files={"SKILL.md": main().encode(), bad: b"x"})
    assert not result["success"] and result["code"] == "unsafe_package"
    assert not (catalog / "new-copy").exists()


@pytest.mark.parametrize("kind", ["symlink-file", "symlink-dir", "fifo", "source-link"])
def test_local_package_rejects_links_and_special_files(catalog, kind):
    source = package(catalog, "source")
    if kind == "source-link":
        (catalog / "source-alias").symlink_to(source, target_is_directory=True)
        name = "source-alias"
    elif kind == "symlink-dir":
        (source / "linked").symlink_to(catalog, target_is_directory=True)
        name = "source"
    elif kind == "symlink-file":
        (source / "linked").symlink_to(source / "SKILL.md")
        name = "source"
    else:
        os.mkfifo(source / "fifo")
        name = "source"
    result = manager.copy_skill_package(name, target_name="new-copy")
    assert not result["success"], result
    assert not (catalog / "new-copy").exists()


@pytest.mark.parametrize("limit", ["MAX_PACKAGE_FILES", "MAX_PACKAGE_BYTES", "MAX_PACKAGE_FILE_BYTES", "MAX_PACKAGE_ENTRIES"])
def test_package_limits_fail_without_partial_skill(catalog, monkeypatch, limit):
    source = package(catalog, "source")
    (source / "a.bin").write_bytes(b"binary")
    monkeypatch.setattr(skill_package, limit, 1)
    result = manager.copy_skill_package("source", target_name="new-copy")
    assert not result["success"]
    assert not (catalog / "new-copy").exists()


def test_no_overwrite_including_empty_destination_race(catalog, monkeypatch):
    package(catalog, "source")
    real_publish = skill_package.publish_package
    def race(stage, destination):
        assert {entry["name"] for entry in offered()} == {"helper"}
        destination.mkdir()
        real_publish(stage, destination)
    monkeypatch.setattr(skill_package, "publish_package", race)
    result = manager.copy_skill_package("source", target_name="new-copy")
    assert result["code"] == "target_exists"
    assert (catalog / "new-copy").is_dir() and list((catalog / "new-copy").iterdir()) == []
    assert not list((catalog / ".hub").glob("copy-*"))


def test_copy_refuses_alias_and_existing_source_identifier(catalog):
    package(catalog, "ops/source")
    assert manager.copy_skill_package("helper", target_name="helper")["code"] == "target_exists"
    assert manager.copy_skill_package("helper", target_name="source")["code"] == "target_exists"
    assert not (catalog / "helper").exists()


@pytest.mark.parametrize("gate", ["approval", "background", "disabled"])
def test_copy_preserves_write_and_disabled_gates(catalog, monkeypatch, gate):
    from tools import skill_provenance, write_approval
    package(catalog, "source")
    if gate == "approval":
        monkeypatch.setattr(write_approval, "evaluate_gate", lambda *a, **k: write_approval.GateDecision(allow=False, message="Approval required"))
    elif gate == "background":
        monkeypatch.setattr(skill_provenance, "is_background_review", lambda: True)
    else:
        monkeypatch.setattr(skills_tool, "_is_skill_disabled", lambda name: True)
    result = manager.copy_skill_package("source", target_name="new-copy")
    assert not result["success"] and result["code"] == "write_blocked"
    assert not (catalog / "new-copy").exists()
    assert not (catalog.parent / "pending").exists()


def test_failed_stage_or_security_scan_never_publishes(catalog, monkeypatch):
    package(catalog, "source")
    monkeypatch.setattr(manager, "_security_scan_skill", lambda _: "Blocked by scan")
    result = manager.copy_skill_package("source", target_name="new-copy")
    assert result["code"] == "security_scan"
    assert not (catalog / "new-copy").exists()
    assert not list((catalog / ".hub").glob("copy-*"))


def test_stage_write_failure_cleans_all_staged_files(catalog, monkeypatch):
    package(catalog, "source")
    actual_write = Path.write_bytes
    def fail_support(path, data):
        if path.name == "second.bin":
            raise OSError("Disk full")
        return actual_write(path, data)
    monkeypatch.setattr(Path, "write_bytes", fail_support)
    result = manager.copy_skill_package(target_name="new-copy", files={"SKILL.md": main().encode(), "second.bin": b"bytes"})
    assert result["code"] == "copy_failed"
    assert not (catalog / "new-copy").exists()
    assert not list((catalog / ".hub").glob("copy-*"))


@pytest.mark.parametrize("files", [{"other": b"missing-main"}, {"SKILL.md": b"\xff"}, {"SKILL.md": b"not frontmatter"}, {"SKILL.md": "text not bytes"}])
def test_incomplete_or_invalid_remote_package_never_publishes(catalog, files):
    assert not manager.copy_skill_package(target_name="new-copy", files=files)["success"]
    assert not (catalog / "new-copy").exists()


def test_wrong_or_ambiguous_source_never_copies(catalog):
    package(catalog, "one", body="FIRST")
    package(catalog, "two", body="SECOND")
    assert manager.copy_skill_package("helper", target_name="new-copy")["code"] == "ambiguous"
    assert manager.copy_skill_package("../one", target_name="new-copy")["code"] == "invalid_source"
    assert not (catalog / "new-copy").exists()


def test_slash_support_hint_retains_qualified_accepted_identifier(catalog):
    first = package(catalog, "ops/first", body="FIRST")
    package(catalog, "other/second", body="SECOND")
    (first / "root.md").write_text("Support", encoding="utf-8")
    loaded = view("ops/first")
    message = skill_commands._build_skill_message(loaded, first, "Invoke selected skill")
    assert 'skill_view(name="ops/first",' in message
    assert view(**loaded["file_example"])["content"] == "Support"


def test_redirected_local_target_is_not_reported_editable(catalog):
    directory = package(catalog, "original", name="declared")
    (catalog / "alias").symlink_to(directory, target_is_directory=True)
    target = manager.skill_management_target("alias")
    assert target["success"] and not target["editable"]


@pytest.mark.parametrize("name", [None, 7, "../bad", "a/b"])
def test_invalid_copy_name_is_reported_without_writing(catalog, name):
    result = manager.copy_skill_package(target_name=name, files={"SKILL.md": main().encode()})
    assert result["code"] == "invalid_name"
    assert list(catalog.iterdir()) == [catalog / ".bundled_manifest"]
