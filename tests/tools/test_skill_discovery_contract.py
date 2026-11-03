"""Exercise the public discovery contract through the real skill loader."""

import ast
import json
import os
from pathlib import Path

import pytest

from agent import prompt_builder, skill_utils
from tools import skill_inventory, skills_tool


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    root = home / "skills"
    root.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", root)
    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", lambda: [])
    monkeypatch.setattr(skill_utils, "get_all_skills_dirs", lambda: [root])
    prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=True)
    return root


def make_skill(root, relative="ops/helper", extra="", body="Follow these instructions."):
    directory = root / relative
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: helper\ndescription: Test discovery.\n{extra}---\n# Procedure\n{body}",
        encoding="utf-8",
    )
    return directory


def write_support(directory, relative, content=None):
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if content is not None else f"Contents of {relative}", encoding="utf-8")
    return path


def view(name="helper", **args):
    return json.loads(skills_tool.skill_view(name, preprocess=False, **args))


def both_inventories(name="helper"):
    loaded = view(name)
    missing = view(name, file_path="missing-file.md")
    assert loaded["success"] is True, loaded
    assert missing["success"] is False, missing
    assert (loaded["linked_files"] or {}) == missing["available_files"]
    for key in ("file_inventory", "file_example", "usage_hint"):
        assert loaded[key] == missing[key]
    return loaded


def test_root_and_nested_files_are_discoverable_before_a_failed_guess(catalog):
    directory = make_skill(catalog)
    expected = {
        "other": ["notes/details.yaml", "site-identity.md"],
        "references": ["references/deep/guide.csv"],
        "scripts": ["scripts/deep/run.js"],
        "templates": ["templates/config.txt"],
        "assets": ["assets/logo.bin"],
    }
    for paths in expected.values():
        for path in paths:
            write_support(directory, path)
    for path in ("ignore.txt", "notes/SKILL.md"):
        write_support(directory, path)
    loaded = both_inventories()
    assert loaded["linked_files"] == expected
    assert loaded["file_inventory"]["complete"] is True
    assert loaded["file_inventory"]["listed_files"] == 6
    for paths in expected.values():
        for path in paths:
            assert view(file_path=path)["content"] == (directory / path).read_text(encoding="utf-8")
    assert loaded["content"] == (directory / "SKILL.md").read_text(encoding="utf-8")


def test_a_file_named_like_a_support_directory_is_not_a_directory(catalog):
    directory = make_skill(catalog)
    write_support(directory, "references")
    assert both_inventories()["linked_files"] is None


def test_example_executes_with_the_accepted_collision_resolving_identifier(catalog):
    directory = make_skill(catalog, body="Selected instructions.")
    make_skill(catalog, "another/helper", body="Different instructions.")
    relative = "café.md" if os.name == "nt" else 'quote"-slash\\-café.md'
    write_support(directory, relative, "Selected root document.")
    assert view()["success"] is False
    loaded = both_inventories("ops/helper")
    example = loaded["file_example"]
    assert example == {"name": "ops/helper", "file_path": relative}
    call_text = loaded["usage_hint"].split("with ", 1)[1].split("). Use each", 1)[0] + ")"
    call = ast.parse(call_text, mode="eval").body
    assert isinstance(call, ast.Call)
    arguments = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
    assert arguments == example
    assert view(**arguments)["content"] == "Selected root document."


@pytest.mark.parametrize("name", ["connectors/workspace", "stale/helper", "missing", "../ops/helper"])
def test_discovery_never_admits_a_guessed_or_unsafe_identifier(catalog, name):
    make_skill(catalog)
    assert view(name)["success"] is False
    assert view()["success"] is True


@pytest.mark.parametrize("budget,limit", [("entries", "entries"), ("files", "files"), ("characters", "characters")])
def test_each_ceiling_has_matching_truthful_omission_on_both_surfaces(catalog, monkeypatch, budget, limit):
    directory = make_skill(catalog)
    for name in ("a.md", "b.md", "c.md"):
        write_support(directory, name)
    if budget == "entries":
        monkeypatch.setattr(skill_inventory, "MAX_INVENTORY_ENTRIES", 2)
    elif budget == "files":
        monkeypatch.setattr(skill_inventory, "MAX_INVENTORY_FILES", 1)
    else:
        monkeypatch.setattr(skill_inventory, "MAX_INVENTORY_CHARS", 25)
    loaded = both_inventories()
    metadata = loaded["file_inventory"]
    assert metadata["complete"] is False
    assert metadata["limit"] == limit
    paths = [path for group in (loaded["linked_files"] or {}).values() for path in group]
    assert metadata["listed_files"] == len(paths)
    assert metadata["examined_entries"] <= skill_inventory.MAX_INVENTORY_ENTRIES
    assert len(paths) <= skill_inventory.MAX_INVENTORY_FILES
    assert len(json.dumps(loaded["linked_files"] or {}, ensure_ascii=False)) <= skill_inventory.MAX_INVENTORY_CHARS
    if paths:
        assert view(**loaded["file_example"])["success"] is True
    else:
        assert loaded["file_example"] is None
        assert loaded["usage_hint"] is None


def test_a_complete_inventory_at_the_file_and_character_ceiling_is_not_reported_missing(catalog, monkeypatch):
    directory = make_skill(catalog)
    write_support(directory, "a.md")
    monkeypatch.setattr(skill_inventory, "MAX_INVENTORY_FILES", 1)
    monkeypatch.setattr(skill_inventory, "MAX_INVENTORY_CHARS", len(json.dumps({"other": ["a.md"]})))
    loaded = both_inventories()
    assert loaded["linked_files"] == {"other": ["a.md"]}
    assert loaded["file_inventory"]["complete"] is True


def test_an_empty_partial_directory_does_not_choose_files_from_scandir_order(catalog, monkeypatch):
    directory = make_skill(catalog)
    for name in ("a.md", "b.md", "c.md"):
        write_support(directory, name)
    monkeypatch.setattr(skill_inventory, "MAX_INVENTORY_ENTRIES", 2)
    original = skill_inventory.os.scandir
    first = both_inventories()

    class ReversedEntries:
        def __init__(self, entries):
            self.entries = iter(reversed(entries))

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.entries)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def close(self):
            pass

    def reversed_entries(path):
        if Path(path) != directory:
            return original(path)
        with original(path) as stream:
            return ReversedEntries(list(stream))

    monkeypatch.setattr(skill_inventory.os, "scandir", reversed_entries)
    second = both_inventories()
    assert first["linked_files"] is None
    assert second["linked_files"] == first["linked_files"]
    assert second["file_inventory"] == first["file_inventory"]


def test_unreadable_directory_is_partial_without_failing_the_main_skill(catalog, monkeypatch):
    directory = make_skill(catalog)
    write_support(directory, "root.md")
    original = skill_inventory.os.scandir

    def unreadable(path):
        if Path(path) == directory:
            raise PermissionError("Synthetic unreadable directory")
        return original(path)

    monkeypatch.setattr(skill_inventory.os, "scandir", unreadable)
    loaded = both_inventories("ops/helper")
    assert loaded["file_inventory"]["limit"] == "unreadable"
    assert loaded["file_inventory"]["complete"] is False
    assert loaded["file_example"] is None


def test_containment_excludes_escape_and_directory_links_but_keeps_contained_files(catalog, tmp_path):
    directory = make_skill(catalog)
    inside = write_support(directory, "inside.md")
    outside = write_support(tmp_path / "outside", "private.md", "Outside fixture.")
    try:
        (directory / "safe-link.md").symlink_to(inside)
        (directory / "escape.md").symlink_to(outside)
        (directory / "linked-directory").symlink_to(outside.parent, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlinks unavailable: {error}")
    loaded = both_inventories()
    assert loaded["linked_files"] == {"other": ["inside.md", "safe-link.md"]}
    assert view(file_path="safe-link.md")["content"] == inside.read_text(encoding="utf-8")
    assert view(file_path="escape.md")["success"] is False


def test_external_skill_inventory_keeps_the_exact_requested_identifier(catalog, tmp_path, monkeypatch):
    external = tmp_path / "external"
    directory = make_skill(external, "external/helper")
    write_support(directory, "root.md")
    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", lambda: [external])
    loaded = both_inventories("external/helper")
    assert view(**loaded["file_example"])["content"] == "Contents of root.md"


def test_inventory_tracks_the_active_profile(catalog, tmp_path, monkeypatch):
    directory = make_skill(catalog)
    write_support(directory, "first.md")
    other_home = tmp_path / "other-profile"
    other = make_skill(other_home / "skills")
    write_support(other, "second.md")
    assert both_inventories()["linked_files"] == {"other": ["first.md"]}
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_tool._SKILLS_DIR_AT_IMPORT)
    monkeypatch.setenv("HERMES_HOME", str(other_home))
    assert both_inventories()["linked_files"] == {"other": ["second.md"]}


def test_configured_disabled_explicit_load_keeps_its_refusal(catalog):
    make_skill(catalog)
    (catalog.parent / "config.yaml").write_text("skills:\n  disabled: [helper]\n", encoding="utf-8")
    loaded = view()
    assert loaded["success"] is False
    assert "disabled" in loaded["error"]


def test_environment_hidden_skill_still_accepts_an_explicit_load(catalog, monkeypatch):
    make_skill(catalog, extra="environments: [kanban]\n")
    monkeypatch.setattr(skill_utils, "_detect_environment", lambda environment: False)
    assert skill_utils.skill_matches_environment({"environments": ["kanban"]}) is False
    assert view()["success"] is True


@pytest.mark.parametrize("compact", [False, True])
def test_full_and_compact_indexes_separate_labels_on_cold_snapshot_and_warm_paths(catalog, compact):
    make_skill(catalog, "connectors/workspace/helper")
    kwargs = {"compact_categories": frozenset({"connectors"})} if compact else {}
    cold = prompt_builder.build_skills_system_prompt(**kwargs)
    marker = "connectors/workspace " + ("[names only] " if compact else "") + "[category; not callable]:"
    assert marker in cold
    assert "Pass an exact listed skill name" in cold
    assert "helper" in cold
    assert view("connectors/workspace")["success"] is False
    assert view()["success"] is True
    prompt_builder.clear_skills_system_prompt_cache()
    assert prompt_builder.build_skills_system_prompt(**kwargs) == cold
    assert prompt_builder.build_skills_system_prompt(**kwargs) == cold


def test_incomplete_body_recovery_keeps_the_discovery_contract(catalog):
    from tools.skill_delivery import _skill_view_incomplete_result

    directory = make_skill(catalog, body="Required instructions.\n" * 5_000)
    write_support(directory, "root.md")
    loaded = view()
    receipt = json.loads(_skill_view_incomplete_result(json.dumps(loaded)))
    assert receipt["load_status"] == "incomplete"
    for key in ("linked_files", "file_inventory", "file_example", "usage_hint"):
        assert receipt[key] == loaded[key]
    assert len(json.dumps(receipt, ensure_ascii=False)) <= 20_000
    assert view(**receipt["file_example"])["content"] == "Contents of root.md"


def test_slash_consumer_uses_root_and_nested_files_from_the_loader(catalog):
    from agent.skill_commands import _build_skill_message

    directory = make_skill(catalog)
    for path in ("root.md", "references/deep/guide.md"):
        write_support(directory, path)
    loaded = view()
    message = _build_skill_message(loaded, directory, "Activated")
    assert "- root.md  ->" in message
    assert "- references/deep/guide.md  ->" in message


def test_empty_partial_inventory_is_not_rescanned_by_the_slash_consumer(catalog, monkeypatch):
    from agent.skill_commands import _build_skill_message

    directory = make_skill(catalog)
    write_support(directory, "references/deep/guide.md")
    monkeypatch.setattr(skill_inventory, "MAX_INVENTORY_CHARS", 10)
    loaded = view()
    assert loaded["linked_files"] is None
    message = _build_skill_message(loaded, directory, "Activated")
    assert "Supporting-file inventory is incomplete" in message
    assert "references/deep/guide.md" not in message
    assert "[This skill has supporting files:]" not in message


def test_omitted_delivery_inventory_does_not_leave_an_unlisted_example(catalog):
    from tools.skill_delivery import _skill_view_incomplete_result

    directory = make_skill(catalog, body="Required instructions.\n" * 5_000)
    instructions = directory / "SKILL.md"
    instructions.write_text(
        instructions.read_text(encoding="utf-8").replace("Test discovery.", "d" * 18_000),
        encoding="utf-8",
    )
    for index in range(200):
        write_support(directory, f"{index:03d}-{'document' * 8}.md")
    loaded = view()
    assert loaded["linked_files"]
    receipt = json.loads(_skill_view_incomplete_result(json.dumps(loaded)))
    assert "linked_files" not in receipt
    assert receipt["file_inventory"]["complete"] is False
    assert receipt["file_inventory"]["listed_files"] == 0
    assert receipt["file_inventory"]["limit"] == "delivery"
    assert receipt.get("file_example") is None
    assert receipt.get("usage_hint") is None
    assert receipt["metadata_omitted"] is True
    assert len(json.dumps(receipt, ensure_ascii=False)) <= 20_000
