"""File reader/writer conflicts in the real metadata-only batch planner."""

import json
from types import SimpleNamespace

import pytest

from agent.tool_dispatch_helpers import _plan_tool_batch_segments


def call(name, **args):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=json.dumps(args)))


@pytest.mark.parametrize("left", ["read_file", "search_files", "write_file", "patch"])
@pytest.mark.parametrize("right", ["read_file", "search_files", "write_file", "patch"])
@pytest.mark.parametrize("paths", [("dir", "dir/file"), ("dir/file", "dir/file")])
def test_overlapping_paths_conflict_only_when_either_call_writes(left, right, paths):
    calls = [call(left, path=paths[0]), call(right, path=paths[1])]
    writer = left in {"write_file", "patch"} or right in {"write_file", "patch"}

    assert _plan_tool_batch_segments(calls) == [("sequential" if writer else "parallel", calls)]


@pytest.mark.parametrize("left", ["read_file", "search_files", "write_file", "patch"])
@pytest.mark.parametrize("right", ["read_file", "search_files", "write_file", "patch"])
def test_disjoint_paths_share_a_segment_including_writes(left, right):
    calls = [call(left, path="dir/a"), call(right, path="dir/b")]

    assert _plan_tool_batch_segments(calls) == [("parallel", calls)]


@pytest.mark.parametrize("reverse", [False, True])
def test_default_search_root_uses_execution_directory(tmp_path, monkeypatch, reverse):
    monkeypatch.chdir(tmp_path.parent)
    search = call("search_files", pattern="x")
    calls = [search, call("write_file", path=str(tmp_path / "a"))]
    if reverse:
        calls.reverse()

    assert _plan_tool_batch_segments(calls, execution_cwd=tmp_path) == [("sequential", calls)]
    independent = [search, call("write_file", path=str(tmp_path.parent / "outside"))]
    assert _plan_tool_batch_segments(independent, execution_cwd=tmp_path) == [("parallel", independent)]


def test_default_search_root_uses_process_directory_without_execution_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = [call("search_files", pattern="x"), call("patch", path="a")]

    assert _plan_tool_batch_segments(calls) == [("sequential", calls)]


def test_path_aliases_reserve_the_same_tree(tmp_path):
    (tmp_path / "dir").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "dir", target_is_directory=True)
    calls = [call("search_files", path="alias"), call("write_file", path="dir/./a")]

    assert _plan_tool_batch_segments(calls, execution_cwd=tmp_path) == [("sequential", calls)]


@pytest.mark.parametrize("header", ["*** ", "***"])
@pytest.mark.parametrize("target", ["first", "second", "deleted", "source", "destination"])
@pytest.mark.parametrize("reverse", [False, True])
def test_v4a_patch_reserves_every_target_and_both_move_endpoints(header, target, reverse):
    body = (
        "*** Begin Patch\n"
        f"{header}Add File: first\n+one\n"
        f"{header}Update File: second\n@@\n-old\n+new\n"
        f"{header}Delete File: deleted\n"
        f"{header}Move File: source -> destination\n"
        "*** End Patch"
    )
    # path is a decoy: V4A writes are defined by the patch body.
    patch_call = call("patch", mode="patch", path="decoy", patch=body)
    calls = [patch_call, call("read_file", path=target)]
    if reverse:
        calls.reverse()

    assert _plan_tool_batch_segments(calls) == [("sequential", calls)]
    independent = [patch_call, call("write_file", path="elsewhere")]
    assert _plan_tool_batch_segments(independent) == [("parallel", independent)]


@pytest.mark.parametrize("name,args", [
    ("read_file", {}),
    ("read_file", {"path": 123}),
    ("write_file", {"path": []}),
    ("write_file", {"path": "a\x00b"}),
    ("search_files", {"path": None}),
    ("search_files", {"path": ""}),
    ("search_files", {"path": "  "}),
    ("search_files", {"path": {}}),
    ("patch", {"mode": "unknown", "path": "decoy"}),
    ("patch", {"mode": "patch", "path": "decoy"}),
    ("patch", {"mode": "patch", "path": "decoy", "patch": 123}),
    ("patch", {"mode": "patch", "path": "decoy", "patch": "not a patch"}),
    ("patch", {"mode": "patch", "patch": "*** Add File: a\n+x\n*** Update File: b"}),
])
def test_unknown_or_malformed_scope_is_an_ordered_barrier_even_with_plugin_opt_in(name, args):
    before = [call("read_file", path="a"), call("read_file", path="b")]
    unknown = call(name, **args)
    after = [call("read_file", path="c"), call("read_file", path="d")]
    calls = [*before, unknown, *after]

    assert _plan_tool_batch_segments(calls, extra_parallel_safe=frozenset({name})) == [
        ("parallel", before), ("sequential", [unknown]), ("parallel", after)
    ]


def test_unresolvable_path_fails_closed():
    calls = [call("read_file", path="~hermes_planner_nonexistent_user/a"), call("read_file", path="a")]

    assert _plan_tool_batch_segments(calls) == [("sequential", calls)]


def test_conflicts_close_the_whole_run_without_reordering_across_barriers():
    calls = [
        call("read_file", path="tree/a"),
        call("read_file", path="tree/a"),
        call("write_file", path="tree/a"),
        call("write_file", path="other/a"),
        call("search_files", path="tree"),
        call("read_file", path="tree/b"),
        call("request_user_input"),
        call("read_file", path="tree/a"),
        call("read_file", path="tree/a"),
    ]

    assert _plan_tool_batch_segments(calls) == [
        ("parallel", calls[:2]),
        ("parallel", calls[2:4]),
        ("parallel", calls[4:6]),
        ("sequential", calls[6:7]),
        ("parallel", calls[7:]),
    ]
