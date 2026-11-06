"""Tests for per-file consecutive patch-failure tracking.

When the agent repeatedly fails to patch the same file with similar but
non-matching old_strings, it's usually stuck in a loop with a stale view
of the file.  After 3 consecutive failures on the same path, the patch
tool injects an escalating ``_hint`` that tells the model to break out
of the loop (re-read, use longer context, or fall back to write_file).

See issue #507 (Roo Code deep-dive, item 2f).
"""

import json

import pytest


@pytest.fixture
def hermes_home(monkeypatch, tmp_path):
    """Isolate HERMES_HOME and clear module-level caches afterward so the
    real shell-out side effects from _handle_patch don't leak into
    subsequent tests (see test_line_ending_preservation.py for details)."""
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home
    try:
        from tools.file_tools import clear_file_ops_cache, _read_tracker_lock, _read_tracker
        clear_file_ops_cache()
        with _read_tracker_lock:
            _read_tracker.clear()
    except Exception:
        pass
    try:
        from tools.terminal_tool import _active_environments, _env_lock
        with _env_lock:
            _active_environments.clear()
    except Exception:
        pass


@pytest.fixture
def fresh_tracker():
    """Reset the module-level tracker before each test so the count starts
    at zero regardless of prior test order."""
    from tools.file_tools import _patch_failure_tracker, _patch_failure_lock

    with _patch_failure_lock:
        _patch_failure_tracker.clear()
    yield
    with _patch_failure_lock:
        _patch_failure_tracker.clear()


class TestPatchFailureEscalation:
    @pytest.mark.parametrize("inert_position", [0, 1, 2])
    def test_identical_hunk_does_not_block_real_edits(
        self, hermes_home, tmp_path, fresh_tracker, inert_position
    ):
        from tools.file_tools import _handle_patch

        target = tmp_path / "guide.md"
        target.write_text("A = 1\nB = 2\nC = 3\n")
        hunks = ["-B = 2\n+B = 4", "-C = 3\n+C = 5"]
        hunks.insert(inert_position, "-A = 1\n+A = 1")
        patch = (
            f"*** Begin Patch\n*** Update File: {target}\n@@\n"
            + "\n@@\n".join(hunks)
            + "\n*** End Patch"
        )

        result = json.loads(_handle_patch(
            {"mode": "patch", "patch": patch}, task_id="identical-hunk"
        ))

        assert result.get("success") is True, result
        assert target.read_text() == "A = 1\nB = 4\nC = 5\n"

    def test_first_two_failures_use_normal_hint(self, hermes_home, tmp_path, fresh_tracker):
        from tools.file_tools import _handle_patch

        target = tmp_path / "f.py"
        target.write_text("def foo():\n    return 1\n")

        for _i in range(2):
            result = _handle_patch(
                {
                    "mode": "replace",
                    "path": str(target),
                    "old_string": f"NONEXISTENT_{_i}_XYZQQQ",
                    "new_string": "x",
                },
                task_id="esc_t1",
            )
            d = json.loads(result)
            hint = d.get("_hint", "") or ""
            assert "failure #" not in hint, (
                f"Escalating hint fired too early on attempt {_i + 1}: {hint!r}"
            )

    def test_third_consecutive_failure_escalates(self, hermes_home, tmp_path, fresh_tracker):
        from tools.file_tools import _handle_patch

        target = tmp_path / "f.py"
        target.write_text("def foo():\n    return 1\n")

        last_hint = ""
        for _i in range(3):
            result = _handle_patch(
                {
                    "mode": "replace",
                    "path": str(target),
                    "old_string": f"DOES_NOT_EXIST_{_i}_FOOFOOFOO",
                    "new_string": "x",
                },
                task_id="esc_t2",
            )
            d = json.loads(result)
            last_hint = d.get("_hint", "") or ""

        assert "failure #3" in last_hint, repr(last_hint)
        assert "Stop retrying" in last_hint
        assert "write_file" in last_hint, (
            "Escalating hint should mention write_file fallback"
        )

    def test_success_clears_failure_counter(self, hermes_home, tmp_path, fresh_tracker):
        from tools.file_tools import _handle_patch

        target = tmp_path / "f.py"
        target.write_text("def foo():\n    return 1\n")

        # Three failures: counter at 3.
        for _i in range(3):
            _handle_patch(
                {
                    "mode": "replace",
                    "path": str(target),
                    "old_string": f"GHOST_{_i}_ABCABC",
                    "new_string": "x",
                },
                task_id="esc_t3",
            )

        # Successful patch: clears the counter.
        result = _handle_patch(
            {
                "mode": "replace",
                "path": str(target),
                "old_string": "return 1",
                "new_string": "return 99",
            },
            task_id="esc_t3",
        )
        d = json.loads(result)
        assert not d.get("error"), d

        # Next failure should be back to "attempt 1" — generic hint only.
        result = _handle_patch(
            {
                "mode": "replace",
                "path": str(target),
                "old_string": "STILL_GHOST_XYZ",
                "new_string": "x",
            },
            task_id="esc_t3",
        )
        d = json.loads(result)
        hint = d.get("_hint", "") or ""
        assert "failure #" not in hint, (
            f"Counter should have been reset after success: {hint!r}"
        )

    def test_different_paths_have_independent_counters(
        self, hermes_home, tmp_path, fresh_tracker
    ):
        from tools.file_tools import _handle_patch

        a = tmp_path / "a.py"
        a.write_text("x = 1\n")
        b = tmp_path / "b.py"
        b.write_text("y = 2\n")

        # Three failures on a.py.
        for _i in range(3):
            _handle_patch(
                {
                    "mode": "replace",
                    "path": str(a),
                    "old_string": f"NONE_A_{_i}_ZZZ",
                    "new_string": "x",
                },
                task_id="esc_t4",
            )

        # One failure on b.py — should NOT inherit a.py's count.
        result = _handle_patch(
            {
                "mode": "replace",
                "path": str(b),
                "old_string": "NONE_B_ZZZ",
                "new_string": "x",
            },
            task_id="esc_t4",
        )
        d = json.loads(result)
        hint = d.get("_hint", "") or ""
        assert "failure #" not in hint, (
            f"b.py's hint inherited a.py's count: {hint!r}"
        )

    def test_different_tasks_have_independent_counters(
        self, hermes_home, tmp_path, fresh_tracker
    ):
        from tools.file_tools import _handle_patch

        target = tmp_path / "shared.py"
        target.write_text("z = 0\n")

        # Three failures under task A.
        for _i in range(3):
            _handle_patch(
                {
                    "mode": "replace",
                    "path": str(target),
                    "old_string": f"GHOST_A_{_i}_QWE",
                    "new_string": "x",
                },
                task_id="task_A",
            )

        # First failure under task B — should NOT see escalation.
        result = _handle_patch(
            {
                "mode": "replace",
                "path": str(target),
                "old_string": "GHOST_B_QWE",
                "new_string": "x",
            },
            task_id="task_B",
        )
        d = json.loads(result)
        hint = d.get("_hint", "") or ""
        assert "failure #" not in hint, (
            f"task_B's hint cross-contaminated from task_A: {hint!r}"
        )


def v4a_update(path, hunks="-shared\n+changed"):
    return f"*** Update File: {path}\n@@\n{hunks}\n"


def v4a_call(body, task="v4a-recovery"):
    from tools.file_tools import _handle_patch

    return json.loads(_handle_patch(
        {"mode": "patch", "patch": f"*** Begin Patch\n{body}*** End Patch"},
        task_id=task,
    ))


@pytest.mark.usefixtures("hermes_home", "fresh_tracker")
class TestV4ARecovery:
    @pytest.mark.parametrize("mode", ["patch", "replace"])
    @pytest.mark.parametrize("missing", [False, True])
    def test_matching_failures_escalate_without_writes(self, tmp_path, mode, missing):
        from tools.file_tools import _handle_patch

        target = tmp_path / "repeat.txt"
        original = "first\nshared\nsecond\nshared\n"
        target.write_text(original)
        old = "missing" if missing else "shared"
        for attempt in range(1, 6):
            if mode == "patch":
                result = v4a_call(v4a_update(target, f"-{old}\n+changed"))
            else:
                result = json.loads(_handle_patch(
                    {"mode": mode, "path": str(target), "old_string": old,
                     "new_string": "changed"}, task_id="v4a-recovery",
                ))
            assert result["success"] is False
            assert target.read_text() == original
            assert "replace_all" not in result["error"]
            hint = result.get("_hint", "")
            if mode == "patch":
                assert "surrounding lines prefixed by a space" in hint
                assert "bare @@" in hint
            if attempt < 3:
                assert "failure #" not in hint
            else:
                assert f"failure #{attempt}" in hint
                assert "Stop retrying" in hint
            assert "match_failure_paths" not in result

    def test_unique_context_recovers_one_occurrence_and_resets_counter(self, tmp_path):
        target = tmp_path / "repeat.txt"
        target.write_text("first\nshared\nsecond\nshared\n")
        for _ in range(3):
            assert v4a_call(v4a_update(target))["success"] is False
        corrected = v4a_call(v4a_update(target, " first\n-shared\n+changed\n second"))
        assert corrected["success"] is True, corrected
        assert target.read_text() == "first\nchanged\nsecond\nshared\n"
        next_failure = v4a_call(v4a_update(target, "-missing\n+changed"))
        assert next_failure["success"] is False
        assert "failure #" not in next_failure["_hint"]

    def test_multi_file_validation_counts_only_failing_file_once(self, tmp_path):
        bad = tmp_path / "bad.txt"
        good = tmp_path / "good.txt"
        original = "first\nshared\nsecond\nshared\n"
        bad.write_text(original)
        good.write_text("unique\n")
        bad_hunks = v4a_update(bad) + v4a_update(bad)
        body = v4a_update(good, "-unique\n+edited") + bad_hunks
        for attempt in range(1, 4):
            result = v4a_call(body)
            assert result["success"] is False
            assert bad.read_text() == original
            assert good.read_text() == "unique\n"
            if attempt == 3:
                assert "failure #3" in result["_hint"]
                assert str(bad.resolve()) in result["_hint"]
                assert str(good.resolve()) not in result["_hint"]
            else:
                assert "failure #" not in result["_hint"]
        # A neighbour that validated successfully has no accumulated failures.
        neighbour = v4a_call(v4a_update(good, "-missing\n+edited"))
        assert "failure #" not in neighbour["_hint"]
        assert v4a_call(v4a_update(good, "-unique\n+edited"))["success"] is True
        # Its later success does not clear the actual failing target.
        still_failing = v4a_call(v4a_update(bad))
        assert "failure #4" in still_failing["_hint"]

    def test_equivalent_paths_share_counts_but_tasks_and_files_do_not(
        self, tmp_path, monkeypatch
    ):
        from tools import terminal_tool

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        monkeypatch.setattr(terminal_tool, "_session_cwd", {})
        for task in ("v4a-recovery", "other-task"):
            terminal_tool.record_session_cwd(task, str(tmp_path))
        target = tmp_path / "repeat.txt"
        target.write_text("shared\nshared\n")
        assert "failure #" not in v4a_call(v4a_update("repeat.txt"))["_hint"]
        assert "failure #" not in v4a_call(v4a_update(target))["_hint"]
        assert "failure #3" in v4a_call(v4a_update("repeat.txt"))["_hint"]
        assert "failure #" not in v4a_call(v4a_update(target), "other-task")["_hint"]
        other = tmp_path / "other" / "repeat.txt"
        other.parent.mkdir()
        other.write_text("shared\nshared\n")
        assert "failure #" not in v4a_call(v4a_update(other))["_hint"]
        # Success clears only this task's canonical identity.
        assert v4a_call(v4a_update(target, "-shared\n-shared\n+fixed"))["success"] is True
        assert "failure #" not in v4a_call(v4a_update("repeat.txt"))["_hint"]

    def test_each_matching_failure_in_multi_file_patch_receives_its_own_count(self, tmp_path):
        paths = [tmp_path / "first.txt", tmp_path / "second.txt"]
        for path in paths:
            path.write_text("shared\nshared\n")
        body = "".join(v4a_update(path) for path in paths)
        for _ in range(3):
            result = v4a_call(body)
        assert result["_hint"].count("failure #3") == 2
        for path in paths:
            assert str(path.resolve()) in result["_hint"]
            assert path.read_text() == "shared\nshared\n"

    def test_successful_write_fallback_resets_only_its_task_and_file(self, tmp_path, monkeypatch):
        from tools.file_operations import WriteResult
        from tools.file_tools import _get_file_ops, _patch_failure_tracker, write_file_tool

        target = tmp_path / "repeat.txt"
        other = tmp_path / "other.txt"
        for path in (target, other):
            path.write_text("shared\nshared\n")
            for _ in range(3):
                v4a_call(v4a_update(path))
        v4a_call(v4a_update(target), "other-task")
        with monkeypatch.context() as injection:
            injection.setattr(
                _get_file_ops("v4a-recovery"), "write_file",
                lambda *_: WriteResult(error="write refused"),
            )
            failed = json.loads(write_file_tool(str(target), "rebuilt\n", task_id="v4a-recovery"))
        assert failed.get("error") == "write refused"
        assert _patch_failure_tracker["v4a-recovery"][str(target.resolve())] == 3
        result = json.loads(write_file_tool(str(target), "rebuilt\n", task_id="v4a-recovery"))
        assert not result.get("error"), result
        assert target.read_text() == "rebuilt\n"
        assert _patch_failure_tracker["v4a-recovery"] == {str(other.resolve()): 3}
        assert _patch_failure_tracker["other-task"] == {str(target.resolve()): 1}
        assert "failure #" not in v4a_call(v4a_update(target))["_hint"]

    @pytest.mark.parametrize("hint", ["shared", "absent"])
    def test_addition_only_matching_failures_have_mode_correct_help(self, tmp_path, hint):
        target = tmp_path / "repeat.txt"
        target.write_text("shared\nshared\n")
        body = f"*** Update File: {target}\n@@ {hint} @@\n+inserted\n"
        for _ in range(3):
            result = v4a_call(body)
        assert result["success"] is False
        assert "failure #3" in result["_hint"]
        assert "replace_all" not in json.dumps(result)
        assert target.read_text() == "shared\nshared\n"

    def test_non_matching_errors_do_not_increment_or_clear_counts(self, tmp_path, monkeypatch):
        from tools.file_operations import WriteResult
        from tools.file_tools import _get_file_ops, _patch_failure_tracker

        target = tmp_path / "repeat.txt"
        target.write_text("shared\nshared\n")
        v4a_call(v4a_update(target))
        with monkeypatch.context() as injection:
            injection.setattr(
                _get_file_ops("v4a-recovery"), "write_file",
                lambda *_: WriteResult(error="Toolbox HTTP 502"),
            )
            apply_error = v4a_call(v4a_update(target, "-shared\n-shared\n+changed"))
        assert "Apply phase failed" in apply_error["error"]
        assert "Toolbox HTTP 502" in apply_error["error"]
        assert "_hint" not in apply_error
        assert target.read_text() == "shared\nshared\n"
        no_op = v4a_call(v4a_update(target, "-shared\n+shared"))
        assert "no changes" in no_op["error"]
        assert "_hint" not in no_op
        missing_file = v4a_call(v4a_update(tmp_path / "not-there.txt"))
        assert missing_file["success"] is False
        assert _patch_failure_tracker["v4a-recovery"] == {str(target.resolve()): 1}
        assert "failure #" not in v4a_call(v4a_update(target))["_hint"]
        assert "failure #3" in v4a_call(v4a_update(target))["_hint"]
