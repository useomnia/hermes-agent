import ast
import io
import sys
from pathlib import Path

import pytest

from hermes_cli import boot_clock

REPO_ROOT = Path(__file__).resolve().parent.parent

LINUX_ONLY = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="procfs is Linux-only"
)


@pytest.fixture(autouse=True)
def _clear_checkpoints():
    boot_clock._checkpoints.clear()
    yield
    boot_clock._checkpoints.clear()


class _FakeProcFs:
    """Serves synthetic /proc/self/stat and /proc/uptime for the parser."""

    def __init__(self, stat: str, uptime: str) -> None:
        self._content = {"/proc/self/stat": stat, "/proc/uptime": uptime}

    def open(self, path, *_args, **_kwargs):
        try:
            return io.StringIO(self._content[path])
        except KeyError:
            raise OSError(f"unexpected path {path}") from None


@LINUX_ONLY
def test_process_elapsed_is_positive_and_plausible():
    elapsed = boot_clock.process_elapsed_seconds()

    assert elapsed is not None
    # This process is seconds-to-minutes old, never negative and never a year.
    assert 0 <= elapsed < 86_400


def test_parses_starttime_from_procfs_layout(monkeypatch):
    # starttime is field 22 of /proc/self/stat. Fields 1-2 are pid and comm, and
    # comm is parenthesised and may itself contain spaces and ')' — hence the
    # rsplit. Fields 3.. are positional after it, putting starttime at index 19.
    fields_after_comm = [
        "S", "1", "1", "0", "-1", "0", "0", "0", "0", "0", "0",
        "0", "0", "0", "0", "20", "0", "1", "0",
        "500",  # starttime (index 19)
        "999", "999",
    ]
    stat = "4242 (hermes gateway) " + " ".join(fields_after_comm)
    monkeypatch.setattr("builtins.open", _FakeProcFs(stat, "1000.0 1.0").open)
    monkeypatch.setattr("os.sysconf", lambda _name: 100)

    # Uptime 1000 s; the process started 500 ticks (5 s) after boot, so it is
    # 995 s old.
    assert boot_clock.process_elapsed_seconds() == pytest.approx(995.0)


def test_returns_none_when_procfs_is_unreadable(monkeypatch):
    # Non-Linux hosts and locked-down containers must degrade, never raise —
    # timing instrumentation may not be able to break a boot.
    def refuse(*_args, **_kwargs):
        raise OSError("no procfs")

    monkeypatch.setattr("builtins.open", refuse)

    assert boot_clock.process_elapsed_seconds() is None


def test_returns_none_when_clock_ticks_are_unusable(monkeypatch):
    stat = "1 (x) " + " ".join(["0"] * 22)
    monkeypatch.setattr("builtins.open", _FakeProcFs(stat, "10.0 1.0").open)
    monkeypatch.setattr("os.sysconf", lambda _name: 0)

    assert boot_clock.process_elapsed_seconds() is None


def test_preamble_is_empty_without_a_clock(monkeypatch):
    monkeypatch.setattr(boot_clock, "process_elapsed_seconds", lambda: None)

    assert boot_clock.format_preamble() == ""


def test_preamble_reports_every_checkpoint_in_order(monkeypatch):
    ages = iter([0.5, 0.6, 2.0, 2.1, 3.0])
    monkeypatch.setattr(boot_clock, "process_elapsed_seconds", lambda: next(ages))
    for name in ("main", "skills", "cli_import", "dispatch"):
        boot_clock.mark(name)

    # Ages, not durations: a phase is the gap between two neighbours, which is
    # what makes a missing middle checkpoint harmless.
    assert boot_clock.format_preamble() == (
        "to_main_ms=500 to_skills_ms=600 to_cli_import_ms=2000 "
        "to_dispatch_ms=2100 to_start_ms=3000"
    )


def test_preamble_reports_only_start_when_nothing_was_marked(monkeypatch):
    monkeypatch.setattr(boot_clock, "process_elapsed_seconds", lambda: 3.25)

    assert boot_clock.format_preamble() == "to_start_ms=3250"


def _marked_names(relative_path: str, *callees: str) -> set[str]:
    """Checkpoint names this file stamps via any of ``callees``.

    Read from the AST rather than importing gateway modules: the imports are
    intentionally heavyweight, and this test is pinning call sites. A dropped
    mark otherwise fails silently and misattributes the phase to a neighbour.
    """
    tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    return {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in callees
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }


def test_every_declared_gateway_checkpoint_is_still_marked():
    expected_by_file = {
        "hermes_cli/main.py": {
            "main",
            "skills",
            "cli_import",
        },
        "hermes_cli/gateway.py": {
            "dispatch",
            "pre_start",
        },
        "gateway/run.py": {
            "plugins",
            "relay",
            "session_recovery",
            "platforms",
            "runtime_ready",
            "fingerprint",
            "dup_guard",
            "skills_resync",
            "logging",
            "audit",
            "runner_init",
            "pid_lock",
            # Discovery is joined by the first agent build, not gateway boot.
            "mcp_discovery_start",
        },
        "gateway/platforms/api_server.py": {
            "api_connect",
            "api_routes",
            "api_bound",
            "api_ready",
            "api_approvals_start",
        },
    }
    observed = set()
    for path, expected in expected_by_file.items():
        names = _marked_names(path, "mark", "_boot_mark")
        assert names == expected
        observed.update(names)

    assert (
        observed
        == boot_clock.BOOT_CHECKPOINT_NAMES
        == {
            "main",
            "skills",
            "cli_import",
            "dispatch",
            "pre_start",
            "plugins",
            "relay",
            "session_recovery",
            "platforms",
            "runtime_ready",
            "fingerprint",
            "dup_guard",
            "skills_resync",
            "logging",
            "audit",
            "runner_init",
            "pid_lock",
            "mcp_discovery_start",
            "api_connect",
            "api_routes",
            "api_bound",
            "api_ready",
            "api_approvals_start",
        }
    )


def test_timeline_uses_the_callers_end_name(monkeypatch):
    ages = iter([0.25, 1.5])
    monkeypatch.setattr(boot_clock, "process_elapsed_seconds", lambda: next(ages))
    boot_clock.mark("spawned")

    assert boot_clock.format_timeline("ready") == "to_spawned_ms=250 to_ready_ms=1500"


@pytest.mark.parametrize("name", ["", "Ready", "not-valid", "has space"])
def test_timeline_rejects_unstructured_end_names(name):
    with pytest.raises(ValueError, match="lowercase identifier"):
        boot_clock.format_timeline(name)


def test_a_checkpoint_reached_twice_is_not_collapsed(monkeypatch):
    ages = iter([1.0, 2.0, 9.0])
    monkeypatch.setattr(boot_clock, "process_elapsed_seconds", lambda: next(ages))
    boot_clock.mark("dispatch")
    boot_clock.mark("dispatch")

    # A repeat means the boot took a path we did not expect; hiding it behind a
    # dict would make that invisible.
    assert boot_clock.format_preamble() == (
        "to_dispatch_ms=1000 to_dispatch_ms=2000 to_start_ms=9000"
    )


def test_mark_records_nothing_without_a_clock(monkeypatch):
    monkeypatch.setattr(boot_clock, "process_elapsed_seconds", lambda: None)

    boot_clock.mark("main")

    assert boot_clock._checkpoints == []
