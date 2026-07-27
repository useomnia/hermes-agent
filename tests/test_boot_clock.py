import sys

import pytest

from hermes_cli import boot_clock


LINUX_ONLY = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="procfs is Linux-only"
)


@LINUX_ONLY
def test_process_elapsed_is_positive_and_plausible():
    elapsed = boot_clock.process_elapsed_seconds()

    assert elapsed is not None
    # This process is seconds-to-minutes old, never negative and never a year.
    assert 0 <= elapsed < 86_400


def test_process_elapsed_returns_none_when_procfs_is_unreadable(monkeypatch):
    # Non-Linux hosts and locked-down containers must degrade, never raise —
    # timing instrumentation may not be able to break a boot.
    def refuse(*_args, **_kwargs):
        raise OSError("no procfs")

    monkeypatch.setattr("builtins.open", refuse)

    assert boot_clock.process_elapsed_seconds() is None


def test_preamble_is_empty_without_a_clock(monkeypatch):
    monkeypatch.setattr(boot_clock, "process_elapsed_seconds", lambda: None)

    assert boot_clock.format_preamble() == ""


def test_preamble_reports_both_phases(monkeypatch):
    monkeypatch.setattr(boot_clock, "process_elapsed_seconds", lambda: 9.5)
    monkeypatch.setattr(boot_clock, "elapsed_at_main", 8.0)

    # to_main_ms is the interpreter plus the CLI import graph; the difference up
    # to to_start_ms is the CLI's own post-import work.
    assert boot_clock.format_preamble() == "to_main_ms=8000 to_start_ms=9500"


def test_preamble_omits_main_phase_when_entry_point_never_marked(monkeypatch):
    monkeypatch.setattr(boot_clock, "process_elapsed_seconds", lambda: 3.25)
    monkeypatch.setattr(boot_clock, "elapsed_at_main", None)

    assert boot_clock.format_preamble() == "to_start_ms=3250"


@LINUX_ONLY
def test_mark_main_records_the_entry_point(monkeypatch):
    monkeypatch.setattr(boot_clock, "elapsed_at_main", None)

    boot_clock.mark_main()

    assert boot_clock.elapsed_at_main is not None


class _FakeProcFs:
    """Serves synthetic /proc/self/stat and /proc/uptime for the parser."""

    def __init__(self, stat: str, uptime: str) -> None:
        self._content = {"/proc/self/stat": stat, "/proc/uptime": uptime}

    def open(self, path, *_args, **_kwargs):
        import io

        try:
            return io.StringIO(self._content[path])
        except KeyError:
            raise OSError(f"unexpected path {path}") from None


def test_parses_starttime_from_procfs_layout(monkeypatch):
    # starttime is field 22 of /proc/self/stat. Fields 1-2 are pid and comm, and
    # comm is parenthesised and may itself contain spaces and ')' — hence the
    # rsplit. Fields 3.. are positional after it, putting starttime at index 19.
    fields_after_comm = [
        "S", "1", "1", "0", "-1", "0", "0", "0", "0", "0", "0",  # state..cmajflt
        "0", "0", "0", "0", "20", "0", "1", "0",                  # utime..itrealvalue
        "500",                                                     # starttime (index 19)
        "999", "999",
    ]
    stat = "4242 (hermes gateway) " + " ".join(fields_after_comm)
    monkeypatch.setattr("builtins.open", _FakeProcFs(stat, "1000.0 1.0").open)
    monkeypatch.setattr("os.sysconf", lambda _name: 100)

    # Uptime 1000 s; the process started 500 ticks (5 s) after boot, so it is
    # 995 s old.
    assert boot_clock.process_elapsed_seconds() == pytest.approx(995.0)


def test_returns_none_when_clock_ticks_are_unusable(monkeypatch):
    stat = "1 (x) " + " ".join(["0"] * 22)
    monkeypatch.setattr("builtins.open", _FakeProcFs(stat, "10.0 1.0").open)
    monkeypatch.setattr("os.sysconf", lambda _name: 0)

    assert boot_clock.process_elapsed_seconds() is None
