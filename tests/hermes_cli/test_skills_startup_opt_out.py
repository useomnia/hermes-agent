"""Cold-start coverage for the bundled-skills opt-out fast path."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

from hermes_cli import main


def test_opted_out_profile_skips_the_heavy_skills_sync_import(tmp_path, monkeypatch):
    sync = Mock()
    (tmp_path / ".no-bundled-skills").touch()
    monkeypatch.setattr(main, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "tools.skills_sync",
        SimpleNamespace(sync_skills=sync),
    )

    main._sync_bundled_skills_quietly()

    sync.assert_not_called()


def test_profile_without_opt_out_still_syncs_bundled_skills(tmp_path, monkeypatch):
    sync = Mock()
    monkeypatch.setattr(main, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "tools.skills_sync",
        SimpleNamespace(sync_skills=sync),
    )

    main._sync_bundled_skills_quietly()

    sync.assert_called_once_with(quiet=True)


def test_gateway_run_leaves_bundled_skill_sync_to_gateway_start(monkeypatch):
    sync = Mock()
    dispatch = Mock()
    monkeypatch.setattr(main, "_sync_bundled_skills_quietly", sync)
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.gateway",
        SimpleNamespace(gateway_command=dispatch),
    )

    main.cmd_gateway(SimpleNamespace(gateway_command="run"))

    sync.assert_not_called()
    dispatch.assert_called_once()
