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
