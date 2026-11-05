"""Per-command credentials must not be restored from the bash session snapshot.

Omnio's backend injects a fresh, short-lived ``GH_TOKEN`` into every command's
environment. The session snapshot is sourced before each command and re-dumped
after it, so a ``GH_TOKEN`` captured in the snapshot would override the fresh
value for the whole life of the environment.
"""

import os
import re
import sys

import pytest

from tools.environments.base import (
    _SNAPSHOT_EXCLUDED_ENV_REGEX,
    _export_dump_excluding_session_vars,
)


def test_regex_excludes_gh_token():
    assert re.search(_SNAPSHOT_EXCLUDED_ENV_REGEX, 'declare -x GH_TOKEN="ghs_example"')


def test_regex_preserves_names_that_only_start_with_gh_token():
    assert not re.search(_SNAPSHOT_EXCLUDED_ENV_REGEX, 'declare -x GH_TOKEN_HELPER="x"')


def test_export_snippet_unsets_gh_token():
    snippet = _export_dump_excluding_session_vars("/tmp/snap.tmp.$BASHPID")
    assert re.search(r"\bGH_TOKEN\b", snippet.split("export -p")[0])


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
class TestLocalEnvironmentGhToken:
    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        import tools.env_passthrough
        from tools.environments.local import LocalEnvironment

        # LocalEnvironment strips GH_TOKEN unless passed through; a backend that
        # injects it per command is the case under test.
        monkeypatch.setattr(
            tools.env_passthrough, "is_env_passthrough", lambda name: name == "GH_TOKEN"
        )
        monkeypatch.setenv("GH_TOKEN", "token-at-session-start")
        environment = LocalEnvironment(cwd=str(tmp_path), timeout=30)
        environment.init_session()
        environment.execute('echo "[$GH_TOKEN]"')
        yield environment
        environment.cleanup()

    def test_later_command_sees_its_own_token(self, env, monkeypatch):
        monkeypatch.setenv("GH_TOKEN", "token-for-this-command")

        output = env.execute('echo "[$GH_TOKEN]"').get("output", "")

        assert "[token-for-this-command]" in output

    def test_snapshot_does_not_store_the_token(self, env):
        with open(env._snapshot_path) as snapshot:
            assert "GH_TOKEN" not in snapshot.read()

    def test_user_exports_still_persist(self, env):
        env.execute("export OMNIO_SNAPSHOT_PROBE=kept")

        output = env.execute('echo "[$OMNIO_SNAPSHOT_PROBE]"').get("output", "")

        assert "[kept]" in output
