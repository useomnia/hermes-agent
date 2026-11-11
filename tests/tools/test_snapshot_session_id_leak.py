"""Cross-session HERMES_SESSION_ID leak via the shared bash snapshot.

Regression coverage for the bug where a single long-lived backend serves many
sessions through ONE ``_active_environments["default"]`` LocalEnvironment (the
messaging gateway, TUI, and desktop/web dashboard all collapse the terminal to
"default"). That environment persists a bash *session snapshot* file and
``source``s it before every command. ``export -p`` dumped the FIRST session's
``HERMES_SESSION_ID`` into the snapshot, so every LATER session ``source``d that
stale value and its ``echo $HERMES_SESSION_ID`` reported a FOREIGN session's id
— overriding the correct per-command Popen env injected by
``_inject_session_context_env``.

The fix strips the per-session bridged vars (HERMES_SESSION_* / UI /
CRON_AUTO_DELIVER_) from the snapshot at both dump sites in
``tools/environments/base.py``; they are re-injected fresh on every command.
"""

import os
import re
import sys

import pytest

from tools.environments.base import (
    _SNAPSHOT_EXCLUDED_ENV_REGEX,
    _export_dump_excluding_session_vars,
)


# ---------------------------------------------------------------------------
# Unit: the exclusion regex matches exactly the bridged vars, nothing else.
# ---------------------------------------------------------------------------

def test_regex_matches_bridged_session_vars():
    rx = re.compile(_SNAPSHOT_EXCLUDED_ENV_REGEX)
    # Every var the gateway bridges must be excluded.
    from gateway.session_context import _VAR_MAP

    for name in _VAR_MAP:
        line = f'declare -x {name}="whatever"'
        assert rx.search(line), f"{name} should be excluded from the snapshot"


def test_regex_preserves_user_env():
    rx = re.compile(_SNAPSHOT_EXCLUDED_ENV_REGEX)
    for line in (
        'declare -x PATH="/usr/bin:/bin"',
        'declare -x HOME="/home/user"',
        'declare -x HERMES_HOME="/home/user/.hermes"',  # NOT a session var
        'declare -x HERMESX="x"',
        'declare -x MY_HERMES_SESSION_ID="x"',  # prefix must anchor after "declare -x "
    ):
        assert not rx.search(line), f"{line!r} must be preserved in the snapshot"


def test_export_snippet_shape():
    snippet = _export_dump_excluding_session_vars("/tmp/snap.tmp.$BASHPID")
    assert "export -p" in snippet
    # Unset-by-name (not line-grep): multi-line declare values must not leave
    # continuation lines in the snapshot (issue #71296).
    assert "unset" in snippet
    assert "${!HERMES_SESSION_*}" in snippet
    assert "${!HERMES_CRON_AUTO_DELIVER_*}" in snippet
    assert "HERMES_UI_SESSION_ID" in snippet
    assert "HERMES_INTERACTION_POLICY" in snippet
    assert "grep -vE" not in snippet
    assert "/tmp/snap.tmp.$BASHPID" in snippet
    # The redirection must be attached to a brace group wrapping the dump,
    # NOT to a pipeline segment: a redirect on a pipeline segment expands
    # $BASHPID inside that segment's subshell (a different PID than the parent
    # that expands the follow-up ``mv`` operand), silently orphaning the dump
    # and breaking snapshot env persistence entirely.
    assert snippet.lstrip().startswith("{ ")
    assert "|| true; }" in snippet
    assert snippet.rstrip().endswith("> /tmp/snap.tmp.$BASHPID")


# ---------------------------------------------------------------------------
# Integration: real LocalEnvironment, two sessions, no cross-contamination.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
@pytest.mark.parametrize("next_conversation_id", ["CONVBBB", ""])
def test_shared_snapshot_no_cross_session_leak(tmp_path, next_conversation_id):
    import threading

    from gateway.session_context import _VAR_MAP, _UNSET, set_session_vars
    from tools.environments.local import LocalEnvironment

    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    try:
        def run_as(sid, conversation_id, *, initialize=False):
            out = {}

            def worker():
                for v in _VAR_MAP.values():
                    v.set(_UNSET)
                set_session_vars(
                    session_key="k" + sid, session_id=sid, source="desktop",
                    origin_turn_id="turn-" + sid,
                    origin_conversation_id=conversation_id,
                )
                if initialize:
                    env.init_session()
                    with open(env._snapshot_path, encoding="utf-8") as snapshot:
                        out["bootstrap"] = snapshot.read()
                out["r"] = env.execute(
                    'printf "[%s][%s][%s]\\n" "$HERMES_SESSION_ID" '
                    '"$HERMES_ORIGIN_TURN_ID" "$HERMES_ORIGIN_CONVERSATION_ID"'
                )

            t = threading.Thread(target=worker)
            t.start()
            t.join()
            return out

        result_a = run_as("SIDAAA", "CONVAAA", initialize=True)
        result_b = run_as("SIDBBB", next_conversation_id)
        out_a = result_a["r"].get("output", "")
        out_b = result_b["r"].get("output", "")

        assert "[SIDAAA][turn-SIDAAA][CONVAAA]" in out_a
        # The core assertion: B must see its OWN id, not A's leaked via snapshot.
        assert f"[SIDBBB][turn-SIDBBB][{next_conversation_id}]" in out_b
        assert "SIDAAA" not in out_b, f"session B leaked A's id: {out_b!r}"
        assert "CONVAAA" not in out_b, f"session B leaked A's conversation: {out_b!r}"

        for name in ("HERMES_SESSION_ID", "HERMES_ORIGIN_TURN_ID", "HERMES_ORIGIN_CONVERSATION_ID"):
            assert name not in result_a["bootstrap"]

        # The refreshed snapshot must not carry the session identity either.
        snap = env._snapshot_path
        if os.path.exists(snap):
            with open(snap, encoding="utf-8") as f:
                snapshot = f.read()
            for name in ("HERMES_SESSION_ID", "HERMES_ORIGIN_TURN_ID", "HERMES_ORIGIN_CONVERSATION_ID"):
                assert name not in snapshot
    finally:
        env.cleanup()
