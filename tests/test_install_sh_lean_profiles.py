"""Installer contract tests for Omnia's lean, fork-aware profiles."""

import subprocess
from pathlib import Path


INSTALL_SH = Path(__file__).parents[1] / "scripts" / "install.sh"


def test_lean_install_flags_are_documented_and_shell_valid():
    subprocess.run(["bash", "-n", str(INSTALL_SH)], check=True)
    help_result = subprocess.run(
        ["bash", str(INSTALL_SH), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--extras LIST" in help_result.stdout
    assert "--skip-ffmpeg" in help_result.stdout
    assert "--no-node-workspaces" in help_result.stdout
    assert "HERMES_REPO=owner/repo" in help_result.stdout
    assert "HERMES_BRANCH=name" in help_result.stdout


def test_lean_profiles_control_each_expensive_install_phase():
    source = INSTALL_SH.read_text(encoding="utf-8")

    assert 'BRANCH="${HERMES_BRANCH:-main}"' in source
    assert 'HERMES_REPO="${HERMES_REPO:-useomnia/hermes-agent}"' in source
    assert 'if [ "$SKIP_FFMPEG" = true ]; then' in source
    assert 'npm install --workspaces=false --silent' in source
    assert 'if [ "$PYTHON_EXTRAS" = "none" ]; then' in source
    assert 'printf \'%s\\n\' "$PYTHON_EXTRAS" > "$INSTALL_DIR/.install-extras"' in source
