"""Installer contract tests for Omnia's lean, fork-aware profiles."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


INSTALL_SH = Path(__file__).parents[1] / "scripts" / "install.sh"


def _write_executable(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _run_installer(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(INSTALL_SH), *args],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **env},
    )


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


@pytest.mark.parametrize(
    ("profile", "expected_spec", "expected_marker"),
    [
        ("all", ".[all]", None),
        ("none", ".", "none\n"),
        ("mcp,web", ".[mcp,web]", "mcp,web\n"),
    ],
)
def test_python_deps_stage_applies_extras_profile(
    tmp_path: Path,
    profile: str,
    expected_spec: str,
    expected_marker: str | None,
):
    install_dir = tmp_path / "install"
    hermes_home = tmp_path / "home"
    command_log = tmp_path / "commands.log"
    install_dir.mkdir()
    (install_dir / "venv" / "bin").mkdir(parents=True)
    (install_dir / "venv" / "bin" / "python").symlink_to(sys.executable)
    (install_dir / "pyproject.toml").write_text(
        """[project]
name = "installer-fixture"
version = "0.0.0"

[project.optional-dependencies]
all = []
""",
        encoding="utf-8",
    )
    _write_executable(
        hermes_home / "bin" / "uv",
        """#!/bin/sh
printf '%s\\n' "$*" >> "$COMMAND_LOG"
if [ "$1" = "--version" ]; then
    echo "uv 0.0.0-test"
elif [ "$1" = "python" ] && [ "$2" = "find" ]; then
    printf '%s\\n' "$TEST_PYTHON"
fi
""",
    )

    result = _run_installer(
        "--stage",
        "python-deps",
        "--json",
        "--dir",
        str(install_dir),
        "--hermes-home",
        str(hermes_home),
        "--extras",
        profile,
        env={
            "COMMAND_LOG": str(command_log),
            "TEST_PYTHON": sys.executable,
        },
    )

    assert f"pip install -e {expected_spec}" in command_log.read_text(encoding="utf-8")
    marker = install_dir / ".install-extras"
    if expected_marker is None:
        assert not marker.exists()
    else:
        assert marker.read_text(encoding="utf-8") == expected_marker
    assert '"ok":true,"stage":"python-deps"' in result.stdout


def test_node_deps_stage_limits_install_to_root_workspace(tmp_path: Path):
    install_dir = tmp_path / "install"
    hermes_home = tmp_path / "home"
    stub_bin = tmp_path / "bin"
    command_log = tmp_path / "commands.log"
    install_dir.mkdir()
    (install_dir / "package.json").write_text("{}\n", encoding="utf-8")
    _write_executable(stub_bin / "node", "#!/bin/sh\necho 'v22.12.0'\n")
    _write_executable(
        stub_bin / "npm",
        "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$COMMAND_LOG\"\n",
    )

    result = _run_installer(
        "--stage",
        "node-deps",
        "--json",
        "--dir",
        str(install_dir),
        "--hermes-home",
        str(hermes_home),
        "--no-node-workspaces",
        "--skip-browser",
        env={
            "COMMAND_LOG": str(command_log),
            "PATH": f"{stub_bin}{os.pathsep}{os.environ['PATH']}",
        },
    )

    assert command_log.read_text(encoding="utf-8").splitlines() == [
        "install --workspaces=false --silent"
    ]
    assert "Skipping Playwright/Chromium install" in result.stdout
    assert '"ok":true,"stage":"node-deps"' in result.stdout


def test_skip_ffmpeg_avoids_package_install(tmp_path: Path):
    hermes_home = tmp_path / "home"
    stub_bin = tmp_path / "bin"
    command_log = tmp_path / "commands.log"
    for command in ("head", "tr", "uname"):
        executable = Path("/usr/bin") / command
        if not executable.exists():
            executable = Path("/bin") / command
        (stub_bin).mkdir(parents=True, exist_ok=True)
        (stub_bin / command).symlink_to(executable)
    _write_executable(
        stub_bin / "rg",
        "#!/bin/sh\nprintf '%s\\n' \"rg $*\" >> \"$COMMAND_LOG\"\necho 'ripgrep 0.0.0-test'\n",
    )

    result = _run_installer(
        "--ensure",
        "ffmpeg",
        "--skip-ffmpeg",
        "--hermes-home",
        str(hermes_home),
        env={
            "COMMAND_LOG": str(command_log),
            "PATH": str(stub_bin),
        },
    )

    assert "Skipping ffmpeg (--skip-ffmpeg)" in result.stdout
    assert command_log.read_text(encoding="utf-8").splitlines() == ["rg --version"]
