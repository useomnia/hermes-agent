"""Exercise exact-commit installation through Bash and real local Git remotes."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


INSTALL_SH = Path(__file__).resolve().parents[1] / "scripts" / "install.sh"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            "user.name=Installer fixture",
            "-c",
            "user.email=fixture@example.invalid",
            *args,
        ],
        cwd=repo,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"},
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
    ).strip()


@pytest.fixture
def origin(tmp_path: Path) -> tuple[Path, list[str], str]:
    repo = tmp_path / "origin"
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main")
    commits = []
    for index in range(6):
        (repo / "fixture.txt").write_text(f"fixture {index}\n")
        _git(repo, "add", "fixture.txt")
        _git(repo, "commit", "--no-gpg-sign", "-m", f"fixture {index}")
        commits.append(_git(repo, "rev-parse", "HEAD"))
    _git(repo, "checkout", "-b", "side", commits[1])
    (repo / "side.txt").write_text("side branch\n")
    _git(repo, "add", "side.txt")
    _git(repo, "commit", "--no-gpg-sign", "-m", "side commit")
    side = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "main")
    return repo, commits, side


def _install(
    tmp_path: Path, origin: Path, install: Path, pin: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "/bin/bash",
            str(INSTALL_SH),
            "--stage",
            "repository",
            "--dir",
            str(install),
            "--branch",
            "main",
            "--commit",
            pin,
            "--json",
        ],
        env={
            **os.environ,
            "HERMES_HOME": str(tmp_path / "home"),
            "HERMES_REPO_URL_SSH": origin.as_uri(),
            "HERMES_REPO_URL_HTTPS": origin.as_uri(),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )


def _assert_exact_tree(install: Path, origin: Path, pin: str) -> None:
    assert _git(install, "rev-parse", "HEAD") == pin
    assert _git(install, "rev-parse", "HEAD^{tree}") == _git(
        origin, "rev-parse", f"{pin}^{{tree}}"
    )
    assert _git(install, "status", "--porcelain") == ""
    assert (
        subprocess.run(
            ["git", "symbolic-ref", "-q", "HEAD"], cwd=install, capture_output=True
        ).returncode
        != 0
    )


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("present", [False, True])
def test_shallow_install_pins_exact_tree_without_fetching_ancestors(
    tmp_path: Path,
    origin: tuple[Path, list[str], str],
    existing: bool,
    present: bool,
) -> None:
    remote, commits, _ = origin
    install = tmp_path / "install"
    pin = commits[-1] if present else commits[-2]
    if existing:
        _git(
            tmp_path,
            "clone",
            "--depth",
            "1",
            "--branch",
            "main",
            remote.as_uri(),
            str(install),
        )

    result = _install(tmp_path, remote, install, pin)

    assert result.returncode == 0, result.stdout + result.stderr
    assert '"ok":true' in result.stdout
    _assert_exact_tree(install, remote, pin)
    assert _git(install, "rev-parse", "--is-shallow-repository") == "true"
    # Every requested tree is complete, but its unneeded parent history stays absent.
    assert _git(install, "rev-list", "HEAD", "--count") == "1"


@pytest.mark.parametrize("present", [False, True])
def test_full_install_retains_history_when_pinning(
    tmp_path: Path,
    origin: tuple[Path, list[str], str],
    present: bool,
) -> None:
    remote, commits, side = origin
    install = tmp_path / "install"
    _git(
        tmp_path,
        "clone",
        "--single-branch",
        "--branch",
        "main",
        remote.as_uri(),
        str(install),
    )
    pin = commits[-2] if present else side
    if not present:
        assert (
            subprocess.run(
                ["git", "cat-file", "-e", pin], cwd=install, capture_output=True
            ).returncode
            != 0
        )

    result = _install(tmp_path, remote, install, pin)

    assert result.returncode == 0, result.stdout + result.stderr
    _assert_exact_tree(install, remote, pin)
    assert _git(install, "rev-parse", "--is-shallow-repository") == "false"
    assert _git(install, "rev-list", "HEAD", "--count") == _git(
        remote, "rev-list", pin, "--count"
    )
    for commit in commits:
        _git(install, "cat-file", "-e", f"{commit}^{{commit}}")


def test_missing_pin_fails_repository_stage(
    tmp_path: Path,
    origin: tuple[Path, list[str], str],
) -> None:
    remote, commits, _ = origin
    install = tmp_path / "install"

    result = _install(tmp_path, remote, install, "f" * 40)

    assert result.returncode != 0
    assert '"ok":false' in result.stdout
    assert _git(install, "rev-parse", "HEAD") == commits[-1]


def test_checkout_failure_fails_stage_and_preserves_untracked_file(
    tmp_path: Path,
    origin: tuple[Path, list[str], str],
) -> None:
    remote, commits, side = origin
    install = tmp_path / "install"
    _git(
        tmp_path,
        "clone",
        "--depth",
        "1",
        "--branch",
        "main",
        remote.as_uri(),
        str(install),
    )
    (install / "side.txt").write_text("keep my local file\n")

    result = _install(tmp_path, remote, install, side)

    assert result.returncode != 0
    assert '"ok":false' in result.stdout
    assert (
        "untracked working tree files would be overwritten by checkout" in result.stderr
    )
    assert (install / "side.txt").read_text() == "keep my local file\n"
    assert _git(install, "rev-parse", "HEAD") == commits[-1]
