"""Behavioral coverage for the additive exact-source installer mode."""

from __future__ import annotations

import io
import os
import subprocess
import tarfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
REVISION = "0123456789abcdef0123456789abcdef01234567"


def _source_archive(
    path: Path,
    *,
    revision: str = REVISION,
    symlink: bool = False,
    traversal: bool = False,
) -> None:
    with tarfile.open(path, "w:gz") as archive:
        manifest = f"format=hermes-source-v1\nrevision={revision}\n".encode()
        info = tarfile.TarInfo(".hermes-source-manifest")
        info.mode = 0o644
        info.mtime = 0
        info.size = len(manifest)
        archive.addfile(info, io.BytesIO(manifest))

        if symlink:
            link = tarfile.TarInfo("unsafe-link")
            link.type = tarfile.SYMTYPE
            link.linkname = "/etc/passwd"
            archive.addfile(link)
            return

        if traversal:
            unsafe = tarfile.TarInfo("../escape.txt")
            unsafe.mode = 0o644
            unsafe.mtime = 0
            unsafe.size = 1
            archive.addfile(unsafe, io.BytesIO(b"x"))
            return

        source = b"print('archive source')\n"
        info = tarfile.TarInfo("hermes_cli/archive_fixture.py")
        info.mode = 0o755
        info.mtime = 0
        info.size = len(source)
        archive.addfile(info, io.BytesIO(source))


def _run_repository_stage(
    tmp_path: Path,
    archive: Path,
    install_dir: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HERMES_HOME": str(tmp_path / "home")}
    return subprocess.run(
        [
            "/bin/bash",
            str(INSTALL_SH),
            "--stage",
            "repository",
            "--source-archive",
            str(archive),
            "--commit",
            REVISION,
            "--dir",
            str(install_dir),
            "--json",
            *extra,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


def test_source_archive_stage_extracts_exact_tree_and_stamps_archive_method(tmp_path: Path) -> None:
    archive = tmp_path / "hermes.tar.gz"
    install_dir = tmp_path / "install"
    _source_archive(archive)

    result = _run_repository_stage(tmp_path, archive, install_dir)

    assert result.returncode == 0, result.stderr
    assert '"ok":true,"stage":"repository"' in result.stdout
    assert (install_dir / "hermes_cli/archive_fixture.py").read_text() == "print('archive source')\n"
    assert (install_dir / ".install_method").read_text() == "archive\n"


def test_source_archive_requires_matching_exact_commit(tmp_path: Path) -> None:
    archive = tmp_path / "hermes.tar.gz"
    install_dir = tmp_path / "install"
    _source_archive(archive, revision="fedcba9876543210fedcba9876543210fedcba98")

    result = _run_repository_stage(tmp_path, archive, install_dir)

    assert result.returncode != 0
    assert not install_dir.exists()
    assert "manifest does not match" in (result.stdout + result.stderr)


def test_source_archive_rejects_links_before_extraction(tmp_path: Path) -> None:
    archive = tmp_path / "hermes.tar.gz"
    install_dir = tmp_path / "install"
    _source_archive(archive, symlink=True)

    result = _run_repository_stage(tmp_path, archive, install_dir)

    assert result.returncode != 0
    assert not install_dir.exists()
    assert "link or special entry" in (result.stdout + result.stderr)


def test_source_archive_rejects_traversal_before_extraction(tmp_path: Path) -> None:
    archive = tmp_path / "hermes.tar.gz"
    install_dir = tmp_path / "install"
    _source_archive(archive, traversal=True)

    result = _run_repository_stage(tmp_path, archive, install_dir)

    assert result.returncode != 0
    assert not install_dir.exists()
    assert "unsafe path" in (result.stdout + result.stderr)


def test_source_archive_never_replaces_existing_git_checkout(tmp_path: Path) -> None:
    archive = tmp_path / "hermes.tar.gz"
    install_dir = tmp_path / "install"
    _source_archive(archive)
    install_dir.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=install_dir, check=True)
    (install_dir / "user-file.txt").write_text("keep me")

    result = _run_repository_stage(tmp_path, archive, install_dir)

    assert result.returncode != 0
    assert "existing Git checkout" in (result.stdout + result.stderr)
    assert (install_dir / "user-file.txt").read_text() == "keep me"


def test_source_archive_refuses_filesystem_root(tmp_path: Path) -> None:
    archive = tmp_path / "hermes.tar.gz"
    _source_archive(archive)

    result = _run_repository_stage(tmp_path, archive, Path("/"))

    assert result.returncode != 0
    assert "filesystem root" in (result.stdout + result.stderr)


def test_source_archive_atomically_replaces_a_managed_archive_install(tmp_path: Path) -> None:
    archive = tmp_path / "hermes.tar.gz"
    install_dir = tmp_path / "install"
    _source_archive(archive)
    first = _run_repository_stage(tmp_path, archive, install_dir)
    assert first.returncode == 0, first.stderr
    (install_dir / "obsolete.txt").write_text("old")

    second = _run_repository_stage(tmp_path, archive, install_dir)

    assert second.returncode == 0, second.stderr
    assert not (install_dir / "obsolete.txt").exists()
    assert list(tmp_path.glob("install.archive-old-*")) == []
