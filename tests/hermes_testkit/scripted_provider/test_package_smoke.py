"""Smoke the packaged scripted-provider module without a network install."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
UV = shutil.which("uv")


@pytest.mark.skipif(UV is None, reason="uv is required for the local package smoke")
def test_wheel_artifact_exposes_scripted_provider_cli(tmp_path: Path) -> None:
    assert UV is not None
    uv = UV
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for filename in ("setup.py", "pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(PROJECT_ROOT / filename, source_dir / filename)
    shutil.copytree(PROJECT_ROOT / "hermes_testkit", source_dir / "hermes_testkit")
    build_env = os.environ.copy()
    # The repository intentionally guards regular wheels; this is the same
    # explicit opt-in used by its Nix build, kept local and dependency-free.
    build_env["HERMES_NIX_BUILD"] = "1"
    build_env.pop("PYTHONPATH", None)
    isolated_env = os.environ.copy()
    isolated_env.pop("HERMES_NIX_BUILD", None)
    isolated_env.pop("PYTHONPATH", None)
    build = subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_wheel; "
            f"build_wheel(r'{wheel_dir}')",
        ],
        cwd=source_dir,
        env=build_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1

    venv_dir = tmp_path / "venv"
    create_venv = subprocess.run(
        [uv, "venv", str(venv_dir), "--python", sys.executable],
        env=isolated_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert create_venv.returncode == 0, create_venv.stderr
    venv_python = (
        venv_dir / "Scripts" / "python.exe"
        if os.name == "nt"
        else venv_dir / "bin" / "python"
    )
    install = subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--python",
            str(venv_python),
            str(wheels[0]),
        ],
        env=isolated_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert install.returncode == 0, install.stderr

    cli = subprocess.run(
        [str(venv_python), "-m", "hermes_testkit.scripted_provider", "--help"],
        cwd=tmp_path,
        env=isolated_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert cli.returncode == 0, cli.stderr
    assert "--script" in cli.stdout
