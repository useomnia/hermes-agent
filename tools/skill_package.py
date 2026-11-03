"""Bounded, complete skill-package copies into one active profile.

Unlike support inventories, this path rejects omissions, links and special
files. It preserves binary support files and publishes a staged directory
with an operating-system no-replace rename.
"""

import ctypes
import errno
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping

MAX_PACKAGE_FILES = 256
MAX_PACKAGE_BYTES = 4 * 1024 * 1024
MAX_PACKAGE_FILE_BYTES = 1024 * 1024
MAX_PACKAGE_ENTRIES = 4096


class PackageError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def validate_files(files):
    """Validate a complete byte mapping without normalizing unsafe spellings."""
    from tools.skills_hub import _validate_bundle_rel_path

    if not isinstance(files, Mapping) or not files:
        raise PackageError("invalid_package", "A complete nonempty package file mapping is required.")
    if len(files) > MAX_PACKAGE_FILES:
        raise PackageError("package_limit", f"Package exceeds {MAX_PACKAGE_FILES} files.")
    checked = {}
    total = 0
    for name, data in files.items():
        try:
            normalized = _validate_bundle_rel_path(name)
        except (TypeError, ValueError) as exc:
            raise PackageError("unsafe_package", str(exc)) from exc
        if (normalized != name or "\\" in name or "\x00" in name
                or any(part in {"", ".", ".."} for part in name.split("/"))):
            raise PackageError("unsafe_package", f"Non-canonical package path: {name!r}.")
        if not isinstance(data, bytes):
            raise PackageError("invalid_package", f"Package file {name!r} must contain bytes.")
        if len(data) > MAX_PACKAGE_FILE_BYTES:
            raise PackageError("package_limit", f"Package file {name!r} exceeds {MAX_PACKAGE_FILE_BYTES} bytes.")
        total += len(data)
        if total > MAX_PACKAGE_BYTES:
            raise PackageError("package_limit", f"Package exceeds {MAX_PACKAGE_BYTES} bytes.")
        checked[name] = data
    if "SKILL.md" not in checked:
        raise PackageError("invalid_package", "The complete package must include root SKILL.md.")
    for name in checked:
        if any(parent.as_posix() in checked for parent in PurePosixPath(name).parents if str(parent) != "."):
            raise PackageError("unsafe_package", f"A package file is also a parent directory of {name!r}.")
    return checked


def collect_package(directory: Path, root: Path):
    """Read every package file using no-follow descriptor-relative traversal."""
    if not all(hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW")):
        raise PackageError("copy_failed", "Safe local package collection is unsupported on this platform.")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(root, flags)
    try:
        for part in directory.relative_to(root).parts:
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        files = {}
        entries = 0
        total = 0

        def visit(current, prefix=""):
            nonlocal entries, total
            directory_before = os.fstat(current)
            with os.scandir(current) as listing:
                for entry in listing:
                    entries += 1
                    if entries > MAX_PACKAGE_ENTRIES:
                        raise PackageError("package_limit", "Package enumeration exceeded its entry limit.")
                    relative = prefix + entry.name
                    before = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(before.st_mode):
                        child_fd = os.open(entry.name, flags, dir_fd=current)
                        try:
                            visit(child_fd, relative + "/")
                        finally:
                            os.close(child_fd)
                    elif stat.S_ISREG(before.st_mode):
                        if len(files) >= MAX_PACKAGE_FILES or before.st_size > MAX_PACKAGE_FILE_BYTES:
                            raise PackageError("package_limit", "Package exceeded a file count or size limit.")
                        file_fd = os.open(entry.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=current)
                        try:
                            opened = os.fstat(file_fd)
                            if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                                raise PackageError("unsafe_package", "Package changed during collection.")
                            with os.fdopen(file_fd, "rb", closefd=False) as stream:
                                data = stream.read(MAX_PACKAGE_FILE_BYTES + 1)
                            after = os.fstat(file_fd)
                            if (len(data) > MAX_PACKAGE_FILE_BYTES or after.st_size != len(data)
                                    or (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                                    != (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)):
                                raise PackageError("invalid_package", "Package file changed or exceeded its limit during collection.")
                        finally:
                            os.close(file_fd)
                        total += len(data)
                        if total > MAX_PACKAGE_BYTES:
                            raise PackageError("package_limit", "Package exceeded its total byte limit.")
                        files[relative] = data
                    else:
                        raise PackageError("unsafe_package", f"Package contains a symlink or special file: {relative}.")
            directory_after = os.fstat(current)
            if (directory_before.st_mtime_ns, directory_before.st_ctime_ns) != (directory_after.st_mtime_ns, directory_after.st_ctime_ns):
                raise PackageError("invalid_package", "Package directory changed during collection.")
        visit(fd)
        return validate_files(files)
    finally:
        os.close(fd)


def publish_package(staging: Path, destination: Path):
    """Atomically publish without replacing even an empty destination directory."""
    if os.name == "nt":
        os.rename(staging, destination)  # Windows rename refuses an existing target.
        return
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(staging)
    target_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        status = rename(source_bytes, target_bytes, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        status = rename(-100, source_bytes, -100, target_bytes, 1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise OSError(errno.ENOTSUP, "Atomic no-replace package publication is unsupported.")
    if status != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def stage_package(files, root, target_name, scan):
    """Build outside active discovery; an exception leaves no partial skill."""
    staging_root = root / ".hub"
    staging_root.mkdir(parents=True, exist_ok=True)
    # The metadata root is trusted, but never follow a caller-injected redirect.
    if staging_root.is_symlink() or not staging_root.resolve().is_relative_to(root.resolve()):
        raise PackageError("unsafe_package", "The package staging directory is redirected.")
    staging = Path(tempfile.mkdtemp(prefix="copy-", dir=staging_root))
    try:
        for name, data in files.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        error = scan(staging)
        if error:
            raise PackageError("security_scan", error)
        publish_package(staging, root / target_name)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
