"""Contained, bounded support-file discovery for native directory skills."""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path

from tools.path_security import validate_within_dir


MAX_INVENTORY_ENTRIES = 4_096
MAX_INVENTORY_FILES = 256
MAX_INVENTORY_CHARS = 12_000
_SUPPORT_DIRECTORIES = {"references", "templates", "assets", "scripts"}
_OTHER_EXTENSIONS = {".md", ".py", ".yaml", ".yml", ".json", ".tex", ".sh"}


@dataclass
class SkillFileInventory:
    files: dict[str, list[str]] = field(default_factory=dict)
    examined_entries: int = 0
    limit: str | None = None

    def response_fields(self, name: str, files_key: str) -> dict:
        paths = [path for group in self.files.values() for path in group]
        example = {"name": name, "file_path": min(paths)} if paths else None
        hint = None
        if example:
            hint = (
                f"Load a listed file with skill_view(name={json.dumps(name, ensure_ascii=False)}, "
                f"file_path={json.dumps(example['file_path'], ensure_ascii=False)}). "
                "Use each listed path exactly; do not add a directory prefix."
            )
        return {
            files_key: self.files if files_key == "available_files" else self.files or None,
            "file_inventory": {
                "complete": self.limit is None,
                "examined_entries": self.examined_entries,
                "listed_files": len(paths),
                "limit": self.limit,
            },
            "file_example": example,
            "usage_hint": hint,
        }

    def add(self, path: str) -> bool:
        if sum(len(group) for group in self.files.values()) >= MAX_INVENTORY_FILES:
            self.limit = "files"
            return False
        directory = path.split("/", 1)[0]
        category = (
            directory if "/" in path and directory in _SUPPORT_DIRECTORIES else "other"
        )
        group = self.files.setdefault(category, [])
        group.append(path)
        if len(json.dumps(self.files, ensure_ascii=False)) > MAX_INVENTORY_CHARS:
            group.pop()
            if not group:
                del self.files[category]
            self.limit = "characters"
            return False
        return True


def _directory_entries(directory: Path, inventory: SkillFileInventory) -> list[os.DirEntry[str]]:
    # Process a directory only after its complete entry list fits the scan
    # budget, so a partial scandir order cannot choose the advertised subset.
    entries = []
    with os.scandir(directory) as stream:
        for entry in stream:
            if inventory.examined_entries >= MAX_INVENTORY_ENTRIES:
                inventory.limit = "entries"
                return []
            inventory.examined_entries += 1
            entries.append(entry)
    return sorted(entries, key=lambda entry: entry.name)


def _support_path(entry: os.DirEntry[str], root: Path) -> str | None:
    if entry.name == "SKILL.md" or not entry.is_file():
        return None
    path = Path(entry.path)
    relative = path.relative_to(root).as_posix()
    directory = relative.split("/", 1)[0]
    if not ("/" in relative and directory in _SUPPORT_DIRECTORIES) and (
        path.suffix not in _OTHER_EXTENSIONS
    ):
        return None
    if validate_within_dir(path, root):
        return None
    return relative


def discover_skill_files(root: Path | None) -> SkillFileInventory:
    inventory = SkillFileInventory()
    if root is None:
        return inventory
    pending = [root]
    try:
        while pending and inventory.limit is None:
            entries = _directory_entries(pending.pop(), inventory)
            children = []
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    children.append(Path(entry.path))
                    continue
                relative = _support_path(entry, root)
                if relative is not None and not inventory.add(relative):
                    break
            pending.extend(reversed(children))
    except OSError:
        inventory.limit = "unreadable"
    inventory.files = {
        category: sorted(paths) for category, paths in sorted(inventory.files.items())
    }
    return inventory
