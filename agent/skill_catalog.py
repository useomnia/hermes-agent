"""One native skill identity projection, with no implicit source precedence.

Callers supply the already active roots and retain their own offer/read gates.
Repository instructions in a remote Toolbox are deliberately not roots here.
"""

import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath

from agent.skill_utils import (
    is_skill_support_path,
    iter_skill_index_files,
    parse_frontmatter,
)


def skill_lookup_path_error(name):
    """Shared read/write validation before any identifier joins a root."""
    from tools.path_security import has_traversal_component

    if not isinstance(name, str):
        return "Skill name must be a string."
    candidate = name.strip()
    if (PurePosixPath(candidate).is_absolute() or PureWindowsPath(candidate).is_absolute()
            or PureWindowsPath(candidate).drive):
        return "Skill name must be a relative path within the skills directory."
    if has_traversal_component(candidate):
        return "Skill name cannot contain '..' path traversal components."
    if "\x00" in candidate:
        return "Skill name cannot contain a null byte."
    return None


def is_package_owned_markdown(path: Path, root: Path) -> bool:
    """A Markdown file below SKILL.md belongs to that package, not discovery."""
    parent = path.parent
    while parent != root and parent.is_relative_to(root):
        if (parent / "SKILL.md").is_file():
            return True
        parent = parent.parent
    return (root / "SKILL.md").is_file()


def deduplicate_same_root_skills(candidates, roots):
    """Keep the existing byte-identical, strictly shallower same-root winner."""
    if len(candidates) < 2:
        return candidates
    owners = {
        max((root for root in roots if path.is_relative_to(root)),
            key=lambda root: len(root.parts), default=None)
        for _, path in candidates
    }
    if len(owners) != 1 or None in owners:
        return candidates
    try:
        if len({hashlib.sha256(path.read_bytes()).digest() for _, path in candidates}) != 1:
            return candidates
    except OSError:
        return candidates
    root = owners.pop()

    def rank(candidate):
        path = candidate[1]
        return path.name != "SKILL.md", len(path.relative_to(root).parts)

    ranked = sorted(candidates, key=rank)
    return [ranked[0]] if rank(ranked[0]) != rank(ranked[1]) else candidates


class SkillCatalog:
    """An immutable per-operation candidate set used for lookup and projection."""

    def __init__(self, roots):
        self.roots = list(dict.fromkeys(Path(root) for root in roots))
        self.aliases = {}
        self.entries = {}
        for root in self.roots:
            if not root.exists():
                continue
            for path in iter_skill_index_files(root, "SKILL.md"):
                try:
                    frontmatter, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError):
                    frontmatter = {}
                relative = path.parent.relative_to(root).as_posix()
                names = {path.parent.name, relative}
                declared = frontmatter.get("name")
                if not isinstance(declared, str) or not declared:
                    declared = path.parent.name
                names.add(declared)
                candidate = (path.parent, path)
                self.entries[path] = (declared or path.parent.name, relative)
                for name in names:
                    self.aliases.setdefault(name, []).append(candidate)
            # Preserve real standalone Markdown, without promoting package docs.
            for path in root.rglob("*.md"):
                if (path.name == "SKILL.md" or is_skill_support_path(path)
                        or is_package_owned_markdown(path, root)):
                    continue
                relative = path.relative_to(root).with_suffix("").as_posix()
                for name in {path.stem, relative}:
                    self.aliases.setdefault(name, []).append((None, path))

    def select(self, name, *, fallback=None):
        """Return all equally eligible matches; callers must refuse ambiguity."""
        candidates = []
        seen = set()

        def record(candidate):
            try:
                key = candidate[1].resolve()
            except (OSError, RuntimeError):
                key = candidate[1]
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)

        for identifier in (name, fallback):
            if not identifier or skill_lookup_path_error(identifier):
                continue
            for candidate in self.aliases.get(identifier, []):
                record(candidate)
            # Exact reads historically accept explicit excluded paths and skill
            # symlinks. Offer filters do not become additional read restrictions.
            for root in self.roots:
                direct = root / identifier
                if (not is_skill_support_path(direct) and direct.is_dir()
                        and (direct / "SKILL.md").is_file()):
                    record((direct, direct / "SKILL.md"))
                flat = direct.with_suffix(".md")
                if (flat.is_file() and not is_skill_support_path(flat)
                        and not is_package_owned_markdown(flat, root)):
                    record((None, flat))
        return deduplicate_same_root_skills(candidates, self.roots)

    def load_name(self, path):
        """Choose an advertised name only if lookup returns this exact entry."""
        path = Path(path)
        names = self.entries.get(path)
        if names is None:
            return None
        for name in dict.fromkeys(names):
            # A native package must not advertise the plugin dispatch spelling.
            # Its ordinary relative directory remains a possible native name.
            if not isinstance(name, str) or ":" in name or skill_lookup_path_error(name):
                continue
            matches = self.select(name)
            if len(matches) == 1 and matches[0][1] == path:
                return name
        return None


def resolve_skill_catalog(entries, roots):
    """Project visible metadata onto identifiers accepted by native reads."""
    catalog = SkillCatalog(roots)
    result = []
    for entry in entries:
        name = catalog.load_name(entry["skill_md_path"])
        if name:
            result.append({**entry, "name": name})
    return result
