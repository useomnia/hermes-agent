"""Filesystem identity for installations with split system/custom skills.

Hermes historically stores every local skill below ``$HERMES_HOME/skills``.
That remains the default. Deployments that set ``skills.source_layout: split``
use two explicit children instead::

    skills/system/<category>/<skill>/SKILL.md
    skills/custom/<category>/<skill>/SKILL.md

The split is a policy boundary, not provenance metadata: system skills are
release-owned and immutable at runtime; custom skills are user-owned and are
the only local skills the manager and curator may mutate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

SkillSource = Literal["system", "custom", "legacy", "external"]

SPLIT_SOURCE_LAYOUT = "split"
SYSTEM_SOURCE = "system"
CUSTOM_SOURCE = "custom"


def split_source_layout_enabled() -> bool:
    """Whether the active profile uses the source-qualified directory layout."""
    try:
        from hermes_cli.config import cfg_get, load_config

        value = cfg_get(load_config(), "skills", "source_layout")
    except Exception:
        return False
    return str(value or "").strip().lower() == SPLIT_SOURCE_LAYOUT


def system_skills_dir(skills_root: Path) -> Path:
    return skills_root / SYSTEM_SOURCE if split_source_layout_enabled() else skills_root


def custom_skills_dir(skills_root: Path) -> Path:
    return skills_root / CUSTOM_SOURCE if split_source_layout_enabled() else skills_root


def classify_skill_path(skill_dir: Path, skills_root: Path) -> SkillSource:
    """Classify a local skill directory from its structural source boundary."""
    try:
        rel = skill_dir.resolve().relative_to(skills_root.resolve())
    except (OSError, ValueError):
        return "external"
    if split_source_layout_enabled() and rel.parts:
        if rel.parts[0] == SYSTEM_SOURCE:
            return "system"
        if rel.parts[0] == CUSTOM_SOURCE:
            return "custom"
    return "legacy"


def skill_relative_path(skill_dir: Path, skills_root: Path) -> str:
    """Return a stable source-relative path for a skill directory."""
    rel = skill_dir.resolve().relative_to(skills_root.resolve())
    if (
        split_source_layout_enabled()
        and rel.parts
        and rel.parts[0]
        in {
            SYSTEM_SOURCE,
            CUSTOM_SOURCE,
        }
    ):
        rel = Path(*rel.parts[1:])
    return rel.as_posix()


def canonical_skill_id(skill_dir: Path, skills_root: Path) -> str:
    """Return ``source:path`` in split mode and the historical name otherwise."""
    source = classify_skill_path(skill_dir, skills_root)
    if source in {"system", "custom"}:
        return f"{source}:{skill_relative_path(skill_dir, skills_root)}"
    return skill_dir.name


def direct_skill_path(identifier: str, skills_root: Path) -> Path | None:
    """Resolve an explicitly qualified/path identifier without a recursive scan."""
    raw = (identifier or "").strip()
    if not raw:
        return None
    if ":" in raw:
        source, rel = raw.split(":", 1)
        if source not in {SYSTEM_SOURCE, CUSTOM_SOURCE} or not rel:
            return None
        source_root = skills_root / source
        candidate = source_root / rel
    elif "/" in raw:
        if split_source_layout_enabled() and not raw.startswith((
            f"{SYSTEM_SOURCE}/",
            f"{CUSTOM_SOURCE}/",
        )):
            candidate = custom_skills_dir(skills_root) / raw
        else:
            candidate = skills_root / raw
    else:
        return None
    try:
        resolved = candidate.resolve()
        boundary = source_root if ":" in raw else skills_root
        resolved.relative_to(boundary.resolve())
    except (OSError, ValueError):
        return None
    return resolved
