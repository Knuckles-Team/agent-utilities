"""Fleet-wide ``SKILL.md`` discovery.

Walks a set of repository roots and returns one :class:`SkillRecord` per
``SKILL.md`` found, skipping VCS/build/venv noise so a repo checkout with an
active ``.venv`` (which itself vendors third-party packages that ship their
own ``SKILL.md`` fixtures) doesn't pollute the fleet inventory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Directory name components that mean "not part of this repo's own tree".
_SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        "build",
        "dist",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".cargo-target",
        ".hypothesis",
    }
)


def _is_noise(relative_path: Path) -> bool:
    for part in relative_path.parts:
        if part in _SKIP_DIR_NAMES:
            return True
        if part.endswith(".egg-info") or part.endswith(".dist-info"):
            return True
    return False


@dataclass(frozen=True)
class SkillRecord:
    """One discovered ``SKILL.md`` and the repo it belongs to."""

    skill_md: Path
    skill_dir: Path
    repo_root: Path
    repo_name: str

    @property
    def relative_path(self) -> str:
        """Path relative to the owning repo root, for stable reporting."""
        return str(self.skill_md.relative_to(self.repo_root))

    @property
    def directory_name(self) -> str:
        return self.skill_dir.name


def discover_skills(roots: list[Path]) -> list[SkillRecord]:
    """Discover every ``SKILL.md`` under ``roots``.

    Roots are scanned independently; a symlink cycle or a root reachable via
    two different paths is de-duplicated by resolved ``SKILL.md`` path.
    Returns records sorted by ``(repo_name, relative_path)`` for a
    deterministic report ordering.
    """
    records: list[SkillRecord] = []
    seen: set[Path] = set()
    for raw_root in roots:
        root = raw_root.resolve()
        if not root.is_dir():
            continue
        for skill_md in root.rglob("SKILL.md"):
            if not skill_md.is_file():
                continue
            relative = skill_md.relative_to(root)
            if _is_noise(relative):
                continue
            resolved = skill_md.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            records.append(
                SkillRecord(
                    skill_md=resolved,
                    skill_dir=resolved.parent,
                    repo_root=root,
                    repo_name=root.name,
                )
            )
    records.sort(key=lambda r: (r.repo_name, r.relative_path))
    return records
