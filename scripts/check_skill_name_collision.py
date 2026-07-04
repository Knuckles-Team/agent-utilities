#!/usr/bin/env python3
"""Fleet gate: skill + prompt names are globally unique across the whole environment.

CONCEPT:AU-OS.deployment.agent-factory-autoload / OS-5.82. When every package's
``agent_utilities.skill_providers`` skills are installed together with the
``universal-skills`` atomic skills and the hub's own ``agent_utilities/skills``,
two skills that share a ``name:`` frontmatter value would shadow each other in the
agent's skill directory. This gate makes that a build break.

Scope — the *installable* skill namespace only:

  * ``agents/<pkg>/<module>/skills/**/SKILL.md``   (fleet skill_providers)
  * ``skills/universal-skills/**/SKILL.md``          (universal atomic skills)
  * ``agent-utilities/agent_utilities/skills/**/SKILL.md`` (hub-own skills)

``skill_graphs`` / ``skill-graphs`` directories are KG-ingestion **reference
corpora** (the same skill legitimately appears in many bundles), not installed
skills, so they are excluded from the collision namespace.

Two checks:
  1. **Uniqueness (hard).** No two installable SKILL.md share a ``name:``. This is
     the guarantee "install every skill, no overlap" the fleet needs. Baseline-gated
     like ``check_prompt_schema`` / ``check_no_env_sprawl`` so a pre-existing dup can
     be grandfathered while it is burned down.
  2. **Prompt task uniqueness (hard).** Within one package's ``prompts/`` no two
     blueprints share a ``task`` (they would collide under ``prompt:<source>/<task>``).

Plus an **advisory** (report-only): an ``agents/<pkg>`` skill whose ``name:`` is not
prefixed with the package slug is a collision *risk* — a package-slug prefix
(``<pkg>-<capability>``) is the recommended convention. Capability-domain names
(e.g. ``dns-record-manager``, ``ipmi-bmc-manager``) are allowed as long as they stay
globally unique, so this is a nudge, not a failure.

Usage::

    python scripts/check_skill_name_collision.py               # report (baseline-gated)
    python scripts/check_skill_name_collision.py --strict      # fail on ANY collision
    python scripts/check_skill_name_collision.py --convention  # also fail on prefix advisory
    python scripts/check_skill_name_collision.py --update-baseline
    python scripts/check_skill_name_collision.py --root /path/to/agent-packages
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

_NAME_RE = re.compile(r"^name:\s*(.+?)\s*$", re.MULTILINE)
_TASK_RE = re.compile(r'"task"\s*:\s*"([^"]+)"')
_EXCLUDE_SEGMENTS = ("skill_graphs", "skill-graphs")

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "scripts" / "skill_collision_baseline.txt"


def _find_fleet_root() -> Path | None:
    """Locate the ``agent-packages`` root (has both ``agents/`` and ``skills/``)."""
    override = os.environ.get("AGENT_PACKAGES_ROOT")
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    # Walk up from this repo and from cwd.
    for start in (REPO, Path.cwd()):
        for p in (start, *start.parents):
            candidates.append(p)
    candidates.append(Path("/home/apps/workspace/agent-packages"))
    for c in candidates:
        if (c / "agents").is_dir() and (c / "skills").is_dir():
            return c
    return None


def _clean_name(raw: str) -> str:
    return raw.strip().strip("'").strip('"').strip()


def _iter_skill_files(root: Path) -> list[Path]:
    roots = [
        root / "agents",
        root / "skills" / "universal-skills",
        root / "agent-utilities" / "agent_utilities" / "skills",
    ]
    files: list[Path] = []
    for r in roots:
        if not r.is_dir():
            continue
        for f in r.rglob("SKILL.md"):
            if any(seg in _EXCLUDE_SEGMENTS for seg in f.parts):
                continue
            files.append(f)
    return sorted(files)


def _pkg_slug(path: Path, root: Path) -> str | None:
    """Return the agents/<pkg> slug for a skill under agents/, else None."""
    try:
        rel = path.relative_to(root / "agents")
    except ValueError:
        return None
    return rel.parts[0] if rel.parts else None


def _short(slug: str) -> str:
    """Package slug with its last ``-token`` dropped (servicenow-api -> servicenow).

    Mirrors ``retrofit_fleet_contribution.py``'s ``short = repo_name.rsplit("-",1)[0]``
    derivation exactly, so the gate accepts precisely the ``<short>-starter`` prefix
    the scaffolder emits (audio-transcriber -> audio, media-downloader -> media).
    """
    return slug.rsplit("-", 1)[0] if "-" in slug else slug


def _load_baseline() -> set[str]:
    if not BASELINE.exists():
        return set()
    return {
        ln.strip()
        for ln in BASELINE.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    }


def scan(
    root: Path,
) -> tuple[dict[str, list[Path]], list[tuple[str, Path]], dict[str, list[Path]]]:
    """Return (name->paths, convention_offenders, prompt_task collisions)."""
    by_name: dict[str, list[Path]] = defaultdict(list)
    convention: list[tuple[str, Path]] = []
    for f in _iter_skill_files(root):
        m = _NAME_RE.search(f.read_text(encoding="utf-8"))
        if not m:
            continue
        name = _clean_name(m.group(1))
        by_name[name].append(f)
        slug = _pkg_slug(f, root)
        if slug is not None:
            short = _short(slug)
            if not (name.startswith(slug) or name.startswith(short)):
                convention.append((name, f))

    # Prompt task collisions within a single package's prompts/ dir.
    prompt_dups: dict[str, list[Path]] = {}
    for pkg_prompts in (root / "agents").glob("*/*/prompts"):
        seen: dict[str, list[Path]] = defaultdict(list)
        for pj in pkg_prompts.glob("*.json"):
            mt = _TASK_RE.search(pj.read_text(encoding="utf-8"))
            if mt:
                seen[mt.group(1)].append(pj)
        for task, paths in seen.items():
            if len(paths) > 1:
                prompt_dups[f"{pkg_prompts.parent.name}/{task}"] = paths
    return by_name, convention, prompt_dups


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="agent-packages fleet root (auto-detected).")
    parser.add_argument("--strict", action="store_true", help="Fail on any collision.")
    parser.add_argument(
        "--convention", action="store_true", help="Also fail on the prefix advisory."
    )
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else _find_fleet_root()
    if root is None:
        print(
            "WARNING: could not locate agent-packages fleet root "
            "(no dir with both agents/ and skills/); skipping collision scan.",
            file=sys.stderr,
        )
        return 0

    by_name, convention, prompt_dups = scan(root)
    collisions = {n: ps for n, ps in by_name.items() if len(ps) > 1}

    if args.update_baseline:
        BASELINE.write_text(
            "# Skill-name collisions grandfathered by check_skill_name_collision.\n"
            "# Burn this down to empty. CONCEPT:AU-OS.deployment.agent-factory-autoload\n"
            + "".join(f"{n}\n" for n in sorted(collisions)),
            encoding="utf-8",
        )
        print(f"Wrote baseline with {len(collisions)} entries.")
        return 0

    baseline = _load_baseline()
    new_collisions = {n: ps for n, ps in collisions.items() if n not in baseline}

    if collisions:
        print(f"Skill-name collisions ({len(collisions)}):")
        for n, ps in sorted(collisions.items()):
            tag = "" if n not in baseline else "  (baseline)"
            print(f"  {n}{tag}:")
            for p in ps:
                print(f"      {p.relative_to(root)}")
    if prompt_dups:
        print(f"\nPrompt task collisions ({len(prompt_dups)}):")
        for k, ps in sorted(prompt_dups.items()):
            print(f"  {k}: {', '.join(str(p.relative_to(root)) for p in ps)}")
    if convention:
        print(
            f"\nPrefix advisory — {len(convention)} agents/* skill(s) not package-prefixed:"
        )
        for name, p in sorted(convention):
            print(f"  {name}  ({p.relative_to(root)})")

    hard_fail = bool(prompt_dups) or (collisions if args.strict else new_collisions)
    if args.convention and convention:
        hard_fail = True
    if hard_fail:
        parts = []
        if collisions:
            parts.append(f"{len(new_collisions)} new skill collision(s)")
        if prompt_dups:
            parts.append(f"{len(prompt_dups)} prompt-task collision(s)")
        if args.convention and convention:
            parts.append(f"{len(convention)} convention violation(s)")
        print(f"\nFAIL: {', '.join(parts)}.", file=sys.stderr)
        return 1

    scanned = sum(len(ps) for ps in by_name.values())
    print(
        f"\nOK: {scanned} installable skills, {len(by_name)} unique names, "
        f"{len(collisions)} grandfathered, 0 new collisions "
        f"({len(convention)} prefix advisories)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
