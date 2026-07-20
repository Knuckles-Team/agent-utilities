#!/usr/bin/env python3
"""Collapse each provider's skill triggers into one progressive-disclosure skill.

The migration preserves every prior workflow beside its existing scripts,
references, and assets as ``WORKFLOW.md``.  Only the single
``<provider>-operations/SKILL.md`` remains discoverable.  Generated content is
privacy-sanitized and contains only repository-relative links.

The command is deterministic and dry-run by default::

    python scripts/consolidate_provider_skills.py --agents-root AGENTS
    python scripts/consolidate_provider_skills.py --agents-root AGENTS --apply

After applying, generate ``agents/openai.yaml`` for each remaining skill with
the skill-creator ``generate_openai_yaml.py`` helper, then run
``quick_validate.py`` over the resulting folders.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

_FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)
_URL = re.compile(r"(?i)\b(?:https?|file)://[^\s)>`]+")
_SKILL_NAME = re.compile(r"[^a-z0-9-]+")


@dataclass(frozen=True)
class LegacySkill:
    folder: Path
    name: str
    description: str
    body: str


def _parse_skill(path: Path) -> LegacySkill:
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER.match(text)
    if not match:
        raise ValueError("skill has no valid YAML frontmatter")
    raw = yaml.safe_load(match.group(1))
    if not isinstance(raw, dict):
        raise ValueError("skill frontmatter is not an object")
    name = str(raw.get("name") or path.parent.name).strip()
    description = str(raw.get("description") or "").strip()
    return LegacySkill(
        folder=path.parent,
        name=name,
        description=description,
        body=text[match.end() :].strip(),
    )


def _safe_name(provider: str) -> str:
    base = _SKILL_NAME.sub("-", provider.lower()).strip("-")
    suffix = "-operations"
    return (base[: 64 - len(suffix)].rstrip("-") + suffix).strip("-")


def _module_skills_dir(repo: Path) -> Path:
    candidates = sorted(
        path
        for path in repo.glob("*/skills")
        if path.is_dir() and not path.parent.name.startswith(".")
    )
    if len(candidates) == 1:
        return candidates[0]

    normalized = repo.name.replace("-", "_")
    preferred = repo / normalized / "skills"
    if preferred.parent.is_dir():
        return preferred

    packages = sorted(
        path
        for path in repo.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    )
    if len(packages) == 1:
        return packages[0] / "skills"
    raise ValueError("provider Python package could not be resolved uniquely")


def _sanitize(text: str, guard: PersistencePrivacyGuard) -> str:
    clean, _ = guard.sanitize_text(text)
    return _URL.sub("[configured-endpoint]", clean).strip()


def _description(provider: str, skills: list[LegacySkill]) -> str:
    topics = [skill.name.replace("-", " ") for skill in skills]
    visible = ", ".join(topics[:8])
    if len(topics) > 8:
        visible += ", and related workflows"
    scope = visible or "discovery, governed operations, ingestion, and verification"
    description = (
        f"Operate {provider} through its governed MCP and GraphOS capabilities, "
        f"including {scope}. Use when a request requires this provider's read, "
        "change, automation, ingestion, troubleshooting, or evidence workflows."
    )
    return description[:1024].rstrip()


def _main_skill(provider: str, name: str, description: str) -> str:
    title = provider.replace("-", " ").title()
    return f"""---
name: {name}
description: >-
  {description}
---

# {title} Operations

Use the provider's governed MCP tools through GraphOS delegation.

## Workflow

1. Establish the verified GraphSession and tenant before discovery or retrieval.
2. Discover the current condensed tool surface; never assume a stale tool name or schema.
3. Prefer read-only inspection first. For changes, present impact and use the provider's
   dry-run or preview mode when available.
4. Execute mutations as fenced WorkItems so retries remain idempotent and auditable.
5. Ingest source data only through the signed connector preset and ChangeEnvelope path.
6. Verify the durable result and its trace/evidence before reporting completion.

## Safety contract

- Never persist credentials, endpoints, raw personal identifiers, hostnames, or local paths.
- Resolve TLS trust and verification from environment/configuration; never hardcode bypasses.
- Treat unknown ACL, tenant, schema, or tool-contract state as a hard failure.
- Require explicit approval for destructive, externally visible, or irreversible actions.
- Keep runtime traces policy-scoped and privacy-sanitized.

## Specialized workflows

Read [the workflow catalog](references/catalog.md) only when the request needs a
provider-specific procedure, parameter map, script, or reference asset.
"""


def _workflow_document(skill: LegacySkill, guard: PersistencePrivacyGuard) -> str:
    title = skill.name.replace("-", " ").title()
    description = _sanitize(skill.description, guard)
    body = _sanitize(skill.body, guard)
    return f"# {title}\n\n{description}\n\n{body}\n"


def _catalog(
    target: Path, skills: list[LegacySkill], guard: PersistencePrivacyGuard
) -> str:
    lines = [
        "# Provider workflow catalog",
        "",
        "Load only the workflow relevant to the current request.",
        "",
    ]
    for skill in skills:
        workflow = skill.folder / "WORKFLOW.md"
        relative = Path(os.path.relpath(workflow, target / "references")).as_posix()
        summary = _sanitize(skill.description, guard).replace("\n", " ")
        lines.append(f"- [{skill.name}]({relative}): {summary}")
    lines.append("")
    return "\n".join(lines)


def consolidate_repo(repo: Path, *, apply: bool) -> tuple[int, Path]:
    skills_dir = _module_skills_dir(repo)
    legacy_paths = sorted(skills_dir.glob("*/SKILL.md"))
    target_name = _safe_name(repo.name)
    target = skills_dir / target_name

    # An idempotent re-run keeps the already-consolidated entry and its catalog.
    if len(legacy_paths) == 1 and legacy_paths[0].parent == target:
        return 1, target

    legacy = [_parse_skill(path) for path in legacy_paths]
    guard = PersistencePrivacyGuard()
    description = _description(repo.name, legacy)
    if not apply:
        return len(legacy), target

    skills_dir.mkdir(parents=True, exist_ok=True)
    init = skills_dir / "__init__.py"
    if not init.exists():
        init.write_text('"""Provider-owned agent skills."""\n', encoding="utf-8")

    for skill in legacy:
        workflow = skill.folder / "WORKFLOW.md"
        workflow.write_text(_workflow_document(skill, guard), encoding="utf-8")
        (skill.folder / "SKILL.md").unlink()

    target.mkdir(parents=True, exist_ok=True)
    (target / "references").mkdir(parents=True, exist_ok=True)
    (target / "SKILL.md").write_text(
        _main_skill(repo.name, target_name, description), encoding="utf-8"
    )
    (target / "references" / "catalog.md").write_text(
        _catalog(target, legacy, guard), encoding="utf-8"
    )
    return len(legacy), target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    repos = sorted(
        path
        for path in args.agents_root.iterdir()
        if path.is_dir() and (path / "pyproject.toml").is_file()
    )
    migrated = 0
    legacy_total = 0
    failures: list[str] = []
    for repo in repos:
        try:
            legacy_count, _target = consolidate_repo(repo, apply=args.apply)
        except Exception:  # noqa: BLE001 - never emit paths or source content
            failures.append(repo.name)
            continue
        legacy_total += legacy_count
        migrated += 1

    mode = "migrated" if args.apply else "would migrate"
    print(f"{mode} {migrated} providers from {legacy_total} skill trigger(s)")
    if failures:
        print(f"skill consolidation failed for {len(failures)} provider(s)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
