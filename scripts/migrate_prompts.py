#!/usr/bin/env python3
"""One-time migration of prompt JSON blueprints to the canonical schema.

CONCEPT:AU-ORCH.routing.resolve-body-single-canonical. Moves every system-prompt blueprint to the single canonical
body location (``instructions.core_directive``), stamps ``schema_version`` and a
``source`` provenance label, and ensures ``type == "prompt"``. The legacy flat
``content``/``input`` keys are removed once their value has been folded into
``instructions.core_directive``.

This is the sanctioned *data* migration (read-old → write-new), not an API
back-compat shim: it converts the persisted blueprints on disk so the
transitional fallbacks in ``resolve_body`` can later be removed.

Usage::

    python scripts/migrate_prompts.py [ROOT ...] [--check] [--verbose]

With no ROOT, migrates the packaged ``agent_utilities/prompts``. Pass one or
more directories (e.g. an agent-package repo) to migrate the fleet. ``--check``
reports what would change and exits non-zero if anything is non-canonical,
without writing (suitable for CI / a baseline burn-down check).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Keys whose presence (with a string value) marks a JSON file as a prompt.
_PROMPT_MARKERS = ("content", "input", "instructions", "task")
_LEGACY_BODY_KEYS = ("content", "input")


def _looks_like_prompt(data: dict) -> bool:
    if data.get("type") == "prompt":
        return True
    return any(k in data for k in _PROMPT_MARKERS)


def _derive_source(path: Path) -> str | None:
    """Best-effort provenance label from the owning package directory.

    For ``agent_utilities/prompts/*.json`` → ``agent-utilities:base``. For a
    fleet package ``agents/<pkg>/<pkg_module>/.../main_agent.json`` → ``<pkg>``.
    """
    parts = path.parts
    if "prompts" in parts:
        idx = parts.index("prompts")
        if idx > 0 and parts[idx - 1] == "agent_utilities":
            return "agent-utilities:base"
    if "agents" in parts:
        idx = parts.index("agents")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def migrate_data(data: dict, *, source: str | None) -> tuple[dict, list[str]]:
    """Return ``(migrated_data, changes)``. Pure transform; does not write."""
    changes: list[str] = []
    out = dict(data)

    instructions = out.get("instructions")
    if not isinstance(instructions, dict):
        instructions = {}

    core = instructions.get("core_directive")
    has_core = isinstance(core, str) and core.strip()

    if not has_core:
        for key in _LEGACY_BODY_KEYS:
            value = out.get(key)
            if isinstance(value, str) and value.strip():
                instructions = {"core_directive": value, **instructions}
                changes.append(f"moved '{key}' -> instructions.core_directive")
                break

    # Drop legacy body keys once their content lives in core_directive.
    new_core = instructions.get("core_directive")
    if isinstance(new_core, str) and new_core.strip():
        for key in _LEGACY_BODY_KEYS:
            if key in out:
                del out[key]
                changes.append(f"removed legacy '{key}'")

    if instructions:
        out["instructions"] = instructions

    if out.get("type") != "prompt":
        out["type"] = "prompt"
        changes.append("set type='prompt'")

    if not str(out.get("schema_version") or "").strip():
        out["schema_version"] = "1.0"
        changes.append("set schema_version='1.0'")

    if not str(out.get("source") or "").strip() and source:
        out["source"] = source
        changes.append(f"set source='{source}'")

    # A canonical blueprint needs a `task`; fall back to `name` then stem.
    if not str(out.get("task") or "").strip():
        name = out.get("name")
        if isinstance(name, str) and name.strip():
            out["task"] = name
            changes.append("set task from 'name'")

    return out, changes


def _iter_prompt_files(root: Path):
    if root.is_file() and root.suffix == ".json":
        yield root
        return
    for path in sorted(root.rglob("*.json")):
        # Skip obvious non-prompt config files by name.
        if path.name in {
            "mcp_config.json",
            "a2a.json",
            "package.json",
            "tsconfig.json",
        }:
            continue
        yield path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", help="Directories/files to migrate.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report changes and exit non-zero if any; do not write.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    roots = [Path(r) for r in args.roots] or [
        Path(__file__).resolve().parent.parent / "agent_utilities" / "prompts"
    ]

    total = 0
    changed = 0
    for root in roots:
        if not root.exists():
            print(f"skip (missing): {root}", file=sys.stderr)
            continue
        for pfile in _iter_prompt_files(root):
            try:
                raw = pfile.read_text(encoding="utf-8")
                data = json.loads(raw)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict) or not _looks_like_prompt(data):
                continue
            total += 1
            migrated, changes = migrate_data(data, source=_derive_source(pfile))
            if not changes:
                continue
            changed += 1
            rel = pfile
            if args.check:
                print(f"NEEDS MIGRATION: {rel}\n    - " + "\n    - ".join(changes))
            else:
                pfile.write_text(
                    json.dumps(migrated, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                if args.verbose:
                    print(f"migrated: {rel}\n    - " + "\n    - ".join(changes))

    verb = "would migrate" if args.check else "migrated"
    print(f"{verb} {changed}/{total} prompt blueprint(s).")
    if args.check and changed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
