#!/usr/bin/env python3
"""Reject unregistered harness modules and cross-owner console scripts."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

OWNERS = {"meristem", "arbora", "scion-sdk", "medulla"}


def _paths(value: Any) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str)
        and item.startswith("agent_utilities/")
        and ".." not in PurePosixPath(item).parts
        and str(PurePosixPath(item)) == item
        for item in value
    ):
        raise ValueError("invalid owner paths")
    if len(value) != len(set(value)):
        raise ValueError("duplicate owner paths")
    return value


def check(root: Path) -> list[str]:
    """Check one checkout using only its own registry and project metadata."""
    registry = json.loads((root / "architecture/component-registry.yml").read_text())
    if registry.get("schema") != "au-boundary-owners/v1":
        raise ValueError("invalid owner registry schema")
    owners = registry.get("owners")
    if not isinstance(owners, dict) or set(owners) != OWNERS:
        raise ValueError("invalid owner registry owners")
    modules = _paths(owners["meristem"].get("modules"))
    if not all(path.endswith(".py") for path in modules):
        raise ValueError("harness module is not a Python file")
    foreign = {
        owner: _paths(owners[owner].get("prefixes")) for owner in OWNERS - {"meristem"}
    }
    actual = {
        path.relative_to(root).as_posix()
        for path in (root / "agent_utilities").rglob("*.py")
    }
    findings = [
        f"unregistered module: {path}" for path in sorted(actual - set(modules))
    ]
    findings.extend(
        f"missing registered module: {path}" for path in sorted(set(modules) - actual)
    )
    for path in sorted(actual):
        for owner, prefixes in foreign.items():
            for prefix in prefixes:
                if path == prefix or path.startswith(prefix.rstrip("/") + "/"):
                    findings.append(f"{owner}-owned module in meristem: {path}")
    project = tomllib.loads((root / "pyproject.toml").read_text())
    scripts = project.get("project", {}).get("scripts", {})
    if not isinstance(scripts, dict):
        raise ValueError("invalid console scripts")
    for name, target in scripts.items():
        if not isinstance(target, str) or ":" not in target:
            findings.append(f"invalid console script target: {name}")
            continue
        module = target.split(":", 1)[0].replace(".", "/")
        if f"{module}.py" not in modules and f"{module}/__init__.py" not in modules:
            findings.append(f"unowned console script target: {name}")
    return findings


def main() -> int:
    try:
        findings = check(Path(__file__).resolve().parents[1])
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        print(f"boundary owner registry invalid: {exc}", file=sys.stderr)
        return 1
    for finding in findings:
        print(finding, file=sys.stderr)
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
