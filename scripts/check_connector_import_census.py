#!/usr/bin/env python3
"""Census retired Meristem imports in connector checkouts, without network access."""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

BANNED = (
    "agent_utilities.mcp",
    "agent_utilities.core.config",
    "agent_utilities.core.exceptions",
    "agent_utilities.core.transport_security",
    "agent_utilities.base_utilities",
)
SKIP = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "tests",
    "test",
    "build",
    "dist",
    "node_modules",
}
RANGES = {"fj": ("f", "j"), "tz": ("t", "z")}


def _banned(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in BANNED)


def _attribute_name(node: ast.AST, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        parent = _attribute_name(node.value, aliases)
        return f"{parent}.{node.attr}" if parent else None
    return None


@dataclass(frozen=True)
class Finding:
    """Static violations and dynamic imports that cannot be resolved safely."""

    hits: tuple[str, ...]
    unresolved: tuple[str, ...]


def inspect_file(path: Path) -> Finding:
    """Inspect one file; unreadable or invalid source is an error, never clean."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: set[str] = set()
    unresolved: set[str] = set()
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.extend(
                (node.module, *(f"{node.module}.{alias.name}" for alias in node.names))
            )
            if node.module == "agent_utilities" and any(
                alias.name == "*" for alias in node.names
            ):
                unresolved.add(f"{path}:{node.lineno}")
        elif isinstance(node, ast.Attribute):
            name = _attribute_name(node, aliases)
            if name:
                names.append(name)
        elif isinstance(node, ast.Call):
            func = node.func
            func_name = _attribute_name(func, aliases)
            if func_name in {"exec", "eval", "builtins.exec", "builtins.eval"}:
                unresolved.add(f"{path}:{node.lineno}")
            dynamic = func_name in {
                "__import__",
                "builtins.__import__",
                "importlib.import_module",
            }
            if dynamic:
                arg = (
                    node.args[0]
                    if node.args
                    else next(
                        (item.value for item in node.keywords if item.arg == "name"),
                        None,
                    )
                )
                if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
                    unresolved.add(f"{path}:{node.lineno}")
                else:
                    names.append(arg.value)
                    if arg.value.startswith(".") or (
                        arg.value == "agent_utilities"
                        and (len(node.args) > 1 or node.keywords)
                    ):
                        unresolved.add(f"{path}:{node.lineno}")
            elif (
                isinstance(func, ast.Name)
                and func.id == "getattr"
                and len(node.args) >= 2
            ):
                parent = _attribute_name(node.args[0], aliases)
                if parent and (
                    parent == "agent_utilities" or parent.startswith("agent_utilities.")
                ):
                    attr = node.args[1]
                    if isinstance(attr, ast.Constant) and isinstance(attr.value, str):
                        names.append(f"{parent}.{attr.value}")
                    else:
                        unresolved.add(f"{path}:{node.lineno}")
        hits.update(name for name in names if _banned(name))
    return Finding(tuple(sorted(hits)), tuple(sorted(unresolved)))


def census(root: Path, selected: str | None = None) -> dict[str, Finding]:
    """Classify every checkout in the selected letter ranges, including clean ones."""
    ranges = [RANGES[selected]] if selected else list(RANGES.values())
    packages = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and any(lo <= path.name[:1].lower() <= hi for lo, hi in ranges)
    )
    if not packages:
        raise ValueError("no connector checkouts in the selected range")
    result: dict[str, Finding] = {}
    for package in packages:
        files = [
            path
            for path in package.rglob("*.py")
            if not any(
                part in SKIP or part.endswith(".egg-info")
                for part in path.relative_to(package).parts[:-1]
            )
        ]
        if not files:
            raise ValueError(f"no Python source in {package.name}")
        findings = [inspect_file(path) for path in files]
        result[package.name] = Finding(
            tuple(sorted({hit for item in findings for hit in item.hits})),
            tuple(
                sorted({item for finding in findings for item in finding.unresolved})
            ),
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleet-root", required=True, type=Path)
    parser.add_argument("--range", choices=sorted(RANGES))
    args = parser.parse_args()
    try:
        result = census(args.fleet_root, args.range)
    except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
        print(f"census incomplete: {exc}", file=sys.stderr)
        return 2
    for name, finding in result.items():
        state = (
            ", ".join(finding.hits)
            if finding.hits
            else ("indeterminate" if finding.unresolved else "clean")
        )
        print(f"{name}: {state}")
        for location in finding.unresolved:
            print(f"  unresolved dynamic import: {location}")
    return int(any(finding.hits or finding.unresolved for finding in result.values()))


if __name__ == "__main__":
    raise SystemExit(main())
