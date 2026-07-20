#!/usr/bin/env python3
"""Reject public HTTP handlers that bypass the guarded graph facade."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PUBLIC_ROOTS = (
    "agent_utilities/gateway",
    "agent_utilities/server/routers",
    "agent_utilities/mcp",
    "agent_utilities/tools",
)


def _mentions_backend(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute) and node.attr == "backend":
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return (
            node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "backend"
        )
    return any(_mentions_backend(child) for child in ast.iter_child_nodes(node))


def violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    backend_aliases: set[str] = set()
    findings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _mentions_backend(node.value):
            backend_aliases.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        if isinstance(node, ast.ImportFrom) and any(
            alias.name == "GraphCore" for alias in node.names
        ):
            findings.append(f"{path}:{node.lineno}: public handler imports GraphCore")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        receiver = node.func.value
        direct = _mentions_backend(receiver)
        aliased = isinstance(receiver, ast.Name) and receiver.id in backend_aliases
        if (direct or aliased) and node.func.attr.startswith("execute"):
            findings.append(
                f"{path}:{node.lineno}: public handler calls a backend execute primitive"
            )
    return findings


def check(root: Path) -> list[str]:
    findings: list[str] = []
    for relative in PUBLIC_ROOTS:
        public_path = root / relative
        if public_path.is_file():
            findings.extend(violations(public_path))
        elif public_path.is_dir():
            for path in sorted(public_path.rglob("*.py")):
                findings.extend(violations(path))
    return findings


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    root = Path(args[0]).resolve() if args else Path(__file__).resolve().parents[1]
    findings = check(root)
    if findings:
        print("Public graph-boundary gate failed:")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("Public graph-boundary gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
