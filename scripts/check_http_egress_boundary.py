#!/usr/bin/env python3
"""Reject HTTP client construction outside the governed egress factory."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "agent_utilities"
FACTORY = Path("agent_utilities/core/http_client.py")

_BLOCKED = {
    "aiohttp.ClientSession",
    "http.client.HTTPConnection",
    "http.client.HTTPSConnection",
    "httpx.AsyncClient",
    "httpx.Client",
    "requests.Session",
    "requests.delete",
    "requests.get",
    "requests.head",
    "requests.options",
    "requests.patch",
    "requests.post",
    "requests.put",
    "requests.request",
    "urllib.request.urlopen",
    "urllib3.PoolManager",
}


def _imports(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".")[0]] = item.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"
    return aliases


def _qualified(node: ast.expr, aliases: dict[str, str]) -> str:
    parts: list[str] = []
    cursor: ast.expr = node
    while isinstance(cursor, ast.Attribute):
        parts.append(cursor.attr)
        cursor = cursor.value
    if not isinstance(cursor, ast.Name):
        return ""
    root = aliases.get(cursor.id)
    if root is None:
        return ""
    return ".".join([root, *reversed(parts)])


def validate(package: Path = PACKAGE) -> list[str]:
    """Return stable violations for direct outbound HTTP constructors/calls."""

    errors: list[str] = []
    for path in sorted(package.rglob("*.py")):
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            relative = Path(package.name) / path.relative_to(package)
        if relative == FACTORY or "__pycache__" in relative.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
        except (OSError, SyntaxError) as exc:
            errors.append(f"{relative.as_posix()}: parse failed ({type(exc).__name__})")
            continue
        aliases = _imports(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = _qualified(node.func, aliases)
            if target in _BLOCKED:
                errors.append(
                    f"{relative.as_posix()}:{node.lineno}: direct {target}; "
                    "use core.http_client"
                )
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("HTTP egress boundary failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("HTTP egress boundary passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
