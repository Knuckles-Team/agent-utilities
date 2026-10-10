"""Shared import-scanning helpers for the deletion/importer census gates."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path


def imports_any(path: Path, needles: tuple[str, ...]) -> bool:
    """True if any import statement in ``path`` names a module containing a needle."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        else:
            continue
        if any(needle in name for name in names for needle in needles):
            return True
    return False


def files_importing(
    roots: Iterable[Path],
    needles: tuple[str, ...],
    base: Path,
    exclude: tuple[str, ...] = (),
) -> frozenset[str]:
    """Posix paths (relative to ``base``) of files under ``roots`` importing a needle."""
    hits: set[str] = set()
    for root in roots:
        for path in root.rglob("*.py"):
            rel = path.relative_to(base).as_posix()
            if any(rel == ex or rel.startswith(ex) for ex in exclude):
                continue
            if imports_any(path, needles):
                hits.add(rel)
    return frozenset(hits)
