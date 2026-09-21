"""Source discovery and exact import-target resolution."""

from __future__ import annotations

import ast
from pathlib import Path

from scripts._git_scan import repo_root_of, tracked_or_walked
from scripts.layer_direction.imports import from_base

_SKIP_DIRS = {"__pycache__", ".venv", "build", "dist", "tests", "test", ".git"}


def targets(
    node: ast.Import | ast.ImportFrom,
    *,
    package_name: str,
    current_package: str,
    known_modules: frozenset[str],
) -> list[str]:
    """Resolve one internal target per imported alias."""
    if isinstance(node, ast.Import):
        raw = [alias.name for alias in node.names]
    else:
        raw = from_targets(
            node,
            package_name=package_name,
            current_package=current_package,
            known_modules=known_modules,
        )
    return [
        target
        for name in raw
        if (target := strip_package(name, package_name)) is not None
    ]


def from_targets(
    node: ast.ImportFrom,
    *,
    package_name: str,
    current_package: str,
    known_modules: frozenset[str],
) -> list[str]:
    """Disambiguate imported attributes from real imported submodules."""
    base = from_base(node, current_package)
    if not base:
        return []
    resolved: list[str] = []
    for alias in node.names:
        extended = f"{base}.{alias.name}"
        target = strip_package(extended, package_name)
        resolved.append(extended if target in known_modules else base)
    return resolved


def module_name(relative_path: Path) -> str:
    """Return the dotted name represented by a package-relative Python path."""
    parts = relative_path.with_suffix("").parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def python_paths(package_root: Path) -> list[Path]:
    """Return product Python files without ambient-git fixture confusion."""
    repo_root = repo_root_of(package_root)
    candidates = (
        tracked_or_walked(package_root, "*.py", root=repo_root)
        if repo_root
        else package_root.rglob("*.py")
    )
    return [
        path
        for path in sorted(candidates)
        if not any(part in _SKIP_DIRS for part in path.parts)
    ]


def known_modules(paths: list[Path], package_root: Path) -> frozenset[str]:
    """Include discovered modules and namespace-package parent prefixes."""
    modules = {
        name for path in paths if (name := module_name(path.relative_to(package_root)))
    }
    return frozenset(prefix for module in modules for prefix in module_prefixes(module))


def module_prefixes(module: str) -> tuple[str, ...]:
    """Return a module and every dotted parent package."""
    parts = module.split(".")
    return tuple(".".join(parts[:length]) for length in range(1, len(parts) + 1))


def strip_package(dotted: str, package_name: str) -> str | None:
    """Remove the scanned package prefix, or reject an external target."""
    prefix = f"{package_name}."
    return dotted[len(prefix) :] if dotted.startswith(prefix) else None
