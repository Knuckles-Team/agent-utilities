#!/usr/bin/env python3
"""Entrypoint engine-authority gate (CONCEPT:AU-ECO.ui.one-engine-authority).

Enforces the *Universal capability — ONE core, thin entrypoints* rule in
``AGENTS.md``: every user/system-facing surface (the messaging stack, the A2A
protocol layer every ``agents/*/agent_server.py`` shares, ``agent-webui``,
``agent-terminal-ui``, ``geniusbot``) is a THIN TRANSPORT that reaches the graph
through the ONE process-wide engine authority
(``IntelligenceGraphEngine.get_active()`` / ``.get_or_create()``) — never a
second, hand-rolled construction path.

This is not speculative: it is the exact shape of a real, found bug (D-WD-7).
``agent_webui.api_extensions.get_engine()``'s lazy-init fallback used to call
``create_backend(backend_type='ladybug', db_path=...)`` and construct
``IntelligenceGraphEngine(...)`` directly. ``IntelligenceGraphEngine`` is a
process-wide singleton (``_ACTIVE_ENGINE``) that ANY caller can win the
construction race for — so whichever entrypoint reached that hand-rolled branch
FIRST silently became the engine authority for the entire process, handing
every other route a disconnected, empty local LadybugDB instead of the real
operational graph ("Workflows shows nothing"). ``get_or_create()`` cannot be
raced around this way: it returns the existing singleton if one exists, and
otherwise builds the one sanctioned operational-authority backend
(``create_backend()`` called with NO ``backend_type`` — the epistemic-graph
engine plus configured mirrors).

So an entrypoint file may call ``IntelligenceGraphEngine.get_active()`` /
``.get_or_create()`` freely, but must NEVER:

  1. Call the ``IntelligenceGraphEngine`` constructor directly
     (``IntelligenceGraphEngine(...)``) — that bypasses the singleton
     arbitration ``get_or_create()`` provides.
  2. Call ``create_backend(...)`` with an explicit ``backend_type=`` — that is
     reserved for connection-registry source adapters, projection
     construction, and focused backend tests; passing it from an entrypoint
     requests a DIFFERENT backend than the operational authority
     ``create_backend()`` (no ``backend_type``) resolves.

Test files are exempt (a test legitimately builds a throwaway engine over an
in-memory backend as a fixture) — this gate is about the SERVING path, not test
scaffolding.

Usage:
  python3 scripts/check_entrypoint_engine_construction.py            # check
  python3 scripts/check_entrypoint_engine_construction.py --list-trees  # show what's scanned

Exit 0 = every entrypoint reaches the graph through get_active()/get_or_create()
only, 1 = a divergent construction path was found.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

CONSTRUCTED_CLASS = "IntelligenceGraphEngine"
BACKEND_FACTORY = "create_backend"

SKIP_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    "build",
    "dist",
    ".ruff_cache",
    ".mypy_cache",
    ".hypothesis",
}
# A path segment anywhere in the relative path marks a file as test scaffolding,
# exempt from this gate (see module docstring).
TEST_MARKERS = {"tests", "test", "__tests__"}


def _is_test_path(rel_parts: tuple[str, ...]) -> bool:
    return any(part in TEST_MARKERS for part in rel_parts) or any(
        part.startswith("test_") or part.endswith("_test.py") for part in rel_parts
    )


def _imported_bare_names(tree: ast.Module) -> set[str]:
    """Names imported directly (``from x import Y`` / ``... as Y``), bare in scope.

    Both ``CONSTRUCTED_CLASS`` and ``BACKEND_FACTORY`` are only ever invoked as
    a bare name (``create_backend(...)``, ``IntelligenceGraphEngine(...)``)
    when they mean the graph engine/backend factory in this codebase's own
    convention -- an attribute call like ``registry.create_backend(...)``
    (``MessagingRegistry.create_backend``, an unrelated per-platform messaging
    backend factory of the same name) is a different symbol entirely and must
    NOT be flagged. Gating on a real bare import of the name is what tells
    them apart without hand-maintaining a module allowlist.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _violations_in_source(source: str) -> list[str]:
    """Return human-readable violation descriptions found in one file's AST."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    imported = _imported_bare_names(tree)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id

        if name == CONSTRUCTED_CLASS and CONSTRUCTED_CLASS in imported:
            violations.append(
                f"line {node.lineno}: direct `{CONSTRUCTED_CLASS}(...)` construction "
                "-- bypasses get_or_create()'s singleton arbitration; use "
                f"`{CONSTRUCTED_CLASS}.get_or_create()` (or `.get_active()` for a "
                "read-only lookup)"
            )
        elif name == BACKEND_FACTORY and BACKEND_FACTORY in imported:
            has_backend_type = any(
                kw.arg == "backend_type" for kw in node.keywords
            ) or bool(node.args)
            if has_backend_type:
                violations.append(
                    f"line {node.lineno}: `{BACKEND_FACTORY}(backend_type=...)` "
                    "-- entrypoints must acquire the operational authority via "
                    f"`{CONSTRUCTED_CLASS}.get_or_create()`, never request a "
                    "specific backend flavor directly (D-WD-7)"
                )
    return violations


def _tracked_or_walked_py_files(root: Path) -> list[Path]:
    """``.py`` files under ``root``, preferring the git-tracked set (BUG-043).

    A raw ``rglob`` also picks up gitignored, generated build output, which
    can carry a stale hand-rolled-construction violation no longer in real
    source. Falls back to a filesystem walk only when ``root`` is not inside
    a git working tree (e.g. a synthetic test fixture).
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--", "*.py"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        tracked = [root / line for line in out.splitlines() if line]
        if tracked:
            return [p for p in tracked if p.is_file()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return sorted(root.rglob("*.py"))


def scan_tree(root: Path) -> dict[str, list[str]]:
    """Return {relpath: [violation, ...]} for every offending file under ``root``."""
    found: dict[str, list[str]] = {}
    for path in _tracked_or_walked_py_files(root):
        rel_parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS for part in rel_parts):
            continue
        if _is_test_path(rel_parts):
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        violations = _violations_in_source(source)
        if violations:
            found[path.relative_to(root).as_posix()] = violations
    return found


def entrypoint_trees(repo_root: Path) -> dict[str, Path]:
    """The five entrypoint surfaces named in AGENTS.md's *Universal capability* rule.

    ``repo_root`` is ``agent-packages/`` (this script's grandparent — mirrors
    ``check_coupling.py``'s convention). Each tree is scanned only if present,
    so this gate degrades gracefully on a checkout that doesn't have every
    sibling repo cloned.
    """
    au_root = repo_root / "agent-utilities"
    trees = {
        "messaging (agent-utilities)": au_root / "agent_utilities" / "messaging",
        "A2A protocol layer (agent-utilities, shared by every agent_server.py)": (
            au_root / "agent_utilities" / "protocols"
        ),
        "agent-webui": repo_root / "agent-webui" / "agent" / "agent_webui",
        "agent-terminal-ui": repo_root / "agent-terminal-ui" / "agent_terminal_ui",
        "geniusbot": repo_root / "geniusbot" / "geniusbot",
    }
    return {name: path for name, path in trees.items() if path.exists()}


def agent_server_files(repo_root: Path) -> list[Path]:
    """Every ``agents/*/…/agent_server.py`` (the A2A/HTTP entrypoint per connector)."""
    agents_root = repo_root / "agents"
    if not agents_root.exists():
        return []
    return sorted(
        p
        for p in agents_root.glob("*/*/agent_server.py")
        if not any(part in SKIP_DIRS for part in p.parts)
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]

    if "--list-trees" in sys.argv:
        for name, path in entrypoint_trees(repo_root).items():
            print(f"{name}: {path}")
        for f in agent_server_files(repo_root):
            print(f"agent_server.py: {f}")
        return 0

    all_violations: dict[str, dict[str, list[str]]] = {}
    for name, path in entrypoint_trees(repo_root).items():
        found = scan_tree(path)
        if found:
            all_violations[name] = {f"{path.name}/{rel}": v for rel, v in found.items()}

    for server_file in agent_server_files(repo_root):
        source = server_file.read_text(encoding="utf-8", errors="ignore")
        violations = _violations_in_source(source)
        if violations:
            all_violations.setdefault("agent_server.py entrypoints", {})[
                server_file.as_posix()
            ] = violations

    if all_violations:
        print("Entrypoint engine-authority gate FAILED:", file=sys.stderr)
        print(
            "  every entrypoint must reach the graph ONLY through "
            f"{CONSTRUCTED_CLASS}.get_active() / .get_or_create() -- see the "
            "module docstring (D-WD-7) for why a second construction path is "
            "unsafe, not just undesirable.",
            file=sys.stderr,
        )
        for surface, files in all_violations.items():
            print(f"\n  {surface}:", file=sys.stderr)
            for rel, violations in files.items():
                for v in violations:
                    print(f"    - {rel}: {v}", file=sys.stderr)
        return 1

    scanned = ", ".join(entrypoint_trees(repo_root)) or "(none found on this checkout)"
    print(
        "OK: every entrypoint reaches the graph only through "
        f"{CONSTRUCTED_CLASS}.get_active()/.get_or_create(). Scanned: {scanned}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
