#!/usr/bin/env python3
"""Import-graph wiring check — the Wire-First step-4 developer tool.

Referenced by AGENTS.md ("Wire-First — reachable != invoked", step 4: "Run
``check_wiring.py`` (import-graph, <=3 hops)"). Adapted from the
agent-utilities-evolution skill's ``wiring_sweep.py``, trimmed to its
import-graph reachability core.

What it does
------------
* Parses every module under ``agent_utilities/`` with ``ast`` and builds a
  static module-to-module import graph (including ``__init__.py``
  re-exports as edges).
* Seeds reachability roots from the live entry points: the package root,
  ``[project.scripts]`` console-script targets in ``pyproject.toml``, and
  ``__main__``-style modules.
* BFS-walks the graph and reports, per module, the minimum hop distance
  from any root. Modules with no path from a root are flagged as
  potentially unwired; ``--max-hops`` (default 3) additionally flags
  modules that are only reachable through long chains.
* With ``--module``, prints the shortest import chain from a root to one
  target module — the quick "is my new file actually on a live path?"
  question.

Known blind spots (do NOT treat a flag here as proof of dead code)
------------------------------------------------------------------
This is a *static import* view only. Per AGENTS.md, it cannot see:

* **Decorator / pkgutil dynamic registration** — ``@register_source`` +
  ``pkgutil.iter_modules`` discovery, ``@adaptor``, plugin entry-points.
  A self-registering module that nothing imports statically is a false
  positive; verify the discovery call runs on a live path instead.
* **Console scripts and external callers** — anything launched by name
  (cron, compose files, docs, other repos). ``[project.scripts]`` targets
  are seeded as roots, but ``[project.entry-points.*]`` plugins and
  out-of-repo callers are not.
* **Lazy / string-based imports** — ``importlib.import_module(f"...")``,
  imports inside function bodies are captured, but dynamically composed
  module paths are invisible.
* **Reachable != invoked** — an import edge proves loadability, not that
  the hot path ever *calls* the code. Wire-First steps 1-3 (trace the live
  path, default the integration on, write a live-path test) remain the
  real gate; this tool only catches the grossest "nothing even imports
  it" misses.

Because of this false-positive rate, the script is a developer aid, not a
pre-commit gate: it always exits 0 unless ``--fail-on-unreachable`` is
passed explicitly.

Usage::

    python scripts/check_wiring.py                       # summary report
    python scripts/check_wiring.py --max-hops 3          # same, explicit
    python scripts/check_wiring.py --module agent_utilities/foo/bar.py
    python scripts/check_wiring.py --json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "agent_utilities"
PYPROJECT = ROOT / "pyproject.toml"

# Modules that are live roots even without an inbound import edge.
DEFAULT_ROOT_PATTERNS = (
    "agent_utilities/__init__.py",
    "agent_utilities/__main__.py",
)


def module_name_to_path(modname: str, modules: set[str]) -> str | None:
    """Resolve a dotted module name to a repo-relative file path, if local."""
    if not modname.startswith("agent_utilities"):
        return None
    as_file = modname.replace(".", "/") + ".py"
    if as_file in modules:
        return as_file
    as_pkg = modname.replace(".", "/") + "/__init__.py"
    if as_pkg in modules:
        return as_pkg
    return None


def path_to_module_name(rel_path: str) -> str:
    name = rel_path[: -len(".py")] if rel_path.endswith(".py") else rel_path
    if name.endswith("/__init__"):
        name = name[: -len("/__init__")]
    return name.replace("/", ".")


def resolve_relative(rel_path: str, node: ast.ImportFrom) -> str | None:
    """Resolve a relative ``from . import x`` to an absolute dotted name."""
    pkg_parts = path_to_module_name(rel_path).split(".")
    # For a module (not package __init__), the package is the parent.
    if not rel_path.endswith("__init__.py"):
        pkg_parts = pkg_parts[:-1]
    # level=1 means current package, each extra level goes one up.
    up = node.level - 1
    if up > len(pkg_parts):
        return None
    base = pkg_parts[: len(pkg_parts) - up]
    if node.module:
        base = base + node.module.split(".")
    return ".".join(base) if base else None


def collect_imports(rel_path: str, tree: ast.AST) -> set[str]:
    """All dotted module names a file imports (absolute + resolved relative)."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                resolved = resolve_relative(rel_path, node)
                if resolved:
                    found.add(resolved)
                    # ``from .pkg import mod`` may target submodules.
                    for alias in node.names or []:
                        found.add(f"{resolved}.{alias.name}")
            elif node.module:
                found.add(node.module)
                for alias in node.names or []:
                    found.add(f"{node.module}.{alias.name}")
    return found


def load_console_script_roots(modules: set[str]) -> set[str]:
    """Seed roots from ``[project.scripts]`` targets in pyproject.toml."""
    roots: set[str] = set()
    if not PYPROJECT.exists():
        return roots
    in_scripts = False
    for line in PYPROJECT.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_scripts = stripped == "[project.scripts]"
            continue
        if in_scripts:
            m = re.match(r'[\w\-]+\s*=\s*"([\w\.]+):[\w\.]+"', stripped)
            if m:
                path = module_name_to_path(m.group(1), modules)
                if path:
                    roots.add(path)
    return roots


def build_graph() -> tuple[dict[str, set[str]], set[str]]:
    """Return (import_graph, modules) over agent_utilities/."""
    modules: set[str] = set()
    file_imports: dict[str, set[str]] = {}

    for py_file in sorted(SRC_DIR.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue
        rel = py_file.relative_to(ROOT).as_posix()
        modules.add(rel)
        try:
            tree = ast.parse(
                py_file.read_text(encoding="utf-8", errors="ignore"), filename=rel
            )
        except SyntaxError:
            file_imports[rel] = set()
            continue
        file_imports[rel] = collect_imports(rel, tree)

    graph: dict[str, set[str]] = defaultdict(set)
    for rel, imps in file_imports.items():
        for modname in imps:
            target = module_name_to_path(modname, modules)
            if target and target != rel:
                graph[rel].add(target)
                # Importing a module also executes every ancestor package
                # __init__ — model those edges so package inits are not
                # falsely orphaned when only deep submodules are imported.
                parent = Path(target).parent
                while parent != Path("."):
                    init = (parent / "__init__.py").as_posix()
                    if init in modules and init != rel:
                        graph[rel].add(init)
                    parent = parent.parent
    return graph, modules


def bfs_hops(graph: dict[str, set[str]], roots: set[str]) -> dict[str, int]:
    """Minimum hop distance from any root, following import edges."""
    dist: dict[str, int] = {r: 0 for r in roots}
    queue: deque[str] = deque(roots)
    while queue:
        cur = queue.popleft()
        for nxt in graph.get(cur, ()):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)
    return dist


def shortest_chain(
    graph: dict[str, set[str]], roots: set[str], target: str
) -> list[str] | None:
    """Shortest import chain from any root to ``target``, or None."""
    prev: dict[str, str | None] = {r: None for r in roots}
    queue: deque[str] = deque(roots)
    while queue:
        cur = queue.popleft()
        if cur == target:
            chain = [cur]
            while prev[cur] is not None:
                cur = prev[cur]  # type: ignore[assignment]
                chain.append(cur)
            return list(reversed(chain))
        for nxt in graph.get(cur, ()):
            if nxt not in prev:
                prev[nxt] = cur
                queue.append(nxt)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import-graph wiring check (Wire-First step 4)."
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=3,
        help="Flag modules farther than this from any entry-point root (default 3).",
    )
    parser.add_argument(
        "--module",
        type=str,
        default=None,
        help="Repo-relative module path; print its shortest import chain from a root.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    parser.add_argument(
        "--fail-on-unreachable",
        action="store_true",
        help="Exit 1 if any module is unreachable (off by default: high "
        "false-positive rate on self-registering modules).",
    )
    args = parser.parse_args()

    if not SRC_DIR.exists():
        print(f"source directory not found: {SRC_DIR}", file=sys.stderr)
        return 2

    graph, modules = build_graph()
    roots = {p for p in DEFAULT_ROOT_PATTERNS if p in modules}
    roots |= load_console_script_roots(modules)
    dist = bfs_hops(graph, roots)

    if args.module:
        target = args.module
        if target not in modules:
            print(f"unknown module: {target}", file=sys.stderr)
            return 2
        chain = shortest_chain(graph, roots, target)
        if chain is None:
            print(
                f"{target}: NO static import path from any entry-point root.\n"
                "Check the blind-spot list in this script's docstring before "
                "concluding it is dead (decorator/pkgutil registration, "
                "entry-points, lazy imports)."
            )
        else:
            print(f"{target}: reachable in {len(chain) - 1} hop(s):")
            for i, hop in enumerate(chain):
                print(f"  {' ' * i}{hop}")
        return 0

    unreachable = sorted(m for m in modules if m not in dist)
    far = sorted(
        (m, d) for m, d in dist.items() if d > args.max_hops
    )

    if args.json:
        print(
            json.dumps(
                {
                    "roots": sorted(roots),
                    "total_modules": len(modules),
                    "reachable": len(dist),
                    "unreachable": unreachable,
                    "beyond_max_hops": [
                        {"module": m, "hops": d} for m, d in far
                    ],
                    "max_hops": args.max_hops,
                },
                indent=2,
            )
        )
    else:
        print(f"roots ({len(roots)}):")
        for r in sorted(roots):
            print(f"  {r}")
        print(
            f"\nmodules: {len(modules)}  reachable: {len(dist)}  "
            f"unreachable: {len(unreachable)}"
        )
        if far:
            print(f"\nreachable only beyond {args.max_hops} hops ({len(far)}):")
            for m, d in far:
                print(f"  {d:>2}  {m}")
        if unreachable:
            print(
                f"\nno static import path from a root ({len(unreachable)}) — "
                "verify against the blind-spot list before treating as dead:"
            )
            for m in unreachable:
                print(f"  {m}")

    if args.fail_on_unreachable and unreachable:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
