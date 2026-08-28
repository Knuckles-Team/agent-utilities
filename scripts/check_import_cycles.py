#!/usr/bin/env python3
"""Eager-import cycle gate — ABSOLUTE ceiling of ZERO. (BUG-CX-004 / WD5-ARCH-02)

★★★ READ THIS BEFORE "SIMPLIFYING" THIS SCRIPT ★★★

This gate counts ONLY import statements that actually execute at
module-LOAD time ("eager" / top-level imports). It deliberately EXCLUDES:

  * imports nested inside a ``def``/``async def`` body (the "deferred"
    idiom this codebase already uses extensively to break cycles), and
  * imports guarded by ``if TYPE_CHECKING:`` (never executed at runtime
    at all — they exist purely for static type checkers).

This distinction is not a shortcut — it is the entire point. `ARCH-import-
cycles.md` (plans/complex/reports/) measured `agent_utilities`'s raw,
everything-counts import graph and found a genuine 821-module strongly
connected component (48.5% of the codebase, confirmed three independent
ways: a hand-written AST parser, `tach map`, and `ast-metrics` coupling
communities). But 46% of the ~13k raw import statements in this codebase
never execute at import time — restricting the graph to load-time-executing
edges only collapses that 821-module SCC to **3 modules plus one separate
2-module cycle**. Gating on the RAW (all-imports) graph is NOT adoptable:
shrinking it below a useful threshold costs cutting roughly 500 of 3,486
edges in the giant subgraph — there is no small first cut. Gating on the
EAGER graph is adoptable *today*, because the debt there is already gone
(see the two commits that closed WD5-ARCH-02).

If a future change to this script makes it count deferred or
TYPE_CHECKING-guarded imports as cycle-forming edges, it will immediately
refire against the (unrelated, unfixable-in-one-sitting) raw-SCC debt and
either have to be reverted or grandfathered with a baseline — exactly the
ratchet this program forbids (see plans/complex/ NO RATCHETS rule). Keep
the eager/deferred/TYPE_CHECKING split. Do not "simplify" it away.

★ THIS GATE IS THE NAMED OWNER OF THREE kiss METRICS (WD10-B-004, 2026-08-28)

``.kiss/kiss.toml`` parks kiss's ``cycle_size``, ``dependency_depth`` and
``indirect_dependencies`` at 999999. That is NOT "we gave up": it is the same
delegation this repo already applies to ``branches_per_function`` (owned by
``check_complexity.py``) and ``orphan_module_enabled`` (owned by
``check_liveness.py``) -- two gates, two definitions, one word is a guaranteed
disagreement, so exactly one tool owns each metric. THIS gate is that tool for
all three, because kiss measures them on the ALL-IMPORTS graph and this one
measures them on the graph that executes.

The three are one finding, not three: every module inside an N-module cycle
necessarily has >= N-1 indirect dependencies and a dependency depth at least
the cycle's radius. Measured on the eager graph (2026-08-28, au ``5bfb6d0ee``):
cycle_size 0, indirect_dependencies p50 2 / max 192, dependency_depth p50 1 /
max 8 -- against kiss's all-imports p50 1133 / p50 11. ``_report_reach`` prints
the live distribution on every run so none of it can go quiet. Full map, ranked
cut plan and the rejected alternatives:
``plans/complex/designs/BUG-CX-004-import-cycle.md``.

What this gate does NOT see (known blind spots, same class as
``check_import_safety.py``'s):

  * dynamic imports (``importlib.import_module("...")``, ``__import__``)
    — invisible to a static AST walk, by construction.
  * imports inside a class body execute at class-definition time (i.e. at
    module-load time too), so they are correctly counted as eager here —
    this is NOT the same exemption as a function body.
  * this only detects cycles *within* the scanned package; a cycle that
    round-trips through an external package is out of scope.

Usage::

    python3 scripts/check_import_cycles.py                    # scans agent_utilities/
    python3 scripts/check_import_cycles.py path/to/some/pkg    # scans an arbitrary package root

Exit 0 = zero eager-import cycles. Exit 1 = at least one found (printed to
stderr, one line per cycle). Exit 2 = could not run (bad package root).
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import networkx as nx

_AU_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_AU_ROOT))
from scripts._git_scan import repo_root_of, tracked_or_walked  # noqa: E402

_SKIP_DIRS = {"__pycache__", ".venv", "build", "dist", "tests", "test", ".git"}


def _is_type_checking_guard(test: ast.expr) -> bool:
    """True for ``if TYPE_CHECKING:`` / ``if typing.TYPE_CHECKING:``."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


class _ImportWalker(ast.NodeVisitor):
    """Collects every ``Import``/``ImportFrom`` node plus whether it is
    "eager" — neither inside a function body nor a ``TYPE_CHECKING`` guard.
    """

    def __init__(self) -> None:
        self.found: list[tuple[ast.Import | ast.ImportFrom, bool]] = []
        self._func_depth = 0
        self._tc_depth = 0

    def _is_eager(self) -> bool:
        return self._func_depth == 0 and self._tc_depth == 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._func_depth += 1
        self.generic_visit(node)
        self._func_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._func_depth += 1
        self.generic_visit(node)
        self._func_depth -= 1

    def visit_If(self, node: ast.If) -> None:
        guarded = _is_type_checking_guard(node.test)
        if guarded:
            self._tc_depth += 1
        self.generic_visit(node)
        if guarded:
            self._tc_depth -= 1

    def visit_Import(self, node: ast.Import) -> None:
        self.found.append((node, self._is_eager()))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.found.append((node, self._is_eager()))


def _module_name(relpath: Path) -> str:
    """Dotted module name a file represents (``__init__.py`` -> its package)."""
    parts = relpath.with_suffix("").parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _current_package(relpath: Path) -> str:
    """Dotted package name relative-imports in this file resolve against."""
    parts = relpath.with_suffix("").parts
    return ".".join(parts[:-1])


def _relative_base(current_package: str, level: int) -> str:
    """Dotted package a relative import's leading dots resolve to.

    ``level == 1`` is the current package itself; each dot beyond that
    strips one trailing segment.
    """
    parts = current_package.split(".") if current_package else []
    if level > 1:
        cut = len(parts) - (level - 1)
        parts = parts[:cut] if cut > 0 else []
    return ".".join(parts)


def _resolve_edges(
    node: ast.Import | ast.ImportFrom, current_package: str
) -> list[str]:
    """Dotted target module(s) an Import/ImportFrom statement depends on.

    ``from PKG import NAME`` and ``from .. import NAME`` each yield TWO
    candidate targets: the resolved package, and ``<package>.NAME``.

    ★ The second is not redundant, and omitting it was a real hole in this
    gate. If ``NAME`` is a SUBMODULE, ``from PKG import NAME`` imports that
    submodule eagerly — a genuine load-time edge to ``PKG.NAME``. Recording
    only the edge to ``PKG`` makes a cycle expressed in that form invisible.
    Proven by planting: ``a: from pkg import b`` / ``b: from pkg import a``
    PASSED this gate, while the dotted, plain-``import`` and relative-dotted
    forms of the same cycle were all correctly caught.

    That is the same blind spot the fleet mirror sweep hit independently on the
    same day: ``from . import mcp as X`` contains no ``pkg.mcp`` substring, so
    a dotted-path scan misses it, and it is a common load-bearing import shape
    in this codebase.

    Emitting the extra candidate is safe when ``NAME`` is a plain attribute
    rather than a submodule: it adds a leaf node with no outgoing edges, and a
    node with no outgoing edges cannot participate in a cycle.
    """
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if node.level == 0:
        if not node.module:
            return []
        return [node.module, *(f"{node.module}.{a.name}" for a in node.names)]
    base = _relative_base(current_package, node.level)
    if node.module:
        target = f"{base}.{node.module}" if base else node.module
        return [target, *(f"{target}.{a.name}" for a in node.names)]
    if not base:
        return []
    return [base, *(f"{base}.{a.name}" for a in node.names)]


def _add_file_edges(
    path: Path,
    module_name: str,
    current_package: str,
    pkg_name: str,
    graph: nx.DiGraph,
    known_modules: frozenset[str],
) -> None:
    try:
        tree = ast.parse(
            path.read_text(encoding="utf-8", errors="ignore"), filename=str(path)
        )
    except SyntaxError:
        return
    walker = _ImportWalker()
    walker.visit(tree)
    for node, is_eager in walker.found:
        if not is_eager:
            continue
        for target in _resolve_edges(node, current_package):
            if target != pkg_name and not target.startswith(f"{pkg_name}."):
                continue
            if target == module_name:
                continue
            # Only real, discovered modules become edges. `_resolve_edges`
            # deliberately over-produces (`from PKG import NAME` yields both
            # `PKG` and `PKG.NAME`, since NAME may be a submodule), and
            # filtering here is what keeps a plain ATTRIBUTE import from
            # inventing a phantom module node. Without this filter the verdict
            # stays correct — a leaf node cannot be in a cycle — but the
            # reported module and edge counts triple, and a gate that prints a
            # false number is a reporting defect even when its verdict is right.
            if target not in known_modules:
                continue
            graph.add_edge(module_name, target)


def build_eager_graph(package_root: Path) -> nx.DiGraph:
    """Directed graph of load-time-executing internal imports under `package_root`."""
    pkg_name = package_root.name
    repo_root = repo_root_of(package_root) or package_root
    paths = [
        p
        for p in sorted(tracked_or_walked(package_root, "*.py", root=repo_root))
        if not any(part in _SKIP_DIRS for part in p.parts)
    ]
    # TWO passes, deliberately: every module must be known before any edge is
    # resolved, so `_add_file_edges` can tell a real submodule import from a
    # plain attribute import of the same syntactic shape.
    discovered = {_module_name(p.relative_to(package_root.parent)): p for p in paths}
    known_modules = frozenset(discovered)
    graph: nx.DiGraph = nx.DiGraph()
    for module_name, path in discovered.items():
        relpath = path.relative_to(package_root.parent)
        graph.add_node(module_name)
        _add_file_edges(
            path,
            module_name,
            _current_package(relpath),
            pkg_name,
            graph,
            known_modules,
        )
    return graph


def eager_reach_stats(graph: nx.DiGraph) -> dict[str, list[int]]:
    """Per-module transitive-closure size and eccentricity on the EAGER graph.

    These are kiss's ``indirect_dependencies`` and ``dependency_depth``, computed
    on the graph that actually executes rather than on every import statement in
    the tree. See ``plans/complex/designs/BUG-CX-004-import-cycle.md``.
    """
    indirect: list[int] = []
    depth: list[int] = []
    for node in graph.nodes:
        levels = nx.single_source_shortest_path_length(graph, node)
        indirect.append(len(levels) - 1)
        depth.append(max(levels.values()))
    return {"indirect_dependencies": indirect, "dependency_depth": depth}


def _pct(values: list[int], fraction: float) -> int:
    """Nearest-rank percentile of a non-empty integer list."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def _report_reach(graph: nx.DiGraph) -> None:
    """Print the eager-graph reach distribution. ALWAYS, pass or fail.

    ★ THIS PRINT IS THE NO-RATCHET POLICY IN CODE (BUG-CX-004 / WD10-B-004).
    ``.kiss/kiss.toml`` parks kiss's ``cycle_size``/``dependency_depth``/
    ``indirect_dependencies`` at 999999 because kiss measures them on the
    ALL-IMPORTS graph, where 63% of the edges are function-local or
    ``TYPE_CHECKING``-guarded and never execute -- i.e. it scores this
    codebase's own cycle-breaking idiom as if it were the cycle. This gate
    owns those three metrics on the definition that can be enforced, and
    printing the real distribution every run is what keeps them from becoming
    invisible debt. Do not make this print conditional or quiet.

    Cycles are gated at an ABSOLUTE ZERO. Depth and indirect-deps are measured
    and reported but NOT yet gated: kiss's own defaults (3 and 10) are exceeded
    by 568 and 582 modules respectively even on the eager graph, and a
    threshold calibrated to what the repo passes today would be a baseline,
    which this program forbids. Picking their regime is CX-004-N1 in the design
    doc; until then the number is published, not frozen.
    """
    stats = eager_reach_stats(graph)
    print("eager-graph reach (reported, not gated -- see CX-004-N1):")
    for metric, values in stats.items():
        print(
            f"  {metric:24s} p50={_pct(values, 0.5):5d} p90={_pct(values, 0.9):5d} "
            f"p99={_pct(values, 0.99):5d} max={max(values):5d}"
        )


def find_cycles(graph: nx.DiGraph) -> list[list[str]]:
    cycles = [sorted(c) for c in nx.strongly_connected_components(graph) if len(c) > 1]
    cycles.sort(key=len, reverse=True)
    return cycles


def _report_failure(graph: nx.DiGraph, cycles: list[list[str]]) -> None:
    print(
        f"FAIL: {len(cycles)} eager-import cycle(s) among {graph.number_of_nodes()} modules "
        "(top-level/module-load-time imports only -- function-local and "
        "TYPE_CHECKING-guarded imports are excluded by design; see this script's "
        "module docstring).",
        file=sys.stderr,
    )
    for cyc in cycles:
        chain = " -> ".join([*cyc, cyc[0]])
        print(f"  cycle ({len(cyc)} modules): {chain}", file=sys.stderr)


def _parse_args(argv: list[str] | None) -> Path:
    ap = argparse.ArgumentParser(description="Eager (load-time) import cycle gate.")
    ap.add_argument(
        "package_root",
        nargs="?",
        default=None,
        help="Path to the package directory to scan (default: agent_utilities).",
    )
    ns = ap.parse_args(argv)
    if ns.package_root:
        return Path(ns.package_root).resolve()
    return _AU_ROOT / "agent_utilities"


def main(argv: list[str] | None = None) -> int:
    package_root = _parse_args(argv)
    if not package_root.is_dir():
        print(
            f"check_import_cycles: CANNOT RUN: {package_root} not found",
            file=sys.stderr,
        )
        return 2
    graph = build_eager_graph(package_root)
    cycles = find_cycles(graph)
    if cycles:
        _report_failure(graph, cycles)
        _report_reach(graph)
        return 1
    _report_reach(graph)
    print(
        f"OK: 0 eager-import cycles among {graph.number_of_nodes()} modules, "
        f"{graph.number_of_edges()} eager top-level import edges "
        "(function-local and TYPE_CHECKING-guarded imports excluded by design)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
