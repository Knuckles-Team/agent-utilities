#!/usr/bin/env python3
"""Backfeed preflight reachability gate (CONCEPT:AU-KG.ingest.backfeed-preflight, DEC-CA-07/P11).

Asserts that every one of ``source_sync.py``'s ``_DELTA_HANDLERS`` entries reaches
:func:`~agent_utilities.knowledge_graph.ontology.sync_conflict.evaluate_backfeed_preflight`
before it can commit -- either directly, or transitively through
``source_sync._apply_with_preflight``/``_apply_with_preflight_one`` (both of which call
it in their own function body). This is the P11 acceptance instrument named by
``CA-22-W06``: it must pass on the wired tree, and it MUST FAIL, naming the offending
handler, when one ``_DELTA_HANDLERS`` entry is rewired to bypass the chokepoint (the
known-bad demonstration required by the lane's acceptance gate 3).

Static AST call-graph walk, not execution: for each handler function, follows bare-name
``Call`` nodes to other module-level functions defined in the SAME file (source_sync.py),
depth-first, until it either finds a direct call to ``evaluate_backfeed_preflight`` (the
target itself is imported and called from ``_apply_with_preflight``, so reaching that
function's body already counts) or exhausts the graph. ``package_install`` -- the sole
``ORCHESTRATION_ONLY_SOURCES`` entry -- is exempt: it is a write-free dispatcher by
construction (its own module docstring/comment), so there is nothing for a backfeed
preflight to gate; a *different*, already-existing gate proves it never builds an
``entities``/``rels`` batch of its own.

Usage:
  python3 scripts/check_handler_preflight.py               # check
  python3 scripts/check_handler_preflight.py --list         # show every handler's status

Exit 0 = every non-exempt ``_DELTA_HANDLERS`` entry reaches the preflight chokepoint.
Exit 1 = one or more handlers do not (named, with the reason).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_AU_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SYNC_PATH = (
    _AU_ROOT / "agent_utilities" / "knowledge_graph" / "core" / "source_sync.py"
)

#: The preflight function itself -- reaching a call to this (directly, or via a
#: helper that calls it) is what "wired" means for this gate.
TARGET_FUNCTION = "evaluate_backfeed_preflight"

#: source keys that never build write material of their own (exempt; see module
#: docstring) -- kept as a static fallback if ``ORCHESTRATION_ONLY_SOURCES``'s own
#: AST literal can't be resolved for some reason (defensive, not the primary path).
_FALLBACK_ORCHESTRATION_ONLY = {"package_install"}


class _GateError(RuntimeError):
    """Raised for a structural problem with the tree itself (not a handler bypass)."""


def _literal_str_set(node: ast.AST) -> set[str]:
    """Best-effort ``{"a", "b"}`` / ``frozenset({"a", "b"})`` literal -> a python set."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "frozenset" and node.args:
            node = node.args[0]
    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        out: set[str] = set()
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.add(elt.value)
        return out
    return set()


def _parse(path: Path) -> ast.Module:
    source = path.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(path))


def _module_functions(
    tree: ast.Module,
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Every module-level (or nested-but-still-def) function, by name -- last def wins,
    matching normal Python name-shadowing semantics for a flat module."""
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node
    return functions


def _delta_handlers(tree: ast.Module) -> dict[str, str]:
    """``{source_key: handler_function_name}`` from the ``_DELTA_HANDLERS`` dict literal.

    Resolves only ``{"key": bare_name, ...}`` entries (every current entry is
    shaped this way) -- a value that isn't a bare ``Name`` (e.g. a subscript,
    call, or attribute) is skipped with a warning printed by the caller, never
    silently treated as compliant.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "_DELTA_HANDLERS" for t in targets
        ):
            continue
        if node.value is None or not isinstance(node.value, ast.Dict):
            raise _GateError(
                "_DELTA_HANDLERS is not a dict literal -- gate can't parse it"
            )
        handlers: dict[str, str] = {}
        for key, value in zip(node.value.keys, node.value.values, strict=True):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            if isinstance(value, ast.Name):
                handlers[key.value] = value.id
            else:
                handlers[key.value] = ""  # unresolved -- reported as a violation below
        return handlers
    raise _GateError("no _DELTA_HANDLERS assignment found in source_sync.py")


def _orchestration_only_sources(tree: ast.Module) -> set[str]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign) and not isinstance(node, ast.Assign):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(t, ast.Name) and t.id == "ORCHESTRATION_ONLY_SOURCES"
            for t in targets
        ):
            continue
        value = node.value
        if value is not None:
            resolved = _literal_str_set(value)
            if resolved:
                return resolved
    return set(_FALLBACK_ORCHESTRATION_ONLY)


def _calls_in(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Every bare-name ``Call`` target inside ``func``'s own body (not nested defs'
    bodies belonging to a DIFFERENT top-level function -- ``ast.walk`` naturally
    stays within this subtree since nested function bodies are still descendants)."""
    names: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.add(node.func.id)
    return names


def _reaches_preflight(
    start: str,
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    *,
    _visited: set[str] | None = None,
) -> bool:
    """DFS: does calling ``start`` (a function defined in this module) transitively
    reach a call to :data:`TARGET_FUNCTION`?"""
    visited = _visited if _visited is not None else set()
    if start in visited:
        return False
    visited.add(start)
    func = functions.get(start)
    if func is None:
        return False  # not a local function (e.g. an imported/stdlib name) -- dead end
    called = _calls_in(func)
    if TARGET_FUNCTION in called:
        return True
    return any(
        _reaches_preflight(callee, functions, _visited=visited)
        for callee in called
        if callee in functions
    )


def check(path: Path = SOURCE_SYNC_PATH) -> dict[str, str]:
    """Return ``{source_key: violation}`` for every non-exempt handler that does NOT
    reach :data:`TARGET_FUNCTION`. Empty dict = gate passes."""
    tree = _parse(path)
    functions = _module_functions(tree)
    handlers = _delta_handlers(tree)
    exempt = _orchestration_only_sources(tree)

    violations: dict[str, str] = {}
    for source_key, fn_name in sorted(handlers.items()):
        if source_key in exempt:
            continue
        if not fn_name:
            violations[source_key] = (
                "_DELTA_HANDLERS value is not a bare function reference -- "
                "cannot verify reachability statically"
            )
            continue
        if fn_name not in functions:
            violations[source_key] = f"handler function {fn_name!r} not found in module"
            continue
        if not _reaches_preflight(fn_name, functions):
            violations[source_key] = (
                f"{fn_name}() does not reach {TARGET_FUNCTION}() -- "
                "directly or via _apply_with_preflight/_apply_with_preflight_one"
            )
    return violations


def main() -> int:
    if "--list" in sys.argv:
        tree = _parse(SOURCE_SYNC_PATH)
        functions = _module_functions(tree)
        handlers = _delta_handlers(tree)
        exempt = _orchestration_only_sources(tree)
        for source_key, fn_name in sorted(handlers.items()):
            if source_key in exempt:
                print(f"{source_key}: EXEMPT (orchestration-only)")
                continue
            ok = (
                bool(fn_name)
                and fn_name in functions
                and _reaches_preflight(fn_name, functions)
            )
            print(f"{source_key} ({fn_name}): {'OK' if ok else 'MISSING'}")
        return 0

    try:
        violations = check()
    except _GateError as exc:
        print(f"backfeed-preflight gate FAILED (structural): {exc}", file=sys.stderr)
        return 1

    if violations:
        print(
            "backfeed-preflight gate FAILED -- the following _DELTA_HANDLERS entries "
            f"do not reach {TARGET_FUNCTION}():",
            file=sys.stderr,
        )
        for source_key, reason in sorted(violations.items()):
            print(f"  - {source_key}: {reason}", file=sys.stderr)
        return 1

    print(f"OK: every non-exempt _DELTA_HANDLERS entry reaches {TARGET_FUNCTION}().")
    return 0


if __name__ == "__main__":
    sys.exit(main())
