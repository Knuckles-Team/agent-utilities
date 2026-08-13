#!/usr/bin/env python3
"""AST-based public-API-surface extraction, shared by the removed-symbol-consumer
gate (``check_removed_symbol_consumers.py``) and its consumer-index generator
(``gen_fleet_symbol_consumers.py``).

Deliberately stdlib-only (``ast`` + ``tomllib``, no third-party imports, no
``import agent_utilities``) so neither caller needs the managed ``.venv`` or a
working install of the package being scanned -- unlike ``check_surface_parity.py``
/ ``check_cpd.py``, which import the live MCP server surface and therefore need
``scripts/_gate_interpreter.py``'s re-exec dance. A gate that scans *source text*
has no such dependency, and avoiding it also avoids the documented
``_gate_interpreter``/``os.execve`` footgun (see that module's docstring and
MEMORY ``realpath-on-uv-venv-execve-kills-pytest``).

Public surface definition
--------------------------
"Public" = module-level ``def``/``class``/assignment not prefixed with ``_``,
PLUS anything listed in ``__all__`` (union, not a restriction). This module
additionally, DELIBERATELY, treats a top-level ``from X import Y [as Z]`` /
``import X [as Z]`` as introducing the local binding ``Z`` (or ``Y``) too --
a re-export -- because a plain import statement IS a module-level name
binding in Python's own semantics, and this codebase leans on that pattern
throughout (e.g. ``agent_utilities/core/config.py`` re-exports ``setting``
from ``agent_utilities/core/_env.py``; ~148 fleet call sites import it as
``agent_utilities.core.config.setting``, never touching ``_env`` directly).
Treating only true local defs as "public" would make the gate blind to
exactly this shape — a facade module whose upstream definition disappears
while its own `from .x import y` line is untouched (the import line still
parses; whether it still RESOLVES depends on the upstream module, which is
why re-export edges are resolved TRANSITIVELY in ``resolve_exposure`` below,
not just recorded one level deep).

A re-export is only as public as what it points to: ``resolve_exposure``
walks the whole per-ref module graph and computes, for each module, the
closure of names actually reachable through it — own definitions, plus
whatever its re-export targets themselves expose (recursively, with a cycle
guard). A name whose upstream definition is deleted stops being exposed
through every facade that only re-exported it, without needing per-facade
special-casing.

Known blind spot: a re-export target outside the scanned package (a 3rd-party
symbol, or a target this scan didn't reach) can't be verified either way, so
it is conservatively treated as still-exposed — this gate does not flag
things it cannot see clearly, per its own "don't overclaim coverage" rule.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: Python stdlib top-level module names (best-effort; used to exclude stdlib
#: imports from the "external package" footprint computed for the transitive
#: optional-dependency-drift check). Built once from ``sys.stdlib_module_names``
#: when available (3.10+); falls back to a short hardcoded core set otherwise.
STDLIB_MODULES: frozenset[str] = frozenset(
    getattr(sys, "stdlib_module_names", ())
) | {"__future__", "_typeshed"}

#: A re-export whose target module couldn't be resolved to a real dotted path
#: (e.g. an unparseable relative import) or that points outside the package
#: sentinel target name meaning "everything the target module exposes"
#: (``from X import *``).
STAR = "*"


@dataclass(frozen=True)
class ReExport:
    local_name: str
    target_module: str
    target_name: str  # STAR for `from X import *`


@dataclass(frozen=True)
class ModuleSurface:
    """One module's own definitions, re-export edges, and eager-import footprint.

    ``public_symbols`` is FILLED IN LAZILY by ``resolve_exposure`` (it starts
    empty from ``parse_module_surface``) — computing it correctly requires the
    whole per-ref module graph, not just this one file, so it can't be known
    at single-file parse time. Callers that need the resolved surface must go
    through ``resolve_exposure``.
    """

    dotted: str  # e.g. "agent_utilities.mcp.server_factory"
    relpath: str  # e.g. "agent_utilities/mcp/server_factory.py"
    is_package: bool = False  # True if sourced from an __init__.py
    own_symbols: frozenset[str] = field(default_factory=frozenset)
    all_names: frozenset[str] = field(default_factory=frozenset)
    reexports: tuple[ReExport, ...] = field(default_factory=tuple)
    #: Top-level external (non-stdlib, non-``agent_utilities``) package names
    #: imported unconditionally at module import time (see ``_eager_external_packages``).
    eager_external_packages: frozenset[str] = field(default_factory=frozenset)
    parse_error: str | None = None


def module_dotted_name(package_root: Path, file_path: Path, package_name: str) -> str:
    """``agent_utilities/mcp/server_factory.py`` -> ``agent_utilities.mcp.server_factory``.

    ``__init__.py`` maps to its containing package (the ``__init__`` component
    is dropped), matching how Python callers actually address it.
    """
    rel = file_path.relative_to(package_root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].removesuffix(".py")
    return ".".join([package_name, *parts]) if parts else package_name


def _resolve_relative_module(
    dotted: str, is_package: bool, level: int, node_module: str | None
) -> str:
    """Absolute dotted target of a relative ``from . import ...`` statement.

    ``level=1`` refers to the module's OWN package if it is itself a package
    (``__init__.py``), else to its containing package. Each further level
    walks up one more package.
    """
    parts = dotted.split(".")
    if not is_package:
        parts = parts[:-1]
    up = level - 1
    if up > 0:
        parts = parts[: len(parts) - up] if up < len(parts) else []
    base = ".".join(parts)
    if node_module:
        return f"{base}.{node_module}" if base else node_module
    return base


def _all_names(tree: ast.Module) -> set[str]:
    """Names listed in a top-level ``__all__ = [...]``/``(...)`` assignment."""
    names: set[str] = set()
    for node in tree.body:
        targets = None
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.target is not None:
            targets = [node.target]
        if not targets:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            continue
        value = node.value
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            for elt in value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    names.add(elt.value)
    return names


def _own_top_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("_"):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign):
            t = node.target
            if isinstance(t, ast.Name) and not t.id.startswith("_"):
                names.add(t.id)
    return names


def _top_level_reexports(
    tree: ast.Module, dotted: str, is_package: bool, package_name: str
) -> list[ReExport]:
    """Top-level import statements as re-export edges (local_name -> target)."""
    out: list[ReExport] = []
    for node in tree.body:
        # `import a.b.c [as d]` is deliberately NOT treated as a symbol
        # re-export: without `as`, it binds only the top-level package name
        # (`a`) in the local namespace, not a member of THIS module; with
        # `as`, it binds a module OBJECT, not a value re-exported from a
        # symbol namespace — a different relationship than `from x import y`.
        # Plain `import agent_utilities.x.y` disappearing entirely (the whole
        # module file deleted) is still caught, separately, by
        # ``find_removed_symbols``'s whole-module-missing-at-head path.
        if isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                target_module = _resolve_relative_module(
                    dotted, is_package, node.level, node.module
                )
            else:
                target_module = node.module or ""
            if not (
                target_module == package_name
                or target_module.startswith(package_name + ".")
            ):
                continue  # external re-export target — not ours to track
            for alias in node.names:
                if alias.name == "*":
                    out.append(ReExport("*", target_module, STAR))
                    continue
                local = alias.asname or alias.name
                if local.startswith("_"):
                    continue
                out.append(ReExport(local, target_module, alias.name))
    return out


def _is_type_checking_guard(node: ast.If) -> bool:
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    return bool(isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


def _eager_external_packages(tree: ast.Module, package_name: str) -> set[str]:
    """Top-level external packages imported unconditionally at module-import time.

    Only walks depth-0 statements, and skips anything inside a ``try:`` block
    (the standard "optional dependency" guard shape) or an
    ``if TYPE_CHECKING:`` block (never executed at runtime). This is a
    deliberately narrow, best-effort signal for the transitive optional-
    dependency-drift check -- see that check's docstring for what it does and
    does not cover.
    """
    out: set[str] = set()

    def _walk_import(node: ast.Import | ast.ImportFrom) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                if top != package_name and top not in STDLIB_MODULES:
                    out.add(top)
        else:  # ImportFrom
            if node.level and node.level > 0:
                return  # relative import — always internal to this package
            mod = node.module or ""
            top = mod.split(".", 1)[0]
            if top and top != package_name and top not in STDLIB_MODULES:
                out.add(top)

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _walk_import(node)
        # Deliberately NOT descending into ast.Try (optional-import guard) or
        # ast.If(TYPE_CHECKING) bodies — those are the two standard "this
        # import does not run unconditionally" shapes. Any other top-level
        # ``if`` (e.g. platform sniffing) is conservatively treated as eager
        # since it can still execute on some/most platforms.
        elif isinstance(node, ast.If) and not _is_type_checking_guard(node):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    _walk_import(sub)
    return out


def parse_module_surface(
    source: str, dotted: str, relpath: str, package_name: str
) -> ModuleSurface:
    is_package = relpath.endswith("__init__.py")
    try:
        tree = ast.parse(source, filename=relpath)
    except SyntaxError as exc:
        return ModuleSurface(
            dotted=dotted, relpath=relpath, is_package=is_package, parse_error=str(exc)
        )
    own = _own_top_level_names(tree)
    allnames = _all_names(tree)
    reexports = _top_level_reexports(tree, dotted, is_package, package_name)
    eager = _eager_external_packages(tree, package_name)
    return ModuleSurface(
        dotted=dotted,
        relpath=relpath,
        is_package=is_package,
        own_symbols=frozenset(own),
        all_names=frozenset(allnames),
        reexports=tuple(reexports),
        eager_external_packages=frozenset(eager),
    )


def surface_from_worktree(
    package_root: Path, package_name: str
) -> dict[str, ModuleSurface]:
    """Per-file surface of every ``*.py`` file under ``package_root`` on disk.

    Returns UNRESOLVED surfaces (``own_symbols``/``reexports``, not the
    transitively-resolved exposure) — pass the result through
    ``resolve_exposure`` before diffing two refs.
    """
    out: dict[str, ModuleSurface] = {}
    for path in sorted(package_root.rglob("*.py")):
        rel = path.relative_to(package_root).as_posix()
        if "/tests/" in f"/{rel}" or rel.startswith("tests/"):
            continue
        dotted = module_dotted_name(package_root, path, package_name)
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            out[dotted] = ModuleSurface(
                dotted=dotted,
                relpath=f"{package_name}/{rel}",
                is_package=rel.endswith("__init__.py"),
                parse_error=str(exc),
            )
            continue
        out[dotted] = parse_module_surface(
            source, dotted, f"{package_name}/{rel}", package_name
        )
    return out


def resolve_exposure(modules: dict[str, ModuleSurface]) -> dict[str, frozenset[str]]:
    """For each module, the full set of names importable as ``module.NAME``.

    Own definitions plus re-export targets resolved TRANSITIVELY (a facade
    re-exporting a name whose upstream definition is gone no longer exposes
    it). Cycle-safe (mutual re-export loops resolve to whatever they can
    prove, never infinite-recurse). A re-export target outside the scanned
    module set is conservatively treated as still providing its name (can't
    see it, won't guess it's gone — see module docstring).
    """
    memo: dict[str, frozenset[str]] = {}
    in_progress: set[str] = set()

    def resolve(m: str) -> frozenset[str]:
        if m in memo:
            return memo[m]
        if m in in_progress:
            return frozenset()  # cycle guard — no new info from here
        surf = modules.get(m)
        if surf is None or surf.parse_error:
            memo[m] = frozenset()
            return memo[m]
        in_progress.add(m)
        names: set[str] = set(surf.own_symbols)
        for edge in surf.reexports:
            if edge.target_name == STAR:
                if edge.target_module in modules:
                    names |= resolve(edge.target_module)
                # else: can't see the star-import source; contributes nothing
                # provable (not the same as "removed" — just unknown).
                continue
            if edge.target_module in modules:
                if edge.target_name in resolve(edge.target_module):
                    names.add(edge.local_name)
                # else: target module IS scanned and does NOT expose this name
                # (deleted/renamed upstream) — correctly do not add it.
            else:
                # Target outside the scanned set: can't verify, don't guess.
                names.add(edge.local_name)
        names |= surf.all_names
        in_progress.discard(m)
        memo[m] = frozenset(names)
        return memo[m]

    return {m: resolve(m) for m in modules}
