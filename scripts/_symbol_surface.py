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

Signature + class-member surface (module-level functions, class methods/attrs)
--------------------------------------------------------------------------------
Alongside plain name presence, this module also extracts, per module-level
public function, its caller-visible parameter list (name, kind, whether it
has a default, and a best-effort text form of the default expression), and
per module-level public class, the set of its public method/attribute names.
Both are captured in the SAME single ``ast.parse`` pass as everything else
above (no re-parsing). ``resolve_callable_surface``/``resolve_class_surface``
walk the re-export graph exactly like ``resolve_exposure`` (same transitive,
cycle-safe shape, via the shared ``_resolve_transitive_map`` helper) so a
facade re-export of a function/class is diffed under every dotted path a
consumer could plausibly import it through — most fleet consumers import the
package-root facade (``agent_utilities.initialize_workspace``), never the
true defining module (``agent_utilities.core.workspace.initialize_workspace``).

Deliberately NOT covered by the signature surface: method signatures on
classes (only method/attribute PRESENCE is diffed, not their parameters —
see ``resolve_class_surface``), decorator-rewritten signatures (e.g. a
``functools.wraps``-preserving decorator whose wrapper takes ``*args,
**kwargs`` is read literally, not resolved to the wrapped signature), and
anything reachable only via a re-export target outside the scanned package
(same blind spot as ``resolve_exposure`` above).
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

#: Python stdlib top-level module names (best-effort; used to exclude stdlib
#: imports from the "external package" footprint computed for the transitive
#: optional-dependency-drift check). Built once from ``sys.stdlib_module_names``
#: when available (3.10+); falls back to a short hardcoded core set otherwise.
STDLIB_MODULES: frozenset[str] = frozenset(getattr(sys, "stdlib_module_names", ())) | {
    "__future__",
    "_typeshed",
}

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
class ParamSignature:
    """One caller-visible parameter of a module-level function.

    ``kind`` mirrors :class:`inspect.Parameter.kind` names
    (``POSITIONAL_ONLY``/``POSITIONAL_OR_KEYWORD``/``VAR_POSITIONAL``/
    ``KEYWORD_ONLY``/``VAR_KEYWORD``). ``default_repr`` is a best-effort
    ``ast.unparse`` of the default expression (``None`` if there is no
    default, or unparse fails) — used only to detect that a default CHANGED,
    never to evaluate it.
    """

    name: str
    kind: str
    has_default: bool
    default_repr: str | None = None


@dataclass(frozen=True)
class CallableSurface:
    """A module-level function's caller-visible signature."""

    name: str
    params: tuple[ParamSignature, ...] = ()
    is_async: bool = False


@dataclass(frozen=True)
class ClassSurface:
    """A module-level class's public method/attribute NAME surface.

    Signatures of individual methods are deliberately not tracked (see module
    docstring) — only whether a public method/attribute is still present.
    ``__init__``/``__call__`` are included despite the leading underscores
    because they are genuine contract surface (constructor / call signature);
    other dunders are excluded as boilerplate.
    """

    name: str
    public_methods: frozenset[str] = field(default_factory=frozenset)
    public_attributes: frozenset[str] = field(default_factory=frozenset)


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
    #: Module-level public (non-``_``-prefixed) function/async-function defs,
    #: by name -- own definitions only, not resolved through re-exports (see
    #: ``resolve_callable_surface`` for the resolved/facade-aware view).
    own_callables: dict[str, CallableSurface] = field(default_factory=dict)
    #: Module-level public classes, by name -- own definitions only (see
    #: ``resolve_class_surface`` for the resolved/facade-aware view).
    own_classes: dict[str, ClassSurface] = field(default_factory=dict)
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


def _reexport_edges_from_importfrom(
    node: ast.ImportFrom,
    dotted: str,
    is_package: bool,
    package_name: str,
    *,
    allowed_names: set[str] | None,
) -> list[ReExport]:
    """Re-export edges contributed by a single ``from X import Y [as Z]``.

    Shared by ``_top_level_reexports`` (``allowed_names=None``: take every
    non-``_``-prefixed local name) and ``_getattr_reexports`` (``allowed_names``
    set to the literal name(s) guarded by the enclosing ``if name == "X"``/
    ``if name in [...]`` branch of a ``__getattr__`` dispatcher: take only
    names that branch actually guards, regardless of leading underscore,
    since the guard itself is the publicity signal there).
    """
    out: list[ReExport] = []
    if node.level and node.level > 0:
        target_module = _resolve_relative_module(
            dotted, is_package, node.level, node.module
        )
    else:
        target_module = node.module or ""
    if not (
        target_module == package_name or target_module.startswith(package_name + ".")
    ):
        return out  # external re-export target — not ours to track
    for alias in node.names:
        if alias.name == "*":
            if allowed_names is None:
                out.append(ReExport("*", target_module, STAR))
            continue
        local = alias.asname or alias.name
        if allowed_names is None:
            if local.startswith("_"):
                continue
        elif local not in allowed_names:
            continue
        out.append(ReExport(local, target_module, alias.name))
    return out


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
            out.extend(
                _reexport_edges_from_importfrom(
                    node, dotted, is_package, package_name, allowed_names=None
                )
            )
    return out


def _getattr_guard_names(test: ast.expr) -> set[str] | None:
    """Literal name(s) tested by ``name == "X"`` or ``name in [...]`` inside a
    module-level ``def __getattr__(name):`` dispatcher branch, else ``None``
    if the test isn't one of these two recognized literal-comparison shapes
    (a computed/non-literal guard is invisible — same "can't see it, won't
    guess" bias as the rest of this module).
    """
    if not (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and len(test.comparators) == 1
    ):
        return None
    left = test.left
    if not (isinstance(left, ast.Name) and left.id == "name"):
        return None
    op = test.ops[0]
    comp = test.comparators[0]
    if (
        isinstance(op, ast.Eq)
        and isinstance(comp, ast.Constant)
        and isinstance(comp.value, str)
    ):
        return {comp.value}
    if isinstance(op, ast.In) and isinstance(comp, (ast.List, ast.Tuple, ast.Set)):
        names = {
            elt.value
            for elt in comp.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        }
        return names or None
    return None


def _getattr_reexports(
    tree: ast.Module, dotted: str, is_package: bool, package_name: str
) -> list[ReExport]:
    """Re-export edges hidden inside a module-level ``def __getattr__(name):``
    lazy-import dispatcher (PEP 562) — a pattern ``agent_utilities/__init__.py``
    itself uses pervasively (dozens of ``elif name == "X":``/``elif name in
    [...]:`` branches, each doing a local ``from .mod import X`` and returning
    it) to avoid the eager-import cost of a plain top-level re-export at
    package-import time. ``_top_level_reexports`` cannot see these — the
    import statement lives inside a function body, not at module top level —
    yet this is the DOMINANT re-export mechanism for this package's own root
    facade, which is also the dotted path most fleet consumers actually
    import through (e.g. ``agent_utilities.initialize_workspace``, never
    ``agent_utilities.core.workspace.initialize_workspace``). Without this,
    the signature/class-surface checks would have almost no real coverage
    against this codebase's actual shape.
    """
    out: list[ReExport] = []
    getattr_fn = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
        ),
        None,
    )
    if getattr_fn is None:
        return out

    def walk_branch(node: ast.If) -> None:
        guard_names = _getattr_guard_names(node.test)
        if guard_names:
            for stmt in node.body:
                if isinstance(stmt, ast.ImportFrom):
                    out.extend(
                        _reexport_edges_from_importfrom(
                            stmt,
                            dotted,
                            is_package,
                            package_name,
                            allowed_names=guard_names,
                        )
                    )
        for sub in node.orelse:
            if isinstance(sub, ast.If):
                walk_branch(sub)

    for node in getattr_fn.body:
        if isinstance(node, ast.If):
            walk_branch(node)
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


def _default_repr(node: ast.expr | None) -> str | None:
    """Best-effort source text of a default-value expression, for CHANGE
    detection only -- never evaluated, never used to infer runtime behavior.
    """
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - ast.unparse is robust; defensive only
        return "<unrepresentable>"


def _params_from_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ParamSignature, ...]:
    """Caller-visible parameter list of a function def, in declaration order.

    Mirrors :class:`inspect.Signature` construction from an AST ``arguments``
    node: positional-only, then positional-or-keyword (both share one
    trailing-aligned ``defaults`` list per Python's own grammar), then
    ``*args``, then keyword-only (each with its own optional default), then
    ``**kwargs``.
    """
    args = node.args
    combined_pos = [*args.posonlyargs, *args.args]
    n_defaults = len(args.defaults)
    default_offset = len(combined_pos) - n_defaults
    params: list[ParamSignature] = []
    for i, a in enumerate(args.posonlyargs):
        has_default = i >= default_offset
        default_node = args.defaults[i - default_offset] if has_default else None
        params.append(
            ParamSignature(
                a.arg, "POSITIONAL_ONLY", has_default, _default_repr(default_node)
            )
        )
    for j, a in enumerate(args.args):
        i = len(args.posonlyargs) + j
        has_default = i >= default_offset
        default_node = args.defaults[i - default_offset] if has_default else None
        params.append(
            ParamSignature(
                a.arg, "POSITIONAL_OR_KEYWORD", has_default, _default_repr(default_node)
            )
        )
    if args.vararg:
        params.append(ParamSignature(args.vararg.arg, "VAR_POSITIONAL", False, None))
    for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        params.append(
            ParamSignature(a.arg, "KEYWORD_ONLY", d is not None, _default_repr(d))
        )
    if args.kwarg:
        params.append(ParamSignature(args.kwarg.arg, "VAR_KEYWORD", False, None))
    return tuple(params)


def _own_callables(tree: ast.Module) -> dict[str, CallableSurface]:
    """Module-level public ``def``/``async def`` statements, by name."""
    out: dict[str, CallableSurface] = {}
    for node in tree.body:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and not node.name.startswith("_"):
            out[node.name] = CallableSurface(
                name=node.name,
                params=_params_from_function(node),
                is_async=isinstance(node, ast.AsyncFunctionDef),
            )
    return out


def _class_surface(node: ast.ClassDef) -> ClassSurface:
    methods: set[str] = set()
    attrs: set[str] = set()
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not item.name.startswith("_") or item.name in ("__init__", "__call__"):
                methods.add(item.name)
        elif isinstance(item, ast.Assign):
            for t in item.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("_"):
                    attrs.add(t.id)
        elif isinstance(item, ast.AnnAssign):
            t = item.target
            if isinstance(t, ast.Name) and not t.id.startswith("_"):
                attrs.add(t.id)
    return ClassSurface(
        name=node.name,
        public_methods=frozenset(methods),
        public_attributes=frozenset(attrs),
    )


def _own_classes(tree: ast.Module) -> dict[str, ClassSurface]:
    """Module-level public class defs, by name, with their method/attr surface."""
    out: dict[str, ClassSurface] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            out[node.name] = _class_surface(node)
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
    reexports = _top_level_reexports(
        tree, dotted, is_package, package_name
    ) + _getattr_reexports(tree, dotted, is_package, package_name)
    eager = _eager_external_packages(tree, package_name)
    return ModuleSurface(
        dotted=dotted,
        relpath=relpath,
        is_package=is_package,
        own_symbols=frozenset(own),
        all_names=frozenset(allnames),
        reexports=tuple(reexports),
        eager_external_packages=frozenset(eager),
        own_callables=_own_callables(tree),
        own_classes=_own_classes(tree),
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


_T = TypeVar("_T")


def _resolve_transitive_map(
    modules: dict[str, ModuleSurface], own: Callable[[ModuleSurface], dict[str, _T]]
) -> dict[str, dict[str, _T]]:
    """Shared re-export graph walk behind ``resolve_callable_surface`` and
    ``resolve_class_surface`` — same transitive, cycle-safe shape as
    ``resolve_exposure`` above, generalized to carry a PAYLOAD (a
    ``CallableSurface``/``ClassSurface``, not just presence) through the
    re-export chain, so a fact about the DEFINING module (its function's
    parameter list, a class's method set) is visible at every facade dotted
    path a consumer might import it through.

    Deliberately does NOT special-case ``__all__`` the way ``resolve_exposure``
    does (a name merely LISTED in ``__all__`` carries no signature/class-shape
    payload to propagate) — this is a narrower, payload-carrying sibling, not
    a drop-in replacement; ``resolve_exposure`` is left untouched.
    """
    memo: dict[str, dict[str, _T]] = {}
    in_progress: set[str] = set()

    def resolve(m: str) -> dict[str, _T]:
        if m in memo:
            return memo[m]
        if m in in_progress:
            return {}  # cycle guard — no new info from here
        surf = modules.get(m)
        if surf is None or surf.parse_error:
            memo[m] = {}
            return memo[m]
        in_progress.add(m)
        out: dict[str, _T] = dict(own(surf))
        for edge in surf.reexports:
            if edge.target_name == STAR:
                if edge.target_module in modules:
                    for name, payload in resolve(edge.target_module).items():
                        out.setdefault(name, payload)
                continue
            if edge.target_module in modules:
                target = resolve(edge.target_module)
                if edge.target_name in target:
                    out[edge.local_name] = target[edge.target_name]
                # else: target module IS scanned and does not (or no longer)
                # carry this payload — correctly leave it unset here too.
            # else: target outside the scanned set — can't verify, don't guess
            # (matches resolve_exposure's conservative bias, but here "don't
            # guess" means "no payload", i.e. this check simply can't compare
            # a signature/class-shape it can't see).
        in_progress.discard(m)
        memo[m] = out
        return out

    return {m: resolve(m) for m in modules}


def resolve_callable_surface(
    modules: dict[str, ModuleSurface],
) -> dict[str, dict[str, CallableSurface]]:
    """For each module, ``{name: CallableSurface}`` for every module-level
    public function reachable as ``module.name`` — own definitions plus
    re-export targets resolved TRANSITIVELY (see ``_resolve_transitive_map``).
    Powers the signature/contract-change check: most fleet consumers import a
    function through a package-root facade, never its true defining module,
    so the facade path needs the SAME signature payload as the origin.
    """
    return _resolve_transitive_map(modules, lambda s: dict(s.own_callables))


def resolve_class_surface(
    modules: dict[str, ModuleSurface],
) -> dict[str, dict[str, ClassSurface]]:
    """For each module, ``{name: ClassSurface}`` for every module-level public
    class reachable as ``module.name`` — own definitions plus re-export
    targets resolved TRANSITIVELY (see ``_resolve_transitive_map``). Powers
    the class/attribute-surface check (removed public methods/attributes on
    an exported class).
    """
    return _resolve_transitive_map(modules, lambda s: dict(s.own_classes))
