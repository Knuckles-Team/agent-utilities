#!/usr/bin/env python3
"""Corrects ``analyze_liveness.py``'s raw ``orphan_modules``/``dead_definitions``
findings before they are ratcheted (CONCEPT:AU-KG.maintenance.periodic-code-health,
GOC-68 D-GOC-68-LIVENESS).

``analyze_liveness.py`` (vendored from ``universal-skills``'s code-enhancer skill,
NOT owned by this repo) has a confirmed, located resolution bug: its
``_module_imported()`` computes a candidate module's dotted name RELATIVE TO
``agent_utilities/`` (e.g. ``kvcache.l2_native_connector``) but compares it
against import statements captured VERBATIM from the source (e.g.
``agent_utilities.kvcache.l2_native_connector`` for this repo's dominant
absolute-import style) — the prefix never matches, so the check is close to
inert for this codebase (see
``plans/graph-os-completion-program/designs/DEAD-CODE-INTENT-RECOVERY.md``
Section 1.2 for the full audit). ``dead_definitions`` has a separate, narrower
blind spot: a symbol reached only through a STRING (an entry-point target, a
``getattr(module, "Name")`` registry, a lazy ``__getattr__`` import map) never
produces an ``ast.Name``/``ast.Attribute`` reference, so it is invisible by
construction to a bare "does this name appear as an identifier" check.

This module does NOT reimplement import-graph resolution — it reuses
``check_wiring.py``'s already-correct ``collect_imports()``/``resolve_relative()``
(the same resolver that already powers ``check_surface_parity.py``'s hard gate),
per the "one capability, one entrypoint" edict and the design doc's own
Section 6.1 recommendation. It only ADDS the dynamic-reference checks
``check_wiring.py`` explicitly disclaims as out of its scope (its own docstring:
"Decorator / pkgutil dynamic registration ... entry-points ... Lazy / string-based
imports ... invisible").

Five mechanisms are checked, in order, for every raw finding:

1. **Corrected static import resolution** — ``check_wiring.collect_imports()``
   over every file (production AND test), matched against the finding's TRUE
   ``agent_utilities.``-prefixed dotted name.
2. **``pyproject.toml`` entry-points** (every ``[project.entry-points.*]`` group
   + ``[project.scripts]``) — plugin-discovery registration
   (``importlib.metadata.entry_points()``), invisible to any import-graph walk.
3. **``getattr(obj, "Name")`` registries** — a class/function reached only by a
   runtime string lookup (``gateway/registry.py``'s widget loader).
4. **``__getattr__`` lazy-import maps (PEP 562)** — a module-level dict literal,
   referenced from a ``__getattr__`` function, whose values are submodule name
   strings fed to ``importlib.import_module()`` — never an ``ast.Import`` node
   at all.
5. **Fail-open for genuinely dynamic, unresolvable import sites** —
   ``importlib.import_module(X)``/``__import__(X)`` where ``X`` is not a string
   literal (e.g. built from ``pkgutil.iter_modules()`` results). The detector
   cannot resolve these statically; a candidate whose own file sits in the same
   package directory as such a call is reported NOT dead rather than guessed at
   (a false positive costs a human five minutes; a false negative deletes
   working code — the task's explicit acceptance bar).

Generated files (header matches "do not edit" / "regenerate with") are excluded
from both categories entirely — see Section 6.1 fix #3 of the design doc: a
generated protocol-completeness surface is not "dead code the team wrote and
forgot", so counting it in the same bucket miscategorizes its lifecycle.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import tomllib
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent


def _load_check_wiring():
    """Load ``check_wiring.py`` by path (``scripts/`` ships an ``__init__.py``
    but this module must work whether or not ``scripts`` is importable as a
    package from the caller's ``sys.path`` — same loading idiom already proven
    by ``tests/gates/test_wire_first_gate.py``)."""
    spec = importlib.util.spec_from_file_location(
        "_check_wiring_for_liveness", REPO / "scripts" / "check_wiring.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_wiring = _load_check_wiring()

SRC_DIR = check_wiring.SRC_DIR
TESTS_DIR = check_wiring.TESTS_DIR
PYPROJECT = check_wiring.PYPROJECT

_GENERATED_MARKER_RE = re.compile(r"do not edit|regenerate with", re.IGNORECASE)
_RELATIVE_SPEC_RE = re.compile(
    r"^\.{1,3}[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)
_BARE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ABSOLUTE_MODULE_RE = re.compile(r"^agent_utilities(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")


def _iter_source_files(*roots: Path) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if root.exists():
            out.extend(check_wiring._tracked_or_walked(root, "*.py"))
    return [p for p in out if "__pycache__" not in p.parts]


def _rel(p: Path) -> str:
    return p.relative_to(REPO).as_posix()


def _parse(p: Path) -> ast.Module | None:
    try:
        return ast.parse(
            p.read_text(encoding="utf-8", errors="ignore"), filename=_rel(p)
        )
    except (OSError, SyntaxError):
        return None


def _module_dotted(mod: str) -> str:
    """``analyze_liveness.py``'s ``mod`` (computed relative to
    ``agent_utilities/``) -> the TRUE full dotted path."""
    return f"agent_utilities.{mod}"


def _module_file(mod: str) -> Path | None:
    as_file = SRC_DIR / (mod.replace(".", "/") + ".py")
    if as_file.exists():
        return as_file
    as_pkg = SRC_DIR / mod.replace(".", "/") / "__init__.py"
    if as_pkg.exists():
        return as_pkg
    return None


def _is_generated(path: Path | None) -> bool:
    """A file is "generated" only if its own MODULE DOCSTRING (not any string
    literal anywhere in the file — a generator script that emits the phrase
    into its OUTPUT, e.g. ``tools/codebase_map_tools.py`` writing "Do not edit
    manually — regenerate with ..." into the markdown file it produces, is not
    itself generated) carries the marker. Matches the confirmed convention
    (``protocols/epistemic_operations/_generated.py``: "Generated ... Regenerate
    with the protocol gate; do not edit.")."""
    if path is None or not path.exists():
        return False
    tree = _parse(path)
    if tree is None:
        return False
    docstring = ast.get_docstring(tree) or ""
    return bool(_GENERATED_MARKER_RE.search(docstring))


# --- mechanism 1: corrected static import resolution (reuses check_wiring.py) --


def _resolved_import_targets() -> set[str]:
    """Every dotted module name resolved CORRECTLY (absolute + relative) from
    an import statement anywhere under ``agent_utilities/`` or ``tests/`` —
    production or test evidence both count as "this module has a caller"
    (matches the design audit's own corrected-reachability methodology,
    DEAD-CODE-INTENT-RECOVERY.md Section 1.3)."""
    targets: set[str] = set()
    for p in _iter_source_files(SRC_DIR, TESTS_DIR):
        tree = _parse(p)
        if tree is None:
            continue
        targets |= check_wiring.collect_imports(_rel(p), tree)
    return targets


def _static_reachable(full: str, resolved: set[str]) -> bool:
    if full in resolved:
        return True
    # `from agent_utilities.foo import bar` also reaches submodule
    # `agent_utilities.foo.bar` (collect_imports already adds the
    # `{module}.{alias}` form for that shape).
    return any(t == full or t.startswith(full + ".") for t in resolved)


# --- mechanism 2: pyproject.toml entry-points (any group) + [project.scripts] --


def _entry_points() -> tuple[set[str], set[str]]:
    """(modules, symbol_names) declared as ``"module.path:Attr"`` anywhere in
    ``pyproject.toml`` — every ``[project.entry-points.*]`` group plus
    ``[project.scripts]``. A dotted-string plugin-discovery registration
    (``importlib.metadata.entry_points().select(...)``) is invisible to any
    import-graph walk by construction."""
    if not PYPROJECT.exists():
        return set(), set()
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data.get("project", {})
    modules: set[str] = set()
    symbols: set[str] = set()
    groups: list[dict[str, Any]] = []
    scripts = project.get("scripts")
    if isinstance(scripts, dict):
        groups.append(scripts)
    entry_points = project.get("entry-points", {})
    if isinstance(entry_points, dict):
        for group in entry_points.values():
            if isinstance(group, dict):
                groups.append(group)
    for group in groups:
        for target in group.values():
            if not isinstance(target, str) or ":" not in target:
                continue
            mod, _, attr = target.partition(":")
            modules.add(mod.strip())
            symbols.add(attr.strip().split(".")[0])
    return modules, symbols


# --- mechanism 3: getattr(module_or_obj, "Name") registries --------------------


def _getattr_symbol_names() -> set[str]:
    """Every literal string passed as the SECOND argument of a ``getattr(...)``
    call anywhere under ``agent_utilities/`` — the shape
    ``gateway/registry.py``'s ``getattr(module, "Widget", None)`` widget loader
    uses to reach a per-module class by a runtime name lookup, never an
    ``ast.Attribute``."""
    names: set[str] = set()
    for p in _iter_source_files(SRC_DIR):
        tree = _parse(p)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Name) and fn.id == "getattr"):
                continue
            if len(node.args) < 2:
                continue
            name_arg = node.args[1]
            if isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str):
                names.add(name_arg.value)
    return names


# --- mechanism 4: __getattr__ lazy-import maps (PEP 562), string-keyed ---------


def _resolve_lazy_spec(rel_path: str, spec: str) -> str | None:
    """Resolve a lazy-import dict VALUE string to a full dotted module name, by
    constructing the same ``ast.ImportFrom``-shaped input
    ``check_wiring.resolve_relative()`` already resolves correctly for a real
    import statement — reused, not reinvented, for a string that isn't one."""
    if _ABSOLUTE_MODULE_RE.match(spec):
        return spec
    if spec.startswith("."):
        level = len(spec) - len(spec.lstrip("."))
        remainder = spec[level:]
        if not remainder or not _RELATIVE_SPEC_RE.match("." * level + remainder):
            return None
        synthetic = ast.ImportFrom(module=remainder, names=[], level=level)
        return check_wiring.resolve_relative(rel_path, synthetic)
    if _BARE_IDENT_RE.match(spec):
        # `importlib.import_module(f".{module}", __name__)` — level=1 relative
        # to the containing file's own package (confirmed shape:
        # `agent_utilities/domains/finance/__init__.py`'s `_SYMBOL_MODULES`).
        synthetic = ast.ImportFrom(module=spec, names=[], level=1)
        return check_wiring.resolve_relative(rel_path, synthetic)
    return None


def _lazy_getattr_targets() -> set[str]:
    """Modules reachable ONLY through a module-level ``__getattr__`` that
    dynamically ``importlib.import_module()``s a submodule named by a STRING
    held in a dict literal — invisible to any AST import-node walk because
    there is no ``ast.Import``/``ast.ImportFrom`` node at all, only a runtime
    string. Confirmed shapes in this repo: ``agent_utilities/tools/__init__.py``'s
    ``_ATTR_MAP`` (values like ``".browser"``, dotted-relative) and
    ``agent_utilities/domains/finance/__init__.py``'s ``_SYMBOL_MODULES``
    (values like ``"alpha_factors"``, bare submodule name, fed to
    ``importlib.import_module(f".{module}", __name__)``).

    Scoped tightly to avoid false rescues: only a module-level dict literal
    whose OWN NAME is referenced inside a ``__getattr__`` function body in the
    SAME file qualifies — not "any dict literal in a file that happens to also
    define ``__getattr__``"."""
    targets: set[str] = set()
    for p in _iter_source_files(SRC_DIR):
        text = p.read_text(encoding="utf-8", errors="ignore")
        if "__getattr__" not in text or (
            "import_module" not in text and "__import__" not in text
        ):
            continue
        tree = _parse(p)
        if tree is None:
            continue
        getattr_fns = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "__getattr__"
        ]
        if not getattr_fns:
            continue
        referenced_names: set[str] = set()
        for fn in getattr_fns:
            referenced_names |= {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
        rel = _rel(p)
        for node in tree.body:
            target = None
            value = None
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                target, value = node.targets[0], node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target, value = node.target, node.value
            if target is None or target.id not in referenced_names:
                continue
            if not isinstance(value, ast.Dict):
                continue
            for v in value.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    resolved = _resolve_lazy_spec(rel, v.value)
                    if resolved:
                        targets.add(resolved)
    return targets


# --- mechanism 5: fail-open for genuinely dynamic, unresolvable import sites ---


def _dynamic_nonliteral_import_dirs() -> tuple[set[str], list[dict[str, Any]]]:
    """Package directories containing a call to
    ``importlib.import_module()``/``__import__()`` whose target is NOT a string
    literal — genuinely dynamic (e.g. an f-string built from
    ``pkgutil.iter_modules()`` results, or a variable). The detector cannot
    resolve these statically at all; per the task's fail-open requirement, a
    candidate whose own file sits in one of these directories is reported NOT
    dead rather than guessed at."""
    dirs: set[str] = set()
    sites: list[dict[str, Any]] = []
    for p in _iter_source_files(SRC_DIR):
        tree = _parse(p)
        if tree is None:
            continue
        rel = _rel(p)
        hit = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name not in ("import_module", "__import__"):
                continue
            if not node.args:
                continue
            arg0 = node.args[0]
            if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                continue  # a literal target is captured by mechanism 1 already
            hit = True
            sites.append({"file": rel, "line": node.lineno})
        if hit:
            dirs.add(str(Path(rel).parent))
    return dirs, sites


# --- mechanism 6: generic absolute string-literal reference --------------------


def _absolute_string_literal_refs() -> set[str]:
    """Every string CONSTANT anywhere (production or test) that is
    syntactically a full ``agent_utilities.``-prefixed dotted module path —
    catches a dict-of-dotted-path registry (``gateway/registry.py``'s
    ``_WIDGET_MODULES``) even outside the specific ``__getattr__``/entry-point
    shapes above."""
    refs: set[str] = set()
    for p in _iter_source_files(SRC_DIR, TESTS_DIR):
        tree = _parse(p)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if _ABSOLUTE_MODULE_RE.match(node.value):
                    refs.add(node.value)
    return refs


# --- composition -----------------------------------------------------------


def reconcile(details: dict[str, list[str]]) -> dict[str, Any]:
    """Re-check every raw ``orphan_modules``/``dead_definitions`` finding
    against the mechanisms above. Returns the corrected still-flagged lists
    plus full provenance (which mechanism rescued each finding, or which
    dynamic-import sites remain genuinely unresolvable) so the caller can
    report the correction transparently rather than silently."""
    orphan_raw = list(details.get("orphan_modules", []))
    dead_raw = list(details.get("dead_definitions", []))

    resolved_imports = _resolved_import_targets()
    ep_modules, ep_symbols = _entry_points()
    getattr_symbols = _getattr_symbol_names()
    lazy_targets = _lazy_getattr_targets()
    absolute_refs = _absolute_string_literal_refs()
    dynamic_dirs, dynamic_sites = _dynamic_nonliteral_import_dirs()

    orphan_still: list[str] = []
    orphan_rescued: list[dict[str, str]] = []
    orphan_excluded_generated: list[str] = []
    for mod in orphan_raw:
        path = _module_file(mod)
        if _is_generated(path):
            orphan_excluded_generated.append(mod)
            continue
        full = _module_dotted(mod)
        mechanism = None
        if _static_reachable(full, resolved_imports):
            mechanism = "corrected-static-import-resolution"
        elif full in ep_modules:
            mechanism = "pyproject-entry-point"
        elif full in lazy_targets:
            mechanism = "__getattr__-lazy-import-map"
        elif full in absolute_refs:
            mechanism = "string-literal-dynamic-reference"
        elif path is not None and str(path.parent.relative_to(REPO)) in dynamic_dirs:
            mechanism = "unresolvable-dynamic-import-fail-open"
        if mechanism:
            orphan_rescued.append({"module": mod, "mechanism": mechanism})
        else:
            orphan_still.append(mod)

    dead_still: list[str] = []
    dead_rescued: list[dict[str, str]] = []
    dead_excluded_generated: list[str] = []
    for entry in dead_raw:
        mod, _, name = entry.partition(":")
        path = _module_file(mod)
        if _is_generated(path):
            dead_excluded_generated.append(entry)
            continue
        mechanism = None
        if name in ep_symbols:
            mechanism = "pyproject-entry-point-suffix"
        elif name in getattr_symbols:
            mechanism = "getattr-registry"
        if mechanism:
            dead_rescued.append({"definition": entry, "mechanism": mechanism})
        else:
            dead_still.append(entry)

    return {
        "orphan_modules": {
            "still": sorted(orphan_still),
            "rescued": orphan_rescued,
            "excluded_generated": sorted(orphan_excluded_generated),
        },
        "dead_definitions": {
            "still": sorted(dead_still),
            "rescued": dead_rescued,
            "excluded_generated": sorted(dead_excluded_generated),
        },
        "dynamic_unresolvable_sites": dynamic_sites,
    }
