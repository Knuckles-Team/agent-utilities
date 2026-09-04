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

Wire-First gate (D-OB-9 / D-OB-13 / D-OB-16)
---------------------------------------------
The module-import BFS above answers "does anything import this file at
all?" — file-level reachability. It is blind to the failure mode that
motivated D-OB-9/13/16: a module that IS imported (its tests import it,
even a sibling production module imports it) but whose specific public
class/method is never actually CALLED from any non-test code — "reachable"
by import, never "invoked". Four extra sweeps close that gap, each
independently runnable and combined by ``--wire-first-report``:

* ``--check-test-collection`` — test files under ``tests/`` that
  ``pytest.ini``'s ``testpaths`` does not collect AND no pre-commit hook /
  CI workflow explicitly points ``pytest`` at (parsed out of
  ``.pre-commit-config.yaml`` / ``.github/workflows/*.yml`` by regex, so
  this can't silently drift from what those files actually run). Enforced
  as an ABSOLUTE ZERO (D-OB-13a) — the repo is measured at zero orphans
  today (see "NO BASELINE HERE ANY MORE" below), so any orphan fails.
* ``--check-mock-hygiene`` — ``MagicMock(spec=[])`` / ``Mock(spec=[])`` /
  ``patch(..., create=True)`` sites in test files. Always informational
  (exit 0): whether a given site is a legitimate object-isolation mock or a
  dangerous "fake an entire module that might not exist" mock needs human
  judgement (see D-OB-13b) — this only makes every site visible for review.
* ``--check-extras-gating`` — ``except ImportError``/``ModuleNotFoundError``
  handlers in test files whose body neither re-raises nor calls
  ``pytest.skip``/``.fail``/``.xfail``/``importorskip`` — i.e. the test
  silently no-ops instead of visibly skipping when an optional extra is
  absent (D-OB-16's failure mode). Always informational (exit 0).
* ``--check-symbol-reachability`` — the core new AST sweep: every public
  top-level class/function and public method defined under
  ``agent_utilities/`` is cross-referenced (word-boundary text search, not
  full type-resolved call-graph — same class of heuristic as
  ``check_swallowed_errors.py``) against every OTHER file under
  ``agent_utilities/`` and every file under ``tests/``. A symbol referenced
  ONLY from test files (zero non-test, non-defining-file references) is a
  "public capability entrypoint with no non-test caller" — exactly the
  D-OB-9 shape (``PolicyEngine``, KV-fork ``snapshot``/``fork``/
  ``branch_get``/``branch_put``, ``AdmissionPolicy.decide``, …).
  DIFF-SCOPED against HEAD (see "NO BASELINE HERE ANY MORE" below) — new
  test-only symbols since HEAD fail, the existing backlog does not.

Known blind spots of the symbol sweep specifically: word-boundary text
matching (not import/type resolution) means an alias (``import X as Y``)
or a same-named symbol in an unrelated module can produce false negatives
or false positives; a handful of extremely generic method names (``run``,
``get``, ``close``, …) are excluded from method-level checking for that
reason — rely on the class-level finding for those. A "0 non-test
references" result is signal for a human to trace the live path (Wire-First
step 1), not proof of dead code to delete on sight.

NO BASELINE HERE ANY MORE (retired ratchet — see check_swallowed_errors.py's
module docstring for the fully-worked-out rationale this gate now follows,
and the "Diff-scoped enforcement" comment further down this file for the
measured counts and the extraction-invariance test that justified keeping
this gate's ``(file, symbol, ordinal)`` key rather than replacing it).

Usage::

    python scripts/check_wiring.py --wire-first-report          # everything, human-readable
    python scripts/check_wiring.py --check-test-collection
    python scripts/check_wiring.py --check-mock-hygiene
    python scripts/check_wiring.py --check-extras-gating
    python scripts/check_wiring.py --check-symbol-reachability

Exit 0 = no new orphan / no new test-only symbol since HEAD, 1 = one was
found (or a degraded HEAD comparison could not be safely made), 2 = a
retired flag was passed.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import tokenize
from collections import Counter, defaultdict, deque
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "agent_utilities"
TESTS_DIR = ROOT / "tests"
PYPROJECT = ROOT / "pyproject.toml"
PYTEST_INI = ROOT / "pytest.ini"
PRECOMMIT_CONFIG = ROOT / ".pre-commit-config.yaml"
WORKFLOWS_DIR = ROOT / ".github" / "workflows"


def _tracked_or_walked(root: Path, pattern: str) -> list[Path]:
    """Files matching ``pattern`` under ``root``, preferring git-tracked (BUG-043).

    A raw ``rglob`` also picks up gitignored, generated build output, which
    can distort the import graph / wire-first sweep with a stale copy of an
    already-fixed source file. Falls back to a filesystem walk only when
    ``root`` is not inside a git working tree (e.g. a synthetic test fixture).

    Anchored at ``ROOT`` (never at ``root`` itself) with a pathspec scoped to
    ``root``'s position under it, because git's ``ls-files`` output-path base
    is NOT reliably "relative to ``-C``'s target" -- it silently reverts to
    the ambient work-tree root whenever ``GIT_DIR``/``GIT_INDEX_FILE`` are
    already present in the process environment, which git itself sets for
    *every* hook subprocess (``git commit``, incl. concluding a merge).
    Confirmed empirically (GOC-70 liveness investigation): under a plain
    invocation, ``git -C agent_utilities ls-files -- '*.py'`` returns
    ``agent_utilities``-relative lines ("__init__.py", ...); with
    ``GIT_DIR``/``GIT_INDEX_FILE`` set (as they are inside every git hook
    subprocess), the SAME command instead returns work-tree-root-relative
    lines ("agent_utilities/__init__.py", ...) while ``-C``'s own
    ``rev-parse --show-toplevel``/``--show-prefix`` simultaneously claim
    ``agent_utilities`` itself is the toplevel with an empty prefix -- an
    internally inconsistent, ambient-env-dependent git quirk, not a rare
    edge case. The old code reconstructed absolute paths as ``root / line``,
    which under that ambient condition doubled the ``agent_utilities/``
    segment onto a path that never exists on disk; every reconstructed path
    then failed ``is_file()`` and got filtered to an EMPTY list -- but
    because the raw ``tracked`` list was non-empty, the function returned
    that empty list directly instead of falling through to the (correct)
    ``rglob`` fallback. Every rescue mechanism that depends on this function
    (all but the pyproject-entry-point one) then saw an empty import graph,
    manufacturing a false "196 new dead orphan modules" regression on a
    ratchet gate that runs as a git hook -- the exact "component that cannot
    do its job returns a value read as a real signal" shape AGENTS.md's
    *Fail closed* section prohibits. Anchoring at ``ROOT`` sidesteps the
    ambiguity entirely: ``ROOT`` is computed once via pure ``Path`` math
    (``Path(__file__).resolve().parent.parent``), so it is always the
    correct work-tree root regardless of what git's ambient environment
    claims, and reconstructing every path as ``ROOT / line`` is then correct
    under both the plain and the ambient-env-polluted invocation shapes.
    """
    try:
        rel = root.relative_to(ROOT)
        anchor = ROOT
        pathspec = pattern if str(rel) == "." else f"{rel.as_posix()}/{pattern}"
    except ValueError:
        # `root` is not under this repo's ROOT at all (e.g. a synthetic test
        # fixture rooted at its own tmp_path with no relation to ROOT) --
        # nothing to anchor to; preserve the prior `-C root` behavior, which
        # is correct there since no ambient GIT_DIR of THIS repo applies.
        anchor = root
        pathspec = pattern
    try:
        out = subprocess.run(
            ["git", "-C", str(anchor), "ls-files", "--", pathspec],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        tracked = [anchor / line for line in out.splitlines() if line]
        if tracked:
            return [p for p in tracked if p.is_file()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return sorted(root.rglob(pattern))


# Method names common enough (as ordinary verbs on many unrelated classes)
# that a global word-boundary usage count is too noisy to trust at the
# method level. The class-level check still covers these classes.
_GENERIC_METHOD_STOPLIST = {
    "run",
    "start",
    "stop",
    "close",
    "get",
    "set",
    "add",
    "remove",
    "update",
    "delete",
    "execute",
    "process",
    "handle",
    "validate",
    "build",
    "create",
    "init",
    "setup",
    "teardown",
    "load",
    "save",
    "to_dict",
    "from_dict",
    "as_dict",
    "dict",
    "json",
    "str",
    "repr",
    "copy",
    "clone",
    "reset",
    "open",
    "read",
    "write",
    "call",
    "apply",
}

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


def _imports_from_import(node: ast.Import) -> set[str]:
    return {alias.name for alias in node.names}


def _imports_from_relative_from(rel_path: str, node: ast.ImportFrom) -> set[str]:
    """``from . import x`` / ``from ..pkg import mod`` -- resolved via
    :func:`resolve_relative`. ``from .pkg import mod`` may target submodules,
    so both the resolved package and each submodule are recorded."""
    resolved = resolve_relative(rel_path, node)
    if not resolved:
        return set()
    found = {resolved}
    found.update(f"{resolved}.{alias.name}" for alias in node.names or [])
    return found


def _imports_from_absolute_from(node: ast.ImportFrom) -> set[str]:
    if not node.module:
        return set()
    found = {node.module}
    found.update(f"{node.module}.{alias.name}" for alias in node.names or [])
    return found


def collect_imports(rel_path: str, tree: ast.AST) -> set[str]:
    """All dotted module names a file imports (absolute + resolved relative)."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(_imports_from_import(node))
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                found.update(_imports_from_relative_from(rel_path, node))
            else:
                found.update(_imports_from_absolute_from(node))
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


def _parse_file_imports(py_file: Path) -> tuple[str, set[str]] | None:
    """``(rel_path, imports)`` for one source file, or None if it should be
    skipped entirely (``__pycache__``). A parse failure yields an empty
    import set rather than dropping the file, so it still counts as a
    known module."""
    if "__pycache__" in py_file.parts:
        return None
    rel = py_file.relative_to(ROOT).as_posix()
    try:
        tree = ast.parse(
            py_file.read_text(encoding="utf-8", errors="ignore"), filename=rel
        )
    except SyntaxError:
        return rel, set()
    return rel, collect_imports(rel, tree)


def _ancestor_init_edges(target: str, modules: set[str], rel: str) -> set[str]:
    """Importing a module also executes every ancestor package ``__init__``
    — model those edges so package inits are not falsely orphaned when only
    deep submodules are imported."""
    edges: set[str] = set()
    parent = Path(target).parent
    while parent != Path("."):
        init = (parent / "__init__.py").as_posix()
        if init in modules and init != rel:
            edges.add(init)
        parent = parent.parent
    return edges


def _add_import_edges(
    graph: dict[str, set[str]], rel: str, imps: set[str], modules: set[str]
) -> None:
    for modname in imps:
        target = module_name_to_path(modname, modules)
        if target and target != rel:
            graph[rel].add(target)
            graph[rel].update(_ancestor_init_edges(target, modules, rel))


def build_graph() -> tuple[dict[str, set[str]], set[str]]:
    """Return (import_graph, modules) over agent_utilities/."""
    modules: set[str] = set()
    file_imports: dict[str, set[str]] = {}

    for py_file in _tracked_or_walked(SRC_DIR, "*.py"):
        parsed = _parse_file_imports(py_file)
        if parsed is None:
            continue
        rel, imps = parsed
        modules.add(rel)
        file_imports[rel] = imps

    graph: dict[str, set[str]] = defaultdict(set)
    for rel, imps in file_imports.items():
        _add_import_edges(graph, rel, imps, modules)
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


def _load_testpaths(*, pytest_ini: Path = PYTEST_INI) -> list[str]:
    """Parse ``testpaths = ...`` out of ``pytest.ini`` (pytest.ini wins over
    ``pyproject.toml`` per pytest's own ini-file precedence, so that's the
    only file this needs to read to know what ``pytest`` actually collects
    by default). ``pytest_ini`` is overridable (default: the real repo's)
    so a caller testing this in isolation (``tests/gates/test_wire_first_gate.py``)
    can point it at a synthetic ini instead of silently falling back to
    reading THIS repo's real, live ``pytest.ini`` regardless of the fixture
    under test."""
    if not pytest_ini.exists():
        return []
    for line in pytest_ini.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*testpaths\s*=\s*(.+)$", line)
        if m:
            return m.group(1).split()
    return []


def _load_explicit_pytest_paths(
    *,
    precommit_config: Path = PRECOMMIT_CONFIG,
    workflows_dir: Path = WORKFLOWS_DIR,
) -> set[str]:
    """``tests/...`` paths explicitly passed to ``pytest`` in a pre-commit
    hook ``entry:`` or a GitHub Actions ``run:`` step, outside of
    ``testpaths``. These are structurally collected even though
    ``testpaths`` doesn't cover them (e.g. ``tests/gates``, ``tests/docs``,
    ``tests/scale/test_prod_profile_guard.py``) — derived by regex over the
    hook/workflow files themselves so this can't hand-drift from what they
    actually run. ``precommit_config``/``workflows_dir`` are overridable
    (default: the real repo's) for the same isolation reason as
    ``_load_testpaths``'s ``pytest_ini`` parameter.
    """
    paths: set[str] = set()
    texts: list[str] = []
    if precommit_config.exists():
        texts.append(precommit_config.read_text(encoding="utf-8"))
    if workflows_dir.exists():
        for f in sorted(workflows_dir.glob("*.yml")):
            texts.append(f.read_text(encoding="utf-8"))
    for text in texts:
        for m in re.finditer(r"pytest\s+((?:tests/[\w./\-]+\s*)+)", text):
            for tok in m.group(1).split():
                if tok.startswith("tests/"):
                    paths.add(tok.rstrip("/"))
    return paths


def _under_any(rel_path: str, prefixes: set[str]) -> bool:
    return any(
        rel_path == p or rel_path.startswith(p.rstrip("/") + "/") for p in prefixes
    )


def find_orphaned_test_files(
    *,
    tests_dir: Path = TESTS_DIR,
    display_root: Path = ROOT,
    pytest_ini: Path = PYTEST_INI,
    precommit_config: Path = PRECOMMIT_CONFIG,
    workflows_dir: Path = WORKFLOWS_DIR,
) -> list[str]:
    """Test files under ``tests/`` collected by NEITHER ``testpaths`` NOR an
    explicit pytest invocation in pre-commit/CI (D-OB-13a). ``tests_dir``/
    ``display_root``/``pytest_ini``/``precommit_config``/``workflows_dir``
    are all overridable (default: the real repo) so
    ``tests/gates/test_wire_first_gate.py`` can prove this trips on a fully
    synthetic fixture, isolated from whatever this repo's OWN live
    ``pytest.ini``/``.pre-commit-config.yaml``/``.github/workflows`` happen
    to say — "a gate that can't fail is not a gate"."""
    if not tests_dir.exists():
        return []
    collected = set(
        _load_testpaths(pytest_ini=pytest_ini)
    ) | _load_explicit_pytest_paths(
        precommit_config=precommit_config, workflows_dir=workflows_dir
    )
    orphans = []
    for py in _tracked_or_walked(tests_dir, "test_*.py"):
        rel = py.relative_to(display_root).as_posix()
        if not _under_any(rel, collected):
            orphans.append(rel)
    return orphans


def _kw_value(call: ast.Call, name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _is_mock_ctor(call: ast.Call) -> bool:
    fn = call.func
    name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
    return name in {"Mock", "MagicMock"}


def _is_patch_call(call: ast.Call) -> bool:
    fn = call.func
    if isinstance(fn, ast.Name):
        return fn.id == "patch"
    if isinstance(fn, ast.Attribute):
        return fn.attr in {"patch", "object"} and (
            (isinstance(fn.value, ast.Name) and fn.value.id == "patch")
            or fn.attr == "patch"
        )
    return False


def find_mock_hygiene_issues(
    *, tests_dir: Path = TESTS_DIR, display_root: Path = ROOT
) -> list[tuple[str, int, str]]:
    """``(file, line, shape)`` for every ``MagicMock(spec=[])`` /
    ``Mock(spec=[])`` / ``patch(..., create=True)`` CALL site under
    ``tests/`` (D-OB-13b) — AST ``ast.Call`` matching, not line-regex, so a
    docstring merely mentioning ``create=True`` (as this codebase's own
    ``test_graph_iter.py`` does, to explain why it does NOT use that shape)
    is never a false positive. Reported for human triage, not
    auto-classified: isolating one unrelated object's attributes with
    ``spec=[]`` is ordinary and fine; using it (or ``create=True``) to fake
    an entire external module that may not actually exist is the dangerous
    shape that hid a dead path in ``test_skill_provider_wiring.py``.
    """
    if not tests_dir.exists():
        return []
    issues: list[tuple[str, int, str]] = []
    for py in _tracked_or_walked(tests_dir, "test_*.py"):
        rel = py.relative_to(display_root).as_posix()
        try:
            tree = ast.parse(
                py.read_text(encoding="utf-8", errors="ignore"), filename=rel
            )
        except (OSError, SyntaxError):
            continue
        issues.extend(_mock_hygiene_issues_in_file(rel, tree))
    return issues


def _mock_ctor_issue(node: ast.Call) -> str | None:
    if not _is_mock_ctor(node):
        return None
    spec = _kw_value(node, "spec")
    return "spec=[]" if isinstance(spec, ast.List) and not spec.elts else None


def _patch_call_issue(node: ast.Call) -> str | None:
    if not _is_patch_call(node):
        return None
    create = _kw_value(node, "create")
    is_true = isinstance(create, ast.Constant) and create.value is True
    return "create=True" if is_true else None


def _mock_hygiene_issues_in_file(
    rel: str, tree: ast.Module
) -> list[tuple[str, int, str]]:
    found: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        shape = _mock_ctor_issue(node) or _patch_call_issue(node)
        if shape:
            found.append((rel, node.lineno, shape))
    return found


def find_silent_import_guards(
    *, tests_dir: Path = TESTS_DIR, display_root: Path = ROOT
) -> list[tuple[str, int]]:
    """``(file, line)`` for every ``except ImportError``/
    ``ModuleNotFoundError`` handler in a test file whose body neither
    re-raises nor calls ``pytest.skip``/``.fail``/``.xfail``/
    ``importorskip`` — the test degrades to a silent pass instead of a
    visible skip when an optional extra is absent (D-OB-16)."""
    if not tests_dir.exists():
        return []
    found: list[tuple[str, int]] = []
    for py in _tracked_or_walked(tests_dir, "test_*.py"):
        rel = py.relative_to(display_root).as_posix()
        try:
            source = py.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source, filename=rel)
        except (OSError, SyntaxError):
            continue
        found.extend(_silent_import_guards_in_file(rel, tree))
    return found


def _handler_catches_import_error(node: ast.ExceptHandler) -> bool:
    type_names: set[str] = set()
    if isinstance(node.type, ast.Tuple):
        type_names = {t.id for t in node.type.elts if isinstance(t, ast.Name)}
    elif isinstance(node.type, ast.Name):
        type_names = {node.type.id}
    return bool(type_names & {"ImportError", "ModuleNotFoundError"})


def _handler_is_silent(node: ast.ExceptHandler) -> bool:
    """True when the handler's body neither re-raises nor visibly skips."""
    body_mod = ast.Module(body=node.body, type_ignores=[])
    has_skip = any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in {"skip", "fail", "xfail"}
        for n in ast.walk(body_mod)
    )
    has_importorskip = "importorskip" in ast.dump(body_mod)
    has_raise = any(isinstance(n, ast.Raise) for n in ast.walk(body_mod))
    return not (has_skip or has_importorskip or has_raise)


def _silent_import_guards_in_file(rel: str, tree: ast.Module) -> list[tuple[str, int]]:
    return [
        (rel, node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
        and _handler_catches_import_error(node)
        and _handler_is_silent(node)
    ]


def _iter_agent_utilities_files(src_dir: Path = SRC_DIR) -> list[Path]:
    return [
        p for p in _tracked_or_walked(src_dir, "*.py") if "__pycache__" not in p.parts
    ]


def _public_top_level_defs(tree: ast.Module) -> list[tuple[str, str, int]]:
    """``(kind, name, lineno)`` for public top-level classes/functions."""
    out: list[tuple[str, str, int]] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            out.append(("class", node.name, node.lineno))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not (
            node.name.startswith("_")
        ):
            out.append(("function", node.name, node.lineno))
    return out


_PROPERTY_DECORATOR_NAMES = {"property", "cached_property"}


def _decorator_name(node: ast.expr) -> str:
    """Best-effort dotted/bare name of a decorator expression — handles
    ``@property``, ``@functools.cached_property``, and a bare call form
    like ``@some_decorator(...)`` by unwrapping to its ``.func``."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _public_methods(tree: ast.Module) -> list[tuple[str, str, int, bool]]:
    """``(class_name, method_name, lineno, is_property)`` for public methods
    of public top-level classes. ``is_property`` flags ``@property`` /
    ``@cached_property`` accessors, whose only legitimate reference syntax
    is bare attribute access (``obj.name``) — never a call (``obj.name()``)
    — see ``find_test_only_symbols`` for why that changes how a reference
    is counted."""
    out: list[tuple[str, str, int, bool]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
            continue
        for sub in node.body:
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and not (
                sub.name.startswith("_")
            ):
                is_property = any(
                    _decorator_name(d) in _PROPERTY_DECORATOR_NAMES
                    for d in sub.decorator_list
                )
                out.append((node.name, sub.name, sub.lineno, is_property))
    return out


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_METHOD_CALL_RE = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _index_file(text: str) -> tuple[Counter[str], Counter[str]]:
    """One tokenize pass over ``text`` -> (NAME-token counts, ``.name(``-call
    counts). Uses ``tokenize`` rather than a raw regex specifically so a
    COMMENT or STRING/docstring token that merely *mentions* a symbol name
    (e.g. "AdmissionPolicy" only ever appearing in a code comment, never
    actually constructed) is NOT counted as a real reference — an earlier
    regex-over-raw-text version of this sweep had exactly that false
    negative. Reused per-file so the sweep is O(total source size) instead
    of O(files x symbols). Falls back to the regex form on a tokenize
    failure (rare: exotic encodings/f-string edge cases) rather than
    dropping the file's contribution entirely.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError, ValueError):
        idents = Counter(_IDENTIFIER_RE.findall(text))
        calls = Counter(_METHOD_CALL_RE.findall(text))
        return idents, calls

    idents = Counter(t.string for t in tokens if t.type == tokenize.NAME)
    calls: Counter[str] = Counter()
    for i, t in enumerate(tokens):
        if t.type == tokenize.NAME and _is_dotted_call_token(tokens, i):
            calls[t.string] += 1
    return idents, calls


def _is_dotted_call_token(tokens: list[tokenize.TokenInfo], i: int) -> bool:
    """True when ``tokens[i]`` is the ``name`` in a ``.name(`` sequence."""
    if i < 2 or i + 1 >= len(tokens):
        return False
    prev, nxt = tokens[i - 1], tokens[i + 1]
    return (
        prev.type == tokenize.OP
        and prev.string == "."
        and nxt.type == tokenize.OP
        and nxt.string == "("
    )


def find_test_only_symbols(
    *,
    src_dir: Path = SRC_DIR,
    tests_dir: Path = TESTS_DIR,
    display_root: Path = ROOT,
) -> list[dict]:
    """Public classes/functions/methods under ``agent_utilities/`` with ZERO
    references outside their own defining file and ``tests/`` — reachable
    by import, never invoked by any live (non-test) caller. This is the
    D-OB-9 shape: unit-tested, never wired. See the module docstring for the
    heuristic's known blind spots (word-boundary token counting, not a
    type-resolved call graph).

    Indexed in one pass per file (identifier counts + ``.name(``-call
    counts) rather than re-scanning every file's text once per candidate
    symbol — the naive O(files x symbols) version is minutes-slow on a
    codebase this size; this is O(total source size).

    ``@property``/``@cached_property`` accessors are excluded from the
    method-level ``.name(``-call check (D-OB-9 method-name collision, see
    ``dual_principal_validation.py``'s module-level ``fingerprint()`` vs.
    three unrelated ``*.fingerprint`` properties on
    ``PromptCacheKey``/``SemanticCacheKey``/``OAuthGrantBinding``): a
    property's only legitimate reference syntax is bare attribute access
    (``obj.name``) — it is never validly invoked with call syntax
    (``obj.name()``), so a real ``.name(`` occurrence elsewhere can never
    actually be a reference to it. Without this exclusion any unrelated
    symbol sharing a property's bare name and invoked with call syntax
    (e.g. a same-named module-level function called as ``mod.fingerprint(
    ...)``) inflates that property's apparent test-reference count with
    zero genuine signal either way — exactly the false positive this fixes.

    Three broader alternatives were tried during development of this fix
    and reverted because each regressed *genuine* D-OB-9 detections:
    import-qualifying the production side of the ``.name(``-call match
    (a file only "reaches" a class if it imports it) flipped >140
    genuinely-wired methods to false positives, because agent_utilities
    leans heavily on factory functions / dependency injection that never
    import the concrete class by name (``get_semantic_cache().
    invalidate(...)``); import-qualifying only the test side still flipped
    dozens more, reached only through a composed/inherited concrete class
    (``engine.add_prompt(...)`` on an ``IntelligenceGraphEngine`` that
    mixes in ``RegistryMixin`` — the test never imports ``RegistryMixin``
    by name); and routing properties to identifier-occurrence counting
    (like top-level defs) instead of skipping them outright is accurate but
    newly *surfaces* ~35 genuinely test-only properties this gate has never
    been able to see before (no ``.name(`` call syntax exists for them to
    match on) — real D-OB-9 backlog, but not the false positive this change
    is scoped to fix; expanding coverage to properties is a deliberate,
    separate change for another day. Skipping properties entirely for this
    one check needs neither import resolution nor a class hierarchy and
    changes nothing about regular (non-property) method matching (same
    precedent as ``_GENERIC_METHOD_STOPLIST``: fall back to the class-level
    finding).

    ``src_dir``/``tests_dir``/``display_root`` are overridable (default: the
    real repo) so ``tests/gates/test_wire_first_gate.py`` can prove this
    trips on a synthetic fixture.
    """
    au_sources = _read_au_sources(src_dir, display_root)
    au_idents, au_calls, total_au_idents, total_au_calls = _index_au_sources(au_sources)
    (
        test_idents_by_file,
        test_calls_by_file,
        test_imports_by_file,
        total_test_idents,
        total_test_calls,
    ) = _index_test_sources(tests_dir, display_root)

    # D-OP-12: caller resolution below is bare-symbol-name text matching,
    # not a type-resolved call graph (see the module/function docstrings).
    # For the overwhelming majority of symbols — unique names across the
    # whole repo — that is harmless: there is only ever one possible
    # definition a reference could mean. But when TWO DIFFERENT symbols
    # share a bare name in different files (e.g. two unrelated classes each
    # defining ``parent_of``), the aggregate ``total_test_*`` counters pool
    # every TEST reference to EITHER definition into one number, so adding
    # a test for symbol A's ``parent_of`` silently CREATES a finding for
    # unrelated symbol B, in a file the change never touched. Fix: pre-detect
    # which names are DEFINED more than once (a "collision") and resolve
    # ONLY the test side with import-scoped counting for those — see
    # ``_colliding_names``/``_scoped_test_count``/``_scoped_test_call_count``
    # for the full rationale (moved out of this docstring to keep it near the
    # code it explains). The production (``other_au``) side deliberately
    # stays pooled/unscoped even for colliding names — see those functions.
    au_trees, top_level_defs_by_file, methods_by_file = _collect_definitions(au_sources)
    colliding_top_level_names, colliding_method_names = _colliding_names(
        top_level_defs_by_file, methods_by_file
    )

    ordinals: dict[tuple[str, str], int] = {}
    findings: list[dict] = []
    for rel in au_trees:
        findings.extend(
            _top_level_symbol_findings(
                rel,
                top_level_defs_by_file,
                au_idents,
                total_au_idents,
                colliding_top_level_names,
                test_idents_by_file,
                test_imports_by_file,
                total_test_idents,
                ordinals,
            )
        )
        findings.extend(
            _method_symbol_findings(
                rel,
                methods_by_file,
                au_calls,
                total_au_calls,
                colliding_method_names,
                test_calls_by_file,
                test_imports_by_file,
                total_test_calls,
                ordinals,
            )
        )
    return findings


def _read_au_sources(src_dir: Path, display_root: Path) -> dict[str, str]:
    au_sources: dict[str, str] = {}
    for py in _iter_agent_utilities_files(src_dir):
        rel = py.relative_to(display_root).as_posix()
        try:
            au_sources[rel] = py.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
    return au_sources


def _all_assignment_values(tree: ast.Module) -> list[ast.expr]:
    """Module-level ``__all__`` assignment right-hand sides."""
    values: list[ast.expr] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            values.append(node.value)
    return values


def _string_elements(value: ast.expr | None) -> list[str]:
    """The string constants of a literal list/tuple/set, else nothing."""
    if not isinstance(value, ast.List | ast.Tuple | ast.Set):
        return []
    return [
        element.value
        for element in value.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    ]


def _declared_reexports(rel: str, text: str) -> Counter[str]:
    """Names a package ``__init__.py`` declares as public exports in ``__all__``.

    A package that re-exports eagerly (``from .mod import Name``) puts ``Name``
    in the file as a NAME token, so :func:`_index_file` counts it as a real
    non-test reference. The lazy ``__getattr__`` + export-map form used by most
    of this package's ``__init__.py`` files (``agent_utilities/__init__.py``,
    ``knowledge_graph/``, ``deployment/``, ``security/`` ...) stores that same
    name as a STRING literal instead, which :func:`_index_file` deliberately
    does not count. Without this, converting a package to lazy exports silently
    turns every symbol whose only non-test reference was that re-export into a
    bogus "test-only" finding.

    Only module-level ``__all__`` entries in an ``__init__.py`` count: that is a
    DECLARED public export, not a passing mention in a comment or docstring, so
    this restores parity with the eager form without reintroducing the
    string-mention false positive ``_index_file``'s tokenize pass exists to
    prevent.
    """
    if PurePosixPath(rel).name != "__init__.py":
        return Counter()
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return Counter()
    names: Counter[str] = Counter()
    for value in _all_assignment_values(tree):
        names.update(_string_elements(value))
    return names


def _index_au_sources(
    au_sources: dict[str, str],
) -> tuple[
    dict[str, Counter[str]], dict[str, Counter[str]], Counter[str], Counter[str]
]:
    au_idents: dict[str, Counter[str]] = {}
    au_calls: dict[str, Counter[str]] = {}
    total_au_idents: Counter[str] = Counter()
    total_au_calls: Counter[str] = Counter()
    for rel, source in au_sources.items():
        idents, calls = _index_file(source)
        idents = idents + _declared_reexports(rel, source)
        au_idents[rel] = idents
        au_calls[rel] = calls
        total_au_idents.update(idents)
        total_au_calls.update(calls)
    return au_idents, au_calls, total_au_idents, total_au_calls


def _index_test_sources(
    tests_dir: Path, display_root: Path
) -> tuple[
    dict[str, Counter[str]],
    dict[str, Counter[str]],
    dict[str, set[str]],
    Counter[str],
    Counter[str],
]:
    """Per-file test counters + import sets, alongside aggregate totals
    (D-OP-12): the aggregate is the fast path for a symbol name defined only
    once in the whole repo; a COLLIDING name needs the per-file scoping this
    also returns — see ``_scoped_test_count``/``_scoped_test_call_count``."""
    test_idents_by_file: dict[str, Counter[str]] = {}
    test_calls_by_file: dict[str, Counter[str]] = {}
    test_imports_by_file: dict[str, set[str]] = {}
    total_test_idents: Counter[str] = Counter()
    total_test_calls: Counter[str] = Counter()
    if not tests_dir.exists():
        return (
            test_idents_by_file,
            test_calls_by_file,
            test_imports_by_file,
            total_test_idents,
            total_test_calls,
        )
    for py in _tracked_or_walked(tests_dir, "*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            text = py.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        idents, calls = _index_file(text)
        total_test_idents.update(idents)
        total_test_calls.update(calls)
        trel = py.relative_to(display_root).as_posix()
        test_idents_by_file[trel] = idents
        test_calls_by_file[trel] = calls
        try:
            ttree = ast.parse(text, filename=trel)
        except SyntaxError:
            continue
        test_imports_by_file[trel] = collect_imports(trel, ttree)
    return (
        test_idents_by_file,
        test_calls_by_file,
        test_imports_by_file,
        total_test_idents,
        total_test_calls,
    )


def _collect_definitions(
    au_sources: dict[str, str],
) -> tuple[
    dict[str, ast.Module],
    dict[str, list[tuple[str, str, int]]],
    dict[str, list[tuple[str, str, int, bool]]],
]:
    au_trees: dict[str, ast.Module] = {}
    for rel, source in au_sources.items():
        if rel.endswith("__init__.py"):
            continue
        try:
            au_trees[rel] = ast.parse(source, filename=rel)
        except SyntaxError:
            continue
    top_level_defs_by_file = {
        rel: _public_top_level_defs(tree) for rel, tree in au_trees.items()
    }
    methods_by_file = {rel: _public_methods(tree) for rel, tree in au_trees.items()}
    return au_trees, top_level_defs_by_file, methods_by_file


def _colliding_names(
    top_level_defs_by_file: dict[str, list[tuple[str, str, int]]],
    methods_by_file: dict[str, list[tuple[str, str, int, bool]]],
) -> tuple[set[str], set[str]]:
    """Top-level names / method names DEFINED in more than one file — see
    ``find_test_only_symbols``'s D-OP-12 note for why these need import-
    scoped test-reference counting instead of the pooled fast path."""
    top_level_name_files: dict[str, set[str]] = defaultdict(set)
    for rel, defs in top_level_defs_by_file.items():
        for _kind, name, _lineno in defs:
            top_level_name_files[name].add(rel)
    colliding_top_level = {
        name for name, files in top_level_name_files.items() if len(files) > 1
    }

    method_name_files: dict[str, set[str]] = defaultdict(set)
    for rel, methods in methods_by_file.items():
        for _cls_name, meth_name, _m_lineno, _is_property in methods:
            method_name_files[meth_name].add(rel)
    colliding_methods = {
        name for name, files in method_name_files.items() if len(files) > 1
    }
    return colliding_top_level, colliding_methods


def _scoped_test_count(
    name: str,
    defining_rel: str,
    test_idents_by_file: dict[str, Counter[str]],
    test_imports_by_file: dict[str, set[str]],
) -> int:
    """A colliding name's test-identifier count, scoped to test files that
    actually import ``defining_rel``'s own module — see ``find_test_only_
    symbols``'s D-OP-12 note. A scoped count is always <= the pooled count
    it replaces, so this can only ever REMOVE a spurious finding."""
    defining_module = path_to_module_name(defining_rel)
    return sum(
        idents.get(name, 0)
        for trel, idents in test_idents_by_file.items()
        if defining_module in test_imports_by_file.get(trel, set())
    )


def _scoped_test_call_count(
    name: str,
    defining_rel: str,
    test_calls_by_file: dict[str, Counter[str]],
    test_imports_by_file: dict[str, set[str]],
) -> int:
    """The ``.name(``-call analogue of :func:`_scoped_test_count`."""
    defining_module = path_to_module_name(defining_rel)
    return sum(
        calls.get(name, 0)
        for trel, calls in test_calls_by_file.items()
        if defining_module in test_imports_by_file.get(trel, set())
    )


def _next_ordinal(ordinals: dict[tuple[str, str], int], rel: str, symbol: str) -> int:
    """(file, symbol) -> next ordinal (D-OP-11: the key must be stable under
    pure line motion elsewhere in the file; a bare (file, symbol) pair is
    unique for the overwhelming majority of real findings, but this ordinal
    disambiguates the rare genuine collision — same traversal-order pattern
    already proven for check_swallowed_errors.py's HandlerKey, D-SWG-1)."""
    key = (rel, symbol)
    ordinal = ordinals.get(key, 0)
    ordinals[key] = ordinal + 1
    return ordinal


def _top_level_symbol_findings(
    rel: str,
    top_level_defs_by_file: dict[str, list[tuple[str, str, int]]],
    au_idents: dict[str, Counter[str]],
    total_au_idents: Counter[str],
    colliding_top_level_names: set[str],
    test_idents_by_file: dict[str, Counter[str]],
    test_imports_by_file: dict[str, set[str]],
    total_test_idents: Counter[str],
    ordinals: dict[tuple[str, str], int],
) -> list[dict]:
    findings: list[dict] = []
    for kind, name, lineno in top_level_defs_by_file.get(rel, []):
        other_au = total_au_idents.get(name, 0) - au_idents[rel].get(name, 0)
        if name in colliding_top_level_names:
            test_refs = _scoped_test_count(
                name, rel, test_idents_by_file, test_imports_by_file
            )
        else:
            test_refs = total_test_idents.get(name, 0)
        same_file = au_idents[rel].get(name, 0) - 1  # minus the def line itself
        if other_au == 0 and same_file <= 0 and test_refs > 0:
            findings.append(
                {
                    "kind": kind,
                    "symbol": name,
                    "file": rel,
                    "line": lineno,
                    "test_refs": test_refs,
                    "ordinal": _next_ordinal(ordinals, rel, name),
                }
            )
    return findings


def _method_is_skippable(meth_name: str, is_property: bool) -> bool:
    """Excludes the generic-verb stoplist and ``@property``/
    ``@cached_property`` accessors — see ``find_test_only_symbols``'s
    docstring ("D-OB-9 method-name collision") for why a property can never
    be validly referenced with call syntax, so the ``.name(``-call check
    this loop relies on cannot see a genuine reference to one in either
    direction."""
    return meth_name in _GENERIC_METHOD_STOPLIST or is_property


def _method_symbol_findings(
    rel: str,
    methods_by_file: dict[str, list[tuple[str, str, int, bool]]],
    au_calls: dict[str, Counter[str]],
    total_au_calls: Counter[str],
    colliding_method_names: set[str],
    test_calls_by_file: dict[str, Counter[str]],
    test_imports_by_file: dict[str, set[str]],
    total_test_calls: Counter[str],
    ordinals: dict[tuple[str, str], int],
) -> list[dict]:
    findings: list[dict] = []
    for cls_name, meth_name, m_lineno, is_property in methods_by_file.get(rel, []):
        if _method_is_skippable(meth_name, is_property):
            continue
        other_au = total_au_calls.get(meth_name, 0) - au_calls[rel].get(meth_name, 0)
        if meth_name in colliding_method_names:
            test_refs = _scoped_test_call_count(
                meth_name, rel, test_calls_by_file, test_imports_by_file
            )
        else:
            test_refs = total_test_calls.get(meth_name, 0)
        same_file = au_calls[rel].get(meth_name, 0)
        if other_au == 0 and same_file <= 0 and test_refs > 0:
            symbol = f"{cls_name}.{meth_name}"
            findings.append(
                {
                    "kind": "method",
                    "symbol": symbol,
                    "file": rel,
                    "line": m_lineno,
                    "test_refs": test_refs,
                    "ordinal": _next_ordinal(ordinals, rel, symbol),
                }
            )
    return findings


_FINDING_KEY_SEP = "\t"


def _finding_key(entry: dict) -> str:
    """A (file, symbol, ordinal) key -- stable under pure line motion.

    D-OP-11: the previous ``path:line:symbol`` key re-keyed EVERY finding
    below any inserted line as a spurious "NEW" entry, making
    ``--update-wire-first-baseline`` the only practical way back to green —
    which silently absorbs unrelated pre-existing debt. Ported verbatim from
    the same fix already proven on the sibling gate
    (check_swallowed_errors.py's HandlerKey, D-SWG-1): drop the line number
    from the key entirely; a (file, symbol) pair is unique for the
    overwhelming majority of findings, and ``ordinal`` (assigned in
    traversal order, see find_test_only_symbols) disambiguates the rare
    genuine collision. TAB-separated (not ``:``) so a symbol/path
    containing a colon can never be misparsed.
    """
    return _FINDING_KEY_SEP.join(
        [entry["file"], entry["symbol"], str(entry.get("ordinal", 0))]
    )


# ── Diff-scoped enforcement (the ratchet's replacement) ───────────────────
#
# WHY THERE IS NO BASELINE HERE ANY MORE.
#
# This gate used to freeze both the orphaned-test-file set and the
# test-only-symbol set into ``scripts/wire_first_baseline.json`` and fail
# only on a key absent from that file — a ratchet, which this project does
# not allow (see ``check_swallowed_errors.py``'s module docstring for the
# fully-worked-out incident that retired the sibling gate this one mirrors).
#
# MEASURED (2026-08-28, see the lane report this commit belongs to): the real
# current count for BOTH finding sets is EXACTLY equal to what was frozen —
# 0 orphaned test files, 1107 test-only symbols — even though the complexity-
# collapse program had already extract-refactored ~20 functions in this repo
# by the time of this measurement (the same refactor wave that produced 37
# phantom re-keys on the swallowed-errors gate's OLD enclosing-symbol key).
# Zero drift is not proof the key is safe on its own — it is corroborating
# evidence for the EXTRACTION-INVARIANCE TEST actually run against
# ``_finding_key``: a test-only method's own defining (class, method) name is
# unaffected by moving code INTO a new nested private helper defined inside
# it (the exact ``create_agent`` -> ``create_agent._setup_mcp_url_toolset``
# shape that broke the swallowed-errors key) — nested defs are invisible to
# ``_public_top_level_defs``/``_public_methods`` in the first place, so this
# key never even sees that class of refactor. It IS perturbed by a genuine
# RENAME of the enclosing class (the same class of instability as D-SWG-1) —
# confirmed by construction, not assumed — but that is a different, much
# rarer operation than private-helper extraction, and not the technique this
# program's automation actually applies. So unlike the swallowed-errors key,
# this one is kept: it already carries the finding's own content (which
# class, which method/function), not an incidental enclosing scope a
# refactor tool reaches into.
#
# What replaces the ratchet:
#
#   * an UNCONDITIONAL CENSUS that prints the real totals every run and never
#     fails — nothing is written to disk, so no number can go stale;
#   * an ABSOLUTE-ZERO invariant for orphaned test files (D-OB-13a) — the
#     repo is measured at zero for this today, so no HEAD comparison is
#     needed at all: any orphan is a new violation, full stop;
#   * DIFF-SCOPED enforcement for test-only symbols (D-OB-9), recomputed live
#     against a ``git archive HEAD`` snapshot of ``agent_utilities/`` +
#     ``tests/`` — a finding present now but absent at HEAD is new debt and
#     fails the commit; the rest is pre-existing backlog, printed, not
#     hidden. The snapshot rebuild (the expensive half — this gate's own
#     O(total source size) sweep, a second time) only runs when a
#     ``agent_utilities/**.py`` or ``tests/**.py`` file actually changed
#     since HEAD, mirroring the same scope this gate's own pre-commit
#     ``files:`` pattern already uses;
#   * retired flags (``--update-wire-first-baseline``) exit 2.


def _git(*args: str, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run git from the repo toplevel with repo-relative paths — see
    ``check_swallowed_errors.py``'s identically-named helper for why every
    invocation runs from the resolved toplevel and never ``git -C <subdir>``
    (GIT_DIR/GIT_INDEX_FILE ambient-env hazard)."""
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )


def _repo_root() -> str | None:
    r = _git("rev-parse", "--show-toplevel")
    out = r.stdout.strip()
    return out if r.returncode == 0 and out else None


def _relevant_wire_first_files_changed(root: str) -> bool:
    """True iff a ``agent_utilities/**.py`` or ``tests/**.py`` file differs
    from HEAD (staged or working tree) — nothing this gate's symbol findings
    depend on could have moved otherwise."""
    changed: set[str] = set()
    for args in (
        ("diff", "--cached", "--name-only", "--diff-filter=ACMR", "HEAD"),
        ("diff", "--name-only", "HEAD"),
    ):
        r = _git(*args, cwd=root)
        if r.returncode == 0:
            changed.update(r.stdout.splitlines())
    return any(
        (p.startswith("agent_utilities/") or p.startswith("tests/"))
        and p.endswith(".py")
        for p in changed
    )


def _archive_head_bytes(root: str) -> bytes | None:
    proc = subprocess.run(
        ["git", "archive", "HEAD", "--", "agent_utilities", "tests"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 and proc.stdout else None


def _materialize_head_snapshot(root: str, dest: Path) -> bool:
    """Extract ``agent_utilities/`` + ``tests/`` AS OF HEAD into ``dest``.
    Returns False when the archive could not be produced at all (e.g. a
    shallow/synthetic repo with no HEAD) so the caller treats the comparison
    as unavailable rather than silently diffing against an empty tree."""
    data = _archive_head_bytes(root)
    if data is None:
        return False
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        tf.extractall(dest, filter="data")
    return True


def _current_to_head_renames(root: str) -> dict[str, str] | None:
    """Map a current path to its exact prior path for Git-classified renames.

    The differential key includes the defining path, so a real ``git mv``
    would otherwise manufacture one new finding per unchanged public symbol.
    Copies deliberately do not enter this map: keeping the original while
    adding the same unwired symbol elsewhere is new backlog.
    """
    result = _git(
        "diff",
        "--name-status",
        "-z",
        "--find-renames",
        "HEAD",
        "--",
        "agent_utilities",
        "tests",
        cwd=root,
    )
    if result.returncode != 0:
        return None
    fields = result.stdout.split("\0")
    renames: dict[str, str] = {}
    index = 0
    while index < len(fields) and fields[index]:
        status = fields[index]
        index += 1
        path_count = 2 if status[:1] in {"R", "C"} else 1
        if index + path_count > len(fields):
            return None
        paths = fields[index : index + path_count]
        index += path_count
        if status.startswith("R"):
            renames[paths[1]] = paths[0]
    return renames


def _finding_key_at_head(entry: dict, renames: dict[str, str]) -> str:
    """Key ``entry`` using its prior path when Git proved an exact rename."""
    prior = dict(entry)
    prior["file"] = renames.get(entry["file"], entry["file"])
    return _finding_key(prior)


_AMBIENT_GIT_IDENTITY_VARS = ("GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE")


def _scan_snapshot_for_test_only_symbols(dest: Path) -> list[dict]:
    """``find_test_only_symbols`` over a bare ``git archive`` extraction,
    with the ambient GIT_DIR/GIT_INDEX_FILE/GIT_WORK_TREE identity stripped
    from the subprocess environment for the scan's duration (restored
    immediately after, even on exception).

    ``dest`` is a plain filesystem extraction, not a git repository, so
    ``_tracked_or_walked``'s git-ls-files preference is EXPECTED to fail
    cleanly there and fall through to its own plain-walk fallback -- which
    is correct here, since ``git archive`` already extracted exactly the
    tracked-at-HEAD set. Git itself sets GIT_DIR/GIT_INDEX_FILE/
    GIT_WORK_TREE in every hook subprocess (confirmed BUG-043/BUG-180); left
    in place, ``git -C <dest>/... ls-files`` does NOT fail the way a plain
    invocation does -- it silently resolves against the AMBIENT (real)
    repository instead of erroring "not a git repository", and
    ``_tracked_or_walked``'s ``if tracked: return [filtered]`` branch trusts
    that non-empty-but-wrong result rather than falling through to the
    (correct) rglob fallback, so ``au_sources``/the test-file set for the
    snapshot come back near-empty and EVERY current finding reads as new.
    Confirmed by construction, not assumed: WD4-RAT-02's own GIT_DIR/
    GIT_INDEX_FILE plant proof reported this exact shape -- 1 genuinely new
    finding read as 1108 (the entire backlog) before this fix.
    """
    ambient = {
        k: os.environ.pop(k) for k in _AMBIENT_GIT_IDENTITY_VARS if k in os.environ
    }
    try:
        return find_test_only_symbols(
            src_dir=dest / "agent_utilities",
            tests_dir=dest / "tests",
            display_root=dest,
        )
    finally:
        os.environ.update(ambient)


def _head_symbol_context(
    root: str,
) -> tuple[set[str], dict[str, str]] | None:
    """Return immutable HEAD finding keys plus proven current-path renames."""
    renames = _current_to_head_renames(root)
    if renames is None:
        return None
    with tempfile.TemporaryDirectory(prefix="au-wire-first-head-") as tmp:
        dest = Path(tmp)
        if not _materialize_head_snapshot(root, dest):
            return None
        head_symbols = _scan_snapshot_for_test_only_symbols(dest)
    return {_finding_key(entry) for entry in head_symbols}, renames


def _new_symbol_findings_vs_head(
    root: str, current_symbols: list[dict]
) -> list[dict] | None:
    """Test-only symbol findings present now but absent at HEAD, or None
    when the comparison could not be safely made — the caller must treat
    that as a degraded read, never a silent pass (AGENTS.md "Fail closed").
    Skips the expensive snapshot rebuild entirely (returns ``[]``) when
    nothing this gate's findings could depend on has changed since HEAD."""
    if not _relevant_wire_first_files_changed(root):
        return []
    context = _head_symbol_context(root)
    if context is None:
        return None
    head_keys, renames = context
    return [
        entry
        for entry in current_symbols
        if _finding_key_at_head(entry, renames) not in head_keys
    ]


def _report_orphan_zero(orphans: list[str]) -> bool:
    """The absolute invariant for D-OB-13a: this repo is measured at zero
    orphaned test files today, so any orphan is a violation with no HEAD
    comparison needed. Returns True on violation."""
    if not orphans:
        return False
    print(f"\n{len(orphans)} test file(s) not collected by testpaths/CI (D-OB-13a):")
    for rel in sorted(orphans):
        print(f"  {rel}")
    return True


def _print_census(
    orphans: list[str],
    symbols: list[dict],
    mock_issues: list[tuple[str, int, str]],
    silent_guards: list[tuple[str, int]],
) -> None:
    """Print the real numbers. Always. This never fails the run."""
    print(
        f"Wire-First census: {len(symbols)} test-only symbol(s) (D-OB-9), "
        f"{len(orphans)} orphaned test file(s) (D-OB-13a)"
    )
    for kind, count in Counter(s["kind"] for s in symbols).most_common():
        print(f"  {count:5d}  {kind}")
    for rel, count in Counter(s["file"] for s in symbols).most_common(5):
        print(f"  top: {count:3d}  {rel}")
    print(
        f"\nMock hygiene (D-OB-13b, informational — not enforced): "
        f"{len(mock_issues)} spec=[]/create=True site(s)."
    )
    for rel, line, shape in mock_issues[:20]:
        print(f"  {rel}:{line} [{shape}]")
    if len(mock_issues) > 20:
        print(f"  ... and {len(mock_issues) - 20} more")
    print(
        f"\nSilently-swallowed optional-extra import guards (D-OB-16, "
        f"informational — not enforced): {len(silent_guards)} site(s)."
    )
    for rel, line in silent_guards[:20]:
        print(f"  {rel}:{line}")
    if len(silent_guards) > 20:
        print(f"  ... and {len(silent_guards) - 20} more")


def _print_new_symbols(new_symbols: list[dict]) -> None:
    print(
        f"\nTest-only public symbols / no non-test caller (D-OB-9): "
        f"{len(new_symbols)} NEW finding(s) since HEAD:"
    )
    for e in new_symbols:
        print(
            f"  {e['file']}:{e['line']} {e['kind']} {e['symbol']} "
            f"(referenced in {e['test_refs']} test location(s))"
        )


def wire_first_report() -> int:
    orphans = find_orphaned_test_files()
    mock_issues = find_mock_hygiene_issues()
    silent_guards = find_silent_import_guards()
    symbols = find_test_only_symbols()

    print("=" * 72)
    print("Wire-First gate report (D-OB-9 / D-OB-13 / D-OB-16)")
    print("=" * 72)
    _print_census(orphans, symbols, mock_issues, silent_guards)

    had_new = _report_orphan_zero(orphans)

    root = _repo_root()
    if root is None:
        print("\n(not inside a work tree — diff-scoped symbol check skipped)")
    else:
        new_symbols = _new_symbol_findings_vs_head(root, symbols)
        if new_symbols is None:
            print(
                "\nFAIL — could not build the HEAD-snapshot comparison; a "
                "degraded read must never be treated as a pass. Re-run, or "
                "investigate `git archive HEAD`."
            )
            had_new = True
        elif new_symbols:
            _print_new_symbols(new_symbols)
            had_new = True

    print()
    if had_new:
        print(
            "FAIL — new Wire-First backlog since HEAD. Fix it, or if "
            "genuinely accepted, document the justification in the commit "
            "message (see AGENTS.md)."
        )
        return 1
    print("OK — no NEW Wire-First backlog since HEAD.")
    return 0


def _handle_check_test_collection() -> int:
    orphans = find_orphaned_test_files()
    had_violation = _report_orphan_zero(orphans)
    print(f"Uncollected test files (D-OB-13a): {len(orphans)} total.")
    return 1 if had_violation else 0


def _handle_check_symbol_reachability() -> int:
    symbols = find_test_only_symbols()
    root = _repo_root()
    if root is None:
        print("(not inside a work tree — diff-scoped symbol check skipped)")
        print(
            f"Test-only public symbols / no non-test caller (D-OB-9): {len(symbols)} total."
        )
        return 0
    new_symbols = _new_symbol_findings_vs_head(root, symbols)
    if new_symbols is None:
        print(
            "FAIL — could not build the HEAD-snapshot comparison; a degraded "
            "read must never be treated as a pass."
        )
        return 1
    print(
        f"Test-only public symbols / no non-test caller (D-OB-9): "
        f"{len(symbols)} total, {len(new_symbols)} NEW since HEAD."
    )
    if new_symbols:
        _print_new_symbols(new_symbols)
        return 1
    return 0


def _handle_check_mock_hygiene() -> int:
    issues = find_mock_hygiene_issues()
    print(f"Mock hygiene: {len(issues)} spec=[]/create=True site(s).")
    for rel, line, shape in issues:
        print(f"  {rel}:{line} [{shape}]")
    return 0


def _handle_check_extras_gating() -> int:
    guards = find_silent_import_guards()
    print(f"Silent optional-extra import guards: {len(guards)} site(s).")
    for rel, line in guards:
        print(f"  {rel}:{line}")
    return 0


def _print_module_chain(
    target: str, modules: set[str], graph: dict[str, set[str]], roots: set[str]
) -> int:
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


def _print_reachability_json(
    roots: set[str],
    modules: set[str],
    dist: dict[str, int],
    unreachable: list[str],
    far: list[tuple[str, int]],
    max_hops: int,
) -> None:
    print(
        json.dumps(
            {
                "roots": sorted(roots),
                "total_modules": len(modules),
                "reachable": len(dist),
                "unreachable": unreachable,
                "beyond_max_hops": [{"module": m, "hops": d} for m, d in far],
                "max_hops": max_hops,
            },
            indent=2,
        )
    )


def _print_reachability_text(
    roots: set[str],
    modules: set[str],
    dist: dict[str, int],
    unreachable: list[str],
    far: list[tuple[str, int]],
    max_hops: int,
) -> None:
    print(f"roots ({len(roots)}):")
    for r in sorted(roots):
        print(f"  {r}")
    print(
        f"\nmodules: {len(modules)}  reachable: {len(dist)}  "
        f"unreachable: {len(unreachable)}"
    )
    if far:
        print(f"\nreachable only beyond {max_hops} hops ({len(far)}):")
        for m, d in far:
            print(f"  {d:>2}  {m}")
    if unreachable:
        print(
            f"\nno static import path from a root ({len(unreachable)}) — "
            "verify against the blind-spot list before treating as dead:"
        )
        for m in unreachable:
            print(f"  {m}")


def _build_roots(modules: set[str]) -> set[str]:
    roots = {p for p in DEFAULT_ROOT_PATTERNS if p in modules}
    roots |= load_console_script_roots(modules)
    return roots


def _emit_reachability(
    args: argparse.Namespace,
    roots: set[str],
    modules: set[str],
    dist: dict[str, int],
) -> list[str]:
    unreachable = sorted(m for m in modules if m not in dist)
    far = sorted((m, d) for m, d in dist.items() if d > args.max_hops)
    if args.json:
        _print_reachability_json(roots, modules, dist, unreachable, far, args.max_hops)
    else:
        _print_reachability_text(roots, modules, dist, unreachable, far, args.max_hops)
    return unreachable


def _handle_reachability_report(args: argparse.Namespace) -> int:
    if not SRC_DIR.exists():
        print(f"source directory not found: {SRC_DIR}", file=sys.stderr)
        return 2

    graph, modules = build_graph()
    roots = _build_roots(modules)
    dist = bfs_hops(graph, roots)

    if args.module:
        return _print_module_chain(args.module, modules, graph, roots)

    unreachable = _emit_reachability(args, roots, modules, dist)
    return 1 if args.fail_on_unreachable and unreachable else 0


def _build_arg_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--check-test-collection",
        action="store_true",
        help="D-OB-13a: report test files under tests/ not collected by "
        "testpaths nor an explicit pre-commit/CI pytest invocation. "
        "Enforced as an absolute zero (exit 1 on any orphan).",
    )
    parser.add_argument(
        "--check-mock-hygiene",
        action="store_true",
        help="D-OB-13b: report MagicMock(spec=[])/patch(create=True) sites "
        "in tests/ for human triage. Always exits 0.",
    )
    parser.add_argument(
        "--check-extras-gating",
        action="store_true",
        help="D-OB-16: report except ImportError handlers in tests/ that "
        "silently no-op instead of visibly skipping. Always exits 0.",
    )
    parser.add_argument(
        "--check-symbol-reachability",
        action="store_true",
        help="D-OB-9: report public agent_utilities/ classes/functions/"
        "methods referenced only from tests/ (no non-test caller). "
        "Diff-scoped against HEAD (exit 1 on new entries).",
    )
    parser.add_argument(
        "--wire-first-report",
        action="store_true",
        help="Run all four Wire-First checks above and print a combined "
        "report. Exit 1 on a new orphan or a new test-only symbol since HEAD.",
    )
    parser.add_argument(
        "--update-wire-first-baseline",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    if args.update_wire_first_baseline:
        print(
            "--update-wire-first-baseline is RETIRED. This gate has no "
            "baseline: it prints the real census every run and enforces an "
            "absolute zero (orphans) / diff-scoped-against-HEAD (test-only "
            "symbols) check, so there is nothing to freeze. See the module "
            "docstring.",
            file=sys.stderr,
        )
        return 2
    if args.wire_first_report:
        return wire_first_report()
    if args.check_test_collection:
        return _handle_check_test_collection()
    if args.check_mock_hygiene:
        return _handle_check_mock_hygiene()
    if args.check_extras_gating:
        return _handle_check_extras_gating()
    if args.check_symbol_reachability:
        return _handle_check_symbol_reachability()
    return _handle_reachability_report(args)


if __name__ == "__main__":
    sys.exit(main())
