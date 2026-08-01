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
  this can't silently drift from what those files actually run). Ratcheted
  against ``scripts/wire_first_baseline.json`` — new orphans fail, the
  existing backlog does not (see D-OB-13a).
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
  ``security/check_swallowed_errors.py``) against every OTHER file under
  ``agent_utilities/`` and every file under ``tests/``. A symbol referenced
  ONLY from test files (zero non-test, non-defining-file references) is a
  "public capability entrypoint with no non-test caller" — exactly the
  D-OB-9 shape (``PolicyEngine``, KV-fork ``snapshot``/``fork``/
  ``branch_get``/``branch_put``, ``AdmissionPolicy.decide``, …). Ratcheted
  against the same baseline file — new test-only symbols fail, the existing
  backlog does not.

Known blind spots of the symbol sweep specifically: word-boundary text
matching (not import/type resolution) means an alias (``import X as Y``)
or a same-named symbol in an unrelated module can produce false negatives
or false positives; a handful of extremely generic method names (``run``,
``get``, ``close``, …) are excluded from method-level checking for that
reason — rely on the class-level finding for those. A "0 non-test
references" result is signal for a human to trace the live path (Wire-First
step 1), not proof of dead code to delete on sight.

Usage::

    python scripts/check_wiring.py --wire-first-report          # everything, human-readable
    python scripts/check_wiring.py --check-test-collection
    python scripts/check_wiring.py --check-mock-hygiene
    python scripts/check_wiring.py --check-extras-gating
    python scripts/check_wiring.py --check-symbol-reachability
    python scripts/check_wiring.py --update-wire-first-baseline  # freeze current backlog
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import sys
import tokenize
from collections import Counter, defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "agent_utilities"
TESTS_DIR = ROOT / "tests"
PYPROJECT = ROOT / "pyproject.toml"
PYTEST_INI = ROOT / "pytest.ini"
PRECOMMIT_CONFIG = ROOT / ".pre-commit-config.yaml"
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
WIRE_FIRST_BASELINE = ROOT / "scripts" / "wire_first_baseline.json"

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


def _load_testpaths() -> list[str]:
    """Parse ``testpaths = ...`` out of ``pytest.ini`` (pytest.ini wins over
    ``pyproject.toml`` per pytest's own ini-file precedence, so that's the
    only file this needs to read to know what ``pytest`` actually collects
    by default)."""
    if not PYTEST_INI.exists():
        return []
    for line in PYTEST_INI.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*testpaths\s*=\s*(.+)$", line)
        if m:
            return m.group(1).split()
    return []


def _load_explicit_pytest_paths() -> set[str]:
    """``tests/...`` paths explicitly passed to ``pytest`` in a pre-commit
    hook ``entry:`` or a GitHub Actions ``run:`` step, outside of
    ``testpaths``. These are structurally collected even though
    ``testpaths`` doesn't cover them (e.g. ``tests/gates``, ``tests/docs``,
    ``tests/scale/test_prod_profile_guard.py``) — derived by regex over the
    hook/workflow files themselves so this can't hand-drift from what they
    actually run.
    """
    paths: set[str] = set()
    texts: list[str] = []
    if PRECOMMIT_CONFIG.exists():
        texts.append(PRECOMMIT_CONFIG.read_text(encoding="utf-8"))
    if WORKFLOWS_DIR.exists():
        for f in sorted(WORKFLOWS_DIR.glob("*.yml")):
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
    *, tests_dir: Path = TESTS_DIR, display_root: Path = ROOT
) -> list[str]:
    """Test files under ``tests/`` collected by NEITHER ``testpaths`` NOR an
    explicit pytest invocation in pre-commit/CI (D-OB-13a). ``tests_dir``/
    ``display_root`` are overridable (default: the real repo) so
    ``tests/gates/test_wire_first_gate.py`` can prove this trips on a
    synthetic fixture — "a gate that can't fail is not a gate"."""
    if not tests_dir.exists():
        return []
    collected = set(_load_testpaths()) | _load_explicit_pytest_paths()
    orphans = []
    for py in sorted(tests_dir.rglob("test_*.py")):
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
    for py in sorted(tests_dir.rglob("test_*.py")):
        rel = py.relative_to(display_root).as_posix()
        try:
            tree = ast.parse(
                py.read_text(encoding="utf-8", errors="ignore"), filename=rel
            )
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _is_mock_ctor(node):
                spec = _kw_value(node, "spec")
                if isinstance(spec, ast.List) and not spec.elts:
                    issues.append((rel, node.lineno, "spec=[]"))
            if _is_patch_call(node):
                create = _kw_value(node, "create")
                if isinstance(create, ast.Constant) and create.value is True:
                    issues.append((rel, node.lineno, "create=True"))
    return issues


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
    for py in sorted(tests_dir.rglob("test_*.py")):
        rel = py.relative_to(display_root).as_posix()
        try:
            source = py.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source, filename=rel)
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            type_names: set[str] = set()
            if isinstance(node.type, ast.Tuple):
                type_names = {t.id for t in node.type.elts if isinstance(t, ast.Name)}
            elif isinstance(node.type, ast.Name):
                type_names = {node.type.id}
            if not type_names & {"ImportError", "ModuleNotFoundError"}:
                continue
            body_mod = ast.Module(body=node.body, type_ignores=[])
            has_skip = any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in {"skip", "fail", "xfail"}
                for n in ast.walk(body_mod)
            )
            has_importorskip = "importorskip" in ast.dump(body_mod)
            has_raise = any(isinstance(n, ast.Raise) for n in ast.walk(body_mod))
            if not (has_skip or has_importorskip or has_raise):
                found.append((rel, node.lineno))
    return found


def _iter_agent_utilities_files(src_dir: Path = SRC_DIR) -> list[Path]:
    return [p for p in sorted(src_dir.rglob("*.py")) if "__pycache__" not in p.parts]


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


def _public_methods(tree: ast.Module) -> list[tuple[str, str, int]]:
    """``(class_name, method_name, lineno)`` for public methods of public
    top-level classes."""
    out: list[tuple[str, str, int]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
            continue
        for sub in node.body:
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and not (
                sub.name.startswith("_")
            ):
                out.append((node.name, sub.name, sub.lineno))
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
        if (
            t.type == tokenize.NAME
            and i >= 2
            and tokens[i - 1].type == tokenize.OP
            and tokens[i - 1].string == "."
            and i + 1 < len(tokens)
            and tokens[i + 1].type == tokenize.OP
            and tokens[i + 1].string == "("
        ):
            calls[t.string] += 1
    return idents, calls


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

    ``src_dir``/``tests_dir``/``display_root`` are overridable (default: the
    real repo) so ``tests/gates/test_wire_first_gate.py`` can prove this
    trips on a synthetic fixture.
    """
    au_sources: dict[str, str] = {}
    for py in _iter_agent_utilities_files(src_dir):
        rel = py.relative_to(display_root).as_posix()
        try:
            au_sources[rel] = py.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

    au_idents: dict[str, Counter[str]] = {}
    au_calls: dict[str, Counter[str]] = {}
    total_au_idents: Counter[str] = Counter()
    total_au_calls: Counter[str] = Counter()
    for rel, source in au_sources.items():
        idents, calls = _index_file(source)
        au_idents[rel] = idents
        au_calls[rel] = calls
        total_au_idents.update(idents)
        total_au_calls.update(calls)

    total_test_idents: Counter[str] = Counter()
    total_test_calls: Counter[str] = Counter()
    if tests_dir.exists():
        for py in tests_dir.rglob("*.py"):
            if "__pycache__" in py.parts:
                continue
            try:
                text = py.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            idents, calls = _index_file(text)
            total_test_idents.update(idents)
            total_test_calls.update(calls)

    findings: list[dict] = []
    for rel, source in au_sources.items():
        if rel.endswith("__init__.py"):
            continue
        try:
            tree = ast.parse(source, filename=rel)
        except SyntaxError:
            continue

        for kind, name, lineno in _public_top_level_defs(tree):
            other_au = total_au_idents.get(name, 0) - au_idents[rel].get(name, 0)
            same_file = au_idents[rel].get(name, 0) - 1  # minus the def line itself
            test_refs = total_test_idents.get(name, 0)
            if other_au == 0 and same_file <= 0 and test_refs > 0:
                findings.append(
                    {
                        "kind": kind,
                        "symbol": name,
                        "file": rel,
                        "line": lineno,
                        "test_refs": test_refs,
                    }
                )

        for cls_name, meth_name, m_lineno in _public_methods(tree):
            if meth_name in _GENERIC_METHOD_STOPLIST:
                continue
            other_au = total_au_calls.get(meth_name, 0) - au_calls[rel].get(
                meth_name, 0
            )
            same_file = au_calls[rel].get(meth_name, 0)
            test_refs = total_test_calls.get(meth_name, 0)
            if other_au == 0 and same_file <= 0 and test_refs > 0:
                findings.append(
                    {
                        "kind": "method",
                        "symbol": f"{cls_name}.{meth_name}",
                        "file": rel,
                        "line": m_lineno,
                        "test_refs": test_refs,
                    }
                )
    return findings


def _finding_key(entry: dict) -> str:
    return f"{entry['file']}:{entry['line']}:{entry['symbol']}"


def _load_wire_first_baseline() -> dict[str, list[str]]:
    if not WIRE_FIRST_BASELINE.exists():
        return {"orphaned_test_files": [], "test_only_symbols": []}
    return json.loads(WIRE_FIRST_BASELINE.read_text(encoding="utf-8"))


def _write_wire_first_baseline(orphans: list[str], symbols: list[dict]) -> None:
    WIRE_FIRST_BASELINE.write_text(
        json.dumps(
            {
                "_comment": (
                    "Frozen Wire-First backlog (ratchet, D-OB-9/13/16) — burn down "
                    "toward zero. New entries beyond this baseline fail "
                    "`check_wiring.py --check-test-collection` / "
                    "`--check-symbol-reachability`. Refresh with "
                    "`--update-wire-first-baseline` only after genuinely fixing or "
                    "deliberately accepting a new backlog item, never to silence a "
                    "regression."
                ),
                "orphaned_test_files": sorted(orphans),
                "test_only_symbols": sorted(_finding_key(e) for e in symbols),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _print_ratchet_result(
    label: str, current_keys: list[str], baseline_keys: list[str]
) -> bool:
    """Prints a swallowed-errors-gate-style ratchet report. Returns True if
    there is at least one NEW (unbaselined) entry."""
    baseline_set = set(baseline_keys)
    current_set = set(current_keys)
    new = sorted(current_set - baseline_set)
    fixed = sorted(baseline_set - current_set)
    if new:
        print(f"\n{label}: {len(new)} NEW finding(s) beyond baseline:")
        for k in new:
            print(f"  {k}")
    removed_note = f", {len(fixed)} fixed since baseline" if fixed else ""
    print(
        f"{label}: {len(current_set)} total "
        f"({len(baseline_set & current_set)} baselined{removed_note})."
    )
    return bool(new)


def wire_first_report(update_baseline: bool = False) -> int:
    orphans = find_orphaned_test_files()
    mock_issues = find_mock_hygiene_issues()
    silent_guards = find_silent_import_guards()
    symbols = find_test_only_symbols()

    if update_baseline:
        _write_wire_first_baseline(orphans, symbols)
        print(
            f"Wire-First baseline updated: {len(orphans)} orphaned test file(s), "
            f"{len(symbols)} test-only symbol(s) -> {WIRE_FIRST_BASELINE.name}"
        )
        return 0

    baseline = _load_wire_first_baseline()

    print("=" * 72)
    print("Wire-First gate report (D-OB-9 / D-OB-13 / D-OB-16)")
    print("=" * 72)

    had_new = _print_ratchet_result(
        "Uncollected test files (D-OB-13a)",
        orphans,
        baseline.get("orphaned_test_files", []),
    )
    had_new |= _print_ratchet_result(
        "Test-only public symbols / no non-test caller (D-OB-9)",
        [_finding_key(e) for e in symbols],
        baseline.get("test_only_symbols", []),
    )

    print(
        f"\nMock hygiene (D-OB-13b, informational — not ratcheted): "
        f"{len(mock_issues)} spec=[]/create=True site(s)."
    )
    for rel, line, shape in mock_issues[:20]:
        print(f"  {rel}:{line} [{shape}]")
    if len(mock_issues) > 20:
        print(f"  ... and {len(mock_issues) - 20} more")

    print(
        f"\nSilently-swallowed optional-extra import guards (D-OB-16, "
        f"informational — not ratcheted): {len(silent_guards)} site(s)."
    )
    for rel, line in silent_guards[:20]:
        print(f"  {rel}:{line}")
    if len(silent_guards) > 20:
        print(f"  ... and {len(silent_guards) - 20} more")

    print()
    if had_new:
        print(
            "FAIL — new Wire-First backlog beyond the frozen baseline. Fix it, or "
            "if genuinely accepted, refresh with --update-wire-first-baseline and "
            "say why in the commit message."
        )
        return 1
    print("OK — no NEW Wire-First backlog beyond the frozen baseline.")
    return 0


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
    parser.add_argument(
        "--check-test-collection",
        action="store_true",
        help="D-OB-13a: report test files under tests/ not collected by "
        "testpaths nor an explicit pre-commit/CI pytest invocation. "
        "Ratcheted against the Wire-First baseline (exit 1 on new orphans).",
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
        "Ratcheted against the Wire-First baseline (exit 1 on new entries).",
    )
    parser.add_argument(
        "--wire-first-report",
        action="store_true",
        help="Run all four Wire-First checks above and print a combined "
        "report. Exit 1 if either ratcheted check has a new backlog entry.",
    )
    parser.add_argument(
        "--update-wire-first-baseline",
        action="store_true",
        help="Freeze the CURRENT orphaned-test-file + test-only-symbol sets "
        "as the new baseline (scripts/wire_first_baseline.json). Only after "
        "genuinely fixing or deliberately accepting a new backlog item.",
    )
    args = parser.parse_args()

    if args.update_wire_first_baseline:
        return wire_first_report(update_baseline=True)

    if args.wire_first_report:
        return wire_first_report()

    if args.check_test_collection:
        orphans = find_orphaned_test_files()
        baseline = _load_wire_first_baseline().get("orphaned_test_files", [])
        had_new = _print_ratchet_result(
            "Uncollected test files (D-OB-13a)", orphans, baseline
        )
        return 1 if had_new else 0

    if args.check_mock_hygiene:
        issues = find_mock_hygiene_issues()
        print(f"Mock hygiene: {len(issues)} spec=[]/create=True site(s).")
        for rel, line, shape in issues:
            print(f"  {rel}:{line} [{shape}]")
        return 0

    if args.check_extras_gating:
        guards = find_silent_import_guards()
        print(f"Silent optional-extra import guards: {len(guards)} site(s).")
        for rel, line in guards:
            print(f"  {rel}:{line}")
        return 0

    if args.check_symbol_reachability:
        symbols = find_test_only_symbols()
        baseline = _load_wire_first_baseline().get("test_only_symbols", [])
        had_new = _print_ratchet_result(
            "Test-only public symbols / no non-test caller (D-OB-9)",
            [_finding_key(e) for e in symbols],
            baseline,
        )
        if had_new:
            print("\nFull current findings (file:line kind symbol test_refs):")
            for e in symbols:
                print(
                    f"  {e['file']}:{e['line']} {e['kind']} {e['symbol']} "
                    f"(referenced in {e['test_refs']} test location(s))"
                )
        return 1 if had_new else 0

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
    far = sorted((m, d) for m, d in dist.items() if d > args.max_hops)

    if args.json:
        print(
            json.dumps(
                {
                    "roots": sorted(roots),
                    "total_modules": len(modules),
                    "reachable": len(dist),
                    "unreachable": unreachable,
                    "beyond_max_hops": [{"module": m, "hops": d} for m, d in far],
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
