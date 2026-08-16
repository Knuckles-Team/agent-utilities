#!/usr/bin/env python3
"""Log-exception-redaction contract gate (loggers outside the privacy boundary).

**This gate exists because of what it does NOT need to check.** Every
``agent_utilities.*``-namespaced logger already passes through a process-wide
privacy boundary: ``agent_utilities/core/log_privacy.py``'s
``install_log_privacy_boundary()``, called unconditionally at package import
time (``agent_utilities/__init__.py``). Its ``LogRecord`` factory sanitizes
``record.msg``/``record.args`` for any logger whose ``.name`` is
``"agent_utilities"`` or starts with ``"agent_utilities."`` — stripping
endpoint URLs, filesystem paths, host:port pairs, and email-shaped callers
via regex, and nulling ``exc_info``/``exc_text``/``stack_info`` outright. For
a caught exception specifically, ``_sanitize_value`` renders it as
``f"{type(exc).__name__}: {sanitize_log_text(str(exc))}"`` — the message
survives, sanitized, not collapsed to the bare type. That is also why
``scripts/check_swallowed_errors.py`` (the OTHER exception-logging gate in
this repo) treats ``type(exc).__name__``-only logging as a violation to
flag, not a fix: collapsing to the type name throws away information the
privacy boundary has already made safe to keep. **A gate that told call
sites under that boundary to hand-redact their exception text would
contradict an established, tested, gate-enforced convention** — see
``tests/unit/test_gateway_daemon_identity.py::
test_sink_install_failure_logs_the_real_cause_not_just_its_type``, which
asserts the real failure message survives verbatim (minus any
path/host/email substrings) through exactly this boundary.

So this gate's job is narrower and more useful: catch the loggers that fall
OUTSIDE that boundary — a logger built with an explicit name that does not
start with ``"agent_utilities."`` (or the bare root logger,
``logging.getLogger()``) never reaches ``install_log_privacy_boundary``'s
factory filter, so a raw caught-exception's text reaches the sink verbatim,
untouched by any redaction. In this repository's network-facing surface
(``agent_utilities/mcp``, ``agent_utilities/gateway``, ``agent_utilities/server``)
exactly two such loggers exist today: ``"mcp_multiplexer"``
(``agent_utilities/mcp/multiplexer.py``) and ``"mcp_multiplexer.child"``
(``agent_utilities/mcp/child_resilience.py``) — both chosen, evidently, to
group MCP-child/multiplexer diagnostics under their own namespace rather
than the package's, which is a legitimate operational choice but has the
side effect of opting out of the privacy boundary. The established,
already-applied fix at the one such site this lane could edit
(``child_resilience.py``) is ``type(exc).__name__`` plus
``redact_for_log(exc)`` (``agent_utilities/security/log_redaction.py`` — a
deterministic, non-reversible hash tag, the SAME helper demonstrated at
``agent_utilities/knowledge_graph/ontology/document_processing.py``'s
``_verbatim_source_text`` for a raw ``path``) — giving back the type and a
stable correlation tag without the raw text a privacy-boundary-covered
logger would have gotten sanitized automatically.

**What counts as an exception-like name in scope** (a static heuristic, not a
dataflow prover, matching every other ``scripts/check_*.py`` gate in this
repo):

* ``except <Type> as name:`` — ``name`` is exception-like for that handler's
  body only (mirrors Python's own implicit ``del name`` at handler exit).
* ``name = <expr>.exception()`` — the ``asyncio.Task``/``Future`` idiom for
  retrieving a task's exception outside an ``except`` block (e.g.
  ``exc = task.exception()``) — ``name`` is exception-like for the rest of
  the enclosing scope from that assignment forward.

**What counts as a logger outside the privacy boundary** — a module-level
``<name> = logging.getLogger(<arg>)`` (or ``getLogger(<arg>)``) assignment
where ``<arg>`` is:

* absent (the bare root logger), or
* a string literal that is not ``"agent_utilities"`` and does not start with
  ``"agent_utilities."``, or
* anything else this gate cannot statically resolve to a literal (a dynamic
  expression) — treated conservatively as OUTSIDE the boundary, since this
  gate cannot prove otherwise.

``logging.getLogger(__name__)`` is treated as INSIDE the boundary: every
candidate file lives under the ``agent_utilities`` package, so its
``__name__`` always starts with ``"agent_utilities."``. A log call is only
in scope for this gate when its receiver is a bare ``Name`` resolving to a
logger variable found OUTSIDE the boundary by this analysis — ``self.logger``,
an imported logger, or any other call shape this gate cannot trace to a
local ``getLogger(...)`` assignment is simply not scanned (a known,
documented false-negative surface, not a soundness claim).

**What counts as safe** even on an out-of-boundary logger (does not flag
even though it references the exception-like name): ``type(name).__name__``,
``name.__class__.__name__``, and ``redact_for_log(name)`` (or any call to
``redact_for_log`` that contains ``name`` anywhere in its argument — the
whole point of that helper is to consume the sensitive value and emit
something safe).

**Fail-closed vs. an honest absence** (the same distinction
``check_cypher_write_subset.py`` documents, itself following
``run_contract_checks`` in ``agent_utilities/governance/merge_queue.py``):
this gate raises ``LogExceptionRedactionGateError`` — and ``main()`` exits 1
— if it cannot read the tree it was asked to check (the repository root is
absent, or a candidate file can't be read/decoded/parsed). It exits 0,
reporting the count, when the scan completes and finds zero NEW violations —
a clean scan is success, not something to be suspicious of. A target
subdirectory that does not exist under the given root is a legitimate empty,
not a degraded read: it contributes zero candidate files and is never
treated as an error.

**A ratchet baseline** (``log_exception_redaction_baseline.txt``, next to
this script) covers the sites this lane found but could NOT fix without
colliding with another in-flight lane: 15 sites in
``agent_utilities/mcp/multiplexer.py`` (owned by the MCP-session-isolation
lane, logger ``"mcp_multiplexer"`` — genuinely outside the privacy boundary,
not a false positive). Registered as ``D-LR-1`` in the deferred registry —
frozen here rather than either silently ignored or used to block every
future merge. Only a violation NOT already in the baseline fails the gate;
fixing a baselined site shrinks the baseline on the next
``--update-baseline`` run.

D-MT-2: the baseline key was ``(path, line)`` — the same line-anchor
fragility ``check_cypher_write_subset.py``'s ratchet baseline had (D-F6-2)
and its ``_ALLOWLIST`` had before that (D-MW-11): any unrelated commit
landing above a baselined site shifts its line and the gate reports a false
"NEW violation". This baseline happens to be empty as of this writing (its
15 sites were fixed), but the mechanism is the landmine, not the current
entry count — the next lane to add an entry here would inherit the same
churn. The key is now ``(path, symbol, logger_name, content_hash)`` —
matching ``check_cypher_write_subset.py``'s D-F6-2 fix exactly (the same
convention, reused rather than a second scheme): ``symbol`` is the
enclosing function's dotted qualname from a manual parent-tracking DFS
(mirrors that gate's ``_scope_units`` / ``check_swallowed_errors.py``'s
``_iter_except_handlers_with_scope``, D-SWG-1), ``logger_name`` narrows to
the specific out-of-boundary logger, and ``content_hash`` is a hash of the
flagged snippet's own text — content, not position, disambiguates two
violations on the same logger in the same symbol, and is unaffected by
unrelated violations elsewhere in that symbol appearing/disappearing.

Usage:
  python3 scripts/security/check_log_exception_redaction.py
  python3 scripts/security/check_log_exception_redaction.py --repository-root DIR
  python3 scripts/security/check_log_exception_redaction.py --self-check
  python3 scripts/security/check_log_exception_redaction.py --update-baseline

Exit 0 = no NEW violation (or self-check passed too), 1 = a new violation was
found or the gate could not read the tree.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from scripts._git_scan import tracked_or_walked  # noqa: E402

#: Same method set as
#: ``tests/unit/security/test_log_location_privacy_static_gate.py``, for the
#: same reason: these are the stdlib ``logging.Logger`` methods (plus
#: ``log()``) that emit a record from their arguments.
_LOG_METHODS = {
    "critical",
    "debug",
    "error",
    "exception",
    "info",
    "log",
    "warning",
}

#: The network-facing surface this lane swept. Relative to
#: ``--repository-root``.
_DEFAULT_TARGET_DIRS = (
    Path("agent_utilities/mcp"),
    Path("agent_utilities/gateway"),
    Path("agent_utilities/server"),
)

_SHAPE = "raw_exception_in_log"

#: The one prefix ``install_log_privacy_boundary`` filters on
#: (``agent_utilities/core/log_privacy.py``).
_PRIVACY_BOUNDARY_PREFIX = "agent_utilities"


class LogExceptionRedactionGateError(RuntimeError):
    """The gate could not read the tree it was asked to check (fail-closed)."""


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    logger_name: str
    snippet: str
    #: D-MT-2: enclosing function/method's dotted qualname (or
    #: ``"<module>"``), used for the content-anchored baseline key instead
    #: of ``line`` — see ``_baseline_key``.
    symbol: str = "<module>"
    #: A hash of the flagged snippet's own text — disambiguates two
    #: violations on the same logger in the same ``symbol`` by CONTENT, the
    #: same convention ``check_cypher_write_subset.py`` uses (D-F6-2).
    content_hash: str = ""

    def render(self) -> str:
        return f"{self.path}:{self.line} [{_SHAPE}] logger={self.logger_name!r} {self.snippet}"


def _is_type_dunder_name(node: ast.AST, name: str) -> bool:
    """``type(name).__name__`` or ``name.__class__.__name__``."""
    if not (isinstance(node, ast.Attribute) and node.attr == "__name__"):
        return False
    value = node.value
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "type"
        and len(value.args) == 1
        and isinstance(value.args[0], ast.Name)
        and value.args[0].id == name
    ):
        return True
    return bool(
        isinstance(value, ast.Attribute)
        and value.attr == "__class__"
        and isinstance(value.value, ast.Name)
        and value.value.id == name
    )


def _is_redact_call(node: ast.AST, name: str) -> bool:
    """``redact_for_log(...)`` where ``name`` appears anywhere in its args —
    the helper consumes the sensitive value and returns a safe tag, so
    anything inside its call is exempt regardless of shape."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
        return False
    if node.func.id != "redact_for_log":
        return False
    for arg in node.args:
        for sub in ast.walk(arg):
            if isinstance(sub, ast.Name) and sub.id == name:
                return True
    for kw in node.keywords:
        for sub in ast.walk(kw.value):
            if isinstance(sub, ast.Name) and sub.id == name:
                return True
    return False


def _contains_unsafe_reference(node: ast.AST, name: str) -> bool:
    """True if ``node`` references ``name`` OUTSIDE a safe wrapper.

    A safe wrapper's subtree is never descended into, so a reference to
    ``name`` inside ``type(name).__name__`` or ``redact_for_log(name, ...)``
    does not itself count.
    """
    if _is_type_dunder_name(node, name) or _is_redact_call(node, name):
        return False
    if isinstance(node, ast.Name) and node.id == name:
        return True
    for child in ast.iter_child_nodes(node):
        if _contains_unsafe_reference(child, name):
            return True
    return False


def _is_task_exception_call(node: ast.AST) -> bool:
    """``<expr>.exception()`` — the asyncio Task/Future idiom for pulling a
    completed task's exception outside an ``except`` block."""
    return bool(
        isinstance(node, ast.Call)
        and not node.args
        and not node.keywords
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "exception"
    )


def _is_getlogger_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "getLogger"
    if isinstance(func, ast.Attribute):
        return func.attr == "getLogger"
    return False


def _logger_coverage(tree: ast.Module) -> dict[str, bool]:
    """Map logger variable name -> ``True`` if it is INSIDE the
    ``agent_utilities.*`` privacy boundary, scanning every
    ``<name> = logging.getLogger(<arg>)`` assignment in the module (any
    scope — a nested/local logger is rare but not impossible)."""
    coverage: dict[str, bool] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Name) and _is_getlogger_call(node.value)):
            continue
        call = node.value
        if not isinstance(call, ast.Call):  # narrows for the type checker
            continue
        if not call.args:
            coverage[target.id] = False  # bare root logger
            continue
        arg = call.args[0]
        if isinstance(arg, ast.Name) and arg.id == "__name__":
            coverage[target.id] = True
            continue
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            coverage[target.id] = arg.value == _PRIVACY_BOUNDARY_PREFIX or (
                arg.value.startswith(_PRIVACY_BOUNDARY_PREFIX + ".")
            )
            continue
        coverage[target.id] = False  # unresolvable — conservative: out of boundary
    return coverage


def _log_call_violation(call: ast.Call, exc_names: frozenset[str]) -> str | None:
    """Return a snippet if ``call`` is a log call that unsafely references
    one of ``exc_names``, else ``None``. Caller has already established the
    receiver is an out-of-boundary logger variable."""
    if not (isinstance(call.func, ast.Attribute) and call.func.attr in _LOG_METHODS):
        return None
    if not exc_names:
        return None
    values: list[ast.expr] = list(call.args) + [kw.value for kw in call.keywords]
    for value in values:
        for name in exc_names:
            if _contains_unsafe_reference(value, name):
                try:
                    return ast.unparse(value)[:160]
                except Exception:  # noqa: BLE001 — best-effort snippet only
                    return "<unparsable>"
    return None


def _out_of_boundary_receiver(call: ast.Call, coverage: dict[str, bool]) -> str | None:
    """If ``call`` is ``<logger_var>.<method>(...)`` where ``logger_var`` is
    a KNOWN, out-of-boundary logger, return its name; else ``None``. A
    receiver this gate cannot trace to a local ``getLogger(...)`` assignment
    (``self.logger``, an imported logger, ...) is not scanned — see the
    module docstring's documented false-negative surface."""
    if not isinstance(call.func, ast.Attribute):
        return None
    receiver = call.func.value
    if not isinstance(receiver, ast.Name):
        return None
    if coverage.get(receiver.id) is False:
        return receiver.id
    return None


def _walk_scope(
    body: list[ast.stmt],
    exc_names: frozenset[str],
    coverage: dict[str, bool],
    violations: list[tuple[int, str, str]],
) -> None:
    """Walk statements IN SOURCE ORDER, threading a growing set of
    exception-like names through sibling statements — the same "track
    locals in source order" idiom ``check_cypher_write_subset.py`` uses for
    its ``local_strings`` dict, adapted to a name-safety set. Does NOT
    descend into a nested function/async-function: it gets its own fresh,
    empty scope from ``_iter_scopes`` instead."""
    current = set(exc_names)
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue  # own scope; handled by _iter_scopes
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            logger_name = _out_of_boundary_receiver(node, coverage)
            if logger_name is None:
                continue
            snippet = _log_call_violation(node, frozenset(current))
            if snippet is not None:
                violations.append((node.lineno, logger_name, snippet))
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name) and _is_task_exception_call(stmt.value):
                current.add(target.id)
        if isinstance(stmt, ast.Try):
            _walk_scope(stmt.body, frozenset(current), coverage, violations)
            for handler in stmt.handlers:
                handler_names = set(current)
                if handler.name:
                    handler_names.add(handler.name)
                _walk_scope(
                    handler.body, frozenset(handler_names), coverage, violations
                )
            _walk_scope(stmt.orelse, frozenset(current), coverage, violations)
            _walk_scope(stmt.finalbody, frozenset(current), coverage, violations)
        else:
            for field in ("body", "orelse", "finalbody"):
                nested = getattr(stmt, field, None)
                if (
                    isinstance(nested, list)
                    and nested
                    and isinstance(nested[0], ast.stmt)
                ):
                    _walk_scope(nested, frozenset(current), coverage, violations)


def _iter_scopes(tree: ast.Module) -> list[tuple[list[ast.stmt], str]]:
    """(scope body, dotted qualname) for the module and every
    function/async-function.

    D-MT-2: a manual parent-tracking DFS (mirrors
    ``check_cypher_write_subset.py``'s ``_scope_units``, D-F6-2) rather than
    a flat ``ast.walk`` — gives each scope a stable qualname
    (``"<module>"``, ``"f"``, ``"ClassName.method"``) for the
    content-anchored baseline key, with no dependence on line position.
    """
    scopes: list[tuple[list[ast.stmt], str]] = [(tree.body, "<module>")]

    def walk(node: ast.AST, scope_stack: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = ".".join(scope_stack + (child.name,))
                scopes.append((child.body, qualname))
                walk(child, scope_stack + (child.name,))
            elif isinstance(child, ast.ClassDef):
                walk(child, scope_stack + (child.name,))
            else:
                walk(child, scope_stack)

    walk(tree, ())
    return scopes


def _find_violations(rel: str, tree: ast.Module) -> list[Violation]:
    coverage = _logger_coverage(tree)
    if not any(covered is False for covered in coverage.values()):
        return []  # every logger in this module is inside the boundary
    raw: list[tuple[int, str, str, str]] = []  # (lineno, logger_name, snippet, symbol)
    for scope_body, symbol in _iter_scopes(tree):
        scope_raw: list[tuple[int, str, str]] = []
        _walk_scope(scope_body, frozenset(), coverage, scope_raw)
        raw.extend(
            (lineno, logger_name, snippet, symbol)
            for lineno, logger_name, snippet in scope_raw
        )
    # A single log call can carry more than one unsafe reference (e.g. both
    # a positional `exc` and a keyword `exc_info=exc`) but is ONE violation
    # site — dedupe by (symbol, line), keep the first snippet found, in
    # discovery order (each scope's own statements are visited in source
    # order, and scopes themselves are visited module-first then DFS order).
    seen: dict[tuple[str, int], tuple[str, str]] = {}
    order: list[tuple[str, int]] = []
    for lineno, logger_name, snippet, symbol in raw:
        key = (symbol, lineno)
        if key not in seen:
            seen[key] = (logger_name, snippet)
            order.append(key)
    violations: list[Violation] = []
    for symbol, lineno in order:
        logger_name, snippet = seen[(symbol, lineno)]
        #: D-MT-2: content_hash over (logger_name, snippet) — the same
        #: "hash the flagged content" convention
        #: ``check_cypher_write_subset.py`` uses (D-F6-2), reused rather
        #: than inventing a second scheme.
        content_hash = hashlib.sha256(
            f"{logger_name}\x00{snippet}".encode()
        ).hexdigest()[:16]
        violations.append(
            Violation(rel, lineno, logger_name, snippet, symbol, content_hash)
        )
    return sorted(violations, key=lambda v: v.line)


def _candidate_files(root: Path, target_dirs: tuple[Path, ...]) -> list[Path]:
    """``*.py`` files under any of ``target_dirs`` (resolved under ``root``).

    A target directory that does not exist contributes nothing and is not an
    error (see the module docstring's fail-closed/honest-absence section);
    ``root`` itself not existing IS an error, raised by the caller before
    this is reached.
    """
    root = root.resolve()
    files: list[Path] = []
    for rel_dir in target_dirs:
        d = root / rel_dir
        if not d.is_dir():
            continue
        files.extend(_tracked_or_walked_py_files(d, root))
    return sorted(set(files))


def _tracked_or_walked_py_files(target_dir: Path, root: Path) -> list[Path]:
    """``.py`` files under ``target_dir``, preferring git-tracked (BUG-043).

    A raw ``rglob`` also picks up gitignored, generated build output, which
    can carry a stale, already-fixed raw-exception log-site violation. Falls
    back to a filesystem walk only when ``target_dir`` is not inside a git
    working tree (e.g. a synthetic test fixture). ``root`` is the resolved
    ``--repository-root`` this call's ``target_dir`` was resolved under --
    the anchor for git's ambient-env-immune pathspec, per
    ``scripts/_git_scan.py``.
    """
    return sorted(tracked_or_walked(target_dir, "*.py", root=root))


def scan(
    root: Path, target_dirs: tuple[Path, ...] = _DEFAULT_TARGET_DIRS
) -> list[Violation]:
    """Scan ``root``'s network-facing surface for raw-exception log sites on
    loggers that fall OUTSIDE the ``agent_utilities.*`` privacy boundary.

    Fail-closed: raises :class:`LogExceptionRedactionGateError` if ``root``
    doesn't exist, or if a candidate file can't be read/decoded/parsed.
    """
    if not root.is_dir():
        raise LogExceptionRedactionGateError(
            f"repository root is not a directory: {root}"
        )
    violations: list[Violation] = []
    for path in _candidate_files(root, target_dirs):
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise LogExceptionRedactionGateError(
                f"could not read candidate file {path}: {exc}"
            ) from exc
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            raise LogExceptionRedactionGateError(
                f"could not parse candidate file {path}: {exc}"
            ) from exc
        try:
            rel = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = path.as_posix()
        violations.extend(_find_violations(rel, tree))
    return sorted(violations, key=lambda v: (v.path, v.line))


#: See the module docstring's "A ratchet baseline" section: the multiplexer
#: sites this lane could not touch without colliding with a concurrent lane.
BASELINE = Path(__file__).resolve().parent / "log_exception_redaction_baseline.txt"

_BASELINE_SEP = "\t"


#: D-MT-2: (path, enclosing symbol, logger_name, content_hash) — NOT line.
#: Stable under unrelated code landing above/below the flagged call, AND
#: under unrelated violations appearing/disappearing elsewhere in the same
#: symbol (content_hash, not position, disambiguates). Mirrors
#: ``check_cypher_write_subset.py``'s ``BaselineKey`` (D-F6-2).
BaselineKey = tuple[str, str, str, str]


def _baseline_key(v: Violation) -> BaselineKey:
    return (v.path, v.symbol, v.logger_name, v.content_hash)


def _load_baseline() -> set[BaselineKey]:
    if not BASELINE.exists():
        return set()
    out: set[BaselineKey] = set()
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(_BASELINE_SEP)
        if len(fields) < 4 or not fields[3]:
            continue
        out.add((fields[0], fields[1], fields[2], fields[3]))
    return out


def _write_baseline(violations: list[Violation]) -> None:
    lines = [
        f"{v.path}{_BASELINE_SEP}{v.symbol}{_BASELINE_SEP}{v.logger_name}"
        f"{_BASELINE_SEP}{v.content_hash}{_BASELINE_SEP}# line {v.line}: {v.snippet}"
        for v in sorted(violations, key=lambda v: (v.path, v.symbol, v.line))
    ]
    BASELINE.write_text(
        "# Frozen baseline of pre-existing raw-exception log violations this\n"
        "# lane (D-W2-5, log-redaction sweep) found in\n"
        '# agent_utilities/mcp/multiplexer.py (logger "mcp_multiplexer", OUTSIDE\n'
        "# the agent_utilities.* privacy boundary installed by\n"
        "# agent_utilities/core/log_privacy.py — see this script's module\n"
        "# docstring) but could not fix without colliding with the concurrent\n"
        "# MCP-session-isolation lane that owns that file. Registered as D-LR-1\n"
        "# in the deferred registry — a ratchet, not a permanent exemption:\n"
        "# burn down toward zero once that lane lands. TAB-separated:\n"
        "# path\\tsymbol\\tlogger_name\\tcontent_hash\\t# line N: snippet (D-MT-2:\n"
        "# keyed on the enclosing symbol + logger + a hash of the flagged\n"
        "# snippet text, NOT line -- the trailing 'line N' in the comment is\n"
        "# informational only, not parsed, and WILL drift as the codebase\n"
        "# changes -- expected and harmless). A NEW entry (not here) fails\n"
        "# check_log_exception_redaction.py. Fixing a baselined site shrinks\n"
        "# this file on the next --update-baseline; never hand-add an entry\n"
        "# to make a real violation disappear.\n"
        + "\n".join(lines)
        + ("\n" if lines else ""),
        encoding="utf-8",
    )


def check(root: Path) -> dict[str, Any]:
    violations = scan(root)
    baseline = _load_baseline()
    new = [v for v in violations if _baseline_key(v) not in baseline]
    if new:
        raise LogExceptionRedactionGateError(
            "NEW raw-exception log violation(s) found (not in "
            f"{BASELINE.name}):\n" + "\n".join(f"  {v.render()}" for v in new)
        )
    current_keys = {_baseline_key(v) for v in violations}
    fixed = len(baseline - current_keys)
    return {
        "ok": True,
        "newViolations": 0,
        "totalInScope": len(violations),
        "baselined": len(current_keys),
        "fixedSinceBaseline": fixed,
    }


# ---------------------------------------------------------------------------
# Self-check: prove the gate actually catches each known-bad shape, and
# doesn't flag the safe idioms — including the majority case (a
# privacy-boundary-covered logger, which must NEVER be flagged, matching
# check_swallowed_errors.py's opposite-direction rule for that case).
# ---------------------------------------------------------------------------

_GOOD_FIXTURE = """
import logging
logger = logging.getLogger(__name__)

def f():
    try:
        risky()
    except Exception as exc:
        logger.warning("op failed: %s", exc)

def g():
    try:
        risky()
    except Exception as exc:
        logger.error("op failed", exc_info=True)
"""

_BAD_FIXTURES: dict[str, str] = {
    "root_logger_positional": """
import logging
logger = logging.getLogger()

def f():
    try:
        risky()
    except Exception as exc:
        logger.warning("op failed: %s", exc)
""",
    "named_logger_fstring": """
import logging
logger = logging.getLogger("mcp_multiplexer")

def f():
    try:
        risky()
    except Exception as exc:
        logger.warning(f"op failed: {exc}")
""",
    "named_logger_exc_info_kwarg": """
import logging
logger = logging.getLogger("mcp_multiplexer.child")

def f():
    try:
        risky()
    except Exception as exc:
        logger.error("op failed", exc_info=exc)
""",
    "named_logger_task_exception_idiom": """
import logging
logger = logging.getLogger("mcp_multiplexer")

def f(task):
    exc = task.exception()
    if exc is not None:
        logger.error("task failed: %s", exc)
""",
}

_SAFE_ON_UNCOVERED_LOGGER = """
import logging
logger = logging.getLogger("mcp_multiplexer")

def f():
    try:
        risky()
    except Exception as exc:
        logger.warning("op failed (%s: %s)", type(exc).__name__, redact_for_log(exc))
"""


def self_check() -> None:
    """Prove: (1) each known-bad shape (an out-of-boundary logger with a raw
    exception reference) is caught, (2) an ``agent_utilities.*``-covered
    logger is NEVER flagged even with a raw exception reference — the
    central distinction this gate exists to draw, (3) the safe idioms on an
    UNcovered logger are not flagged, (4) a baselined site is tolerated
    while a different line is not, (5) a missing root fails closed, (6) an
    honest absence (no target subdirectories) scans clean."""
    good_tree = ast.parse(_GOOD_FIXTURE)
    good_hits = _find_violations("fixture_good.py", good_tree)
    if good_hits:
        raise LogExceptionRedactionGateError(
            "self-check: a privacy-boundary-covered logger was flagged: "
            f"{[v.render() for v in good_hits]}"
        )

    for shape, fixture in _BAD_FIXTURES.items():
        tree = ast.parse(fixture)
        hits = _find_violations(f"fixture_{shape}.py", tree)
        if not hits:
            raise LogExceptionRedactionGateError(
                f"self-check: known-bad shape {shape!r} was NOT caught"
            )

    safe_tree = ast.parse(_SAFE_ON_UNCOVERED_LOGGER)
    safe_hits = _find_violations("fixture_safe.py", safe_tree)
    if safe_hits:
        raise LogExceptionRedactionGateError(
            "self-check: type()/redact_for_log() on an out-of-boundary "
            f"logger was still flagged: {[v.render() for v in safe_hits]}"
        )

    # Fail-closed: a nonexistent root must raise, not report a clean scan.
    missing = Path("/nonexistent-log-exception-redaction-gate-fixture-root")
    try:
        scan(missing)
    except LogExceptionRedactionGateError:
        pass
    else:
        raise LogExceptionRedactionGateError(
            "self-check: scanning a nonexistent root did not fail closed"
        )

    # The ratchet baseline: a violation present in the baseline is
    # tolerated (burn-down, not a block), but a violation in a DIFFERENT
    # symbol is treated as genuinely new — even though both happen to
    # render at a line, the key never looks at it.
    baselined = Violation("fixture_baselined.py", 10, "mcp_multiplexer", "x", "f")
    new_one = Violation("fixture_baselined.py", 20, "mcp_multiplexer", "y", "g")
    baseline_keys = {_baseline_key(baselined)}
    still_new = [
        v for v in (baselined, new_one) if _baseline_key(v) not in baseline_keys
    ]
    if still_new != [new_one]:
        raise LogExceptionRedactionGateError(
            "self-check: baseline ratchet did not distinguish a baselined "
            f"site from a new one (got {still_new!r})"
        )

    # D-MT-2: the SAME baselined site's key must be byte-identical whether
    # or not unrelated code has landed above it in the file (no line-number
    # anchor to drift).
    fixture = _BAD_FIXTURES["named_logger_fstring"]
    unpadded_tree = ast.parse(fixture)
    unpadded_hits = _find_violations("fixture_d_mt_2.py", unpadded_tree)
    padded_source = ("\n" * 7) + fixture
    padded_tree = ast.parse(padded_source)
    padded_hits = _find_violations("fixture_d_mt_2.py", padded_tree)
    if not unpadded_hits or not padded_hits:
        raise LogExceptionRedactionGateError(
            "self-check: D-MT-2 fixture produced no violations to compare"
        )
    if padded_hits[0].line == unpadded_hits[0].line:
        raise LogExceptionRedactionGateError(
            "self-check: D-MT-2 padding fixture did not actually move the "
            "violation's line -- the drift-survival proof is vacuous"
        )
    if _baseline_key(unpadded_hits[0]) != _baseline_key(padded_hits[0]):
        raise LogExceptionRedactionGateError(
            "self-check: baseline key changed when unrelated lines landed "
            f"above the site (line-anchor drift survived the D-MT-2 fix): "
            f"{_baseline_key(unpadded_hits[0])!r} != "
            f"{_baseline_key(padded_hits[0])!r}"
        )

    # An honest absence: a root with none of the target subdirectories
    # present scans clean (zero candidate files), not an error.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        empty_root = Path(tmp)
        empty_violations = scan(empty_root)
        if empty_violations:
            raise LogExceptionRedactionGateError(
                "self-check: a root with no target subdirectories produced "
                f"violations: {[v.render() for v in empty_violations]}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check-log-exception-redaction")
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Freeze the current scan as the new ratchet baseline and exit 0.",
    )
    args = parser.parse_args(argv)
    if args.update_baseline:
        try:
            violations = scan(args.repository_root)
        except LogExceptionRedactionGateError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
            return 1
        _write_baseline(violations)
        print(
            json.dumps(
                {"ok": True, "baselineUpdated": True, "entries": len(violations)},
                sort_keys=True,
            )
        )
        return 0
    try:
        if args.self_check:
            self_check()
        report = check(args.repository_root)
        if args.self_check:
            report["selfCheck"] = True
    except LogExceptionRedactionGateError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    except Exception as exc:  # noqa: BLE001 — gate boundary: any unexpected failure must fail closed (exit 1), never be swallowed into a false "clean scan"
        print(
            json.dumps(
                {"ok": False, "error": f"unexpected {type(exc).__name__}: {exc}"},
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
